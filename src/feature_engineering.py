from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return upper, ma, lower


@dataclass(frozen=True)
class FeatureConfig:
    ma_windows: tuple[int, ...] = (10, 20, 50, 100)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    boll_window: int = 20
    boll_std: float = 2.0
    rolling_std_window: int = 20
    return_windows: tuple[int, ...] = (1, 7, 30, 60)
    vol_ma_window: int = 20


def add_technical_indicators(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Expects columns: Date, Open, High, Low, Close, Adj_Close, Volume
    Returns a new DataFrame with indicator columns added.
    """
    cfg = cfg or FeatureConfig()
    out = df.copy()
    out = out.sort_values("Date").reset_index(drop=True)

    close = out["Adj_Close"] if "Adj_Close" in out.columns else out["Close"]

    for w in cfg.ma_windows:
        out[f"MA{w}"] = close.rolling(w).mean()

    out["RSI14"] = rsi(close, period=cfg.rsi_period)

    macd_line, signal_line, hist = macd(close, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal)
    out["MACD"] = macd_line
    out["MACD_signal"] = signal_line
    out["MACD_hist"] = hist

    bb_u, bb_m, bb_l = bollinger(close, window=cfg.boll_window, num_std=cfg.boll_std)
    out["BB_upper"] = bb_u
    out["BB_mid"] = bb_m
    out["BB_lower"] = bb_l
    out["BB_width"] = (bb_u - bb_l) / bb_m.replace(0, np.nan)

    out["RollingStd20"] = close.rolling(cfg.rolling_std_window).std()

    # Returns
    out["ret_1d"] = close.pct_change(1)
    for w in cfg.return_windows:
        if w == 1:
            continue
        out[f"ret_{w}d"] = close.pct_change(w)

    # Volume
    out["vol_change"] = out["Volume"].pct_change(1)
    out["vol_ma20"] = out["Volume"].rolling(cfg.vol_ma_window).mean()
    out["vol_vs_ma20"] = out["Volume"] / out["vol_ma20"].replace(0, np.nan)

    # Price range features
    out["hl_range"] = (out["High"] - out["Low"]) / close.replace(0, np.nan)
    out["oc_change"] = (out["Close"] - out["Open"]) / out["Open"].replace(0, np.nan)

    # Clean infs
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def make_label(
    df: pd.DataFrame,
    horizon_days: int,
    threshold_return: float,
    price_col: str = "Adj_Close",
) -> pd.Series:
    """
    Label = 1 if (future price / current price - 1) >= threshold_return within horizon_days.
    Uses *close-to-close* return at exact horizon (not intrahorizon max).
    """
    if price_col not in df.columns:
        price_col = "Close"
    px = df[price_col]
    fut = px.shift(-horizon_days)
    fut_ret = (fut / px) - 1.0
    return (fut_ret >= threshold_return).astype(int)


def current_indicator_snapshot(df_with_indicators: pd.DataFrame) -> dict[str, float]:
    """
    Returns latest available (non-null) values for a small set of indicators.
    """
    cols = [
        "MA10",
        "MA20",
        "MA50",
        "MA100",
        "RSI14",
        "MACD",
        "MACD_signal",
        "BB_upper",
        "BB_mid",
        "BB_lower",
        "BB_width",
        "RollingStd20",
        "ret_7d",
        "ret_30d",
        "ret_60d",
        "vol_change",
        "vol_vs_ma20",
        "hl_range",
        "oc_change",
    ]
    snap: dict[str, float] = {}
    for c in cols:
        if c not in df_with_indicators.columns:
            continue
        v = df_with_indicators[c].dropna()
        if len(v) == 0:
            continue
        val = float(v.iloc[-1])
        if math.isfinite(val):
            snap[c] = val
    return snap

