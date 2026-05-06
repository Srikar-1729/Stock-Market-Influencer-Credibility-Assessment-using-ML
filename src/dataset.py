from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, ensure_dirs
from .feature_engineering import FeatureConfig, add_technical_indicators, make_label


FEATURE_COLUMNS = [
    # MAs
    "MA10",
    "MA20",
    "MA50",
    "MA100",
    # Momentum
    "RSI14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    # Volatility
    "BB_upper",
    "BB_mid",
    "BB_lower",
    "BB_width",
    "RollingStd20",
    # Returns
    "ret_1d",
    "ret_7d",
    "ret_30d",
    "ret_60d",
    # Volume
    "vol_change",
    "vol_ma20",
    "vol_vs_ma20",
    # Range
    "hl_range",
    "oc_change",
]


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_examples_for_ticker(
    ticker: str,
    horizon_days: int,
    threshold_return: float,
    feature_cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Build supervised examples for one ticker.
    Each row uses features at time t, label is based on t+horizon_days.
    """
    csv_path = RAW_DIR / f"{ticker.replace('.', '_')}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found for {ticker}. Expected {csv_path}. Run data_loader first.")

    df = load_raw_csv(csv_path)
    df["ticker"] = ticker
    df = add_technical_indicators(df, cfg=feature_cfg)
    df["label"] = make_label(df, horizon_days=horizon_days, threshold_return=threshold_return, price_col="Adj_Close")

    # Keep only rows with full feature set and non-null label (last horizon is null due to shift).
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df[["Date", "ticker"] + FEATURE_COLUMNS + ["label"]]


def build_dataset(
    tickers: Iterable[str],
    horizon_days: int,
    threshold_return: float,
    out_path: Path | None = None,
    feature_cfg: FeatureConfig | None = None,
) -> Path:
    ensure_dirs()
    frames: list[pd.DataFrame] = []
    for t in tickers:
        frames.append(build_examples_for_ticker(t, horizon_days, threshold_return, feature_cfg=feature_cfg))

    ds = pd.concat(frames, ignore_index=True)
    ds = ds.sort_values(["Date", "ticker"]).reset_index(drop=True)

    # Optional: clip extreme values to reduce tree overfit / numeric issues.
    for c in FEATURE_COLUMNS:
        if c in ds.columns and np.issubdtype(ds[c].dtype, np.number):
            q1, q99 = ds[c].quantile([0.01, 0.99])
            ds[c] = ds[c].clip(lower=q1, upper=q99)

    out_path = out_path or (PROCESSED_DIR / f"dataset_h{horizon_days}_thr{int(threshold_return*100)}.csv")
    ds.to_csv(out_path, index=False)
    return out_path


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)
    return X, y

