from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

from .config import MODELS_DIR, ensure_dirs
from .credibility_score import probability_to_credibility
from .feature_engineering import FeatureConfig, add_technical_indicators, current_indicator_snapshot
from .dataset import FEATURE_COLUMNS


@dataclass(frozen=True)
class PredictionOutput:
    stock_symbol: str
    expected_change_percent: float
    horizon_days: int
    threshold_percent_used: float
    probability: float
    credibility_score: float
    confidence_level: str
    indicators: dict[str, float]


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) for c in df.columns.to_list()]
    return df


def _latest_artifact_dir(horizon_days: int, threshold_percent: float) -> Path:
    return MODELS_DIR / f"h{horizon_days}_thr{int(threshold_percent)}"


def load_model(horizon_days: int, threshold_percent: float) -> object:
    artifact_dir = _latest_artifact_dir(horizon_days, threshold_percent)
    model_path = artifact_dir / "model_latest.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first: python -m src.train_model --horizons {horizon_days} --threshold {threshold_percent}"
        )
    return joblib.load(model_path)


def fetch_latest_data(symbol: str, lookback_years: int = 5) -> pd.DataFrame:
    end = date.today().isoformat()
    start_year = date.today().year - lookback_years
    start = f"{start_year}-01-01"
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for symbol={symbol}")
    df = _flatten_yfinance_columns(df)
    df = df.reset_index()
    df.columns = [str(c).replace(" ", "_") for c in df.columns]
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def _build_latest_feature_row(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    df_ind = add_technical_indicators(df, cfg=FeatureConfig())
    indicators = current_indicator_snapshot(df_ind)
    latest = df_ind.dropna(subset=FEATURE_COLUMNS).iloc[[-1]][FEATURE_COLUMNS].copy()
    return latest, indicators


def evaluate_prediction(
    stock_symbol: str,
    expected_change_percent: float,
    horizon_days: int,
    threshold_percent_for_model: float | None = None,
) -> PredictionOutput:
    """
    Evaluate a finfluencer-style prediction.

    By default, this uses a model trained for threshold == expected_change_percent,
    so the returned probability aligns with "reach expected change within horizon".
    """
    ensure_dirs()
    threshold_percent_for_model = float(
        expected_change_percent if threshold_percent_for_model is None else threshold_percent_for_model
    )

    try:
        model = load_model(horizon_days=horizon_days, threshold_percent=threshold_percent_for_model)
    except FileNotFoundError:
        # Graceful fallback to a common baseline model if available.
        fallback_thr = 5.0
        model = load_model(horizon_days=horizon_days, threshold_percent=fallback_thr)
        threshold_percent_for_model = fallback_thr

    latest_df = fetch_latest_data(stock_symbol)
    X_latest, indicators = _build_latest_feature_row(latest_df)

    prob = float(model.predict_proba(X_latest)[:, 1][0])
    cred = probability_to_credibility(prob)

    return PredictionOutput(
        stock_symbol=stock_symbol,
        expected_change_percent=float(expected_change_percent),
        horizon_days=int(horizon_days),
        threshold_percent_used=float(threshold_percent_for_model),
        probability=prob,
        credibility_score=cred.credibility_score,
        confidence_level=cred.confidence_level,
        indicators=indicators,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a stock prediction and output credibility score.")
    p.add_argument("--symbol", type=str, required=True, help="Yahoo ticker, e.g. RELIANCE.NS")
    p.add_argument("--expected_change", type=float, required=True, help="Expected change in percent, e.g. 10")
    p.add_argument("--horizon", type=int, required=True, help="Horizon in days, e.g. 60")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for model threshold percent (default: expected_change).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = evaluate_prediction(
        stock_symbol=args.symbol,
        expected_change_percent=args.expected_change,
        horizon_days=args.horizon,
        threshold_percent_for_model=args.threshold,
    )
    print(json.dumps(asdict(out), indent=2))


if __name__ == "__main__":
    main()

