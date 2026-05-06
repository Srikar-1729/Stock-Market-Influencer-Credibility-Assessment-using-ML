from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is importable regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import evaluate_prediction, fetch_latest_data
from src.config import DEFAULT_TICKERS, RAW_DIR


st.set_page_config(page_title="Finfluencer Credibility Analyzer", layout="wide")

st.title("Finfluencer Credibility Analyzer")
st.caption("Credibility scoring using only historical market data + technical indicators.")


def _symbol_options() -> list[str]:
    symbols = set(DEFAULT_TICKERS)
    if RAW_DIR.exists():
        for p in RAW_DIR.glob("*.csv"):
            symbols.add(p.stem.replace("_", "."))
    return sorted(symbols)


with st.sidebar:
    st.subheader("Prediction input")
    symbol_options = _symbol_options()
    selected_symbol = st.selectbox(
        "Stock symbol",
        options=symbol_options,
        index=symbol_options.index("RELIANCE.NS") if "RELIANCE.NS" in symbol_options else 0,
        help="Pick from list or type in the box to quickly search matching symbols.",
    )

    expected_change_input = st.text_input(
        "Expected price change (%)",
        value="10",
        help="Type any value, e.g. 7.5 or 10",
    )
    horizon_input = st.text_input(
        "Time horizon (days)",
        value="60",
        help="Type any integer number of days, e.g. 7, 30, 60",
    )

    with st.expander("Advanced"):
        model_threshold_input = st.text_input(
            "Override model threshold (%)",
            value="",
            placeholder="Leave empty to use expected change",
            help="Set this only if you want to force a specific threshold model.",
        )
    run = st.button("Evaluate prediction", type="primary")


def _price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Date",
        yaxis_title="Price",
        height=420,
    )
    return fig


if run:
    symbol = selected_symbol

    try:
        expected_change = float(expected_change_input.strip())
    except ValueError:
        st.error("Expected price change must be a number (e.g. 10 or 7.5).")
        st.stop()

    try:
        horizon = int(horizon_input.strip())
        if horizon <= 0:
            raise ValueError("horizon must be positive")
    except ValueError:
        st.error("Time horizon must be a positive integer (e.g. 7, 30, 60).")
        st.stop()

    model_threshold = None
    if model_threshold_input.strip():
        try:
            model_threshold = float(model_threshold_input.strip())
        except ValueError:
            st.error("Override model threshold must be numeric if provided.")
            st.stop()

    try:
        out = evaluate_prediction(
            stock_symbol=symbol.strip(),
            expected_change_percent=float(expected_change),
            horizon_days=int(horizon),
            threshold_percent_for_model=model_threshold,
        )
    except Exception as e:
        st.error(f"Failed to evaluate prediction: {e}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Credibility Score", f"{out.credibility_score:.1f}%")
    col2.metric("Prediction Probability", f"{out.probability:.3f}")
    col3.metric("Confidence Level", out.confidence_level)

    st.info(
        f"Interpretation: P( return ≥ {out.threshold_percent_used:.0f}% within {out.horizon_days} days ) "
        f"for `{out.stock_symbol}`."
    )
    if abs(out.threshold_percent_used - float(expected_change)) > 1e-9:
        st.warning(
            f"You asked for {expected_change:.2f}%, but model threshold {out.threshold_percent_used:.2f}% was used "
            "(fallback model). Train that exact threshold for a more aligned credibility score."
        )

    left, right = st.columns([1.2, 0.8])

    with left:
        st.subheader("Recent price trend")
        df = fetch_latest_data(out.stock_symbol, lookback_years=3)
        st.plotly_chart(_price_chart(df.tail(260)), use_container_width=True)

    with right:
        st.subheader("Technical indicators (latest)")
        st.json(out.indicators)

    st.caption(f"Generated at {datetime.now().isoformat(timespec='seconds')}")
else:
    st.write("Enter a prediction in the sidebar and click **Evaluate prediction**.")

