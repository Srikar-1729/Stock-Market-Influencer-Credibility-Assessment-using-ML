# Finfluencer Credibility Analyzer

An end-to-end ML system that scores the credibility of a stock prediction using **only historical price/volume data + technical indicators** (no influencer datasets).

## What it does

Given:

- Stock symbol (e.g., `RELIANCE.NS`)
- Expected price change (e.g., `+10%`)
- Time horizon (e.g., `60` days)

It outputs:

- Prediction probability (model-estimated)
- Credibility score (0–100)
- Confidence band (Low/Medium/High)
- A small technical-indicator summary + price chart (Streamlit)

## Setup

```bash
cd finfluencer-analyzer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Download data (NIFTY 50 universe subset by default)

```bash
python -m src.data_loader --start 2010-01-01 --end today
```

CSV files are stored in `data/raw/`.

## 2) Build dataset + train models

This trains models for multiple horizons and saves the best model per horizon.

```bash
python -m src.train_model --horizons 7 30 60 --threshold 5
```

Artifacts are stored in `models/`.

## 3) Evaluate a prediction from CLI

```bash
python -m src.predict --symbol RELIANCE.NS --expected_change 10 --horizon 60
```

By default, the predictor tries to use a model trained for `threshold == expected_change`.
If that specific model is unavailable, it falls back to threshold `5%` model (if present).

## 4) Run the web app

```bash
streamlit run app/app.py
```

## Notes

- The default label is: **1 if future return ≥ 5% within horizon**, else 0.
- If you want the model to directly match a different threshold (e.g., 10%), retrain with `--threshold 10`.

