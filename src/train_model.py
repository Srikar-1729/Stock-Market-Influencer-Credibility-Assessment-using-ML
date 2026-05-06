from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .config import DEFAULT_TICKERS, MODELS_DIR, ensure_dirs
from .dataset import FEATURE_COLUMNS, build_dataset, load_dataset


@dataclass(frozen=True)
class TrainConfig:
    horizons: tuple[int, ...]
    threshold_percent: float
    tickers: tuple[str, ...]
    n_splits: int = 5
    random_state: int = 42


def _models(random_state: int) -> dict[str, object]:
    # Use relatively conservative defaults so it trains on laptops.
    models: dict[str, object] = {
        "logreg": LogisticRegression(max_iter=2000, n_jobs=None, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
    }

    # Optional: XGBoost
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgb"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    except Exception:
        pass

    # Optional: LightGBM (often requires libomp on macOS)
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        models["lgbm"] = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
        )
    except Exception:
        pass

    return models


def _preprocess_pipeline(model: object, numeric_features: list[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[("num", numeric, numeric_features)],
        remainder="drop",
    )

    # Calibrate probabilities for better "credibility score" behavior.
    # CalibratedClassifierCV wraps the base estimator and will refit inside each split.
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    return Pipeline(steps=[("pre", pre), ("clf", calibrated)])


def _evaluate_cv(
    X: pd.DataFrame, y: pd.Series, pipeline: Pipeline, n_splits: int
) -> dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_proba_all: list[float] = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())
        y_proba_all.extend(proba.tolist())

    y_true = np.asarray(y_true_all)
    y_pred = np.asarray(y_pred_all)
    y_proba = np.asarray(y_proba_all)

    # If a split yields a single class, roc_auc can fail. Guard it.
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


def train_for_horizon(cfg: TrainConfig, horizon_days: int) -> dict[str, object]:
    ensure_dirs()
    thr = cfg.threshold_percent / 100.0

    ds_path = build_dataset(cfg.tickers, horizon_days=horizon_days, threshold_return=thr)
    X, y = load_dataset(ds_path)

    results: dict[str, dict[str, float]] = {}
    best_name = None
    best_score = -1e9
    best_pipeline: Pipeline | None = None

    for name, model in _models(cfg.random_state).items():
        pipe = _preprocess_pipeline(model, numeric_features=FEATURE_COLUMNS)
        metrics = _evaluate_cv(X, y, pipe, n_splits=cfg.n_splits)
        results[name] = metrics

        # Selection rule: prioritize ROC AUC, then F1.
        auc = metrics["roc_auc"]
        score = (auc if not np.isnan(auc) else 0.0) + 0.25 * metrics["f1"]
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipe

    assert best_name is not None and best_pipeline is not None

    # Fit final model on full dataset.
    best_pipeline.fit(X, y)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = MODELS_DIR / f"h{horizon_days}_thr{int(cfg.threshold_percent)}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / f"model_{best_name}_{stamp}.joblib"
    joblib.dump(best_pipeline, model_path)

    meta = {
        "trained_at": stamp,
        "horizon_days": horizon_days,
        "threshold_percent": cfg.threshold_percent,
        "tickers": list(cfg.tickers),
        "best_model": best_name,
        "cv_results": results,
        "dataset_path": str(ds_path),
        "feature_columns": FEATURE_COLUMNS,
    }
    meta_path = artifact_dir / f"metadata_{stamp}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # Convenience pointer to "latest" for this (horizon,threshold).
    latest_model = artifact_dir / "model_latest.joblib"
    latest_meta = artifact_dir / "metadata_latest.json"
    joblib.dump(best_pipeline, latest_model)
    latest_meta.write_text(json.dumps(meta, indent=2))

    return {"model_path": model_path, "meta_path": meta_path, "best_model": best_name, "cv_results": results}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train models for credibility analyzer.")
    p.add_argument("--horizons", type=int, nargs="+", default=[7, 30, 60])
    p.add_argument("--threshold", type=float, default=5.0, help="Return threshold in percent, e.g. 5 for +5%%.")
    p.add_argument("--tickers", type=str, nargs="*", default=None)
    p.add_argument("--n_splits", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tickers = tuple(args.tickers) if args.tickers else tuple(DEFAULT_TICKERS)
    cfg = TrainConfig(
        horizons=tuple(args.horizons),
        threshold_percent=float(args.threshold),
        tickers=tickers,
        n_splits=int(args.n_splits),
    )

    print("Training config:")
    print(json.dumps(asdict(cfg), indent=2))

    for h in cfg.horizons:
        print(f"\n=== Training for horizon={h} days, threshold={cfg.threshold_percent:.1f}% ===")
        out = train_for_horizon(cfg, horizon_days=h)
        print(f"Saved best model: {out['best_model']} -> {out['model_path']}")
        print("CV metrics:")
        print(json.dumps(out["cv_results"], indent=2))


if __name__ == "__main__":
    main()

