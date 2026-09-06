"""Packaging models preserve mixed input types and proxy-label provenance."""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from semiyield.common.catboost import BASE_PARAMS, TRIALS, catboost_classifier, sample_candidates
from semiyield.common.catboost import SCHEMA_VERSION as CATBOOST_SCHEMA_VERSION
from semiyield.packaging.data import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    validate_data,
)
from semiyield.packaging.labeling import proxy_labels, resolve_threshold
from semiyield.packaging.preprocessing import build_preprocessor


def build_pipeline(model="logistic", seed=42, catboost_params=None):
    if model == "dummy":
        classifier = DummyClassifier(strategy="prior")
    elif model == "logistic":
        classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    elif model == "catboost":
        classifier = catboost_classifier(seed=seed, params=catboost_params)
    else:
        raise ValueError(f"Unknown packaging model: {model}")
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", classifier),
        ]
    )


@dataclass
class PackagingArtifact:
    estimator: Pipeline
    model_name: str
    throughput_threshold: float
    metadata: dict
    probability_threshold: float = 0.5
    schema_version: str = "semiyield-packaging-v1"
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURES))

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise TypeError("Expected a packaging model artifact")
        return artifact

    def predict(self, frame: pd.DataFrame):
        data = validate_data(frame, require_target=False)
        scores = self.estimator.predict_proba(data[self.feature_columns])[:, 1]
        result = data[[c for c in ("batch_id", "unit_id") if c in data]].copy()
        result["proxy_failure_probability"] = scores
        result["predicted_proxy_failed"] = (scores >= self.probability_threshold).astype(int)
        return result


def train_model(
    frame,
    *,
    model="logistic",
    seed=42,
    threshold=None,
    quantile=None,
    probability_threshold=0.5,
    data_sha256=None,
    catboost_params=None,
    catboost_params_sha256=None,
):
    data = validate_data(frame)
    value, provenance = resolve_threshold(data.Y, threshold=threshold, quantile=quantile)
    target = proxy_labels(data.Y, value)
    if len(np.unique(target)) < 2:
        raise ValueError("Training requires both proxy pass and proxy fail samples")
    if not np.isfinite(probability_threshold) or not 0 <= probability_threshold <= 1:
        raise ValueError("probability_threshold must be between 0 and 1")
    estimator = build_pipeline(model, seed, catboost_params).fit(data[FEATURES], target)
    return PackagingArtifact(
        estimator,
        model,
        value,
        {
            "threshold": provenance,
            **(
                {
                    "catboost_params": {**BASE_PARAMS, **(catboost_params or {})},
                    "catboost_params_sha256": catboost_params_sha256,
                }
                if model == "catboost"
                else {}
            ),
            "seed": seed,
            "data_sha256": data_sha256,
            "input_schema": {"categorical": CATEGORICAL, "numeric": NUMERIC},
            "label_rule": "Y < throughput_threshold",
            "target_role": "low_throughput_proxy_failure",
            "training_rows": len(data),
        },
        probability_threshold,
    )


def tune_catboost(
    frame, *, data_sha256: str, seed: int = 42, trials: int = TRIALS, threshold=None, quantile=None
):
    """Tune with proxy labels and quantile thresholds fitted separately in every inner fold."""
    data = validate_data(frame)
    # Split on a provisional training-only-equivalent label solely to make stratification possible.
    value, _ = resolve_threshold(data.Y, threshold=threshold, quantile=quantile)
    provisional = proxy_labels(data.Y, value)
    minority = int(np.bincount(provisional).min())
    folds = min(3, minority)
    if folds < 2:
        raise ValueError("CatBoost tuning requires at least two samples in each proxy class")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for params in sample_candidates(seed, trials):
        scores = []
        for train_idx, valid_idx in splitter.split(data, provisional):
            train, valid = data.iloc[train_idx], data.iloc[valid_idx]
            fold_threshold, _ = resolve_threshold(train.Y, threshold=threshold, quantile=quantile)
            train_y = proxy_labels(train.Y, fold_threshold)
            valid_y = proxy_labels(valid.Y, fold_threshold)
            estimator = build_pipeline("catboost", seed, params).fit(train[FEATURES], train_y)
            scores.append(
                float(
                    average_precision_score(valid_y, estimator.predict_proba(valid[FEATURES])[:, 1])
                )
            )
        rows.append(
            {"params": params, "mean_pr_auc": float(np.mean(scores)), "fold_pr_auc": scores}
        )
    rows.sort(
        key=lambda row: (-row["mean_pr_auc"], row["params"]["iterations"], row["params"]["depth"])
    )
    from datetime import datetime, timezone

    return {
        "schema_version": CATBOOST_SCHEMA_VERSION,
        "task": "packaging",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "trials": trials,
        "folds": folds,
        "objective": "pr_auc",
        "data_sha256": data_sha256,
        "best_params": rows[0]["params"],
        "best_mean_pr_auc": rows[0]["mean_pr_auc"],
        "candidates": rows,
    }
