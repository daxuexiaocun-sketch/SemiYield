"""Manufacturing evaluation protocols and business-facing metrics."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from .modeling import ModelArtifact


def inspection_capture_rate(y_true, scores, budget: float = 0.1) -> float:
    y = np.asarray(y_true)
    if y.sum() == 0:
        return float("nan")
    count = max(1, int(np.ceil(len(y) * budget)))
    selected = np.argsort(np.asarray(scores))[::-1][:count]
    return float(y[selected].sum() / y.sum())


def classification_metrics(y_true, probabilities, threshold: float = 0.5, budget: float = 0.1):
    y = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predicted = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, probabilities)),
        "failure_recall": float(recall_score(y, predicted, zero_division=0)),
        "failure_precision": float(precision_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "capture_at_budget": inspection_capture_rate(y, probabilities, budget),
    }


def evaluate_model(
    artifact: ModelArtifact,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    protocol: str = "holdout",
    timestamps: pd.Series | None = None,
    folds: int = 5,
    repeats: int = 2,
    inspection_budget: float = 0.1,
) -> pd.DataFrame:
    if protocol == "holdout":
        started = perf_counter()
        scores = artifact.estimator.predict_proba(features)[:, 1]
        values = classification_metrics(target, scores, artifact.threshold, inspection_budget)
        values["inference_seconds"] = perf_counter() - started
        return pd.DataFrame([values])
    if protocol == "temporal":
        if timestamps is None:
            raise ValueError("Temporal evaluation requires timestamps")
        order = np.argsort(np.asarray(timestamps))
        split = int(len(order) * 0.8)
        train_idx, test_idx = order[:split], order[split:]
        estimator = clone(artifact.estimator).fit(features.iloc[train_idx], target.iloc[train_idx])
        scores = estimator.predict_proba(features.iloc[test_idx])[:, 1]
        return pd.DataFrame(
            [
                classification_metrics(
                    target.iloc[test_idx], scores, artifact.threshold, inspection_budget
                )
            ]
        )
    if protocol != "cv":
        raise ValueError("Protocol must be holdout, temporal, or cv")
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, target), 1):
        estimator = clone(artifact.estimator).fit(features.iloc[train_idx], target.iloc[train_idx])
        scores = estimator.predict_proba(features.iloc[test_idx])[:, 1]
        row = classification_metrics(
            target.iloc[test_idx], scores, artifact.threshold, inspection_budget
        )
        row["fold"] = fold
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_table(
    y_true, probabilities, thresholds=None, missed_cost: float = 10, review_cost: float = 1
):
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    y = np.asarray(y_true)
    rows = []
    for threshold in thresholds:
        pred = np.asarray(probabilities) >= threshold
        false_negative = int(((pred == 0) & (y == 1)).sum())
        reviewed = int(pred.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "reviewed": reviewed,
                "missed_failures": false_negative,
                "estimated_cost": false_negative * missed_cost + reviewed * review_cost,
            }
        )
    return pd.DataFrame(rows)
