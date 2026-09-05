"""Leakage-safe, reproducible benchmark orchestration."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from gc import collect
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

from semiyield.common.evaluate import classification_metrics
from semiyield.common.modeling import build_model_pipeline, train_model
from semiyield.common.visuals import write_benchmark_chart
from semiyield.constants import EXPERIMENT_SCHEMA_VERSION, RANDOM_STATE
from semiyield.yield_risk.data import SecomDataset, sha256_file


@dataclass(frozen=True)
class BenchmarkConfig:
    models: tuple[str, ...] = ("dummy", "logistic", "catboost", "tabpfn")
    folds: int = 3
    repeats: int = 1
    calibration_folds: int = 3
    inspection_budget: float = 0.1
    missed_cost: float = 10.0
    review_cost: float = 1.0
    random_state: int = RANDOM_STATE


def select_budget_threshold(probabilities, inspection_budget: float = 0.1) -> float:
    """Select a training-only threshold that sends the top budget fraction to review."""
    if not 0 < inspection_budget <= 1:
        raise ValueError("inspection_budget must be in (0, 1]")
    scores = np.asarray(probabilities, dtype=float)
    return float(np.quantile(scores, 1 - inspection_budget, method="lower"))


def select_threshold(
    target,
    probabilities,
    *,
    missed_cost: float = 10.0,
    review_cost: float = 1.0,
) -> float:
    """Choose the lowest-cost threshold using training-only probabilities."""
    y = np.asarray(target, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    costs = []
    for threshold in candidates:
        predicted = scores >= threshold
        missed = ((predicted == 0) & (y == 1)).sum()
        reviewed = predicted.sum()
        costs.append(missed * missed_cost + reviewed * review_cost)
    return float(candidates[int(np.argmin(costs))])


def _confidence_summary(rows: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    metrics = [column for column in rows.columns if column not in {"fold", "protocol"}]
    for metric in metrics:
        values = pd.to_numeric(rows[metric], errors="coerce").dropna()
        if not len(values):
            continue
        mean = float(values.mean())
        half_width = (
            0.0 if len(values) == 1 else float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
        )
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = 0.0 if len(values) == 1 else float(values.std(ddof=1))
        low, high = mean - half_width, mean + half_width
        if metric not in {"mcc", "threshold", "fit_seconds", "inference_seconds"}:
            low, high = max(0.0, low), min(1.0, high)
        result[f"{metric}_ci95_low"] = low
        result[f"{metric}_ci95_high"] = high
    return result


def _evaluate_split(
    features: pd.DataFrame,
    target: pd.Series,
    train_idx,
    test_idx,
    model_name: str,
    config: BenchmarkConfig,
) -> dict[str, float]:
    train_x, test_x = features.iloc[train_idx], features.iloc[test_idx]
    train_y, test_y = target.iloc[train_idx], target.iloc[test_idx]
    oof = None
    if model_name != "dummy":
        minority = int(train_y.value_counts().min())
        inner_folds = min(config.calibration_folds, minority)
        if inner_folds >= 2:
            inner = StratifiedKFold(
                n_splits=inner_folds, shuffle=True, random_state=config.random_state
            )
            # A fixed review budget depends on ranking. Sigmoid calibration is monotonic,
            # so an uncalibrated base model's OOF ranking selects the same budget without
            # recursively cross-validating an already calibrated CatBoost estimator.
            oof = cross_val_predict(
                build_model_pipeline(model_name, config.random_state),
                train_x,
                train_y,
                cv=inner,
                method="predict_proba",
            )[:, 1]
    started = perf_counter()
    artifact = train_model(
        train_x,
        train_y,
        model_name=model_name,
        calibrate=model_name != "dummy",
        calibration_folds=config.calibration_folds,
        random_state=config.random_state,
    )
    fit_seconds = perf_counter() - started
    threshold = 0.5
    if oof is not None:
        estimator = artifact.estimator
        if isinstance(estimator, CalibratedClassifierCV):
            calibrated = estimator.calibrated_classifiers_[0]
            calibrator = calibrated.calibrators[0]
            oof = calibrator.predict(oof)
        threshold = select_budget_threshold(oof, config.inspection_budget)
    started = perf_counter()
    probabilities = artifact.estimator.predict_proba(test_x)[:, 1]
    inference_seconds = perf_counter() - started
    metrics = classification_metrics(
        test_y, probabilities, threshold=threshold, budget=config.inspection_budget
    )
    metrics.update(
        {
            "threshold": threshold,
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
        }
    )
    return metrics


def run_benchmark(
    dataset: SecomDataset,
    *,
    config: BenchmarkConfig | None = None,
    output_dir: str | Path = "reports/reference/yield",
) -> dict[str, object]:
    """Run repeated stratified CV and chronological holdout with identical model splits."""
    config = config or BenchmarkConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cv = RepeatedStratifiedKFold(
        n_splits=config.folds,
        n_repeats=config.repeats,
        random_state=config.random_state,
    )
    cv_splits = list(cv.split(dataset.features, dataset.target))
    order = np.argsort(np.asarray(dataset.timestamps))
    boundary = int(len(order) * 0.8)
    temporal_split = (order[:boundary], order[boundary:])
    all_rows = []
    model_status = []
    for model_name in config.models:
        model_rows = []
        try:
            for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
                row = _evaluate_split(
                    dataset.features,
                    dataset.target,
                    train_idx,
                    test_idx,
                    model_name,
                    config,
                )
                row.update({"model": model_name, "protocol": "repeated_cv", "fold": fold})
                model_rows.append(row)
            temporal = _evaluate_split(
                dataset.features,
                dataset.target,
                *temporal_split,
                model_name,
                config,
            )
            temporal.update({"model": model_name, "protocol": "temporal", "fold": 1})
            model_rows.append(temporal)
            model_status.append({"model": model_name, "status": "completed", "reason": ""})
        except ImportError as exc:
            model_status.append({"model": model_name, "status": "skipped", "reason": str(exc)})
        except Exception as exc:
            model_status.append({"model": model_name, "status": "failed", "reason": str(exc)})
        all_rows.extend(model_rows)
        del model_rows
        collect()
    fold_metrics = pd.DataFrame(all_rows)
    summaries = []
    if not fold_metrics.empty:
        for (model, protocol), rows in fold_metrics.groupby(["model", "protocol"]):
            summaries.append({"model": model, "protocol": protocol, **_confidence_summary(rows)})
    summary = pd.DataFrame(summaries)
    fold_path = output / "fold_metrics.csv"
    summary_path = output / "summary.csv"
    status_path = output / "model_status.csv"
    fold_metrics.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(model_status).to_csv(status_path, index=False)
    chart_path = (
        write_benchmark_chart(summary, output / "benchmark_pr_auc.svg")
        if not summary.empty
        else None
    )
    artifact_paths = [fold_path, summary_path, status_path]
    if chart_path:
        artifact_paths.append(chart_path)
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "dataset": dataset.metadata,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
        },
        "resource_guard": "not_applied_python_api",
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=list), encoding="utf-8")
    return {"fold_metrics": fold_metrics, "summary": summary, "manifest": manifest}
