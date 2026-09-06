"""Holdout training and matched-fold packaging benchmarks."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from semiyield.common.artifacts import sha256_file
from semiyield.common.metrics import classification_report
from semiyield.packaging.data import validate_data
from semiyield.packaging.labeling import proxy_labels
from semiyield.packaging.modeling import train_model, tune_catboost
from semiyield.packaging.reporting import write_benchmark_charts


def train_holdout(input_csv, output, **kwargs):
    frame = validate_data(pd.read_csv(input_csv))
    train, test = train_test_split(frame, test_size=0.2, random_state=kwargs.get("seed", 42))
    artifact = train_model(train, data_sha256=sha256_file(Path(input_csv)), **kwargs)
    scores = artifact.predict(test).proxy_failure_probability
    report = classification_report(
        proxy_labels(test.Y, artifact.throughput_threshold), scores, artifact.probability_threshold
    )
    artifact.metadata["split"] = {"protocol": "random_holdout", "test_rows": len(test)}
    artifact.metadata["holdout_metrics"] = report
    artifact.save(output)
    Path(output).with_suffix(".json").write_text(
        json.dumps(artifact.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return artifact


def run_benchmark(
    input_csv, output_dir, *, models=("dummy", "logistic"), seed=42, threshold=None, quantile=None
):
    frame = validate_data(pd.read_csv(input_csv))
    if not models or any(m not in {"dummy", "logistic", "catboost"} for m in models):
        raise ValueError("models must contain dummy, logistic, or catboost")
    digest = sha256_file(Path(input_csv))
    rows = []
    details = []
    for fold, (train_idx, test_idx) in enumerate(
        KFold(n_splits=3, shuffle=True, random_state=seed).split(frame), 1
    ):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        for model in models:
            tuned = None
            if model == "catboost":
                fold_digest = hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()
                tuned = tune_catboost(
                    train,
                    data_sha256=fold_digest,
                    seed=seed,
                    threshold=threshold,
                    quantile=quantile,
                )
            artifact = train_model(
                train,
                model=model,
                seed=seed,
                threshold=threshold,
                quantile=quantile,
                data_sha256=digest,
                catboost_params=tuned["best_params"] if tuned else None,
            )
            metrics = classification_report(
                proxy_labels(test.Y, artifact.throughput_threshold),
                artifact.predict(test).proxy_failure_probability,
            )
            details.append(
                {
                    "fold": fold,
                    "model": model,
                    **artifact.metadata,
                    "metrics": metrics,
                    **({"tuning": tuned} if tuned else {}),
                }
            )
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "throughput_threshold": artifact.throughput_threshold,
                    **{k: v for k, v in metrics.items() if k != "undefined_reasons"},
                    **({"tuning_pr_auc": tuned["best_mean_pr_auc"]} if tuned else {}),
                }
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary = metrics.groupby("model").mean(numeric_only=True).drop(columns="fold")
    summary.to_csv(output / "summary.csv")
    charts = write_benchmark_charts(summary, metrics, output)
    artifact_paths = [output / "summary.csv", output / "fold_metrics.csv"]
    artifact_paths.extend(charts)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "semiyield-packaging-benchmark-v1",
                "seed": seed,
                "protocol": "random_3fold_training_only_threshold",
                "scope": (
                    "Low-throughput proxy failure; no batch or time identifiers in source data"
                ),
                "data_sha256": digest,
                "folds": details,
                "artifacts": {artifact.name: sha256_file(artifact) for artifact in artifact_paths},
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
