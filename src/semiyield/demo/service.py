"""Offline cross-business orchestration with shared batch isolation."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from semiyield.common.data import sha256_file
from semiyield.common.metrics import classification_report
from semiyield.demo.data import SENSORS, load_dataset, prepare_output, write_json
from semiyield.demo.reporting import write_charts, write_report
from semiyield.packaging.modeling import train_model as train_packaging
from semiyield.reliability import fit_arrhenius_weibull, fit_weibull, survival_probability
from semiyield.yield_risk.service import train_manufacturing


def run(data_dir, output_dir, *, force=False):
    manifest, tables = load_dataset(data_dir)
    source, destination = Path(data_dir).resolve(), Path(output_dir).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Demo reports and source data must use separate, non-nested directories")
    # Check the only plotting dependency before writing partial reports.
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError("Run `uv sync --locked --extra demo` to generate static charts") from exc
    output = prepare_output(output_dir, force)
    if force:
        # Remove only optional artifacts owned by this runner so skipped fits cannot
        # leave an older successful model/curve in the new report.
        for relative in (
            "weibull_curve.csv",
            "manufacturing/dummy.joblib",
            "manufacturing/logistic.joblib",
            "packaging/dummy.joblib",
            "packaging/logistic.joblib",
        ):
            (output / relative).unlink(missing_ok=True)
    manufacturing, packaging, lifetime, trace = (
        tables[name] for name in ("manufacturing", "packaging", "lifetime", "trace")
    )
    threshold = manifest["throughput_threshold"]
    rows = []
    for stage, data in [("manufacturing", manufacturing), ("packaging", packaging)]:
        train, test = data.loc[data.split.eq("train")], data.loc[data.split.eq("test")]
        for model in ("dummy", "logistic"):
            row = {"stage": stage, "model": model, "dataset_role": "synthetic"}
            try:
                if train.empty or test.empty:
                    raise ValueError("No training or test devices reached this stage")
                if stage == "manufacturing":
                    artifact = train_manufacturing(
                        train[SENSORS], train.failed, model=model, seed=manifest["seed"]
                    )
                    scores = artifact.estimator.predict_proba(test[SENSORS])[:, 1]
                    labels = test.failed
                else:
                    artifact = train_packaging(train, model=model, seed=manifest["seed"])
                    if not np.isclose(artifact.throughput_threshold, threshold, rtol=1e-12):
                        raise ValueError("Generated and fitted packaging thresholds disagree")
                    scores = artifact.predict(test).proxy_failure_probability
                    labels = test.proxy_failed
                artifact.metadata.update(
                    dataset_role="synthetic",
                    split_protocol="batch_holdout",
                    data_sha256=manifest["files"][f"{stage}.csv"]["sha256"],
                )
                if stage == "manufacturing":
                    artifact.schema_version = "semiyield-synthetic-manufacturing-v1"
                artifact.save(output / stage / f"{model}.joblib")
                row.update(classification_report(labels, scores))
            except ValueError as exc:
                row.update(status="skipped", reason=str(exc))
            rows.append(row)
    metrics = pd.DataFrame(rows)
    if "pr_auc" not in metrics:
        metrics["pr_auc"] = np.nan
    metrics.drop(columns=["undefined_reasons"], errors="ignore").to_csv(
        output / "metrics.csv", index=False
    )
    write_json(output / "metrics.json", rows)
    curve = None
    life_train = lifetime.loc[lifetime.split.eq("train")]
    reliability = {
        "status": "skipped",
        "cohort": "training_survivors",
        "train_units": len(life_train),
        "test_units": int(lifetime.split.eq("test").sum()),
        "time_unit": "hours",
        "dataset_role": "synthetic",
    }
    try:
        if len(life_train) < 3 or int(life_train.event_observed.sum()) < 2:
            raise ValueError("Need at least 3 training survivors and 2 observed failures")
        weibull = fit_weibull(life_train)
        reliability.update(status="completed", weibull=asdict(weibull))
        time = np.linspace(0, life_train.time_to_event.max() * 1.1, 200)
        curve = pd.DataFrame(
            {
                "time": time,
                "survival_probability": survival_probability(
                    time, beta=weibull.beta, eta=weibull.eta
                ),
            }
        )
        curve["dataset_role"] = "synthetic"
        curve.to_csv(output / "weibull_curve.csv", index=False)
    except (ValueError, RuntimeError) as exc:
        reliability["reason"] = str(exc)
    counts = life_train.loc[life_train.event_observed.eq(1)].groupby("temperature_c").size()
    eligible = counts[counts >= 3].index
    reliability["arrhenius"] = {
        "status": "skipped",
        "reason": "Need two training stress groups with 3 failures each",
    }
    if len(eligible) >= 2:
        try:
            fitted = fit_arrhenius_weibull(
                life_train.loc[life_train.temperature_c.isin(eligible)], use_temperature_c=55
            )
            reliability["arrhenius"] = {"status": "completed", **asdict(fitted)}
        except (ValueError, RuntimeError) as exc:
            reliability["arrhenius"] = {"status": "skipped", "reason": str(exc)}
    write_json(output / "reliability.json", reliability)
    stage_rows = []
    for stage, data, failed in [
        ("manufacturing", manufacturing, manufacturing.failed),
        ("packaging", packaging, packaging.proxy_failed),
        ("lifetime", lifetime, lifetime.event_observed),
    ]:
        known = int(failed.notna().sum())
        failures = int(failed.sum())
        stage_rows.append(
            {
                "stage": stage,
                "dataset_role": "synthetic",
                "entered": len(data),
                "observed_failures": failures,
                "unknown_outcomes": len(data) - known,
                "passed": known - failures if stage != "lifetime" else None,
                "censored": known - failures if stage == "lifetime" else None,
                "failure_rate": failures / known if known else None,
            }
        )
    stages = pd.DataFrame(stage_rows)
    stages.to_csv(output / "stages.csv", index=False)
    charts = write_charts(output, stages, metrics, packaging, threshold, curve)
    write_report(output, manifest, stages, metrics, reliability, trace, charts)
    write_json(
        output / "manifest.json",
        {
            "schema_version": "semiyield-three-stage-report-v1",
            "dataset_role": "synthetic",
            "seed": manifest["seed"],
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
            "source_files": manifest["files"],
            "throughput_threshold": threshold,
            "artifacts": {
                str(p.relative_to(output)): sha256_file(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return output / "README.md"
