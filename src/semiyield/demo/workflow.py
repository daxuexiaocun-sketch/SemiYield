"""Offline orchestration; this is the only package allowed to join business lines."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from semiyield.common.artifacts import sha256_file
from semiyield.common.metrics import classification_report
from semiyield.common.reporting import write_json_report
from semiyield.common.validation import managed_output_dir, require_distinct_directories
from semiyield.demo.reporting import write_charts, write_report
from semiyield.manufacturing.modeling import train_manufacturing
from semiyield.packaging.labeling import proxy_labels, resolve_threshold
from semiyield.packaging.modeling import train_model as train_packaging
from semiyield.reliability.lifetime import fit_arrhenius_weibull, fit_weibull, survival_probability
from semiyield.simulation.contracts import SENSORS
from semiyield.simulation.generator import generate_lifetime_observations
from semiyield.simulation.validation import load_dataset


def _run_into(data_dir, output: Path):
    manifest, tables = load_dataset(data_dir)
    source = Path(data_dir).resolve()
    # Check the only plotting dependency before writing partial reports.
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError("Run `uv sync --locked --extra demo` to generate static charts") from exc
    manufacturing = tables["manufacturing"]
    packaging = tables["packaging_candidates"].copy()
    training_y = packaging.loc[packaging.split.eq("train"), "Y"]
    threshold, threshold_info = resolve_threshold(training_y)
    packaging["proxy_failed"] = proxy_labels(packaging.Y, threshold)
    lifetime = generate_lifetime_observations(
        packaging.loc[packaging.proxy_failed.eq(0)].copy(), seed=manifest["seed"]
    )
    trace = manufacturing[["batch_id", "unit_id", "split", "dataset_role", "failed"]].copy()
    trace = trace.rename(columns={"failed": "manufacturing_failed"})
    trace["entered_packaging"] = trace.unit_id.isin(packaging.unit_id)
    packaging_labels = packaging.set_index("unit_id")["proxy_failed"]
    trace["packaging_proxy_failed"] = trace.unit_id.map(packaging_labels).astype("Int64")
    trace["entered_lifetime"] = trace.unit_id.isin(lifetime.unit_id)
    lifetime_values = lifetime.set_index("unit_id")[["time_to_event", "event_observed"]]
    trace["time_to_event"] = trace.unit_id.map(lifetime_values["time_to_event"])
    trace["event_observed"] = trace.unit_id.map(lifetime_values["event_observed"]).astype("Int64")
    trace["stage_status"] = np.select(
        [
            trace.manufacturing_failed.eq(1),
            trace.packaging_proxy_failed.eq(1).fillna(False),
            trace.entered_lifetime,
        ],
        ["manufacturing_rejected", "packaging_proxy_rejected", "lifetime_observed"],
        default="packaging_label_unavailable",
    )
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
                    data_sha256=manifest["files"][
                        "manufacturing.csv"
                        if stage == "manufacturing"
                        else "packaging_candidates.csv"
                    ]["sha256"],
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
    write_json_report(rows, output / "metrics.json")
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
    write_json_report(reliability, output / "reliability.json")
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
    packaging.to_csv(output / "packaging_observed.csv", index=False)
    lifetime.to_csv(output / "lifetime_observed.csv", index=False)
    trace.to_csv(output / "trace.csv", index=False)
    charts = write_charts(output, stages, metrics, packaging, lifetime, threshold, curve)
    report_manifest = {**manifest, "throughput_threshold": threshold, "threshold": threshold_info}
    write_report(output, report_manifest, stages, metrics, reliability, charts)
    write_json_report(
        {
            "schema_version": "semiyield-three-stage-report-v1",
            "dataset_role": "synthetic",
            "seed": manifest["seed"],
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
            "source_files": manifest["files"],
            "throughput_threshold": threshold,
            "threshold": threshold_info,
            "artifacts": {
                str(p.relative_to(output)): sha256_file(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
                and p.name != "manifest.json"
                and p.relative_to(output).parts[0] not in {"manufacturing", "packaging"}
                and p.name
                not in {
                    "trace.csv",
                    "packaging_observed.csv",
                    "lifetime_observed.csv",
                }
            },
        },
        output / "manifest.json",
    )
    return output / "README.md"


def run(data_dir, output_dir, *, force=False):
    """Run the cross-business demo and atomically publish a complete report."""
    require_distinct_directories(data_dir, output_dir)
    with managed_output_dir(output_dir, force=force) as output:
        _run_into(data_dir, output)
    return Path(output_dir) / "README.md"
