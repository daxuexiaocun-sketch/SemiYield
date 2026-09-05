"""Reliability reports and deterministic SVG figures."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from semiyield.common.artifacts import sha256_file
from semiyield.reliability.lifetime import (
    benchmark_survival_forest,
    fit_arrhenius_weibull,
    fit_weibull,
    survival_probability,
)


def write_reliability_charts(lifetime: pd.DataFrame, curve: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output = Path(output_dir)
    plt.rcParams["svg.hashsalt"] = "semiyield-v1"
    paths = []
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(curve["time"], curve["survival_probability"], color="#2878B5", linewidth=2)
    axis.set(xlabel="Time", ylabel="Survival probability", title="Weibull survival curve")
    axis.set_ylim(0, 1.02); axis.grid(alpha=0.25); figure.tight_layout()
    survival_path = output / "weibull_survival.svg"
    figure.savefig(survival_path, format="svg", metadata={"Date": None}); plt.close(figure)
    paths.append(survival_path)
    if "temperature_c" in lifetime:
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        for temperature, rows in lifetime.groupby("temperature_c"):
            axis.scatter([temperature] * len(rows), rows["time_to_event"], label=f"{temperature:g} °C", alpha=0.8)
        axis.set(xlabel="Stress temperature (°C)", ylabel="Observed/censored time", title="Accelerated-life observations")
        axis.grid(alpha=0.25); figure.tight_layout()
        accelerated_path = output / "accelerated_life.svg"
        figure.savefig(accelerated_path, format="svg", metadata={"Date": None}); plt.close(figure)
        paths.append(accelerated_path)
    return paths


def write_degradation_chart(features: pd.DataFrame, output: str | Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "semiyield-v1"
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for _, rows in features.groupby("device_id", sort=True):
        rows = rows.sort_values("time_end_s")
        axis.plot(rows["time_end_s"] / 60, rows["rds_on_delta_ohm"], color="#2878B5", alpha=0.16, linewidth=0.8)
    axis.axhline(0.045, color="#E28E2C", linestyle="--", linewidth=1.2, label="0.045 Ω")
    axis.axhline(0.050, color="#D62728", linestyle=":", linewidth=1.2, label="0.050 Ω")
    axis.set(xlabel="Cumulative measured aging time (minutes)", ylabel="Temperature-corrected ΔRDS(on) (Ω)", title="NASA MOSFET all-device degradation trajectories")
    axis.set_ylim(-0.1, 0.25); axis.grid(alpha=0.2); axis.legend(); figure.tight_layout()
    figure.savefig(destination, format="svg", metadata={"Date": None}); plt.close(figure)
    return destination


def write_reliability_report(frame: pd.DataFrame, *, output_dir: str | Path = "reports/reference/reliability", use_temperature_c: float = 55.0, data_sha256: str | None = None) -> dict[str, object]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    weibull = fit_weibull(frame)
    report: dict[str, object] = {"schema_version": "semiyield-reliability-report-v1", "data_sha256": data_sha256, "weibull": asdict(weibull)}
    if "temperature_c" in frame:
        stress_group = (pd.to_numeric(frame["temperature_c"], errors="coerce") / 10).round() * 10
        failures_by_group = frame.loc[frame["event_observed"].astype(bool)].groupby(stress_group).size()
        eligible_groups = failures_by_group[failures_by_group >= 3]
        report["arrhenius_weibull"] = (asdict(fit_arrhenius_weibull(frame.loc[stress_group.isin(eligible_groups.index)], use_temperature_c=use_temperature_c)) if len(eligible_groups) >= 2 else {"status": "skipped", "reason": "Need at least two 10 C stress groups with three failures each", "failures_by_rounded_stress_c": {str(key): int(value) for key, value in failures_by_group.items()}})
    numeric_features = [c for c in frame.select_dtypes(include="number").columns if c not in {"time_to_event", "event_observed", "threshold_ohm", "smoothing_windows", "persistence_windows", "baseline_feature_windows"}]
    report["survival_forest"] = benchmark_survival_forest(frame, feature_columns=numeric_features[:10])
    curve_time = np.linspace(0, float(frame["time_to_event"].max()) * 1.1, 200)
    curve = pd.DataFrame({"time": curve_time, "survival_probability": survival_probability(curve_time, beta=weibull.beta, eta=weibull.eta)})
    curve.to_csv(output / "weibull_curve.csv", index=False)
    chart_paths = write_reliability_charts(frame, curve, output)
    report["artifacts"] = {"weibull_curve.csv": sha256_file(output / "weibull_curve.csv"), **{path.name: sha256_file(path) for path in chart_paths}}
    (output / "reliability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
