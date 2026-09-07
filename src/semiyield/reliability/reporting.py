"""Reliability reports and deterministic SVG figures."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from semiyield.common.artifacts import sha256_file
from semiyield.common.reporting import save_svg
from semiyield.reliability.lifetime import (
    DEFAULT_USE_TEMPERATURES_C,
    benchmark_survival_forest,
    fit_arrhenius_weibull,
    fit_weibull,
    survival_probability,
    sweep_arrhenius_weibull,
)


def _kaplan_meier(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.sort_values("time_to_event")
    survival = 1.0
    points = [(0.0, survival)]
    for time, group in rows.groupby("time_to_event", sort=True):
        at_risk = int((rows.time_to_event >= time).sum())
        survival *= 1 - int(group.event_observed.sum()) / at_risk
        points.append((float(time), survival))
    return pd.DataFrame(points, columns=["time", "survival_probability"])


def write_reliability_charts(
    lifetime: pd.DataFrame,
    curve: pd.DataFrame,
    output_dir: str | Path,
    *,
    use_temperatures_c=DEFAULT_USE_TEMPERATURES_C,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output = Path(output_dir)
    plt.rcParams["svg.hashsalt"] = "semiyield-v1"
    paths = []
    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    empirical = _kaplan_meier(lifetime)
    axis.step(
        empirical.time,
        empirical.survival_probability,
        where="post",
        color="#2878b5",
        linewidth=2,
        label="Kaplan–Meier (empirical)",
    )
    axis.plot(
        curve["time"],
        curve["survival_probability"],
        color="#4f8f6b",
        linestyle="--",
        linewidth=2,
        label="Weibull fit (continuous)",
    )
    censored = lifetime.loc[lifetime.event_observed.eq(0)]
    if len(censored):
        axis.scatter(
            censored.time_to_event,
            np.interp(censored.time_to_event, empirical.time, empirical.survival_probability),
            marker="+",
            color="#172033",
            s=30,
            label=f"Right-censored: {len(censored):,}",
        )
    axis.set(
        xlabel="Stress time (s)",
        ylabel="Survival probability",
        title="NASA MOSFET lifetime · Weibull and Kaplan–Meier",
    )
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    survival_path = output / "weibull_survival.svg"
    axis.legend()
    save_svg(
        figure,
        survival_path,
        title="Lifetime survival with censoring",
        description="Kaplan-Meier survival, Weibull fit, and right-censored observations.",
    )
    plt.close(figure)
    paths.append(survival_path)
    if "temperature_c" in lifetime:
        figure, axis = plt.subplots(figsize=(9.2, 4.8))
        events = lifetime.loc[lifetime.event_observed.eq(1)]
        censored = lifetime.loc[lifetime.event_observed.eq(0)]
        axis.scatter(
            events["temperature_c"],
            events["time_to_event"],
            color="#4f8f6b",
            alpha=0.78,
            s=48,
            label=f"Events: {len(events):,}",
        )
        axis.scatter(
            censored["temperature_c"],
            censored["time_to_event"],
            marker="+",
            color="#172033",
            s=54,
            linewidths=1.6,
            label=f"Right-censored: {len(censored):,}",
        )
        times = pd.to_numeric(lifetime["time_to_event"], errors="coerce")
        if times.gt(0).all() and times.max() / times.min() >= 100:
            axis.set_yscale("log")
            time_label = "Observed/censored time (log scale)"
        else:
            time_label = "Observed/censored time"
        axis.set(
            xlabel="Stress temperature (°C)",
            ylabel=time_label,
            title="NASA MOSFET accelerated-life observations",
        )
        axis.grid(alpha=0.25)
        target_colors = ("#2878b5", "#3b9b8a", "#d47832", "#7252b8")
        for index, temperature in enumerate(use_temperatures_c):
            axis.axvline(
                temperature,
                color=target_colors[index % len(target_colors)],
                linestyle=":",
                linewidth=1.4,
                label=f"{temperature:g} °C extrapolated use target",
            )
        axis.margins(x=0.04, y=0.12)
        axis.legend(loc="upper left", fontsize=8)
        accelerated_path = output / "accelerated_life.svg"
        save_svg(
            figure,
            accelerated_path,
            title="Accelerated-life observations",
            description=(
                "Stress-temperature observations distinguish failed and right-censored units; "
                "declared use-temperature targets are reported as extrapolated model estimates. "
                "The vertical scale is logarithmic "
                "when the observed times span at least two orders of magnitude."
            ),
        )
        plt.close(figure)
        paths.append(accelerated_path)
    return paths


def write_arrhenius_lifetime_sweep(sweep: pd.DataFrame, output_dir: str | Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    output = Path(output_dir)
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    positions = np.arange(len(sweep), dtype=float)
    width = 0.34
    extrapolated = sweep["support_status"].eq("extrapolated").to_numpy()
    b10_bars = axis.bar(
        positions - width / 2,
        sweep["b10_s"],
        width,
        color="#9aa9b8",
        label="B10",
    )
    eta_bars = axis.bar(
        positions + width / 2,
        sweep["eta_s"],
        width,
        color="#2878b5",
        label="η",
    )
    labels = [
        f"{value:,.0f} s{'+' if is_extrapolated else ''}"
        for value, is_extrapolated in zip(sweep["b10_s"], extrapolated, strict=True)
    ]
    axis.bar_label(b10_bars, labels=labels, padding=3, fontsize=7)
    labels = [
        f"{value:,.0f} s{'+' if is_extrapolated else ''}"
        for value, is_extrapolated in zip(sweep["eta_s"], extrapolated, strict=True)
    ]
    axis.bar_label(eta_bars, labels=labels, padding=3, fontsize=7)
    axis.set(
        xlabel="Use junction temperature (°C)",
        ylabel="Lifetime estimate (seconds, log scale)",
        title="Arrhenius–Weibull use-temperature lifetime estimates",
        yscale="log",
        xticks=positions,
        xticklabels=[f"{temperature:g}" for temperature in sweep["use_temperature_c"]],
    )
    axis.grid(axis="y", alpha=0.25)
    axis.margins(x=0.08, y=0.26)
    axis.legend(
        title="Lifetime metric",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.78),
        fontsize=8,
        title_fontsize=8,
    )
    axis.text(
        0.02,
        0.97,
        "B10 = 10% failure time; η = 63.2% failure time\n"
        "+ = declared use-temperature extrapolation / model estimate\n"
        "Arrhenius–Weibull model estimates, not direct observations.",
        transform=axis.transAxes,
        fontsize=8,
        color="#516174",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 2},
    )
    path = output / "arrhenius_lifetime_sweep.svg"
    save_svg(
        figure,
        path,
        title="Arrhenius-Weibull use-temperature lifetime sweep",
        description=(
            "Grouped bars show B10, the 10 percent failure time, and eta, the 63.2 percent "
            "failure time, by use junction temperature. Bar labels with a plus sign identify "
            "declared use-temperature extrapolations; values are "
            "Arrhenius-Weibull model estimates, not direct observations."
        ),
    )
    plt.close(figure)
    return path


def write_degradation_chart(features: pd.DataFrame, output: str | Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "semiyield-v1"
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for _, rows in features.groupby("device_id", sort=True):
        rows = rows.sort_values("time_end_s")
        axis.plot(
            rows["time_end_s"] / 60,
            rows["rds_on_delta_ohm"],
            color="#2878B5",
            alpha=0.16,
            linewidth=0.8,
        )
    axis.axhline(0.045, color="#E28E2C", linestyle="--", linewidth=1.2, label="0.045 Ω")
    axis.axhline(0.050, color="#D62728", linestyle=":", linewidth=1.2, label="0.050 Ω")
    axis.set(
        xlabel="Cumulative measured aging time (minutes)",
        ylabel="Temperature-corrected ΔRDS(on) (Ω)",
        title="NASA MOSFET degradation trajectories",
    )
    axis.set_ylim(-0.1, 0.25)
    axis.grid(alpha=0.2)
    axis.legend()
    save_svg(
        figure,
        destination,
        title="NASA MOSFET degradation trajectories",
        description=(
            "All-device degradation trajectories with two resistance-change threshold lines."
        ),
    )
    plt.close(figure)
    return destination


def write_reliability_report(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path = "reports/reference/reliability",
    use_temperature_c: float = 55.0,
    use_temperatures_c=DEFAULT_USE_TEMPERATURES_C,
    data_sha256: str | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weibull = fit_weibull(frame)
    report: dict[str, object] = {
        "schema_version": "semiyield-reliability-report-v1",
        "data_sha256": data_sha256,
        "weibull": asdict(weibull),
    }
    if "temperature_c" in frame:
        stress_group = (pd.to_numeric(frame["temperature_c"], errors="coerce") / 10).round() * 10
        failures_by_group = (
            frame.loc[frame["event_observed"].astype(bool)].groupby(stress_group).size()
        )
        eligible_groups = failures_by_group[failures_by_group >= 3]
        if len(eligible_groups) >= 2:
            fit_frame = frame.loc[stress_group.isin(eligible_groups.index)]
            single = fit_arrhenius_weibull(fit_frame, use_temperature_c=use_temperature_c)
            sweep = pd.DataFrame(
                sweep_arrhenius_weibull(fit_frame, use_temperatures_c=use_temperatures_c)
            )
            report["arrhenius_weibull"] = asdict(single)
            report["arrhenius_weibull_sweep"] = sweep.to_dict(orient="records")
            sweep.to_csv(output / "arrhenius_lifetime_sweep.csv", index=False)
        else:
            report["arrhenius_weibull"] = {
                "status": "skipped",
                "reason": "Need at least two 10 C stress groups with three failures each",
                "failures_by_rounded_stress_c": {
                    str(key): int(value) for key, value in failures_by_group.items()
                },
            }
    numeric_features = [
        c
        for c in frame.select_dtypes(include="number").columns
        if c
        not in {
            "time_to_event",
            "event_observed",
            "threshold_ohm",
            "smoothing_windows",
            "persistence_windows",
            "baseline_feature_windows",
        }
    ]
    report["survival_forest"] = benchmark_survival_forest(
        frame, feature_columns=numeric_features[:10]
    )
    curve_time = np.linspace(0, float(frame["time_to_event"].max()) * 1.1, 200)
    curve = pd.DataFrame(
        {
            "time": curve_time,
            "survival_probability": survival_probability(
                curve_time, beta=weibull.beta, eta=weibull.eta
            ),
        }
    )
    curve.to_csv(output / "weibull_curve.csv", index=False)
    chart_paths = write_reliability_charts(
        frame,
        curve,
        output,
        use_temperatures_c=use_temperatures_c,
    )
    if "arrhenius_weibull_sweep" in report:
        sweep_chart = write_arrhenius_lifetime_sweep(sweep, output)
        if sweep_chart:
            chart_paths.append(sweep_chart)
    report["artifacts"] = {
        "weibull_curve.csv": sha256_file(output / "weibull_curve.csv"),
        **(
            {"arrhenius_lifetime_sweep.csv": sha256_file(output / "arrhenius_lifetime_sweep.csv")}
            if "arrhenius_weibull_sweep" in report
            else {}
        ),
        **{path.name: sha256_file(path) for path in chart_paths},
    }
    (output / "reliability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
