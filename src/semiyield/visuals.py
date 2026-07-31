"""Small, deterministic SVG artifacts for reports and project documentation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_benchmark_chart(summary: pd.DataFrame, output: str | Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    destination = Path(output)
    plt.rcParams["svg.hashsalt"] = "semiyield-v1"
    selected = summary.loc[
        summary["protocol"].eq("repeated_cv"), ["model", "pr_auc_mean", "pr_auc_std"]
    ].copy()
    if selected.empty:
        return None
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(
        selected["model"],
        selected["pr_auc_mean"],
        yerr=selected["pr_auc_std"],
        capsize=4,
        color="#2878B5",
    )
    axis.set_ylabel("PR-AUC")
    axis.set_title("SECOM repeated cross-validation")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(figure)
    return destination


def write_reliability_charts(
    lifetime: pd.DataFrame,
    curve: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
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
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    survival_path = output / "weibull_survival.svg"
    figure.savefig(survival_path, format="svg", metadata={"Date": None})
    plt.close(figure)
    paths.append(survival_path)
    if "temperature_c" in lifetime:
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        for temperature, rows in lifetime.groupby("temperature_c"):
            axis.scatter(
                [temperature] * len(rows),
                rows["time_to_event"],
                label=f"{temperature:g} °C",
                alpha=0.8,
            )
        axis.set(
            xlabel="Stress temperature (°C)",
            ylabel="Observed/censored time",
            title="Accelerated-life observations",
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        alt_path = output / "accelerated_life.svg"
        figure.savefig(alt_path, format="svg", metadata={"Date": None})
        plt.close(figure)
        paths.append(alt_path)
    return paths


def write_degradation_chart(features: pd.DataFrame, output: str | Path) -> Path | None:
    """Plot all-device corrected RDS(on) trajectories without publishing row-level data."""
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
        title="NASA MOSFET all-device degradation trajectories",
    )
    axis.set_ylim(-0.1, 0.25)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(figure)
    return destination
