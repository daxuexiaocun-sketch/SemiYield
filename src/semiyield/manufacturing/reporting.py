"""Static report artifacts for the manufacturing-yield line."""

from pathlib import Path

import pandas as pd

from semiyield.common.reporting import save_svg


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
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    colors = {"dummy": "#9aa9b8", "logistic": "#2878b5", "catboost": "#d47832"}
    bars = axis.bar(
        selected["model"].str.title(),
        selected["pr_auc_mean"],
        yerr=selected["pr_auc_std"],
        capsize=5,
        color=[colors.get(name, "#516174") for name in selected["model"]],
        edgecolor="#172033",
        linewidth=0.6,
    )
    for bar, value in zip(bars, selected["pr_auc_mean"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_ylabel("PR-AUC")
    axis.set_title("SECOM manufacturing yield · repeated stratified CV")
    axis.set_xlabel("Model (Dummy is baseline, not deployable)")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    result = save_svg(
        figure,
        destination,
        title="SECOM repeated cross-validation PR-AUC",
        description=(
            "Bar chart of PR-AUC means and standard deviations for SECOM manufacturing models; "
            "Dummy is a baseline."
        ),
    )
    plt.close(figure)
    return result
