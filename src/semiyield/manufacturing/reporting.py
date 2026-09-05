"""Static report artifacts for the manufacturing-yield line."""

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
