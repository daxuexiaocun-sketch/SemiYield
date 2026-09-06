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
    colors = ["#9aa9b8" if name == "dummy" else "#2878b5" for name in selected["model"]]
    axis.errorbar(
        selected["model"],
        selected["pr_auc_mean"],
        yerr=selected["pr_auc_std"],
        fmt="none",
        ecolor="#415266",
        capsize=5,
        zorder=1,
    )
    axis.scatter(selected["model"], selected["pr_auc_mean"], color=colors, s=82, zorder=2)
    baseline = float(selected.loc[selected["model"].eq("dummy"), "pr_auc_mean"].iloc[0])
    axis.axhline(baseline, color="#9aa9b8", linestyle="--", linewidth=1, label="Dummy baseline")
    for row in selected.itertuples(index=False):
        axis.annotate(
            f"{row.pr_auc_mean:.3f}",
            (row.model, row.pr_auc_mean),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set_ylabel("PR-AUC")
    axis.set_title("SECOM manufacturing / 制造良率 · repeated CV / 重复交叉验证")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    result = save_svg(
        figure,
        destination,
        title="SECOM repeated cross-validation PR-AUC",
        description=(
            "Point estimates and cross-validation standard deviations for manufacturing models."
        ),
    )
    plt.close(figure)
    return result
