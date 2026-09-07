"""Static report artifacts for the manufacturing-yield line."""

from pathlib import Path

import pandas as pd

from semiyield.common.reporting import write_pr_auc_benchmark_chart


def write_benchmark_chart(summary: pd.DataFrame, output: str | Path) -> Path | None:
    selected = summary.loc[
        summary["protocol"].eq("repeated_cv"), ["model", "pr_auc_mean", "pr_auc_std"]
    ].copy()
    return write_pr_auc_benchmark_chart(
        selected,
        output,
        title="SECOM manufacturing yield · PR-AUC · repeated stratified CV",
        description=(
            "Bar chart of PR-AUC means and standard deviations for SECOM manufacturing models; "
            "Dummy is a baseline."
        ),
        y_max=0.5,
    )
