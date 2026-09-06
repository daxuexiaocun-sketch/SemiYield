"""Static, aggregate-only report artifacts for packaging proxy-failure analysis."""

from pathlib import Path

import pandas as pd

from semiyield.common.reporting import save_svg

PROXY_FAILURE_TERM = "low-throughput proxy failure"
PROXY_FAILURE_NOTE = (
    "This label is an operational throughput proxy and does not establish a physical device "
    "failure."
)


def write_benchmark_charts(
    summary: pd.DataFrame, fold_metrics: pd.DataFrame, output_dir: str | Path
) -> list[Path]:
    """Write aggregate-only matrices and threshold provenance charts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "semiyield-packaging-v1"
    metrics = ["pr_auc", "roc_auc", "recall", "precision", "f1", "mcc"]
    available = [metric for metric in metrics if metric in summary]
    matrix = summary.reindex(columns=available)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    x = range(len(available))
    width = 0.8 / max(1, len(matrix.index))
    colors = {"dummy": "#9aa9b8", "logistic": "#2878b5", "catboost": "#d47832"}
    for position, (model, values) in enumerate(matrix.iterrows()):
        bars = axis.bar(
            [v + (position - (len(matrix.index) - 1) / 2) * width for v in x],
            values.to_numpy(dtype=float),
            width=width,
            label=str(model).title(),
            color=colors.get(str(model), "#516174"),
        )
        for bar, value in zip(bars, values, strict=True):
            if pd.notna(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.02,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axis.set_xticks(list(x), [metric.replace("_", " ").upper() for metric in available])
    axis.set_ylim(0, 1.08)
    axis.set_title("Packaging proxy failure · random three-fold internal validation")
    axis.set_xlabel("Metric (proxy label, not physical failure)")
    axis.set_ylabel("Score")
    axis.legend(title="Model", loc="upper right")
    matrix_path = destination / "benchmark_summary.svg"
    paths = [
        save_svg(
            figure,
            matrix_path,
            title="Packaging proxy-failure model metrics",
            description=(
                "Three-fold internal validation metric bars. The label is a throughput proxy, "
                "not physical device failure."
            ),
        )
    ]
    plt.close(figure)
    thresholds = fold_metrics.loc[:, ["fold", "model", "throughput_threshold"]].drop_duplicates()
    figure, axis = plt.subplots(figsize=(8.6, 4.2))
    offsets = {"dummy": -0.08, "logistic": 0.08, "catboost": 0.0}
    for model, rows in thresholds.groupby("model", sort=True):
        axis.scatter(
            rows["fold"] + offsets.get(str(model), 0),
            rows["throughput_threshold"],
            s=72,
            label=str(model).title(),
        )
    axis.set(
        xlabel="Outer fold",
        ylabel="Training throughput threshold",
        title="Packaging proxy-label threshold provenance",
    )
    axis.set_xticks(sorted(thresholds.fold.unique()))
    axis.grid(alpha=0.25)
    axis.legend(title="Model")
    threshold_path = destination / "thresholds.svg"
    paths.append(
        save_svg(
            figure,
            threshold_path,
            title="Packaging training-fold throughput thresholds",
            description="Each point is calculated only from the corresponding training fold.",
        )
    )
    plt.close(figure)
    return paths
