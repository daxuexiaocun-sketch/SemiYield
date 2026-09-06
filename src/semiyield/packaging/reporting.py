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
    matrix = summary.reindex(columns=available).to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="YlOrBr")
    axis.set_xticks(
        range(len(available)), [metric.replace("_", " ").upper() for metric in available]
    )
    axis.set_yticks(range(len(summary.index)), [str(model).title() for model in summary.index])
    for row, model in enumerate(summary.index):
        for column, metric in enumerate(available):
            value = summary.loc[model, metric]
            label = "N/A" if pd.isna(value) else f"{value:.3f}"
            axis.text(column, row, label, ha="center", va="center", fontsize=10)
    figure.colorbar(image, ax=axis, label="Aggregate score / 聚合分数")
    axis.set_title(
        "Packaging proxy failure / 封测低吞吐代理失效\n"
        "Random three-fold internal validation / 随机三折内部验证"
    )
    axis.set_xlabel("Metrics / 指标")
    axis.set_ylabel("Model / 模型")
    matrix_path = destination / "benchmark_summary.svg"
    paths = [
        save_svg(
            figure,
            matrix_path,
            title="Packaging proxy-failure model metrics",
            description=(
                "Three-fold internal validation metric matrix. The label is a throughput proxy, "
                "not physical device failure."
            ),
        )
    ]
    plt.close(figure)
    thresholds = fold_metrics.loc[:, ["fold", "model", "throughput_threshold"]].drop_duplicates()
    figure, axis = plt.subplots(figsize=(8.6, 4.2))
    for model, rows in thresholds.groupby("model", sort=True):
        axis.scatter(rows["fold"], rows["throughput_threshold"], s=72, label=str(model).title())
    axis.set(
        xlabel="Outer fold / 外层折",
        ylabel="Training throughput threshold / 训练吞吐阈值",
        title="Packaging threshold provenance / 封测阈值溯源",
    )
    axis.set_xticks(sorted(thresholds.fold.unique()))
    axis.grid(alpha=0.25)
    axis.legend(title="Model / 模型")
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
