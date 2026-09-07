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
    """Write a primary PR-AUC chart and an aggregate metrics table."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "semiyield-packaging-v1"
    metrics = ["pr_auc", "roc_auc", "recall", "precision", "f1", "mcc"]
    available = [metric for metric in metrics if metric in fold_metrics]
    preferred_order = ["dummy", "logistic", "catboost"]
    models = [model for model in preferred_order if model in summary.index]
    models.extend(model for model in summary.index if model not in models)
    statistics = fold_metrics.groupby("model")[available].agg(["mean", "std"])
    statistics = statistics.reindex(models)

    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    x = list(range(len(models)))
    colors = {"dummy": "#9aa9b8", "logistic": "#2878b5", "catboost": "#d47832"}
    means = statistics[("pr_auc", "mean")].to_numpy(dtype=float)
    deviations = statistics[("pr_auc", "std")].fillna(0).to_numpy(dtype=float)
    bars = axis.bar(
        x,
        means,
        yerr=deviations,
        capsize=4,
        color=[colors.get(str(model), "#516174") for model in models],
    )
    for bar, value, deviation in zip(bars, means, deviations, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.02, value + deviation + 0.025),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(x, [str(model).title() for model in models])
    axis.set_ylim(0, 1.08)
    axis.set_title("Packaging proxy failure · PR-AUC · random three-fold internal validation")
    axis.set_xlabel("Model")
    axis.set_ylabel("PR-AUC (mean ± standard deviation)")
    axis.legend(
        handles=[
            Patch(facecolor=colors.get(str(model), "#516174"), label=str(model).title())
            for model in models
        ],
        title="Model",
        loc="upper left",
    )
    matrix_path = destination / "benchmark_summary.svg"
    paths = [
        save_svg(
            figure,
            matrix_path,
            title="Packaging proxy-failure PR-AUC by model",
            description=(
                "PR-AUC means with standard-deviation error bars from three-fold internal "
                "validation. The label is a throughput proxy, not physical device failure."
            ),
        )
    ]
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11.0, 1.95 + 0.42 * len(models)))
    axis.axis("off")
    values = []
    for model in models:
        row = [str(model).title()]
        for metric in available:
            mean = statistics.loc[model, (metric, "mean")]
            deviation = statistics.loc[model, (metric, "std")]
            row.append("—" if pd.isna(mean) else f"{mean:.3f} ± {deviation:.3f}")
        values.append(row)
    table = axis.table(
        cellText=values,
        colLabels=["Model", *[metric.replace("_", " ").upper() for metric in available]],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.01, 0.06, 0.98, 0.68],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1d3857")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2:
            cell.set_facecolor("#f3f6f9")
    axis.set_title("Packaging proxy-failure metrics", fontsize=12, pad=10)
    axis.text(
        0.5,
        0.88,
        "Random three-fold internal validation · training-only threshold · n=3 folds",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="#516174",
    )
    table_path = destination / "benchmark_metrics_table.svg"
    paths.append(
        save_svg(
            figure,
            table_path,
            title="Packaging proxy-failure metrics table",
            description=(
                "Mean plus standard deviation across three random internal-validation folds for "
                "the low-throughput proxy-failure label. Thresholds are derived from training "
                "data only."
            ),
        )
    )
    plt.close(figure)
    return paths
