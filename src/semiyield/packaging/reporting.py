"""Static, aggregate-only report artifacts for packaging proxy-failure analysis."""

from pathlib import Path

import pandas as pd

PROXY_FAILURE_TERM = "low-throughput proxy failure"
PROXY_FAILURE_NOTE = (
    "This label is an operational throughput proxy and does not establish a physical device "
    "failure."
)


def write_benchmark_chart(summary: pd.DataFrame, output: str | Path) -> Path | None:
    """Write a portable model-comparison chart without row-level predictions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "semiyield-packaging-v1"
    metrics = ["pr_auc", "roc_auc", "recall", "precision", "f1", "mcc"]
    available = [metric for metric in metrics if metric in summary]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    positions = range(len(available))
    for model, rows in summary.iterrows():
        axis.plot(
            positions,
            [rows.get(metric) for metric in available],
            marker="o",
            linewidth=2,
            label=str(model),
        )
    axis.set_xticks(list(positions), [metric.replace("_", " ").upper() for metric in available])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Aggregate three-fold score")
    axis.set_title("Packaging low-throughput proxy-failure prediction")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Model")
    figure.tight_layout()
    figure.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(figure)
    return destination
