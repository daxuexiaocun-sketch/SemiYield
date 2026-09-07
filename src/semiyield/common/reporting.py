from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

MODEL_ORDER = ("dummy", "logistic", "catboost")
MODEL_COLORS = {
    "dummy": "#9aa9b8",
    "logistic": "#2878b5",
    "catboost": "#d47832",
}


def write_json_report(payload: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return destination


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def save_svg(figure, path: str | Path, *, title: str, description: str) -> Path:
    """Save a deterministic SVG with accessible, portable metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from matplotlib import font_manager, rcParams

        cjk_font = Path("/System/Library/Fonts/STHeiti Medium.ttc")
        if cjk_font.is_file():
            font_manager.fontManager.addfont(cjk_font)
            rcParams["font.sans-serif"] = ["STHeiti", "DejaVu Sans"]
    except ImportError:
        pass
    figure.tight_layout()
    figure.savefig(destination, format="svg", metadata={"Date": None})
    content = destination.read_text(encoding="utf-8")
    marker = content.find(">", content.find("<svg")) + 1
    metadata = f"\n<title>{escape(title)}</title>\n<desc>{escape(description)}</desc>"
    annotated = content[:marker] + metadata + content[marker:]
    destination.write_text(
        "\n".join(line.rstrip() for line in annotated.splitlines()) + "\n", encoding="utf-8"
    )
    return destination


def write_pr_auc_benchmark_chart(
    summary: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
    description: str,
    mean_column: str = "pr_auc_mean",
    std_column: str = "pr_auc_std",
    y_max: float = 1.08,
) -> Path | None:
    """Write a consistently styled PR-AUC benchmark chart for available models."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return None

    if "model" not in summary or mean_column not in summary:
        return None
    selected = summary.loc[:, ["model", mean_column]].copy()
    selected[std_column] = summary.get(std_column, 0.0)
    selected["model"] = selected["model"].astype(str).str.lower()
    ordered_models = [model for model in MODEL_ORDER if model in set(selected["model"])]
    ordered_models.extend(model for model in selected["model"] if model not in ordered_models)
    selected = selected.set_index("model").loc[ordered_models].reset_index()
    if selected.empty:
        return None

    plt.rcParams["svg.hashsalt"] = "semiyield-pr-auc-v1"
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    x = list(range(len(selected)))
    means = selected[mean_column].to_numpy(dtype=float)
    deviations = selected[std_column].fillna(0.0).to_numpy(dtype=float)
    bars = axis.bar(
        x,
        means,
        yerr=deviations,
        capsize=4,
        color=[MODEL_COLORS.get(model, "#516174") for model in selected["model"]],
        edgecolor="#172033",
        linewidth=0.6,
        error_kw={"ecolor": "#172033", "elinewidth": 0.8},
    )
    for bar, value, deviation in zip(bars, means, deviations, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(y_max - 0.02, value + deviation + 0.025),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_xticks(x, [model.title() for model in selected["model"]])
    axis.set_xlim(-0.55, len(selected) - 0.15)
    axis.set_ylim(0, y_max)
    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel("PR-AUC (mean ± standard deviation)")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Patch(facecolor=MODEL_COLORS.get(model, "#516174"), label=model.title())
            for model in selected["model"]
        ],
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    result = save_svg(figure, path, title=title, description=description)
    plt.close(figure)
    return result


def data_quality_report(
    features: pd.DataFrame, target: pd.Series | None = None
) -> dict[str, object]:
    """Return a business-neutral tabular quality summary."""
    missing = features.isna().mean()
    report: dict[str, object] = {
        "rows": len(features),
        "columns": features.shape[1],
        "constant_columns": features.nunique(dropna=True)
        .le(1)
        .loc[lambda values: values]
        .index.tolist(),
        "duplicate_columns": int(features.T.duplicated().sum()),
        "high_missing_columns": missing[missing > 0.5].index.tolist(),
        "overall_missing_rate": float(features.isna().mean().mean()),
    }
    if target is not None:
        report.update({"failure_count": int(target.sum()), "failure_rate": float(target.mean())})
    return report
