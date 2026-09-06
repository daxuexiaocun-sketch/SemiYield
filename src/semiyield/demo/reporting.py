"""Static plots and a portable linked report for the synthetic workflow."""

import numpy as np
import pandas as pd

from semiyield.common.reporting import save_svg


def _kaplan_meier(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute a compact empirical survival curve without an extra dependency."""
    rows = frame.sort_values("time_to_event")
    survival = 1.0
    values = [(0.0, survival)]
    for time, group in rows.groupby("time_to_event", sort=True):
        at_risk = int((rows.time_to_event >= time).sum())
        events = int(group.event_observed.sum())
        if at_risk and events:
            survival *= 1 - events / at_risk
        values.append((float(time), survival))
    return pd.DataFrame(values, columns=["time", "survival_probability"])


def write_charts(output, stages, metrics, packaging, lifetime, threshold, curve):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "semiyield-three-stage-v1"
    paths = []

    def save(figure, name, title, description):
        save_svg(figure, output / name, title=title, description=description)
        plt.close(figure)
        paths.append(name)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    labels = ["Manufacturing\n制造", "Packaging\n封测", "Lifetime\n寿命"]
    values = stages.entered.to_numpy()
    axis.plot(range(3), values, color="#7654b8", linewidth=3, marker="o", markersize=10)
    for position, row in stages.reset_index(drop=True).iterrows():
        if row.stage == "lifetime":
            detail = f"{int(row.observed_failures):,} events / {int(row.censored):,} censored"
        else:
            detail = f"{int(row.observed_failures):,} rejected ({row.failure_rate:.1%})"
        axis.annotate(
            f"{int(row.entered):,}\n{detail}",
            (position, row.entered),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.annotate(
        f"Overall process pass / 总工艺通过率: {values[-1] / values[0]:.2%}",
        (1, max(values) * 0.83),
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.set(
        xticks=range(3),
        xticklabels=labels,
        ylabel="Devices entering stage / 进入阶段器件数",
        title="Synthetic end-to-end observed flow / 合成端到端观测流转",
    )
    axis.set_xlim(-0.25, 2.25)
    axis.set_ylim(0, max(values) * 1.18)
    axis.grid(axis="y", alpha=0.25)
    save(
        figure,
        "stage_flow.svg",
        "Synthetic three-stage device flow",
        "Stage entries, observed process rejections, lifetime events, and censoring.",
    )

    score_columns = ["pr_auc", "roc_auc", "f1", "mcc"]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    matrix = metrics.reindex(columns=score_columns).to_numpy(dtype=float)
    image = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Purples")
    axis.set_xticks(range(len(score_columns)), ["PR-AUC", "ROC-AUC", "F1", "MCC"])
    axis.set_yticks(
        range(len(metrics)),
        [f"{r.stage.title()} / {r.model.title()}" for r in metrics.itertuples()],
    )
    for row, record in enumerate(metrics.itertuples()):
        for column, key in enumerate(score_columns):
            value = getattr(record, key, np.nan)
            axis.text(
                column, row, "N/A" if pd.isna(value) else f"{value:.3f}", ha="center", va="center"
            )
    figure.colorbar(image, ax=axis, label="Holdout score / 留出集分数")
    axis.set(title="Batch-isolated model evaluation / 批次隔离模型评估")
    save(
        figure,
        "model_metrics.svg",
        "Synthetic holdout model metric matrix",
        "Dummy and logistic model scores for manufacturing and packaging; "
        "N/A values remain visible.",
    )

    figure, axis = plt.subplots(figsize=(9.2, 4.5))
    for split, rows in packaging.groupby("split", sort=True):
        axis.hist(rows.Y, bins=40, alpha=0.58, label=f"{split.title()} / {len(rows):,}")
    if threshold is not None:
        axis.axvline(
            threshold,
            color="#d47832",
            linewidth=2,
            label=f"Training threshold / 训练阈值: {threshold:.1f}",
        )
    failures = int(packaging.proxy_failed.sum())
    axis.legend(title=f"Proxy failures / 代理失效: {failures:,}")
    axis.set(
        xlabel="Synthetic throughput (units/hour)",
        ylabel="Devices",
        title="Low-throughput proxy label / 低吞吐代理标签",
    )
    save(
        figure,
        "throughput.svg",
        "Synthetic packaging throughput threshold",
        "Training and test throughput distributions with a training-only proxy-failure threshold.",
    )
    figure, axis = plt.subplots(figsize=(9.2, 4.5))
    training = lifetime.loc[lifetime.split.eq("train")]
    empirical = _kaplan_meier(training) if len(training) else None
    if empirical is not None:
        axis.step(
            empirical.time,
            empirical.survival_probability,
            where="post",
            color="#7654b8",
            linewidth=2,
            label="Kaplan–Meier / 经验生存",
        )
    if curve is not None:
        axis.plot(
            curve.time,
            curve.survival_probability,
            color="#4f8f6b",
            linestyle="--",
            linewidth=2,
            label="Weibull fit / Weibull 拟合",
        )
    if len(training):
        censored = training.loc[training.event_observed.eq(0)]
        if len(censored):
            axis.scatter(
                censored.time_to_event,
                np.interp(censored.time_to_event, empirical.time, empirical.survival_probability),
                marker="+",
                color="#172033",
                s=28,
                label=f"Censored / 删失: {len(censored):,}",
            )
        axis.text(
            0.98,
            0.08,
            f"Events / 事件: {int(training.event_observed.sum()):,}\n"
            f"Training / 训练: {len(training):,}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
        )
    else:
        axis.text(0.5, 0.5, "Lifetime fit unavailable; see reliability.json", ha="center")
    axis.set(
        xlabel="Time (hours)",
        ylabel="Survival probability",
        ylim=(0, 1.02),
        title="Synthetic lifetime with censoring / 合成寿命与右删失",
    )
    axis.legend(loc="upper right")
    save(
        figure,
        "survival.svg",
        "Synthetic lifetime survival with censoring",
        "Empirical training survival, Weibull fit, and right-censored observations.",
    )
    return paths


def markdown_table(frame):
    # Avoid an optional tabulate dependency in the offline demo.
    def cell(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "N/A"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def write_report(output, manifest, stages, metrics, reliability, charts):
    total = int(stages.entered.iloc[0])
    shipped = int(stages.entered.iloc[2])
    summary = metrics.reindex(
        columns=["stage", "model", "status", "pr_auc", "roc_auc", "f1", "mcc", "reason"]
    )
    threshold_text = (
        f"`Y < {manifest['throughput_threshold']}` (synthetic units/hour). Equality passes."
        if manifest["throughput_threshold"] is not None
        else "Unavailable: no training devices reached packaging."
    )
    content = f"""# 三环节模拟演示 / Three-stage synthetic demonstration

**全部数据为 synthetic，不代表 SECOM、NASA 或真实封测实验结论。**

Seed: {manifest["seed"]}. Split: nominal 80/20 by batch (test batches rounded up),
shared across all stages.
The packaging label threshold is derived only from manufacturing-passed training devices:
{threshold_text}
No latent variables, IDs, labels, throughput targets or future outcomes are model inputs.

## 阶段流转 / Observed stage flow

{markdown_table(stages)}

Overall process pass fraction: {shipped}/{total} = {shipped / total:.2%}.
Manufacturing and packaging rates are conditional on entry to each stage.
Lifetime events occur after both stages passed; censoring is not a pass label.
Post-shipment lifetime is not combined with process rates into one failure probability.

## 留出评估 / Holdout evaluation

{markdown_table(summary)}

Dummy and logistic models train on training batches only. Model predictions do not route devices.
Undefined metrics and skipped fits carry reasons in [metrics.json](metrics.json).
The Weibull curve is fitted to training survivors, pooled across stress temperatures;
it is descriptive, not a use-condition survival prediction or a holdout accuracy score.
Arrhenius fitting uses eligible training stress groups only; see
[reliability.json](reliability.json) for status and extrapolation limits.
Lifetime analysis status: {reliability["status"]}.

## 图表 / Charts

"""
    content += "\n\n".join(f"![{name.removesuffix('.svg')}]({name})" for name in charts)
    content += "\n\n[Stage counts](stages.csv)"
    content += " · [Metrics](metrics.csv) · [Provenance and hashes](manifest.json)\n"
    (output / "README.md").write_text(content, encoding="utf-8")
