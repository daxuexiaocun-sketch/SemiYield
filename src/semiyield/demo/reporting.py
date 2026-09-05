"""Static plots and a portable linked report for the synthetic workflow."""

import numpy as np
import pandas as pd


def write_charts(output, stages, metrics, packaging, threshold, curve):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "semiyield-three-stage-v1"
    paths = []

    def save(figure, name):
        figure.tight_layout()
        figure.savefig(output / name, format="svg", metadata={"Date": None})
        plt.close(figure)
        paths.append(name)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(["Manufacturing", "Packaging", "Lifetime"], stages.entered, color="#2878b5")
    axis.set(ylabel="Devices entering stage", title="Synthetic stage flow")
    save(figure, "stage_funnel.svg")
    figure, axis = plt.subplots(figsize=(8, 4))
    rates = stages.failure_rate.iloc[:2]
    labels = ["Manufacturing failure", "Packaging proxy failure"]
    for position, rate in enumerate(rates):
        if pd.notna(rate):
            axis.bar(position, rate, color="#d47832")
        else:
            axis.text(position, 0.05, "Unavailable", ha="center")
    axis.set_xticks([0, 1], labels)
    axis.set(
        ylabel="Conditional stage failure fraction",
        ylim=(0, 1),
        title="Synthetic observed process outcomes",
    )
    save(figure, "stage_failure_rates.svg")
    figure, axis = plt.subplots(figsize=(8, 4))
    valid = metrics.loc[metrics.status.eq("completed")].dropna(subset=["pr_auc"])
    if len(valid):
        axis.bar(valid.stage + "/" + valid.model, valid.pr_auc, color="#2878b5")
    else:
        axis.text(0.5, 0.5, "Insufficient holdout classes; see metrics.json", ha="center")
    axis.set(ylabel="Holdout PR-AUC", ylim=(0, 1), title="Batch-isolated synthetic evaluation")
    save(figure, "model_comparison.svg")
    figure, axis = plt.subplots(figsize=(8, 4))
    if len(packaging):
        axis.hist(packaging.Y, bins=40, color="#2878b5", alpha=0.8)
    if threshold is not None:
        axis.axvline(threshold, color="#d47832", label=f"Training 10% threshold: {threshold:.1f}")
        axis.legend()
    axis.set(
        xlabel="Synthetic throughput (units/hour)",
        ylabel="Devices",
        title="Low-throughput proxy label",
    )
    save(figure, "throughput.svg")
    figure, axis = plt.subplots(figsize=(8, 4))
    if curve is not None:
        axis.plot(curve.time, curve.survival_probability)
    else:
        axis.text(0.5, 0.5, "Lifetime fit unavailable; see reliability.json", ha="center")
    axis.set(
        xlabel="Time (hours)",
        ylabel="Survival probability",
        ylim=(0, 1.02),
        title="Synthetic training-cohort Weibull (pooled stress temperatures)",
    )
    save(figure, "survival.svg")
    return paths


def markdown_table(frame):
    # Avoid an optional tabulate dependency in the offline demo.
    def cell(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "N/A"
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
    summary = metrics.reindex(columns=["stage", "model", "status", "pr_auc", "roc_auc", "reason"])
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
