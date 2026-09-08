# Reference Experiment Results

## Scope

This directory contains aggregate reference results for all four SemiYield routes: UCI SECOM
manufacturing screening, local packaging proxy-failure screening, NASA Power MOSFET reliability,
and the linked synthetic three-stage demonstration. Machine-readable manifests retain the applicable
configuration, input identity, split protocol, and artifact hashes.

## Manufacturing yield

SECOM repeated cross-validation: CatBoost PR-AUC **0.167** and 10% review capture **0.269**.
This is a process-risk screening result, not a causal diagnosis.

![Manufacturing PR-AUC by model](yield/benchmark_pr_auc.svg)

[Aggregate metrics](yield/summary.csv) · [fold metrics](yield/fold_metrics.csv) · [manifest](yield/manifest.json)

## Packaging process

Local mixed process data: low-throughput proxy failure is defined as `Y` below each training fold's
10th percentile. Random three-fold internal validation: CatBoost PR-AUC **0.967**, Logistic
**0.882**, and Dummy **0.101**. This is an operational proxy, not physical device failure, and does
not establish generalization to new production batches, machines, recipes, or time periods.

![Packaging proxy-failure PR-AUC by model](packaging/benchmark_summary.svg)

[Aggregate metrics](packaging/summary.csv) · [fold metrics](packaging/fold_metrics.csv) · [manifest](packaging/manifest.json)

## Device lifetime

NASA MOSFET data at ΔRDS(on)=0.045 Ω: Weibull β **0.880**, η **13,897 s**, B10 **1,079 s**.
The 55 °C result is an accelerated-life extrapolation and needs mechanism validation.
The NASA curves use real stress-test MOSFET data and are not comparable to the synthetic Demo survival curve.

![NASA MOSFET degradation trajectories](nasa/degradation_trends.svg)

![NASA MOSFET Weibull and Kaplan-Meier survival](reliability_0045/weibull_survival.svg)

![Arrhenius-Weibull use-temperature lifetime sweep](reliability_0045/arrhenius_lifetime_sweep.svg)

The accelerated-life scatter plot shows observed high-temperature data; the lifetime sweep shows
declared use-temperature model estimates at 55, 85, 105, and 125 °C. All four are reported using
a conservative extrapolation policy; 125 °C lies within the fitted stress-temperature range but is
still not a directly observed 125 °C lifetime.
Each bar is a Weibull-distribution summary rather than one "true lifetime": **B10 = 10% failure
time** and **η = 63.2% failure time**. Both quantities are reported in seconds.
Every reliability report attempts the required RSF predictive module using baseline temperature,
initial RDS(on) level, and initial RDS(on) slope only. Its protected 9-device test-set metrics are
exploratory evidence within observed stress conditions; an unavailable dependency or data condition
is reported explicitly rather than hidden. RSF does not replace Weibull population summaries,
Arrhenius–Weibull extrapolation, or qualification testing.

For the fixed 24/8/9 NASA device split, RSF reports a **Harrell C-index of 0.905** (bootstrap 95% CI
**0.588–1.000**) on the protected 9-device test set. This measures risk-ranking discrimination, for
which 0.5 is approximately random; the wide interval reflects the small protected test sample.
The right-censoring-adjusted **Uno C-index is 0.862**, providing a complementary discrimination
check. The **integrated Brier score is 0.149**; it summarizes survival-probability prediction error
over time, where lower values are better. The observed-failure median-lifetime **MAE is 5,810 s**;
it is calculated only for test devices with observed failures and therefore excludes censored test
devices. These are exploratory, within-observed-stress predictive measurements, not a comparison
against Weibull or Arrhenius–Weibull population inference.

![NASA MOSFET RSF individual predicted survival](reliability_0045/rsf_individual_predicted_survival.svg)

The nine curves are individual RSF predictions for the protected test devices. They are model
predictions rather than Kaplan–Meier observations; observed failure and censoring outcomes are not
drawn. The individual device labels and curves are published under this report's explicit derived-
prediction exception, but they do not establish new-device generalization or a physical mechanism.

[Lifetime report and RSF metrics](reliability_0045/reliability_report.json) · [Weibull curve data](reliability_0045/weibull_curve.csv) · [temperature sweep data](reliability_0045/arrhenius_lifetime_sweep.csv)

## Three-stage synthetic demonstration

Synthetic, batch-isolated manufacturing → packaging → lifetime routing: **80.73%** overall process
pass fraction. It is not real SECOM, packaging, or NASA evidence.

![Synthetic stage flow](demo/stage_flow.svg)

![Synthetic model matrix](demo/model_metrics.svg)

![Synthetic survival and censoring](demo/survival.svg)

[Demo report](demo/README.md) · [stage counts](demo/stages.csv) · [metrics](demo/metrics.csv) · [manifest](demo/manifest.json)

## Reproducibility

- `yield/` contains benchmark metrics, resource summaries, and output manifests.
- `packaging/` contains a real local-input aggregate benchmark, source hash, and no raw rows,
  model files, or individual predictions. Its outcome is a low-throughput proxy label, not a
  physical-device-failure result.
- `nasa/` contains conversion provenance and fixed device partitions.
- `reliability_0045/` and `reliability_0050/` contain lifetime reports and curves, including the
  explicitly authorized protected-test device labels and RSF model-predicted curves; they do not
  contain feature values, observed device outcomes, raw rows, or downloadable predictions.
- `demo/` contains only synthetic aggregate metrics, charts, curves, and provenance; model files
  and individual device records are intentionally excluded.
- `experiment_manifest.json` indexes the published artifacts.

NASA row-level derived data is not included, except for the explicitly published protected-test
device labels and their SVG-only RSF model predictions. Refer to the [NASA data pipeline](../../docs/NASA_PIPELINE.md) and [reliability data card](../../docs/RELIABILITY_DATA_CARD.md) for source handling and scope.
