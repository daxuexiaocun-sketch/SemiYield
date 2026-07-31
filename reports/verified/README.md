# Reference Experiment Results

## Scope

This directory contains aggregate reference results for the UCI SECOM benchmark and the external NASA Power MOSFET analysis. Machine-readable manifests retain configurations, input identities, split assignments, and artifact hashes.

## Results

| Analysis | Reference result |
|---|---:|
| Logistic Regression, SECOM repeated-CV PR-AUC | 0.113 |
| CatBoost, SECOM repeated-CV PR-AUC | 0.167 |
| NASA Weibull β / η at 0.045 Ω | 0.832 / 10,553 s |
| NASA B10 | 706 s |
| NASA device-isolated survival-forest C-index | 0.839 |

![All-device degradation](nasa/degradation_trends.svg)

![Weibull survival](reliability_0045/weibull_survival.svg)

## Reproducibility

- `yield/` contains benchmark metrics, resource summaries, and output manifests.
- `nasa/` contains conversion provenance and fixed device partitions.
- `reliability_0045/` and `reliability_0050/` contain lifetime reports and curves.
- `experiment_manifest.json` indexes the published artifacts.

NASA row-level derived data is not included. Refer to the [NASA data pipeline](../../docs/NASA_PIPELINE.md) and [reliability data card](../../docs/RELIABILITY_DATA_CARD.md) for source handling and scope.
