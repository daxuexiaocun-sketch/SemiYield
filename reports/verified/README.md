# Reference Experiment Results

## Scope

This directory contains aggregate reference results for all four SemiYield routes: UCI SECOM
manufacturing screening, local packaging proxy-failure screening, NASA Power MOSFET reliability,
and the linked synthetic three-stage demonstration. Machine-readable manifests retain the applicable
configuration, input identity, split protocol, and artifact hashes.

## Results

| Analysis | Reference result |
|---|---:|
| Logistic Regression, SECOM repeated-CV PR-AUC | 0.113 |
| CatBoost, SECOM repeated-CV PR-AUC | 0.167 |
| Packaging logistic, three-fold PR-AUC / ROC-AUC | 0.882 / 0.979 |
| Packaging logistic, recall / F1 / MCC | 0.943 / 0.773 / 0.759 |
| NASA Weibull β / η at 0.045 Ω | 0.832 / 10,553 s |
| NASA B10 | 706 s |
| NASA device-isolated survival-forest C-index | 0.839 |
| Synthetic three-stage process pass fraction | 80.73% |

![All-device degradation](nasa/degradation_trends.svg)

![Weibull survival](reliability_0045/weibull_survival.svg)

![Packaging proxy-failure model comparison](packaging/benchmark_summary.svg)

![Synthetic three-stage funnel](demo/stage_funnel.svg)

## Reproducibility

- `yield/` contains benchmark metrics, resource summaries, and output manifests.
- `packaging/` contains a real local-input aggregate benchmark, source hash, and no raw rows,
  model files, or individual predictions. Its outcome is a low-throughput proxy label, not a
  physical-device-failure result.
- `nasa/` contains conversion provenance and fixed device partitions.
- `reliability_0045/` and `reliability_0050/` contain lifetime reports and curves.
- `demo/` contains only synthetic aggregate metrics, charts, curves, and provenance; model files
  and individual device records are intentionally excluded.
- `experiment_manifest.json` indexes the published artifacts.

NASA row-level derived data is not included. Refer to the [NASA data pipeline](../../docs/NASA_PIPELINE.md) and [reliability data card](../../docs/RELIABILITY_DATA_CARD.md) for source handling and scope.
