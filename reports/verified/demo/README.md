# Three-stage synthetic demonstration

**All data are synthetic and do not represent SECOM, NASA, or real packaging experiments.**

Seed: 42. Split: nominal 80/20 by batch (test batches rounded up),
shared across all stages.
The packaging label threshold is derived only from manufacturing-passed training devices:
`Y < 257.5891893426385` (synthetic units/hour). Equality passes.
No latent variables, IDs, labels, throughput targets or future outcomes are model inputs.

## Observed stage flow

| stage | dataset_role | entered | observed_failures | unknown_outcomes | passed | censored | failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| manufacturing | synthetic | 10000 | 992 | 0 | 9008.000 | N/A | 0.099 |
| packaging | synthetic | 9008 | 935 | 0 | 8073.000 | N/A | 0.104 |
| lifetime | synthetic | 8073 | 4305 | 0 | N/A | 3768.000 | 0.533 |

Overall process pass fraction: 8073/10000 = 80.73%.
Manufacturing and packaging rates are conditional on entry to each stage.
Lifetime events occur after both stages passed; censoring is not a pass label.
Post-shipment lifetime is not combined with process rates into one failure probability.

## Holdout evaluation

| stage | model | status | pr_auc | roc_auc | f1 | mcc | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| manufacturing | dummy | completed | 0.108 | 0.500 | 0.000 | N/A | N/A |
| manufacturing | logistic | completed | 0.271 | 0.728 | 0.282 | 0.189 | N/A |
| packaging | dummy | completed | 0.119 | 0.500 | 0.000 | N/A | N/A |
| packaging | logistic | completed | 0.680 | 0.934 | 0.571 | 0.535 | N/A |

Dummy and logistic models train on training batches only. Model predictions do not route devices.
Undefined metrics and skipped fits carry reasons in [metrics.json](metrics.json).
The Weibull curve is fitted to training survivors, pooled across stress temperatures;
it is descriptive, not a use-condition survival prediction or a holdout accuracy score.
Arrhenius fitting uses eligible training stress groups only; see
[reliability.json](reliability.json) for status and extrapolation limits.
Lifetime analysis status: completed.

## Charts

![stage_flow](stage_flow.svg)

![model_metrics](model_metrics.svg)

![survival](survival.svg)

[Stage counts](stages.csv) · [Metrics](metrics.csv) · [Provenance and hashes](manifest.json)
