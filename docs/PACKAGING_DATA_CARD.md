# Packaging process data card / 封测过程数据卡

## Input

Local input: `docs/Dataset_from_semiconductor_processes/mixed_categorical_numerical_data.csv`.
The accompanying readme identifies it as raw data and the `Dummy` CSV as an encoded representation.
The raw CSV contains 13,186 rows, five categorical columns (`X1`–`X5`), eleven numerical columns
(`X6`–`X16`), and target `Y`. Inspection found no missing values or duplicate rows.
The bundled R models and pre-encoded CSV are not used by the Python workflow.

The project uses `Y` as throughput according to the supplied task context. The local source readme
does not specify throughput units, feature meanings, batch/time identifiers or reuse terms.
Do not assign physical units or detailed machine semantics to these anonymous source fields.
The files remain local inputs rather than wheel assets; the synthetic demo does not depend on them.

## Outcome definition

**Low-throughput proxy failure / 低吞吐代理失效** is `Y < throughput_threshold`; equality passes.
This is a process screening label, not evidence of physical device failure or measured package yield.
The default threshold is the linear-interpolated 10th percentile of training `Y`.
A fixed finite engineering threshold can replace the percentile. `--threshold` and `--quantile`
are mutually exclusive. Ties can make the observed failure fraction differ from 10%.

```bash
uv run semiyield packaging data-status
uv run semiyield packaging validate --input-csv /path/to/mixed_categorical_numerical_data.csv
uv run semiyield packaging train --input-csv /path/to/mixed_categorical_numerical_data.csv
uv run semiyield packaging benchmark --input-csv /path/to/mixed_categorical_numerical_data.csv --models dummy,logistic
uv run semiyield packaging benchmark --input-csv /path/to/input.csv --threshold 100 --output-dir reports/packaging_fixed
uv run semiyield packaging train --input-csv /path/to/input.csv --quantile 0.15 --output artifacts/packaging/q15.joblib
uv run semiyield packaging predict artifacts/packaging/model.joblib input.csv
```

The value 100 above is an example, not an engineering recommendation. Every real-data command
requires an explicit `--input-csv`; this avoids silently depending on a Git-ignored local path.
Inference requires `X1`–`X16`, but does not require `Y`.

## Evaluation and persistence

Training uses a seeded 80/20 random holdout and derives its threshold only from training rows.
The persisted model remains trained on those rows, so its saved holdout metrics describe that model.
Benchmarking uses three shuffled outer folds (seed 42), identical for all requested models. Each
fold computes its own threshold only from its training rows. There are no batch/time identifiers
in the source, so this evaluation does not establish generalization to unseen batches or future time.

## Leakage and generalization audit

The Python implementation supplies only `X1`–`X16` to the preprocessing pipeline; `Y`, the derived
proxy label, and optional IDs are excluded. Each benchmark fold resolves its throughput threshold
from that fold's training `Y`, then applies it to the held-out fold. The inspected local source has
no duplicate feature rows.

This prevents direct target and split leakage in the implemented benchmark, but it cannot establish
that the anonymous source features are semantically independent of throughput or that the random
split represents future production. Several numerical features are strongly associated with `Y`,
and the categorical configurations recur in random training and test folds. The source readme does
not provide feature semantics, batch IDs, time ordering, or collection protocol. Treat the published
PR-AUC/ROC-AUC as within-source random-split discrimination only. A deployment claim needs a
time-ordered, batch-held-out, machine-held-out, or recipe-held-out evaluation after those identifiers
are available.

Numerical columns use training-only median imputation and scaling; categorical columns use missing
category imputation and one-hot encoding. Unknown inference categories are ignored by the encoder.
`Y`, generated labels and identifiers never enter model features. No probability calibration or
probability threshold search is performed in this initial packaging baseline; predictions use 0.5.

Outputs include fold metrics, numeric fold means, and a manifest with per-fold threshold provenance,
seed and source hash. Undefined metrics are null with reasons in the manifest, blank in CSV, and
excluded from their summary means. Confusion counts in `summary.csv` are fold means, not totals.
Metrics include PR-AUC (average precision), ROC-AUC, recall, precision, F1, MCC and TN/FP/FN/TP.
Models persist the input schema, both thresholds, label rule, seed and source hash.

Invalid or missing `Y`, infinite inputs, missing feature columns, invalid threshold parameters and
single-class training sets fail with an explicit message. Missing feature values are imputed.
