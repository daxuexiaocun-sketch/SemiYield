# Evaluation Protocol

## Purpose

Define reproducible evaluation for manufacturing yield, packaging proxy failure and device lifetime.

## SECOM yield-risk screening

- Repeated stratified cross-validation measures small-sample variability; a chronological 80/20 split evaluates forward generalization.
- All models use identical outer splits. Cleaning, imputation, feature selection, calibration, and threshold selection are fitted on training data only.
- The default operating point reviews the top 10% of training out-of-fold risk scores. Primary metrics are PR-AUC, failure recall, MCC, Brier score, and failure capture at the review budget.
- CLI benchmarks execute in a monitored process tree. By default, the warning threshold is 70% of physical memory and the termination threshold is 80%; `--soft-memory-gb` and `--memory-limit-gb` explicitly override them.

Generated manifests record configuration, package versions, dataset metadata, and output hashes.

## MOSFET reliability analysis

- All observations from one device remain in one train, validation, or test partition.
- Weibull fitting supports right-censored units; Arrhenius–Weibull is reported only when temperature groups support the estimate.
- The optional survival forest uses device-level splits and reports concordance index.

See [model card](MODEL_CARD.md) and [reliability methods](RELIABILITY_METHODS.md) for interpretation limits.

## Packaging process screening

Run `uv run semiyield packaging benchmark --input-csv /path/to/raw.csv`. The default three shuffled outer folds use identical
splits across models and derive each low-throughput threshold only from that fold's training Y.
Feature processing is fitted only on training rows; target Y is excluded. A fixed engineering
threshold can replace the default training 10th percentile. The probability decision cutoff is 0.5.
See [packaging methods](PACKAGING_DATA_CARD.md) for undefined metrics and source limitations.

## Linked synthetic demonstration

Run `uv run semiyield demo quickstart`. Assign train/test by batch before generating stage outcomes,
then propagate IDs and split assignments through observed manufacturing and packaging gates.
Evaluate classifiers on held-out batches; fit lifetime statistics only on training survivors.
These reports have a synthetic role and their own output directory, independently of reference
SECOM and NASA results. See the [demo data contract and guide](DEMO.md).
