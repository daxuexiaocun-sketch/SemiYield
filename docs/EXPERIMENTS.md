# Evaluation Protocol

## Purpose

Define reproducible evaluation for yield-risk screening and device-reliability analysis.

## SECOM yield-risk screening

- Repeated stratified cross-validation measures small-sample variability; a chronological 80/20 split evaluates forward generalization.
- All models use identical outer splits. Cleaning, imputation, feature selection, calibration, and threshold selection are fitted on training data only.
- The default operating point reviews the top 10% of training out-of-fold risk scores. Primary metrics are PR-AUC, failure recall, MCC, Brier score, and failure capture at the review budget.
- CLI benchmarks execute in a monitored process tree. The default local policy warns at 32 GB RSS and terminates at 48 GB RSS.

Generated manifests record configuration, package versions, dataset metadata, and output hashes.

## MOSFET reliability analysis

- All observations from one device remain in one train, validation, or test partition.
- Weibull fitting supports right-censored units; Arrhenius–Weibull is reported only when temperature groups support the estimate.
- The optional survival forest uses device-level splits and reports concordance index.

See [model card](MODEL_CARD.md) and [reliability methods](RELIABILITY_METHODS.md) for interpretation limits.
