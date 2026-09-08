# Model Card: Manufacturing Yield

## Purpose

Estimate statistical failure risk from anonymous manufacturing measurements.

## Models and outputs

SemiYield provides a majority baseline, weighted logistic regression, CatBoost, and optional TabPFN v2. Models return risk probabilities, review decisions, candidate-variable rankings, and drift reports.

## Evaluation

Use the [evaluation protocol](EXPERIMENTS.md): repeated stratified cross-validation for uncertainty and a chronological split for forward generalization. Report PR-AUC, failure recall, MCC, Brier score, and fixed-budget capture.
CatBoost hyperparameters are governed through explicit, versioned tuning artifacts; published
CatBoost benchmark results use nested validation rather than parameters selected on their test data.

## Limitations

Predictions require revalidation for each fab, product, toolset, process node, prevalence, and inspection cost. Feature attributions are not causal root-cause analysis; drift alerts require engineering review.

## Other business lines

The packaging models use a separate mixed-type pipeline and throughput proxy labels; see the
[packaging data and methods card](PACKAGING_DATA_CARD.md). Their probabilities are not calibrated
physical failure probabilities. Lifetime analysis is described in [reliability methods](RELIABILITY_METHODS.md).
