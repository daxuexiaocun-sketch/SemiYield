# Model Card

## Purpose

Estimate statistical failure risk from anonymous manufacturing measurements.

## Models and outputs

SemiYield provides a majority baseline, weighted logistic regression, CatBoost, and optional TabPFN v2. Models return risk probabilities, review decisions, candidate-variable rankings, and drift reports.

## Evaluation

Use the [evaluation protocol](EXPERIMENTS.md): repeated stratified cross-validation for uncertainty and a chronological split for forward generalization. Report PR-AUC, failure recall, MCC, Brier score, and fixed-budget capture.

## Limitations

Predictions require revalidation for each fab, product, toolset, process node, prevalence, and inspection cost. Feature attributions are not causal root-cause analysis; drift alerts require engineering review.
