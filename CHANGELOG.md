# Changelog

## 1.1.0

- Rename the public manufacturing Python package to `semiyield.manufacturing` while retaining the `semiyield yield` CLI.
- Publish aggregate-only packaging reference artifacts and a results overview.

## 1.0.0

- Separate manufacturing yield, packaging process and device lifetime into pure business packages.
- Remove legacy top-level CLI commands, Python import aliases, `workflows.py`, and old joblib loading paths.
- Preserve the existing SECOM, packaging, and NASA/MOSFET analysis rules under the new command paths.
- Add training-only low-throughput proxy labels, mixed-type packaging models and matched-fold evaluation.
- Standardize development and CI on uv.lock; repair clean-runner star-history updates and calendar spacing.


## 0.3.0

- Added Python 3.13/CatBoost reference experiment results and a locked development environment.
- Added explicit NASA download and embedded-notice inspection commands.
- Added right-censored 0.045/0.05 ohm RDS(on) lifetime sensitivity outputs.
- Added manifest-driven survival splits and a hashed experiment manifest.
- Added separate English and Simplified Chinese READMEs, localized graphical abstracts,
  and a repository-owned star-history workflow.
- Removed nested calibration during OOF threshold selection and switched final calibration to
  a single fitted estimator.
- Added process-tree RSS monitoring, resource reports, and 32/48 GB local memory limits.

## 0.2.0

- Added leakage-safe multi-model benchmark reports and training-only budget thresholds.
- Added NASA archive inventory, normalized signal aggregation, device-level splits, and provenance.
- Added censored Weibull, Arrhenius-Weibull, optional survival forest, plots, and reliability UI.
- Added synthetic example data for CI and local workflow checks.

## 0.1.0

- Initial data, modeling, evaluation, explanation, drift, CLI, and Streamlit implementation.
