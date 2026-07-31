# SemiYield

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/python-3.10--3.13-3776AB">
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

**Semiconductor yield-risk screening and device-reliability analysis.**

SemiYield provides reproducible workflows for high-dimensional manufacturing data and MOSFET thermal-overstress reliability studies. It supports risk scoring, candidate-variable ranking, drift monitoring, device-isolated lifetime analysis, and local interactive review.

![SemiYield graphical abstract](docs/assets/graphical-abstract-en.svg)

## Quick start

Use Python **3.10–3.13**. The first run downloads UCI SECOM and therefore requires network access.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[demo]"

semiyield quickstart
semiyield app
```

`quickstart` downloads SECOM, installs a synthetic local reliability example, and generates results under `reports/reference/`. It does not download the NASA archive. The equivalent expanded commands are:

```bash
semiyield download
semiyield data download-demo
semiyield benchmark --profile quick
semiyield reliability report --profile quick
```

Full NASA preparation is documented in the [NASA data pipeline](docs/NASA_PIPELINE.md).

## Reference results

### SECOM yield-risk screening

Shared outer splits keep preprocessing, feature selection, calibration, and threshold selection within training data.

| Model | PR-AUC | Failure recall | MCC | Capture at 10% review |
|---|---:|---:|---:|---:|
| Dummy | 0.066 | 0.000 | 0.000 | 0.057 |
| Logistic Regression | 0.113 | 0.392 | 0.040 | 0.164 |
| CatBoost | 0.167 | 0.308 | 0.160 | 0.269 |

![SECOM repeated-CV PR-AUC](reports/verified/yield/benchmark_pr_auc.svg)

### MOSFET reliability

The reference NASA analysis covers 41 device-isolated Power MOSFETs. At the temperature-corrected ΔRDS(on) threshold of 0.045 Ω, Weibull β is **0.832**, characteristic life η is **10,553 s**, B10 is **706 s**, and the device-isolated survival-forest C-index is **0.839**.

![NASA MOSFET degradation trajectories](reports/verified/nasa/degradation_trends.svg)

![NASA MOSFET Weibull survival curve](reports/verified/reliability_0045/weibull_survival.svg)

## Data and scope

- **UCI SECOM:** anonymous process variables for rare-failure screening; the source data is downloaded at runtime.
- **NASA Power MOSFET:** the upstream archive remains external. This repository publishes code, aggregate results, and provenance—not row-level NASA-derived data.
- Outputs support engineering review and reliability research. They do not replace process engineering, failure analysis, qualification, or causal root-cause investigation.

## Documentation

- [Evaluation protocol](docs/EXPERIMENTS.md)
- [Data cards](docs/DATA_CARD.md) and [reliability data card](docs/RELIABILITY_DATA_CARD.md)
- [Model card](docs/MODEL_CARD.md) and [reliability methods](docs/RELIABILITY_METHODS.md)
- [NASA data pipeline](docs/NASA_PIPELINE.md)
- [Reference experiment results](reports/verified/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Project activity

The repository workflow records cumulative GitHub Stars over time.

![GitHub stars over time](docs/assets/star-history.svg)

## License and citation

Code is licensed under [Apache-2.0](LICENSE). UCI SECOM, NASA data, and optional dependencies retain their own terms. Cite the project with [`CITATION.cff`](CITATION.cff).
