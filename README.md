# SemiYield

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/python-3.10--3.13-3776AB">
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

**Three reproducible routes: manufacturing yield-risk, packaging process screening, and device lifetime analysis.**

SemiYield separates three business workflows: SECOM manufacturing yield, low-throughput packaging proxy failure, and MOSFET lifetime analysis.

![SemiYield graphical abstract](docs/assets/graphical-abstract-en.svg)

## Quick start

Use Python **3.13** (CI covers 3.10, 3.12 and 3.13). Install [uv](https://docs.astral.sh/uv/) first.

```bash
uv sync --locked --extra charts --extra app --extra dev
uv run semiyield app
```

For real-data routes, pass a local input explicitly where required:

```bash
uv run semiyield packaging data-status
uv run semiyield packaging benchmark --input-csv /path/to/packaging.csv
uv run semiyield yield download
uv run semiyield yield benchmark --profile quick
uv run semiyield reliability example install
uv run semiyield reliability report --profile quick
```

Version 1.1 removes the former top-level commands and Python import aliases. Existing model files
from earlier releases must be re-trained or their old exported results retained. NASA preparation is
documented in the [NASA data pipeline](docs/NASA_PIPELINE.md). `uv.lock` is the dependency source of truth; extras must
be explicitly selected. See [architecture and environments](docs/ARCHITECTURE.md).

## Three-route results

| Route | Main result | Evidence boundary | Reproduce and details |
|---|---|---|---|
| Manufacturing | CatBoost PR-AUC **0.167** | Measured SECOM screening, not causal diagnosis | `semiyield yield benchmark --profile quick` · [report](reports/verified/README.md#manufacturing-yield) |
| Packaging | CatBoost PR-AUC **0.967** | Low-throughput proxy; random internal validation only | `semiyield packaging benchmark --input-csv …` · [report](reports/verified/README.md#packaging-process) |
| Lifetime | Weibull B10 **1,079 s** | NASA stress evidence; use-temperature result is an extrapolation | [report](reports/verified/README.md#device-lifetime) |

The packaging route is a **low-throughput proxy failure** result, not proof of physical device failure. Its source has no batch or time identifier, so the published random three-fold figures do not establish performance on a new production batch, machine, recipe, or time period.

## Data and scope

- **UCI SECOM:** anonymous process variables for rare-failure screening; the source data is downloaded at runtime.
- **Packaging process:** mixed categorical/numerical local inputs; throughput below a training-only threshold is a proxy failure, not a physical device-failure label. See the [data card](docs/PACKAGING_DATA_CARD.md).
- **NASA Power MOSFET:** the upstream archive remains external. This repository publishes code,
  aggregate results, provenance, and an explicitly authorized SVG-only RSF prediction for each
  protected test-device label—not raw row-level NASA-derived data.
- Outputs support engineering review and reliability research. They do not replace process engineering, failure analysis, qualification, or causal root-cause investigation.

## Documentation

- [Architecture / 业务架构](docs/ARCHITECTURE.md)
- [Packaging data and methods / 封测数据与方法](docs/PACKAGING_DATA_CARD.md)
- [Evaluation protocol](docs/EXPERIMENTS.md)
- [Data cards](docs/DATA_CARD.md) and [reliability data card](docs/RELIABILITY_DATA_CARD.md)
- [Model card](docs/MODEL_CARD.md) and [reliability methods](docs/RELIABILITY_METHODS.md)
- [NASA data pipeline](docs/NASA_PIPELINE.md)
- [Reference experiment results](reports/verified/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Project activity

The workflow reconstructs a curve from current stargazers and their timestamps. Removed stars cannot be recovered. See [maintenance and troubleshooting](docs/STAR_HISTORY.md).

![GitHub stars over time](docs/assets/star-history.svg)

## License and citation

Code is licensed under [Apache-2.0](LICENSE). UCI SECOM, NASA data, and optional dependencies retain their own terms. Cite the project with [`CITATION.cff`](CITATION.cff).
