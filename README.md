# SemiYield

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/python-3.10--3.13-3776AB">
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

**Four reproducible routes: manufacturing yield-risk, packaging process screening, device lifetime, and a linked synthetic demonstration.**

SemiYield separates three business workflows—SECOM manufacturing yield, low-throughput packaging proxy failure, and MOSFET lifetime analysis—and presents a fourth route: a reproducible synthetic demonstration linking devices across all three stages.

![SemiYield graphical abstract](docs/assets/graphical-abstract-en.svg)

## Quick start

Use Python **3.13** for the demo (CI covers 3.10, 3.12 and 3.13). Install [uv](https://docs.astral.sh/uv/) first.

```bash
uv sync --locked --extra demo --extra dev
uv run semiyield demo quickstart
uv run semiyield app
```

The synthetic demo runs offline after dependency installation. It generates 100 batches × 100
linked devices through manufacturing, packaging and lifetime, then writes a static report to
`reports/demo/three_stage/README.md`. Existing output requires `--force` to replace.
See the [complete demo guide](docs/DEMO.md).

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

## Four-route reference results

| Route | Input and evidence type | Method | Published aggregate result | Reproduce |
|---|---|---|---|---|
| Manufacturing yield | UCI SECOM, measured process/failure labels | Repeated CV risk screening | CatBoost PR-AUC **0.167**; 10% review capture **0.269** | `semiyield yield benchmark --profile quick` |
| Packaging process | Local mixed process data; throughput proxy label | Training-only 10% threshold, three-fold benchmark | Logistic PR-AUC **0.882**, ROC-AUC **0.979**, recall **0.943**, MCC **0.759** | `semiyield packaging benchmark --input-csv …` |
| Device lifetime | NASA Power MOSFET observations | Weibull and accelerated-life analysis | At ΔRDS(on)=0.045 Ω: β **0.832**, η **10,553 s**, B10 **706 s** | See [NASA pipeline](docs/NASA_PIPELINE.md) |
| Three-stage demo | Synthetic linked batches and units | Batch-isolated routing and holdout evaluation | 80.73% process pass fraction; packaging logistic PR-AUC **0.680** | `semiyield demo quickstart` |

The packaging route is a **low-throughput proxy failure** result, not proof of physical device failure. The demo is entirely **synthetic** and is not SECOM, NASA, or packaging experimental evidence.

![SECOM repeated-CV PR-AUC](reports/verified/yield/benchmark_pr_auc.svg)

![Packaging proxy-failure model comparison](reports/verified/packaging/benchmark_summary.svg)

![NASA MOSFET degradation trajectories](reports/verified/nasa/degradation_trends.svg)

![Synthetic three-stage funnel](reports/verified/demo/stage_funnel.svg)

See [packaging aggregate artifacts](reports/verified/packaging/manifest.json) and the [synthetic three-stage report](reports/verified/demo/README.md) for full methodology and provenance.

## Data and scope

- **UCI SECOM:** anonymous process variables for rare-failure screening; the source data is downloaded at runtime.
- **Packaging process:** mixed categorical/numerical local inputs; throughput below a training-only threshold is a proxy failure, not a physical device-failure label. See the [data card](docs/PACKAGING_DATA_CARD.md).
- **Synthetic three-stage demo:** linked batch/device records, clearly separated from real reference results.
- **NASA Power MOSFET:** the upstream archive remains external. This repository publishes code, aggregate results, and provenance—not row-level NASA-derived data.
- Outputs support engineering review and reliability research. They do not replace process engineering, failure analysis, qualification, or causal root-cause investigation.

## Documentation

- [Architecture / 业务架构](docs/ARCHITECTURE.md)
- [Packaging data and methods / 封测数据与方法](docs/PACKAGING_DATA_CARD.md)
- [Three-stage demo / 三环节演示](docs/DEMO.md)

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
