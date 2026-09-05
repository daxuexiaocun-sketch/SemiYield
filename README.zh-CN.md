# SemiYield

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/python-3.10--3.13-3776AB">
  <a href="LICENSE"><img alt="Apache-2.0 许可证" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

**制造良率风险、封测过程筛查与器件寿命分析。**

SemiYield 按 SECOM 制造良率、低吞吐封测代理失效、MOSFET 寿命三个业务组织代码，支持本地审阅，
并提供同批次器件跨越三个环节的可复现合成演示。

![SemiYield 图形摘要](docs/assets/graphical-abstract-zh-CN.svg)

## 快速开始

演示默认使用 Python **3.13**（CI 覆盖 3.10、3.12、3.13），先安装 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --locked --extra demo --extra dev
uv run semiyield demo quickstart
uv run semiyield app
```

依赖安装后，合成演示可以离线运行。默认生成 100 批次 × 100 个器件，串联制造、封测和寿命阶段，
静态报告位于 `reports/demo/three_stage/README.md`。已有输出需要 `--force` 才能覆盖。
详见[完整演示指南](docs/DEMO.md)。

本地封测 CSV 与已有真实数据流程：

```bash
uv run semiyield packaging validate
uv run semiyield packaging benchmark
uv run semiyield yield download
uv run semiyield yield benchmark --profile quick
uv run semiyield reliability example install
uv run semiyield reliability report --profile quick
```

1.0 版本移除了原有顶层命令、Python 导入别名和旧模型文件兼容层；旧模型需要重新训练，或保留旧版本导出的结果。
NASA 数据准备详见 [NASA 数据流水线](docs/NASA_PIPELINE.md)。
依赖统一以 `uv.lock` 为准，extras 需要显式选择，详见[业务架构与环境说明](docs/ARCHITECTURE.md)。

## 参考结果

### SECOM 良率风险筛查

所有模型共用外层切分，预处理、特征筛选、校准和阈值选择仅在训练数据内完成。

| 模型 | PR-AUC | 失败召回率 | MCC | 10% 复检捕获率 |
|---|---:|---:|---:|---:|
| Dummy | 0.066 | 0.000 | 0.000 | 0.057 |
| 逻辑回归 | 0.113 | 0.392 | 0.040 | 0.164 |
| CatBoost | 0.167 | 0.308 | 0.160 | 0.269 |

![SECOM 重复交叉验证 PR-AUC](reports/verified/yield/benchmark_pr_auc.svg)

### MOSFET 可靠性

参考 NASA 分析覆盖 41 个器件隔离的功率 MOSFET。在温度修正 ΔRDS(on) = 0.045 Ω 阈值下，Weibull β 为 **0.832**，特征寿命 η 为 **10,553 s**，B10 为 **706 s**，器件隔离生存森林 C-index 为 **0.839**。

![NASA MOSFET 退化轨迹](reports/verified/nasa/degradation_trends.svg)

![NASA MOSFET Weibull 生存曲线](reports/verified/reliability_0045/weibull_survival.svg)

## 数据与适用范围

- **UCI SECOM：**匿名过程变量，用于少数类失效风险筛查；源数据在运行时下载。
- **封测过程：**本地混合类型数据，以低于训练期吞吐阈值作为代理失效，不能直接等同于器件物理失效。详见[数据卡](docs/PACKAGING_DATA_CARD.md)。
- **三环节合成演示：**关联批次与器件 ID，与真实参考结果分开。
- **NASA Power MOSFET：**上游原始包保持在仓库外。本仓库发布代码、聚合结果和溯源信息，不发布 NASA 逐样本派生数据。
- 输出用于工程复核和可靠性研究，不替代工艺工程、失效分析、认证流程或因果根因确认。

## 文档

- [Architecture / 业务架构](docs/ARCHITECTURE.md)
- [Packaging data and methods / 封测数据与方法](docs/PACKAGING_DATA_CARD.md)
- [Three-stage demo / 三环节演示](docs/DEMO.md)

- [评估协议](docs/EXPERIMENTS.md)
- [数据卡](docs/DATA_CARD.md) 与 [可靠性数据卡](docs/RELIABILITY_DATA_CARD.md)
- [模型卡](docs/MODEL_CARD.md) 与 [可靠性方法](docs/RELIABILITY_METHODS.md)
- [NASA 数据流水线](docs/NASA_PIPELINE.md)
- [参考实验结果](reports/verified/README.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)

## 项目活跃度

工作流根据当前仍保留的星标及其时间重建曲线，无法恢复已取消的星标。详见[维护与故障排查](docs/STAR_HISTORY.md)。

![GitHub Stars 随时间变化](docs/assets/star-history.svg)

## 许可证与引用

代码采用 [Apache-2.0](LICENSE) 许可证。UCI SECOM、NASA 数据和可选依赖保留各自条款；引用本项目请使用 [`CITATION.cff`](CITATION.cff)。
