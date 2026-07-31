# SemiYield

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/daxuexiaocun-sketch/SemiYield/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/python-3.10--3.13-3776AB">
  <a href="LICENSE"><img alt="Apache-2.0 许可证" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

**面向半导体制造良率风险筛查与器件可靠性分析。**

SemiYield 为高维制造过程数据与 MOSFET 热过应力可靠性研究提供可复现的分析流程，支持风险评分、候选变量排序、漂移监测、器件隔离寿命分析和本地交互式审阅。

![SemiYield 图形摘要](docs/assets/graphical-abstract-zh-CN.svg)

## 快速开始

请使用 Python **3.10–3.13**。首次运行需从 UCI 下载 SECOM，因此需要网络连接。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[demo]"

semiyield quickstart
semiyield app
```

`quickstart` 下载 SECOM、安装本地合成可靠性示例，并将结果生成至 `reports/reference/`；不会下载 NASA 原始包。等价的展开命令为：

```bash
semiyield download
semiyield data download-demo
semiyield benchmark --profile quick
semiyield reliability report --profile quick
```

完整 NASA 数据准备流程见 [NASA 数据流水线](docs/NASA_PIPELINE.md)。

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
- **NASA Power MOSFET：**上游原始包保持在仓库外。本仓库发布代码、聚合结果和溯源信息，不发布 NASA 逐样本派生数据。
- 输出用于工程复核和可靠性研究，不替代工艺工程、失效分析、认证流程或因果根因确认。

## 文档

- [评估协议](docs/EXPERIMENTS.md)
- [数据卡](docs/DATA_CARD.md) 与 [可靠性数据卡](docs/RELIABILITY_DATA_CARD.md)
- [模型卡](docs/MODEL_CARD.md) 与 [可靠性方法](docs/RELIABILITY_METHODS.md)
- [NASA 数据流水线](docs/NASA_PIPELINE.md)
- [参考实验结果](reports/verified/README.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)

## 项目活跃度

仓库工作流记录 GitHub Stars 的累计变化。

![GitHub Stars 随时间变化](docs/assets/star-history.svg)

## 许可证与引用

代码采用 [Apache-2.0](LICENSE) 许可证。UCI SECOM、NASA 数据和可选依赖保留各自条款；引用本项目请使用 [`CITATION.cff`](CITATION.cff)。
