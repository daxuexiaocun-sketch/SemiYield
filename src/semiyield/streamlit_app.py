"""Bilingual local Streamlit interface for engineering analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from semiyield.data import data_quality_report, load_secom
from semiyield.drift import detect_drift
from semiyield.evaluate import threshold_table
from semiyield.explain import explain_prediction
from semiyield.modeling import train_model
from semiyield.predict import predict_risk
from semiyield.reliability import fit_arrhenius_weibull, fit_weibull, survival_probability

st.set_page_config(page_title="SemiYield", page_icon="🔬", layout="wide")

language = st.sidebar.selectbox("Language / 语言", ["中文", "English"])
zh = language == "中文"
pages = {
    "项目说明" if zh else "Overview": "overview",
    "数据质量" if zh else "Data quality": "quality",
    "模型评估" if zh else "Model benchmark": "benchmark",
    "阈值与成本" if zh else "Threshold & cost": "threshold",
    "样本解释" if zh else "Sample explanation": "explain",
    "漂移监测" if zh else "Drift monitoring": "drift",
    "可靠性分析" if zh else "Reliability": "reliability",
}
page = pages[st.sidebar.radio("页面" if zh else "Page", list(pages))]
data_dir = Path(st.sidebar.text_input("Data directory", "data/raw"))


@st.cache_data
def cached_data(path: str):
    return load_secom(path)


@st.cache_resource
def cached_model(path: str, model_name: str):
    dataset = cached_data(path)
    return train_model(dataset.features, dataset.target, model_name=model_name)


st.title("SemiYield · Semiconductor Engineering Analytics")
if page == "overview":
    st.info(
            "用于良率风险筛查、候选变量排序与可靠性分析。"
        if zh
        else (
            "Yield-risk screening, candidate-variable ranking, and reliability analysis."
        )
    )
    st.markdown(
        "SECOM 数据的变量已经匿名化，因此模型解释不能映射到具体设备或工艺步骤。"
        if zh
        else (
            "SECOM variables are anonymized, so explanations cannot identify "
            "a physical tool or process step."
        )
    )
    report_path = Path("reports/verified/experiment_manifest.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        st.success(
            "已加载 NASA 与 SECOM 参考结果。"
            if zh
            else "NASA and SECOM reference results are available."
        )
        st.json(
            {
                "nasa_analysis_status": report.get("nasa_analysis_status"),
                "model_status": report.get("model_status"),
            }
        )
else:
    if page == "reliability":
        real_report_path = Path("reports/verified/reliability_0045/reliability_report.json")
        source_mode = st.sidebar.selectbox(
            "可靠性数据源" if zh else "Reliability data source",
            [
                "NASA 参考结果" if zh else "NASA reference results",
                "本地 CSV" if zh else "Local CSV",
            ],
        )
        if source_mode in {"NASA 参考结果", "NASA reference results"} and real_report_path.exists():
            report = json.loads(real_report_path.read_text(encoding="utf-8"))
            weibull = report["weibull"]
            metrics = st.columns(4)
            metrics[0].metric("β", f"{weibull['beta']:.3f}")
            metrics[1].metric("η", f"{weibull['eta']:.0f} s")
            metrics[2].metric("B10", f"{weibull['b10']:.0f} s")
            metrics[3].metric("C-index", f"{report['survival_forest']['c_index']:.3f}")
            st.warning(report["arrhenius_weibull"].get("extrapolation_warning", ""))
            st.json(report)
            st.stop()
        reliability_path = Path(
            st.sidebar.text_input("Lifetime CSV", "data/demo/mosfet_lifetime_smoke.csv")
        )
        if not reliability_path.exists():
            st.error(
                "请先运行 `semiyield data download-demo`"
                if zh
                else "Run `semiyield data download-demo` first"
            )
            st.stop()
        lifetime = pd.read_csv(reliability_path)
        if "dataset_role" in lifetime and lifetime["dataset_role"].eq("ci_smoke_only").any():
            st.warning(
                "当前为合成 CI 冒烟数据，不代表 NASA 实验结果。"
                if zh
                else "Synthetic example data: this is not a NASA experimental result."
            )
        weibull = fit_weibull(lifetime)
        metrics = st.columns(3)
        metrics[0].metric("β", f"{weibull.beta:.3f}")
        metrics[1].metric("η", f"{weibull.eta:.1f}")
        metrics[2].metric("B10", f"{weibull.b10:.1f}")
        curve_time = pd.Series(range(0, int(lifetime["time_to_event"].max() * 1.1) + 1))
        st.line_chart(
            pd.DataFrame(
                {
                    "time": curve_time,
                    "survival": survival_probability(
                        curve_time, beta=weibull.beta, eta=weibull.eta
                    ),
                }
            ).set_index("time")
        )
        if "temperature_c" in lifetime and lifetime["temperature_c"].nunique() >= 2:
            st.json(fit_arrhenius_weibull(lifetime).__dict__)
        st.stop()
    try:
        dataset = cached_data(str(data_dir))
    except Exception as exc:
        st.error(
            f"请先运行 `semiyield download`: {exc}"
            if zh
            else f"Run `semiyield download` first: {exc}"
        )
        st.stop()
    if page == "quality":
        report = data_quality_report(dataset.features, dataset.target)
        cols = st.columns(4)
        for container, (key, value) in zip(cols, list(report.items())[:4], strict=False):
            container.metric(key, value if not isinstance(value, list) else len(value))
        st.bar_chart(dataset.features.isna().mean().sort_values(ascending=False).head(30))
        st.json(report)
    else:
        model_name = st.sidebar.selectbox("Model", ["logistic", "catboost"])
        try:
            artifact = cached_model(str(data_dir), model_name)
        except ImportError as exc:
            st.error(str(exc))
            st.stop()
        if page == "benchmark":
            verified_path = Path("reports/verified/yield/summary.csv")
            report_path = (
                verified_path
                if verified_path.exists()
                else Path("reports/reference/yield/summary.csv")
            )
            if report_path.exists():
                st.dataframe(pd.read_csv(report_path), use_container_width=True)
                st.caption(
                    "采用统一外层切分与训练期阈值选择。"
                    if zh
                    else "Generated with shared outer splits and training-only threshold selection."
                )
            else:
                st.info(
                    "运行 `semiyield benchmark --profile quick` 生成模型对比。"
                    if zh
                    else "Run `semiyield benchmark --profile quick` for a fair comparison."
                )
        elif page == "threshold":
            probabilities = artifact.estimator.predict_proba(dataset.features)[:, 1]
            missed_cost = st.slider("Missed failure cost", 1, 100, 10)
            review_cost = st.slider("Review cost", 1, 20, 1)
            table = threshold_table(
                dataset.target, probabilities, missed_cost=missed_cost, review_cost=review_cost
            )
            st.line_chart(table.set_index("threshold")[["estimated_cost", "reviewed"]])
            st.dataframe(table, use_container_width=True)
        elif page == "explain":
            index = st.number_input("Sample index", 0, len(dataset.features) - 1, 0)
            result = predict_risk(artifact, dataset.features.iloc[[index]])
            st.metric("Failure probability", f"{result.iloc[0].failure_probability:.2%}")
            st.dataframe(
                explain_prediction(
                    artifact, dataset.features.iloc[[index]], dataset.features, top_k=15
                ),
                use_container_width=True,
            )
        elif page == "drift":
            upload = st.file_uploader("Upload current CSV", type="csv")
            current = (
                pd.read_csv(upload)
                if upload
                else dataset.features.tail(max(100, len(dataset.features) // 5))
            )
            report = detect_drift(
                dataset.features.iloc[: len(dataset.features) - len(current)],
                current,
                artifact=artifact,
            )
            st.bar_chart(report["severity"].value_counts())
            st.dataframe(report, use_container_width=True)
