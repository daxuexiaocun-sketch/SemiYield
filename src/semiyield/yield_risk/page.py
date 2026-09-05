"""Manufacturing review pages."""

from pathlib import Path

import pandas as pd
import streamlit as st

from semiyield.data import data_quality_report, load_secom
from semiyield.drift import detect_drift
from semiyield.evaluate import threshold_table
from semiyield.explain import explain_prediction
from semiyield.modeling import train_model
from semiyield.predict import predict_risk


@st.cache_data
def cached_data(path: str):
    return load_secom(path)


@st.cache_resource
def cached_model(path: str, model_name: str):
    dataset = cached_data(path)
    return train_model(dataset.features, dataset.target, model_name=model_name)


def render(zh):
    pages = {
        "数据质量" if zh else "Data quality": "quality",
        "模型评估" if zh else "Model benchmark": "benchmark",
        "阈值与成本" if zh else "Threshold & cost": "threshold",
        "样本解释" if zh else "Sample explanation": "explain",
        "漂移监测" if zh else "Drift monitoring": "drift",
    }
    page = pages[st.sidebar.radio("页面" if zh else "Page", list(pages))]
    data_dir = Path(st.sidebar.text_input("Data directory", "data/raw"))
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
