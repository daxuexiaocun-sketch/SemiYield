"""Device lifetime review page."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from semiyield.reliability.lifetime import fit_arrhenius_weibull, fit_weibull, survival_probability


def render(zh):
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
            "请先运行 `semiyield reliability example install`"
            if zh
            else "Run `semiyield reliability example install` first"
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
                "survival": survival_probability(curve_time, beta=weibull.beta, eta=weibull.eta),
            }
        ).set_index("time")
    )
    if "temperature_c" in lifetime and lifetime["temperature_c"].nunique() >= 2:
        st.json(fit_arrhenius_weibull(lifetime).__dict__)
    st.stop()
