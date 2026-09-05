"""Bilingual packaging data and report review."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from semiyield.common.reporting import data_quality_report
from semiyield.packaging.data import FEATURES, local_data_status, validate_data


def render(zh):
    st.subheader("封测过程 · 低吞吐代理失效" if zh else "Packaging · Low-throughput proxy failure")
    st.warning(
        "低吞吐率只是过程失效的代理指标，不能直接证明器件物理失效。原始 Y 的单位未说明。"
        if zh
        else "Low throughput is a process-failure proxy, not proof of physical device failure. "
        "The source unit of Y is unspecified."
    )
    input_text = st.sidebar.text_input("CSV", "")
    if not input_text:
        st.info(local_data_status()["message"])
        st.code("uv run semiyield packaging validate --input-csv /path/to/packaging.csv")
        return
    input_path = Path(input_text)
    try:
        data = validate_data(pd.read_csv(input_path))
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    st.json(data_quality_report(data[FEATURES]))
    st.dataframe(data.head(100), use_container_width=True)
    st.caption(
        "标签规则：Y < 训练集第 10 百分位；也可指定固定阈值。等于阈值视为通过。"
        if zh
        else "Label rule: Y < training-set 10th percentile, or an explicit fixed threshold. "
        "Equality passes."
    )
    report_dir = Path(
        st.sidebar.text_input("报告目录" if zh else "Report directory", "reports/packaging")
    )
    summary, manifest = report_dir / "summary.csv", report_dir / "manifest.json"
    if summary.exists() and manifest.exists():
        try:
            st.dataframe(pd.read_csv(summary), use_container_width=True)
            st.json(json.loads(manifest.read_text(encoding="utf-8")))
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    else:
        st.info("请先生成基准报告。" if zh else "Generate a benchmark report first.")
        st.code("uv run semiyield packaging benchmark")
