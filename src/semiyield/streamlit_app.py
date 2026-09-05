"""Bilingual business navigation for the local interface."""

import streamlit as st

from semiyield.manufacturing import page as manufacturing_page
from semiyield.packaging import page as packaging_page
from semiyield.reliability import page as reliability_page

st.set_page_config(page_title="SemiYield", page_icon="🔬", layout="wide")
zh = st.sidebar.selectbox("Language / 语言", ["中文", "English"]) == "中文"
st.title("SemiYield · Semiconductor Engineering Analytics")
pages = {
    "项目说明" if zh else "Overview": None,
    "制造良率" if zh else "Manufacturing yield": manufacturing_page.render,
    "封测过程" if zh else "Packaging process": packaging_page.render,
    "器件寿命" if zh else "Device lifetime": reliability_page.render,
}
render = pages[st.sidebar.radio("业务线路" if zh else "Business line", list(pages))]
if render:
    render(zh)
else:
    st.info(
        "制造良率 → 封测过程 → 器件寿命"
        if zh
        else "Manufacturing yield → Packaging process → Device lifetime"
    )
    st.markdown(
        "SECOM 用于良率风险筛查；封测以低吞吐率作为代理失效；MOSFET 用于寿命分析。"
        "三环节合成演示由 CLI 生成静态报告，独立于真实参考实验。"
        if zh
        else "SECOM screens yield risk; packaging uses low throughput as a proxy failure; "
        "MOSFET supports lifetime analysis. The CLI produces a separate synthetic "
        "three-stage demonstration and static reports."
    )
    st.code("uv run semiyield demo quickstart")
