#!/usr/bin/env python3
# Run locally:
#   streamlit run app.py --server.fileWatcherType poll --server.runOnSave true

import streamlit as st
import tabs.daily_compare as dc
import tabs.live_vs_backtest as lvb

# Must be the first Streamlit call:
st.set_page_config(
    page_title="Trade Tools — Daily Compare & Live vs Backtest",
    layout="wide",
)

st.title("Trade Tools")

# Global compact-table styling (used by Live-vs-Backtest and Daily Compare)
st.markdown(
    """
<style>
.compact-table { width: 100% !important; }
.compact-table table { width: 100% !important; table-layout: auto !important; border-collapse: collapse; }
.compact-table th, .compact-table td { white-space: nowrap; padding: 6px 10px; }
</style>
""",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Daily Compare", "Live vs Backtest"])

with tab1:
    dc.daily_compare_tab()

with tab2:
    lvb.live_vs_backtest_tab()
