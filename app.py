import streamlit as st

import tabs.daily_compare as dc
import tabs.live_vs_backtest as lvb
import tabs.trade_log_analyzer as tla

# Must be the first Streamlit call:
st.set_page_config(
    page_title="Trade Tools — Daily Compare & Live vs Backtest",
    layout="wide",
)

st.title("Trade Tools")

st.markdown(
    """
<style>
/* Make tables more compact */
.compact-table table { width: 100% !important; table-layout: auto !important; border-collapse: collapse; }
.compact-table th, .compact-table td { white-space: nowrap; padding: 6px 10px; }
</style>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Daily Compare", "Live vs Backtest", "Trade Log Analyzer"])

with tab1:
    dc.daily_compare_tab()

with tab2:
    lvb.live_vs_backtest_tab()

with tab3:
    tla.render_trade_log_analyzer_tab()
