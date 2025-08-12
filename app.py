import streamlit as st
import tabs.daily_compare as dc
import tabs.live_vs_backtest as lvb

st.set_page_config(page_title="Trade Tools — Daily Compare & Live vs Backtest", layout="wide")
st.title("Trade Tools")

tabs = st.tabs(["Daily Compare", "Live vs Backtest"])
with tabs[0]:
    # if your function is dc.daily_compare_tab(), this still works
    # if it’s dc.daily_compare(), this also works—adjust the name below to match
    dc.daily_compare_tab() if hasattr(dc, "daily_compare_tab") else dc.daily_compare()

with tabs[1]:
    lvb.live_vs_backtest_tab()
