from datetime import date, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials

def _rerun():
    try:
        st.experimental_rerun()
    except Exception:
        st.rerun()

def open_sheet():
    sa_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    gc = gspread.authorize(creds)
    return gc.open(st.secrets["sheets"]["sheet_name"])

@st.cache_data(ttl=60)
def load_sheets_data():
    sh = open_sheet()
    frames = []
    for ws in sh.worksheets():
        # Only per-user tabs produced by the watchers
        if not ws.title.startswith("Raw_"):
            continue
        df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all")
        if df.empty:
            continue
        df["__source_tab"] = ws.title
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["User","Date","DateTime","FileName","BatchId","Strategy","TotalPremium","ProfitLoss"])

    df = pd.concat(frames, ignore_index=True)

    # Types
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    else:
        df["Date"] = df["DateTime"].dt.date

    for c in ("TotalPremium","ProfitLoss"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["User"] = df.get("User","Unknown").astype(str).str.strip()
    df["Strategy"] = df.get("Strategy","Unknown").astype(str).str.strip()
    return df

def render_daily_card(title, df):
    prem = df["TotalPremium"].sum()
    pnl  = df["ProfitLoss"].sum()
    pct  = (pnl / prem * 100) if prem else np.nan

    st.markdown(f"### {title}")
    st.markdown(f"**{pct:+.1f}% overall**" if pd.notna(pct) else "_No data_")

    by = (
        df.groupby("Strategy", dropna=False)
          .agg(premium=("TotalPremium","sum"), pnl=("ProfitLoss","sum"))
          .assign(pct=lambda x: (x["pnl"]/x["premium"]*100))
          .sort_values("pct", ascending=False)
    )

    preferred = ["EMA Credit Spread","PH","Early Hour","Megatrend","EMA-T","EMA-1x","EMA-B"]
    alias = {"EMA Credit Spread":"EMA","PH":"PH","Early Hour":"EH","Megatrend":"EMA-M","EMA-T":"EMA-T","EMA-1x":"EMA-1x","EMA-B":"EMA-B"}

    order = [k for k in preferred if k in by.index] + [k for k in by.index if k not in preferred]
    lines = [f"{by.loc[k,'pct']:+.1f}% {alias.get(k,k)}" for k in order]
    st.code("\n".join(lines) if lines else "—")

def daily_compare_tab():
    st.subheader("Daily Compare (Google Sheets)")

    # Refresh
    if st.button("Refresh data", key="dc_btn_refresh"):
        load_sheets_data.clear()
        _rerun()

    df = load_sheets_data()
    if df.empty:
        st.info("No data found yet. Start your watchers (Raw_Chad / Raw_Kelly) and click Refresh.")
        return

    # --- date range + quick buttons (ordered) ---
    min_date, max_date = df["Date"].min(), df["Date"].max()

    # One canonical state tuple for this picker
    if "dc_date_range" not in st.session_state or st.session_state.dc_date_range is None:
        st.session_state.dc_date_range = (max_date, max_date)

    def clamp(d: date) -> date:
        return max(min(d, max_date), min_date)

    def set_range(start: date, end: date):
        # Set BEFORE rendering the date_input to avoid Streamlit state error
        st.session_state.dc_date_range = (clamp(start), clamp(end))
        _rerun()

    # Row 1: Today, Yesterday, This Week
    r1c1, r1c2, r1c3 = st.columns(3)
    if r1c1.button("Today", key="dc_btn_today", use_container_width=True):
        t = date.today()
        set_range(t, t)

    if r1c2.button("Yesterday", key="dc_btn_yesterday", use_container_width=True):
        y = date.today() - timedelta(days=1)
        set_range(y, y)

    if r1c3.button("This Week", key="dc_btn_this_week", use_container_width=True):
        d0 = date.today()
        start = d0 - timedelta(days=d0.weekday())
        set_range(start, d0)

    # Row 2: Last Week, This Month, Last Month, YTD
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    if r2c1.button("Last Week", key="dc_btn_last_week", use_container_width=True):
        d0 = date.today()
        this_mon = d0 - timedelta(days=d0.weekday())
        last_mon = this_mon - timedelta(days=7)
        last_sun = last_mon + timedelta(days=6)
        set_range(last_mon, last_sun)

    if r2c2.button("This Month", key="dc_btn_this_month", use_container_width=True):
        d0 = date.today()
        set_range(d0.replace(day=1), d0)

    if r2c3.button("Last Month", key="dc_btn_last_month", use_container_width=True):
        d0 = date.today()
        first_this = d0.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        set_range(last_prev.replace(day=1), last_prev)

    if r2c4.button("YTD", key="dc_btn_ytd", use_container_width=True):
        d0 = date.today()
        set_range(date(d0.year, 1, 1), d0)

    # Now render the date picker AFTER buttons may have updated the state
    start, end = st.session_state.dc_date_range
    st.session_state.dc_date_range = (clamp(start), clamp(end))  # clamp to available data
    picked = st.date_input(
        "Date range",
        value=st.session_state.dc_date_range,
        min_value=min_date,
        max_value=max_date,
        key="dc_date_range",
    )

    # Normalize picked value
    if isinstance(picked, (tuple, list)):
        d1, d2 = picked
    else:
        d1, d2 = st.session_state.dc_date_range

    # ---- FILTER (this defines `view`) ----
    view = df[(df["Date"] >= d1) & (df["Date"] <= d2)].copy()

    # ---- CARDS ----
    users_in_view = view["User"].dropna().unique().tolist()
    preferred_user_order = ["Chad", "Kelly"]
    ordered_users = [u for u in preferred_user_order if u in users_in_view] + \
                    [u for u in sorted(users_in_view) if u not in preferred_user_order]

    if not ordered_users:
        st.warning("Nobody placed any trades for the selected date range.")
        return

    cols = st.columns(len(ordered_users))
    for c, user in zip(cols, ordered_users):
        with c:
            render_daily_card(user, view[view["User"] == user])

    # ---- TABLE ----
    st.divider()
    stats = (
        view.groupby(["User", "Strategy"], dropna=False)
            .agg(Premium=("TotalPremium", "sum"),
                 PnL=("ProfitLoss", "sum"))
            .assign(Pct=lambda x: np.where(x["Premium"] != 0, (x["PnL"] / x["Premium"]) * 100, np.nan))
            .reset_index()
    )
    st.subheader("Strategy breakdown (selected range)")
    st.dataframe(stats, use_container_width=True)
