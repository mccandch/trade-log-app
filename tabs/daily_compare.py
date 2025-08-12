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

    if st.button("Refresh data"):
        load_sheets_data.clear()
        _rerun()

    df = load_sheets_data()
    if df.empty:
        st.info("No data found yet. Start your watchers (Raw_Chad / Raw_Kelly) and click Refresh.")
        return

    min_date, max_date = df["Date"].min(), df["Date"].max()
    d1, d2 = st.date_input(
        "Date range",
        value=(max_date, max_date),
        min_value=min_date, max_value=max_date,
        key="dc_date_range"
    )
    if isinstance(d1, (tuple, list)):
        d1, d2 = d1

    view = df[(df["Date"] >= d1) & (df["Date"] <= d2)].copy()
    users_in_view = view["User"].dropna().unique().tolist()

    preferred_user_order = ["Chad","Kelly"]
    ordered_users = [u for u in preferred_user_order if u in users_in_view] + \
                    [u for u in sorted(users_in_view) if u not in preferred_user_order]

    if not ordered_users:
        st.warning("Nobody placed any trades for the selected date range.")
        return

    cols = st.columns(len(ordered_users))
    for c, user in zip(cols, ordered_users):
        with c:
            render_daily_card(user, view[view["User"] == user])

    st.divider()
    stats = (
        view.groupby(["User","Strategy"])
            .agg(Premium=("TotalPremium","sum"),
                 PnL=("ProfitLoss","sum"))
            .assign(Pct=lambda x: (x["PnL"]/x["Premium"]*100))
            .reset_index()
    )
    st.subheader("Strategy breakdown (selected range)")
    st.dataframe(stats, use_container_width=True)
