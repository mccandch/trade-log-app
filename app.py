import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Daily Options Stats", layout="wide")

st.title("Daily Options Trade Stats")

# ---------- Google auth from Streamlit secrets ----------
# In Streamlit Community Cloud, set secrets:
# [gcp_service_account]
# type = "service_account"
# project_id = "..."
# private_key_id = "..."
# private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
# client_email = "...@...gserviceaccount.com"
# client_id = "..."
# auth_uri = "https://accounts.google.com/o/oauth2/auth"
# token_uri = "https://oauth2.googleapis.com/token"
# auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
# client_x509_cert_url = "..."
#
# [sheets]
# sheet_name = "TradeLog"
# tab_name = "Raw"

def open_ws():
    secrets = st.secrets
    sa_info = dict(secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open(secrets["sheets"]["sheet_name"])
    ws = sh.worksheet(secrets["sheets"]["tab_name"])
    return ws

@st.cache_data(ttl=60)
def load_data():
    ws = open_ws()
    df = get_as_dataframe(ws, evaluate_formulas=True)
    df = df.dropna(how="all")
    if df.empty:
        return pd.DataFrame(columns=["User","Date","FileName","BatchId","Strategy","TotalPremium","ProfitLoss"])
    # Coerce types
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    for c in ["TotalPremium","ProfitLoss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["User"] = df.get("User","Unknown").astype(str)
    df["Strategy"] = df.get("Strategy","Unknown").astype(str)
    return df

def pct_series(df):
    prem = df["TotalPremium"].sum()
    pnl = df["ProfitLoss"].sum()
    return (pnl / prem * 100) if prem else np.nan

def render_card(title, df):
    pct = pct_series(df)
    st.markdown(f"### {title}")
    st.markdown(f"**{pct:+.1f}% overall**" if pd.notna(pct) else "_No data_")
    by = (df.groupby("Strategy")
            .agg(premium=("TotalPremium","sum"), pnl=("ProfitLoss","sum"))
            .assign(pct=lambda x: (x["pnl"]/x["premium"]*100))
            .sort_values("pct", ascending=False))
    # Preferred display order + aliases
    order = ["EMA Credit Spread","PH","Early Hour","Megatrend","EMA-T","EMA-1x","EMA-B"]
    alias = {"EMA Credit Spread":"EMA","PH":"PH","Early Hour":"EH","Megatrend":"EMA-M","EMA-T":"EMA-T","EMA-1x":"EMA-1x","EMA-B":"EMA-B"}
    lines = []
    for k in order:
        if k in by.index:
            lines.append(f"{by.loc[k,'pct']:+.1f}% {alias[k]}")
    st.code("\n".join(lines) if lines else "—")

df = load_data()
if df.empty:
    st.info("No data found. Once your watcher uploads rows to Google Sheets, refresh.")
    st.stop()

# Sidebar filters
users = ["Both"] + sorted(df["User"].dropna().unique().tolist())
user = st.sidebar.selectbox("User", users, index=0)

min_date, max_date = df["Date"].min(), df["Date"].max()
start, end = st.sidebar.date_input("Date range", value=(max_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(start, tuple) or isinstance(start, list):
    start, end = start

mask = (df["Date"] >= start) & (df["Date"] <= end)
if user != "Both":
    mask &= df["User"] == user
view = df.loc[mask].copy()

# Top metrics
c1, c2, c3 = st.columns(3)
with c1: render_card("Selected View", view)
with c2: render_card("User A (filter)", df[(df['User']==users[1])] if len(users)>1 else df.iloc[0:0])
with c3: render_card("User B (filter)", df[(df['User']==users[2])] if len(users)>2 else df.iloc[0:0])

st.divider()

# Table of strategy stats
stats = (view.groupby(["User","Strategy"])
         .agg(Premium=("TotalPremium","sum"),
              PnL=("ProfitLoss","sum"))
         .assign(Pct=lambda x: (x["PnL"]/x["Premium"]*100))
         .reset_index()
        )
st.subheader("Strategy breakdown (selected range)")
st.dataframe(stats, use_container_width=True)

# Recent days overview
st.subheader("Recent days overview")
daily = (df.groupby(["User","Date"])
         .agg(Premium=("TotalPremium","sum"),
              PnL=("ProfitLoss","sum"))
         .assign(Pct=lambda x: (x["PnL"]/x["Premium"]*100))
         .reset_index()
         .sort_values(["Date","User"], ascending=[False, True])
        )
st.dataframe(daily.head(50), use_container_width=True)
