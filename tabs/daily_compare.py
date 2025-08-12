# tabs/daily_compare.py
from __future__ import annotations

from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials


# =========================
# CSS helpers (runtime only)
# =========================
def _aggrid_css() -> None:
    """Inject CSS for centered headers and tight spacing. Call at the start of the tab."""
    st.markdown(
        """
        <style>
          /* Center headers for any AG Grid theme */
          [class^="ag-theme"] .ag-header-cell-label { display:flex; justify-content:center; }
          [class^="ag-theme"] .ag-header-cell-text  { margin-left: 0 !important; }

          /* Optional header class applied via headerClass="dc-center" */
          .dc-center .ag-header-cell-label { justify-content: center; }
          .dc-center .ag-header-cell-text  { margin-left: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _user_header(name: str, overall: float) -> None:
    """Render 'Name  [+/-X.X% overall]' tightly above the grid."""
    if overall > 0:
        color, bg = "#22c55e", "rgba(34,197,94,0.12)"   # green
    elif overall < 0:
        color, bg = "#ef4444", "rgba(239,68,68,0.12)"   # red
    else:
        color, bg = "#9ca3af", "rgba(148,163,184,0.12)" # gray

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin:0 0 4px 0;">
          <span style="font-size:28px;font-weight:900;color:#e5e7eb !important;line-height:1;">
            {name}
          </span>
          <span style="
            display:inline-block;padding:4px 10px;border-radius:12px;
            font-weight:800;font-size:20px;line-height:1;
            color:{color} !important;background:{bg};
            border:1px solid rgba(255,255,255,0.06);
          ">{overall:+.1f}% overall</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =================
# Utility functions
# =================
def _numeric_series(df: pd.DataFrame, candidates, default=0.0) -> pd.Series:
    """
    Return a numeric Series from the first column in `candidates` that exists.
    If none exist, return a Series full of `default` with the same index.
    """
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(default, index=df.index, dtype="float64")


def _rerun():
    try:
        st.experimental_rerun()
    except Exception:
        st.rerun()


# ========================
# Google Sheets I/O loader
# ========================
def open_sheet():
    """Authorize and open the Google Sheet specified in secrets."""
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
def load_sheets_data() -> pd.DataFrame:
    """Read Raw_* tabs and normalize columns for the app."""
    sh = open_sheet()
    frames = []

    def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for c in list(df.columns):
            k = str(c).strip()
            if k in ("ProfitLoss", "P/L", "PL"):
                rename_map[c] = "PnL"
            elif k in ("TotalPremium", "Total Premium", "Premium($)"):
                rename_map[c] = "Premium"
            elif k in ("SourceFile", "Source file"):
                rename_map[c] = "Source"
        return df.rename(columns=rename_map) if rename_map else df

    for ws in sh.worksheets():
        title = ws.title or ""
        if not title.startswith("Raw_"):
            continue

        df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all")
        if df.empty:
            continue

        df = _rename_cols(df)

        # Ensure we have a User value; if missing, derive from tab name (Raw_<user>)
        if "User" not in df.columns or df["User"].isna().all():
            df["User"] = title.replace("Raw_", "", 1)

        # Build a single DateTime from whatever we have
        if "DateTime" in df.columns:
            dt = pd.to_datetime(df["DateTime"], errors="coerce")
        elif "Date" in df.columns:
            dt = pd.to_datetime(df["Date"], errors="coerce")
        else:
            dt = pd.to_datetime(pd.Series([pd.NaT] * len(df)), errors="coerce")

        df["DateTime"] = dt
        df["Date"] = dt.dt.date

        # Numeric fields
        for col in ("Premium", "PnL", "Strike"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalize Right (C/P/Call/Put variants)
        if "Right" in df.columns:
            ser = df["Right"].astype(str).str.strip().str.upper()
            df["Right"] = ser.replace(
                {"CALL": "C", "CALLS": "C", "PUT": "P", "PUTS": "P", "1": "C", "2": "P"}
            )
        else:
            df["Right"] = pd.NA

        keep = [
            "User", "Date", "DateTime", "Strategy", "Right", "Strike",
            "Premium", "PnL", "Source", "BatchID", "TradeID", "Account",
        ]
        cols = [c for c in keep if c in df.columns]
        out = df[cols].copy()

        out["__source_tab"] = title  # helpful for debugging
        frames.append(out)

    if not frames:
        return pd.DataFrame(
            columns=[
                "User", "Date", "DateTime", "Strategy", "Right", "Strike",
                "Premium", "PnL", "Source", "BatchID", "TradeID", "Account",
                "__source_tab",
            ]
        )

    return pd.concat(frames, ignore_index=True)


# ========================
# Presentation + Computeds
# ========================
PREFERRED_ORDER = [
    "EMA Credit Spread", "PH", "Early Hour", "Megatrend", "EMA-T", "EMA-1x", "EMA-B"
]
ALIAS = {
    "EMA Credit Spread": "EMA",
    "PH": "PH",
    "Early Hour": "EH",
    "Megatrend": "EMA-M",
    "EMA-T": "EMA-T",
    "EMA-1x": "EMA-1x",
    "EMA-B": "EMA-B",
}


def _order_key(name: str):
    try:
        return (0, PREFERRED_ORDER.index(name))
    except ValueError:
        return (1, name.lower())


def _pcr(pnl_sum: float, prem_sum: float) -> float:
    return 0.0 if prem_sum == 0 else (pnl_sum / prem_sum) * 100.0


def _user_strategy_pcr(df_user: pd.DataFrame) -> pd.DataFrame:
    """Return Strategy + PCR% for a single user's filtered window."""
    if df_user.empty:
        return pd.DataFrame(columns=["Strategy", "PCR"])

    dfu = df_user.copy()
    dfu["Premium"] = _numeric_series(dfu, ["Premium", "TotalPremium", "Premium($)"])
    dfu["PnL"] = _numeric_series(dfu, ["PnL", "ProfitLoss", "P/L", "PL"])

    g = (
        dfu.groupby("Strategy", dropna=False)[["Premium", "PnL"]]
        .sum()
        .reset_index()
    )
    g["PCR"] = np.where(g["Premium"] != 0, (g["PnL"] / g["Premium"]) * 100.0, 0.0)
    g["Strategy"] = g["Strategy"].astype(str)
    g = g.sort_values("Strategy", key=lambda s: s.map(_order_key))
    return g[["Strategy", "PCR"]]


def render_trades_table(trades: pd.DataFrame, title: str):
    if trades.empty:
        st.caption(f"{title}: no trades")
        return

    cols = []
    if "DateTime" in trades.columns:
        trades = trades.copy()
        trades["Entry time"] = pd.to_datetime(trades["DateTime"], errors="coerce")
        cols.append("Entry time")

    if "Right" in trades.columns:
        cols.append("Right")

    if "Strike" in trades.columns:
        trades["Strike"] = trades["Strike"].apply(
            lambda x: int(x) if pd.notna(x) and float(x).is_integer() else x
        )
        cols.append("Strike")

    cols += [c for c in ["Strategy"] if c in trades.columns]

    if "PnL" in trades.columns:
        trades["PnL"] = pd.to_numeric(trades["PnL"], errors="coerce")
        cols.append("PnL")

    view = trades[cols]
    if "Entry time" in view.columns:
        view = view.sort_values(by="Entry time", ascending=True, na_position="last")

    # Row coloring by PnL
    def _row_style(row):
        try:
            val = float(row.get("PnL", 0))
        except Exception:
            val = 0
        color = "#143d2b" if val > 0 else ("#4b1f1f" if val < 0 else "transparent")
        return [f"background-color: {color}"] * len(row)

    def _fmt_time(x):
        if pd.isna(x):
            return ""
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(x)

    def _fmt_money(x):
        return "" if pd.isna(x) else f"${float(x):,.2f}"

    styled = (
        view.style
        .apply(_row_style, axis=1)
        .format({"Entry time": _fmt_time, "PnL": _fmt_money}, na_rep="")
        .set_table_attributes('class="compact-table"')
        .hide(axis="index")
        .set_properties(**{"text-align": "center"})
    )

    st.dataframe(styled, use_container_width=True)


# ===================================
# AG Grid: per-user strategy selector
# ===================================
def render_user_table_with_toggles(user: str, df_user: pd.DataFrame) -> list[str]:
    # overall badge
    prem_series = _numeric_series(df_user, ["Premium", "TotalPremium", "Premium($)"])
    pnl_series = _numeric_series(df_user, ["PnL", "ProfitLoss", "P/L", "PL"])
    overall = _pcr(float(pnl_series.sum()), float(prem_series.sum()))
    _user_header(user, overall)

    # PCR table with unique columns
    pcr = _user_strategy_pcr(df_user).copy()        # ["Strategy", "PCR"]
    pcr.rename(columns={"Strategy": "Canonical"}, inplace=True)
    pcr["Strategy"] = pcr["Canonical"].map(lambda s: ALIAS.get(s, s))
    pcr["PCR %"] = pcr["PCR"].round(1)
    pcr_for_grid = pcr[["Strategy", "PCR %"]]

    gb = GridOptionsBuilder.from_dataframe(pcr_for_grid)
    gb.configure_selection("multiple", use_checkbox=True)
    gb.configure_default_column(cellStyle={"textAlign": "center"})

    # center header text
    gb.configure_column("Strategy", headerClass="dc-center")
    gb.configure_column(
        "PCR %", header_name="PCR", headerClass="dc-center",
        type=["numericColumn"],
        valueFormatter=JsCode(
            "function(p){return (p.value==null)?'':(Number(p.value).toFixed(1)+'%');}"
        ),
    )

    # row fill color by PCR %
    row_style = JsCode(
        """
        function(params){
            const v = Number(params.data["PCR %"]);
            if (isNaN(v)) return null;
            return { backgroundColor: (v>0) ? "#143d2b" : (v<0 ? "#4b1f1f" : "transparent"),
                     color: "white" };
        }
        """
    )

    go = gb.build()
    go["getRowStyle"] = row_style
    go["headerHeight"] = 32

    # remember selection between reruns
    sel_key = f"dc_sel_{user}"
    _prev = st.session_state.get(sel_key, [])

    grid = AgGrid(
        pcr_for_grid,
        gridOptions=go,
        theme="streamlit",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
    )

    picked_aliases = [r["Strategy"] for r in grid["selected_rows"]]
    st.session_state[sel_key] = picked_aliases

    # map aliases back to canonical names used in your data
    rev_alias = {v: k for k, v in ALIAS.items()}
    return [rev_alias.get(a, a) for a in picked_aliases]


# ===========
# Main (Tab)
# ===========
def daily_compare_tab():
    _aggrid_css()

    st.subheader("Daily Compare (Google Sheets)")

    # Refresh button
    if st.button("Refresh data", key="dc_btn_refresh"):
        load_sheets_data.clear()
        _rerun()

    df = load_sheets_data()
    if df.empty:
        st.info("No data found yet. Start your watchers (Raw_* tabs) and click Refresh.")
        return

    # Date range presets
    min_date, max_date = df["Date"].min(), df["Date"].max()

    if "dc_date_range" not in st.session_state or st.session_state.dc_date_range is None:
        st.session_state.dc_date_range = (max_date, max_date)

    def clamp(d: date) -> date:
        return max(min(d, max_date), min_date)

    def set_range(start: date, end: date):
        st.session_state.dc_date_range = (clamp(start), clamp(end))
        _rerun()

    r1c1, r1c2, r1c3 = st.columns(3)
    if r1c1.button("Today", key="dc_btn_today", use_container_width=True):
        t = date.today(); set_range(t, t)
    if r1c2.button("Yesterday", key="dc_btn_yesterday", use_container_width=True):
        y = date.today() - timedelta(days=1); set_range(y, y)
    if r1c3.button("This Week", key="dc_btn_this_week", use_container_width=True):
        d0 = date.today(); start = d0 - timedelta(days=d0.weekday()); set_range(start, d0)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    if r2c1.button("Last Week", key="dc_btn_last_week", use_container_width=True):
        d0 = date.today()
        this_mon = d0 - timedelta(days=d0.weekday())
        last_mon = this_mon - timedelta(days=7)
        last_sun = last_mon + timedelta(days=6)
        set_range(last_mon, last_sun)
    if r2c2.button("This Month", key="dc_btn_this_month", use_container_width=True):
        d0 = date.today(); set_range(d0.replace(day=1), d0)
    if r2c3.button("Last Month", key="dc_btn_last_month", use_container_width=True):
        d0 = date.today()
        first_this = d0.replace(day=1)
        last_prev  = first_this - timedelta(days=1)
        set_range(last_prev.replace(day=1), last_prev)
    if r2c4.button("YTD", key="dc_btn_ytd", use_container_width=True):
        d0 = date.today(); set_range(date(d0.year, 1, 1), d0)

    # Date picker AFTER buttons
    start, end = st.session_state.dc_date_range
    st.session_state.dc_date_range = (clamp(start), clamp(end))
    picked = st.date_input(
        "Date range",
        value=st.session_state.dc_date_range,
        min_value=min_date,
        max_value=max_date,
        key="dc_date_range",
    )
    if isinstance(picked, (tuple, list)):
        d1, d2 = picked
    else:
        d1, d2 = st.session_state.dc_date_range

    # Filtered window
    view = df[(df["Date"] >= d1) & (df["Date"] <= d2)].copy()

    # Per-user UI: each column shows picker + trades
    users_in_view = view["User"].dropna().unique().tolist()
    preferred_user_order = ["Chad", "Kelly"]
    ordered_users = [u for u in preferred_user_order if u in users_in_view] + \
                    [u for u in sorted(users_in_view) if u not in preferred_user_order]

    if not ordered_users:
        st.warning("Nobody placed any trades for the selected date range.")
        return

    cols = st.columns(len(ordered_users))
    for col, user in zip(cols, ordered_users):
        with col:
            df_user = view[view["User"] == user].copy()

            picked_strats = render_user_table_with_toggles(user, df_user)

            if picked_strats:
                trades = df_user[df_user["Strategy"].isin(picked_strats)].copy()
                render_trades_table(trades, title=f"{user} — selected strategy trades")

    # Global Strategy breakdown (selected range)
    st.divider()

    df_view = view.copy()
    df_view["Premium"] = _numeric_series(df_view, ["Premium", "TotalPremium", "Premium($)"])
    df_view["PnL"] = _numeric_series(df_view, ["PnL", "ProfitLoss", "P/L", "PL"])

    stats = (
        df_view.groupby(["User", "Strategy"], dropna=False)[["Premium", "PnL"]]
        .sum()
        .reset_index()
    )
    stats["PCR %"] = np.where(
        stats["Premium"] != 0, (stats["PnL"] / stats["Premium"]) * 100.0, np.nan
    )
    stats = stats[["User", "Strategy", "PCR %", "Premium", "PnL"]]

    st.subheader("Strategy breakdown (selected range)")
    st.dataframe(stats, use_container_width=True)
