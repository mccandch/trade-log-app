from __future__ import annotations

from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from dateutil.relativedelta import relativedelta
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from typing import Optional, Union, BinaryIO
import altair as alt


# =========================
# CSS helpers (runtime only)
# =========================
def _render_ema_b_schedule(today_only: bool = True) -> None:
    """Show EMA-B time buckets. Default: only the current day."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    today_name = date.today().strftime("%A")

    def _rows(kind: str):
        days = [today_name] if today_only else day_order
        return [{"Day": d, "Start-End (ET)": ", ".join(EMA_B_SCHEDULE[kind].get(d, [])) or "—"} for d in days]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📈 Put Buckets (EMA20 > EMA40)**")
        _df_no_index(pd.DataFrame(_rows("Put")))
    with c2:
        st.markdown("**📉 Call Buckets (EMA20 < EMA40)**")
        _df_no_index(pd.DataFrame(_rows("Call")))


def _df_no_index(df: pd.DataFrame):
    """Render a dataframe without showing its index/row numbers."""
    st.dataframe(
        df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,   # <- this actually hides the 0,1,2 row numbers
    )

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


def _user_header(name: str, pcr_pct: float, win_rate_pct: float, total_pnl: float, total_premium: float = 0.0) -> None:
    """Render 'Name  [PCR: ±X.X% | Win Rate: X.X% | Premium $X,XXX | $±N]' with table-matching colors."""
    # Match the same green/red as your grid rows
    if pcr_pct > 0:
        txt_color, bg = "#ffffff", "#143d2b"   # green
    elif pcr_pct < 0:
        txt_color, bg = "#ffffff", "#4b1f1f"   # red
    else:
        txt_color, bg = "#ffffff", "rgba(148,163,184,0.12)"  # gray

    # Format parts
    pcr_str     = f"{pcr_pct:+.1f}%"
    win_str     = f"{win_rate_pct:.1f}%"
    premium_str = f"Premium ${abs(float(total_premium)):,.0f}"   # <- no cents: $x,xxx
    pnl_str     = f"${abs(float(total_pnl)):,.2f}"
    if total_pnl < 0:
        pnl_str = f"-{pnl_str}"

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin:0 0 4px 0;">
          <span style="font-size:28px;font-weight:900;color:#e5e7eb !important;line-height:1;">
            {name}
          </span>
          <span style="
            display:inline-block;padding:4px 12px;border-radius:12px;
            font-weight:800;font-size:20px;line-height:1;
            color:{txt_color} !important;background:{bg};
            border:1px solid rgba(255,255,255,0.06);
          ">
            PCR: {pcr_str} &nbsp;|&nbsp; Win Rate: {win_str} &nbsp;|&nbsp; {premium_str} &nbsp;|&nbsp; {pnl_str}
          </span>
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
    """Read Raw_* tabs and normalize columns for the app, with a brief stability check."""
    import time
    import pandas as pd

    sh = open_sheet()

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

    # --- one pass over all Raw_* tabs, returning a list of normalized frames ---
    def _read_all_raw_tabs() -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for ws in sh.worksheets():
            title = ws.title or ""
            if not title.startswith("Raw_"):
                continue

            df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all")
            if df.empty:
                continue

            df = _rename_cols(df)

            # Ensure User
            if "User" not in df.columns or df["User"].isna().all():
                df["User"] = title.replace("Raw_", "", 1)

            # Build DateTime / Date
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

            # Normalize Right to {C,P,NA}
            if "Right" in df.columns:
                ser = df["Right"].astype(str).str.strip().str.upper()
                ser = ser.replace({
                    "CALL": "C", "CALLS": "C",
                    "PUT": "P", "PUTS": "P",
                    "1": "C", "2": "P",
                    "NAN": pd.NA, "NONE": pd.NA,
                    "": pd.NA, "NA": pd.NA, "N/A": pd.NA
                })
                ser = ser.where(ser.isin(["C", "P"]), pd.NA)
                df["Right"] = ser
            else:
                df["Right"] = pd.NA

            keep = [
                "User", "Date", "DateTime", "Strategy", "Right", "Strike",
                "Premium", "PnL", "Source", "BatchID", "TradeID", "Account",
            ]
            cols = [c for c in keep if c in df.columns]
            out = df[cols].copy()

            out["__source_tab"] = title
            frames.append(out)
        return frames

    # For stability, compare (rowcount, sum Premium, sum PnL) per tab across two reads
    def _signature(frames: list[pd.DataFrame]) -> list[tuple]:
        sig = []
        for f in frames:
            tab = str(f["__source_tab"].iloc[0]) if not f.empty and "__source_tab" in f.columns else ""
            prem = float(pd.to_numeric(f.get("Premium", pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()) if "Premium" in f.columns else 0.0
            pnl  = float(pd.to_numeric(f.get("PnL", pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()) if "PnL" in f.columns else 0.0
            sig.append((tab, len(f), round(prem, 2), round(pnl, 2)))
        # sort to make ordering irrelevant
        sig.sort(key=lambda x: x[0])
        return sig

    # --- stability retry loop: read -> brief pause -> read again and compare ---
    frames_final: list[pd.DataFrame] = []
    last_frames: list[pd.DataFrame] = []
    for attempt in range(5):
        f1 = _read_all_raw_tabs()
        sig1 = _signature(f1)
        time.sleep(0.7)  # small pause to avoid catching a mid-write snapshot
        f2 = _read_all_raw_tabs()
        sig2 = _signature(f2)

        if sig1 == sig2:
            frames_final = f2
            break
        last_frames = f2  # keep the most recent in case we never stabilize
    else:
        frames_final = last_frames

    if not frames_final:
        return pd.DataFrame(
            columns=[
                "User", "Date", "DateTime", "Strategy", "Right", "Strike",
                "Premium", "PnL", "Source", "BatchID", "TradeID", "Account",
                "__source_tab",
            ]
        )

    return pd.concat(frames_final, ignore_index=True)


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

# ========================
# EMA-B schedule (ET)
# ========================
EMA_B_SCHEDULE = {
    "Put": {
        "Monday":    ["11:00-11:30", "11:30-12:00", "12:00-12:30", "12:30-13:00", "13:00-13:30", "13:30-14:00", "14:00-14:30"],
        "Tuesday":   ["10:00-10:30", "10:30-11:00", "13:30-14:00", "14:00-14:30", "15:30-16:00"],
        "Wednesday": ["12:30-13:00", "14:00-14:30", "15:30-16:00"],
        "Thursday":  ["15:30-16:00"],
        "Friday":    ["11:00-11:30", "12:30-13:00", "14:00-14:30", "15:30-16:00"],
    },
    "Call": {
        "Monday":    ["10:30-11:00", "11:00-11:30", "11:30-12:00", "12:00-12:30", "12:30-13:00", "13:00-13:30", "14:00-14:30", "14:30-15:00"],
        "Tuesday":   ["10:30-11:00", "11:00-11:30"],
        "Wednesday": ["11:30-12:00", "12:30-13:00", "13:00-13:30"],
        "Thursday":  ["10:30-11:00", "11:00-11:30", "13:00-13:30", "13:30-14:00", "14:00-14:30"],
        "Friday":    ["09:30-10:00", "10:00-10:30", "10:30-11:00", "13:00-13:30", "13:30-14:00", "14:30-15:00", "15:00-15:30"],
    },
}



def _order_key(name: str):
    try:
        return (0, PREFERRED_ORDER.index(name))
    except ValueError:
        return (1, name.lower())


def _pcr(pnl_sum: float, prem_sum: float) -> float:
    return 0.0 if prem_sum == 0 else (pnl_sum / prem_sum) * 100.0


def _user_strategy_pcr(df_user: pd.DataFrame) -> pd.DataFrame:
    """Return per-strategy PCR%, Win%, Trades, Winners, Losers for a single user's filtered window."""
    if df_user.empty:
        return pd.DataFrame(columns=["Strategy", "PCR", "WinRate", "Trades", "Winners", "Losers"])  # empty shape

    dfu = df_user.copy()
    dfu["Premium"] = _numeric_series(dfu, ["Premium", "TotalPremium", "Premium($)"])
    dfu["PnL"] = _numeric_series(dfu, ["PnL", "ProfitLoss", "P/L", "PL"])

    g = (
        dfu.groupby("Strategy", dropna=False)
        .agg(
            Premium=("Premium", "sum"),
            PnL=("PnL", "sum"),
            Trades=("PnL", lambda s: (
                (((s != 0) | (dfu.loc[s.index, "Date"] == date.today())) &
                    dfu.loc[s.index, "Right"].isin(["C", "P"])).sum()
            )),
            Winners=("PnL", lambda s: ((s > 0) & dfu.loc[s.index, "Right"].isin(["C", "P"])).sum()),
            Losers=("PnL", lambda s: ((s < 0) & dfu.loc[s.index, "Right"].isin(["C", "P"])).sum()),
        )
        .reset_index()
    )


    # Derived metrics
    g["PCR"] = np.where(g["Premium"] != 0, (g["PnL"] / g["Premium"]) * 100.0, np.nan)
    denom = g["Winners"] + g["Losers"]
    g["WinRate"] = np.where(denom > 0, (g["Winners"] / denom) * 100.0, np.nan)

    g["Strategy"] = g["Strategy"].astype(str)
    g = g.sort_values("Strategy", key=lambda s: s.map(_order_key))

    # Ensure integer dtypes for count columns
    for c in ("Trades", "Winners", "Losers"):
        g[c] = g[c].fillna(0).astype(int)

    return g[["Strategy", "Premium", "PCR", "WinRate", "Trades", "Winners", "Losers"]]


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

    if "Premium" in trades.columns:
        trades["Premium"] = pd.to_numeric(trades["Premium"], errors="coerce")
        cols.append("Premium")

    if "PnL" in trades.columns:
        trades["PnL"] = pd.to_numeric(trades["PnL"], errors="coerce")
        cols.append("PnL")

    view = trades[cols]
    if "Entry time" in view.columns:
        view = view.sort_values(by="Entry time", ascending=True, na_position="last")

    # Drop the index here so IDs like 1450, 1451 don’t appear
    view = view.reset_index(drop=True)
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
        .format({"Entry time": _fmt_time,"Premium": _fmt_money, "PnL": _fmt_money}, na_rep="")
        .set_table_attributes('class="compact-table"')
        .hide(axis="index")
        .set_properties(**{"text-align": "center"})
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)


def _pnl_line_chart(trades: pd.DataFrame, title: str = "PnL over time"):
    """
    Interactive line chart of *daily* cumulative PnL.
    One point per calendar day:
      1) Sum PnL within each day
      2) Plot cumulative sum across days
    Requires DateTime + PnL columns.
    """
    if trades.empty or "DateTime" not in trades.columns or "PnL" not in trades.columns:
        st.caption("No time series to chart.")
        return

    df = trades.copy()
    # Normalize types
    df["Entry time"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Entry time"])

    if df.empty:
        st.caption("No time series to chart.")
        return

    # ---- DAILY ROLLUP ----
    # Convert to calendar day and sum PnL per day
    df["Day"] = df["Entry time"].dt.normalize()  # midnight of that day (keeps tz if present)
    daily = (
        df.groupby("Day", as_index=False)["PnL"]
          .sum()
          .sort_values("Day")
    )
    daily["CumPnL"] = daily["PnL"].cumsum()

    if daily.empty:
        st.caption("No time series to chart.")
        return

    line = (
        alt.Chart(daily)
        .mark_line(point=True)
        .encode(
            x=alt.X("Day:T", title="Day"),
            y=alt.Y("CumPnL:Q", title="Cumulative PnL ($)"),
            tooltip=[
                alt.Tooltip("Day:T", title="Day"),
                alt.Tooltip("PnL:Q", title="Daily PnL ($)", format=",.2f"),
                alt.Tooltip("CumPnL:Q", title="Cum PnL ($)", format=",.2f"),
            ],
        )
        .interactive()
        .properties(height=260, title=title)
    )

    st.altair_chart(line, use_container_width=True)

# ===================================
# AG Grid: per-user strategy selector
# ===================================
def render_user_table_with_toggles(user: str, df_user: pd.DataFrame) -> list[str]:
    # ---- Build per-strategy stats FIRST (so we can safely summarize for the header) ----
    stats = _user_strategy_pcr(df_user).copy()  # columns: Strategy, Premium, PCR, WinRate, Trades, Winners, Losers
    stats.rename(columns={"Strategy": "Canonical"}, inplace=True)
    stats["Strategy"] = stats["Canonical"].map(lambda s: ALIAS.get(s, s))
    stats["PCR %"] = stats["PCR"].round(1)
    stats["Win %"] = stats["WinRate"].round(1)

    # Map PnL by canonical key (prevents misalignment)
    dfu = df_user.copy()
    dfu["PnL"] = _numeric_series(dfu, ["PnL", "ProfitLoss", "P/L", "PL"])
    pnl_map = dfu.groupby("Strategy", dropna=False)["PnL"].sum().to_dict()

    stats["PnL"] = stats["Canonical"].map(pnl_map).fillna(0.0)

    # Table consumed by AG Grid
    stats_for_grid = stats[["Strategy", "PCR %", "Win %", "Premium", "Trades", "Winners", "Losers", "PnL"]].copy()

    # ---- Compute header/badge values from the table the user sees ----
    if stats_for_grid.empty:
        visible_pnl = 0.0
        visible_prem = 0.0
        total_wins = 0
        total_losses = 0
    else:
        visible_pnl  = float(pd.to_numeric(stats_for_grid["PnL"], errors="coerce").fillna(0).sum())
        visible_prem = float(pd.to_numeric(stats_for_grid["Premium"], errors="coerce").fillna(0).sum())
        total_wins   = int(pd.to_numeric(stats_for_grid["Winners"], errors="coerce").fillna(0).sum())
        total_losses = int(pd.to_numeric(stats_for_grid["Losers"],  errors="coerce").fillna(0).sum())

    trade_denom   = total_wins + total_losses
    visible_winrt = (total_wins / trade_denom) * 100.0 if trade_denom > 0 else 0.0
    visible_pcr   = _pcr(visible_pnl, visible_prem)

    _user_header(
        name=user,
        pcr_pct=visible_pcr,
        win_rate_pct=visible_winrt,
        total_pnl=visible_pnl,
        total_premium=visible_prem,
    )

    # If there are no strategies, show a note and return nothing to select
    if stats_for_grid.empty:
        st.caption("No strategies found for this user in the selected date range.")
        return []

    # ---- AG Grid config ----
    gb = GridOptionsBuilder.from_dataframe(stats_for_grid)
    gb.configure_selection("multiple", use_checkbox=True)
    gb.configure_default_column(cellStyle={"textAlign": "center"})

    pctFmt = JsCode("""
      function(p){
        if (p.value==null) return '';
        const v = Number(p.value);
        return isNaN(v) ? '' : (v.toFixed(1) + '%');
      }
    """)
    intFmt = JsCode("""
      function(p){
        if (p.value==null) return '';
        const v = Number(p.value);
        return isNaN(v) ? '' : v.toFixed(0);
      }
    """)
    moneyFmt = JsCode("""
      function(p){
        if (p.value==null) return '';
        const v = Number(p.value);
        if (isNaN(v)) return '';
        const abs = Math.abs(v).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
        return (v < 0 ? '-$' : '$') + abs;
      }
    """)

    gb.configure_column("Strategy", headerClass="dc-center")
    gb.configure_column("PCR %", header_name="PCR", headerClass="dc-center", type=["numericColumn"], valueFormatter=pctFmt)
    gb.configure_column("Win %", header_name="Win rate", headerClass="dc-center", type=["numericColumn"], valueFormatter=pctFmt)
    gb.configure_column("Premium", headerClass="dc-center", type=["numericColumn"], valueFormatter=moneyFmt)
    gb.configure_column("Trades", headerClass="dc-center", type=["numericColumn"], valueFormatter=intFmt)
    gb.configure_column("Winners", headerClass="dc-center", type=["numericColumn"], valueFormatter=intFmt)
    gb.configure_column("Losers", headerClass="dc-center", type=["numericColumn"], valueFormatter=intFmt)
    gb.configure_column("PnL", header_name="PnL", headerClass="dc-center", type=["numericColumn"], valueFormatter=moneyFmt)

    row_style = JsCode("""
      function(params){
        const v = Number(params.data["PCR %"]);
        if (isNaN(v)) return null;
        return { backgroundColor: (v>0) ? "#143d2b" : (v<0 ? "#4b1f1f" : "transparent"),
                 color: "white" };
      }
    """)

    go = gb.build()
    go["getRowStyle"] = row_style
    go["headerHeight"] = 32
    go["suppressSizeToFit"] = True
    go["onFirstDataRendered"] = JsCode("""
      function(params){
        const all = params.columnApi.getAllDisplayedColumns();
        params.columnApi.autoSizeColumns(all, false);
      }
    """)

    # Render grid
    grid = AgGrid(
        stats_for_grid,
        gridOptions=go,
        theme="streamlit",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        key=f"dc_grid_{user}",
    )

    # Robust selection extraction
    def _extract_selected_rows(g):
        rows = None
        if isinstance(g, dict):
            rows = g.get("selected_rows") or g.get("selectedRows")
        if rows is None:
            try:
                rows = getattr(g, "selected_rows", None) or getattr(g, "selectedRows", None)
            except Exception:
                rows = None
        if rows is None:
            return []
        if isinstance(rows, pd.DataFrame):
            return rows.to_dict("records")
        if isinstance(rows, list):
            return rows
        return []

    selected_rows = _extract_selected_rows(grid)
    picked_aliases = [r.get("Strategy") for r in selected_rows if isinstance(r, dict) and r.get("Strategy")]

    # Persist for next rerun
    st.session_state[f"dc_sel_{user}"] = picked_aliases

    # Map back to canonical names
    rev_alias = {v: k for k, v in ALIAS.items()}
    return [rev_alias.get(a, a) for a in picked_aliases]


# ===========
# Main (Tab)
# ===========
def daily_compare_tab():
    _aggrid_css()

    st.subheader("Daily Compare (Google Sheets)")
    # === EMA-B schedule panel (always visible) ===
    show_full = st.checkbox("EMA-B schedule: show full week", value=False, key="dc_sched_fullweek")
    _render_ema_b_schedule(today_only=not show_full)
    st.divider()

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

    # Initialize once, keep Session State as the single source of truth
    default_range = (max_date, max_date)
    st.session_state.setdefault("dc_date_range", default_range)

    def clamp(d: date) -> date:
        return max(min(d, max_date), min_date)

    def set_range(start: date, end: date):
        st.session_state["dc_date_range"] = (clamp(start), clamp(end))
        # also push into individual pickers so they reflect preset changes
        st.session_state["dc_start_date"] = clamp(start)
        st.session_state["dc_end_date"]   = clamp(end)
        _rerun()

    # ---------- Preset rows ----------
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    if r1c1.button("Today", key="dc_btn_today", use_container_width=True):
        t = date.today(); set_range(t, t)
    if r1c2.button("Yesterday", key="dc_btn_yesterday", use_container_width=True):
        y = date.today() - timedelta(days=1); set_range(y, y)
    if r1c3.button("This Week", key="dc_btn_this_week", use_container_width=True):
        d0 = date.today(); start = d0 - timedelta(days=d0.weekday()); set_range(start, d0)
    if r1c4.button("Rolling month", key="dc_btn_rolling_month", use_container_width=True):
        # True month arithmetic: one calendar month back from today (handles 31->30/28/29 correctly)
        d0 = date.today()
        start = d0 - relativedelta(months=1)   # e.g., Oct 31 -> Sep 30; Oct 8 -> Sep 8
        set_range(start, d0)

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

    # ---------- Left-aligned Start/End pickers ----------
    prev_start, prev_end = st.session_state.get("dc_date_range", default_range)

    # Narrow layout: keep both pickers small and aligned to the left
    c_start, c_gap, c_end, _ = st.columns([0.16, 0.02, 0.16, 0.66])

    with c_start:
        start_picked = st.date_input(
            "Start date",
            value=st.session_state.get("dc_start_date", clamp(prev_start)),
            min_value=min_date,
            max_value=max_date,
            key="dc_start_date",
        )

    with c_end:
        end_picked = st.date_input(
            "End date",
            value=st.session_state.get("dc_end_date", clamp(prev_end)),
            min_value=min_date,
            max_value=max_date,
            key="dc_end_date",
        )

    # Clamp & normalize order
    d1 = clamp(start_picked)
    d2 = clamp(end_picked)
    if d1 > d2:
        d1, d2 = d2, d1

    # Persist normalized range for the rest of the app & preset buttons
    st.session_state["dc_date_range"] = (d1, d2)


    # ---- Now compute the filtered view and user list (ALWAYS) ----
    view = df[(df["Date"] >= d1) & (df["Date"] <= d2)].copy()

    users_in_view = view["User"].dropna().unique().tolist()
    preferred_user_order = ["Chad", "Kelly"]
    ordered_users = [u for u in preferred_user_order if u in users_in_view] + \
                    [u for u in sorted(users_in_view) if u not in preferred_user_order]

    if not ordered_users:
        st.warning("Nobody placed any trades for the selected date range.")
        st.stop()  # or return

    # --- User filter (default: show all users in the current view) ---
    users = st.multiselect(
        "Users to display", options=ordered_users, default=ordered_users, key="dc_user_filter"
    )

    if not users:
        st.caption("No users selected.")
        return

    else:
        # --- PASS 1: show headers + strategy grids, and collect per-user trades ---
        user_data = []  # list of tuples: (user, trades_df, has_selection)

        cols = st.columns(len(users))
        for col, user in zip(cols, users):
            with col:
                df_user = view[view["User"] == user].copy()
                picked_strats = render_user_table_with_toggles(user, df_user)

                # Build the trades set used later (table + chart)
                if picked_strats:
                    trades = df_user[df_user["Strategy"].isin(picked_strats)].copy()
                    has_selection = True
                else:
                    # Nothing picked yet -> don’t filter strategies; we’ll still chart all
                    trades = df_user.copy()
                    has_selection = False

                # Keep only option trades
                trades = trades[trades["Right"].isin(["C", "P"])]
                user_data.append((user, trades, has_selection))

        # --- PASS 2: trade tables (aligned across columns) ---
        cols = st.columns(len(users))
        for col, (user, trades, has_selection) in zip(cols, user_data):
            with col:
                if has_selection:
                    if not trades.empty:
                        render_trades_table(trades, title=f"{user} — selected strategy trades")
                    else:
                        st.caption("No trades to show for the selected strategies.")
                else:
                    st.caption("Select strategies above to show the trade table.")

        # --- PASS 3: charts (also aligned across columns) ---
        cols = st.columns(len(users))
        for col, (user, trades, _has_selection) in zip(cols, user_data):
            with col:
                if not trades.empty:
                    _pnl_line_chart(trades, title=f"{user} — PnL over time")
                else:
                    st.caption("No trades to chart.")
    # (Dropped duplicate breakdown table per user request)
    return
