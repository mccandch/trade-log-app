# tabs/daily_compare.py

from __future__ import annotations

from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials

def _inject_dc_compact_css():
    if st.session_state.get("_dc_compact_css_done"):
        return
    st.session_state["_dc_compact_css_done"] = True
    st.markdown("""
    <style>
      /* Scope to our rows wrapper so it won't leak elsewhere */
      .dc-rows div[data-testid="stCheckbox"] { margin: 0 !important; }
      .dc-rows label[data-testid="stWidgetLabel"] { margin: 0 !important; padding: 0 !important; }
      .dc-rows [data-testid="stHorizontalBlock"] { gap: .55rem !important; } /* small col gap */
      .dc-rows .stColumn > div { margin: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important; }
      .dc-pct { display:block; text-align:right; width:6.5rem; font-variant-numeric: tabular-nums; }
    </style>
    """, unsafe_allow_html=True)

def _inject_dc_css():
    if st.session_state.get("_dc_css_done"):
        return
    st.session_state["_dc_css_done"] = True

    st.markdown("""
    <style>
      /* Keep things compact */
      .dc-card .element-container{ margin-bottom: 2px !important; }
      .dc-card [data-testid="stMarkdownContainer"] p{ margin:0 !important; }

      /* Strategy | PCR laid out as two columns with a visible gap */
      .dc-card .dc-row{
        display: grid;
        grid-template-columns: 1fr 6rem;   /* left flexes, right is fixed width */
        column-gap: .75rem;                /* <-- visible spacing between columns */
        align-items: center;
        min-height: 28px;                  /* match checkbox height */
        margin: 2px 0;
      }
      .dc-card .dc-strat{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .dc-card .dc-pcr{ text-align: right; white-space: nowrap; }

      /* Checkbox vertical alignment */
      .dc-card [data-testid="stCheckbox"]{ margin:0 !important; }
      .dc-card [data-testid="stCheckbox"] > div{
        display:flex; align-items:center; min-height:28px;
      }
      .dc-card [data-testid="stCheckbox"] div[role="checkbox"]{
        transform: translateY(-2px);   /* nudge up; increase to -3px if needed */
      }
    </style>
    """, unsafe_allow_html=True)

# ---------- small utility ----------

def _rerun():
    try:
        st.experimental_rerun()
    except Exception:
        st.rerun()


# ---------- Google Sheets I/O ----------

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


@st.cache_data(ttl=60, show_spinner=False)
def load_sheets_data() -> pd.DataFrame:
    """
    Read all tabs that start with Raw_* and normalize to columns:
    User, Date (date), DateTime (datetime), Strategy, TotalPremium, ProfitLoss.

    Time parsing is robust to:
      - Google Sheets time fractions (0..1 of a day)
      - "HH:MM[:SS]" / "h:mm AM/PM" strings
      - integer minutes since midnight (0..1440)
      - HHMM integers (e.g., 931 -> 09:31)
    """
    sh = open_sheet()
    frames: List[pd.DataFrame] = []

    for ws in sh.worksheets():
        if not ws.title.startswith("Raw_"):
            continue

        df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all")
        if df.empty:
            continue

        out = pd.DataFrame()

        # --- User ---
        if "User" in df.columns and df["User"].notna().any():
            out["User"] = df["User"]
        else:
            out["User"] = ws.title.replace("Raw_", "")  # fallback from sheet name

        # --- Date (date only) ---
        out["Date"] = pd.to_datetime(df.get("Date"), errors="coerce").dt.date

        # --- DateTime (default = date at midnight) ---
        out["DateTime"] = pd.to_datetime(out["Date"], errors="coerce")

        # Prefer an existing DateTime column if present
        if "DateTime" in df.columns:
            dt = pd.to_datetime(df["DateTime"], errors="coerce")
            mask = dt.notna()
            out.loc[mask, "DateTime"] = dt[mask]
        else:
            # Try to find a time-of-day column
            time_col = next(
                (c for c in ("EntryTime", "Time", "TimeOpened", "Time Opened", "Entry Time") if c in df.columns),
                None,
            )
            if time_col is not None:
                tser = df[time_col]

                def to_seconds(v):
                    if pd.isna(v):
                        return np.nan
                    if isinstance(v, pd.Timestamp):
                        return v.hour * 3600 + v.minute * 60 + v.second
                    if isinstance(v, (int, float, np.number)):
                        f = float(v)
                        if 0 <= f <= 1:            # fraction of a day
                            return int(round(f * 86400))
                        if 0 <= f < 1440:          # minutes since midnight
                            return int(round(f)) * 60
                        n = int(round(f))
                        if 0 <= n < 2400:          # HHMM
                            h, m = divmod(n, 100)
                            return h * 3600 + m * 60
                        return np.nan
                    if isinstance(v, str):
                        s = v.strip()
                        if not s:
                            return np.nan
                        try:
                            tt = pd.to_datetime("1970-01-01 " + s, errors="raise")
                            return int(tt.hour * 3600 + tt.minute * 60 + tt.second)
                        except Exception:
                            if s.isdigit():
                                n = int(s)
                                if 0 <= n < 1440:
                                    return n * 60
                                if 0 <= n < 2400:
                                    h, m = divmod(n, 100)
                                    return h * 3600 + m * 60
                            return np.nan
                    return np.nan

                secs = tser.apply(to_seconds)
                mask = secs.notna()
                if mask.any():
                    base_dates = pd.to_datetime(out.loc[mask, "Date"], errors="coerce")
                    out.loc[mask, "DateTime"] = base_dates + pd.to_timedelta(secs[mask], unit="s")

        # --- Strategy / Premium / PnL ---
        out["Strategy"] = df.get("Strategy").astype(str)
        out["TotalPremium"] = pd.to_numeric(df.get("TotalPremium"), errors="coerce").fillna(0.0)
        out["ProfitLoss"] = pd.to_numeric(df.get("ProfitLoss"), errors="coerce").fillna(0.0)

        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["User", "Date", "DateTime", "Strategy", "TotalPremium", "ProfitLoss"])

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["Date"].notna()].reset_index(drop=True)
    return all_df


# ---------- presentation helpers ----------

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
    g = (
        df_user.groupby("Strategy", dropna=False)
        .agg(Premium=("TotalPremium", "sum"), PnL=("ProfitLoss", "sum"))
        .reset_index()
    )
    g["PCR"] = g.apply(lambda r: _pcr(r["PnL"], r["Premium"]), axis=1)
    g["Strategy"] = g["Strategy"].astype(str)
    g = g.sort_values("Strategy", key=lambda s: s.map(_order_key))
    return g[["Strategy", "PCR"]]

def render_trades_table(trades_df: pd.DataFrame, *, title: str):
    """Pretty table: Entry time | Strategy | PnL ($), green for wins & red for losses."""
    if trades_df.empty:
        st.info(f"No trades for {title.lower()}.")
        return

    # prefer a datetime-like column if present
    dt_col = "DateTime" if "DateTime" in trades_df.columns else "Date"
    show = (
        trades_df.rename(columns={dt_col: "Entry time", "ProfitLoss": "PnL"})
        .loc[:, ["Entry time", "Strategy", "PnL"]]
        .sort_values("Entry time")
        .copy()
    )
    # Optional: trim seconds
    try:
        show["Entry time"] = pd.to_datetime(show["Entry time"]).dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    # format + color rows
    styler = show.style.format({"PnL": "${:,.2f}"})
    # Hide index across pandas versions
    try:
        styler = styler.hide(axis="index")           # pandas >= 1.4
    except Exception:
        try:
            styler = styler.hide_index()             # older pandas
        except Exception:
            styler = styler.set_table_styles([
                {"selector": "th.row_heading", "props": "display: none;"},
                {"selector": "th.blank",        "props": "display: none;"},
            ], overwrite=False)

    def _row_color(row):
        try:
            v = float(row["PnL"])
        except Exception:
            return [""] * len(row)
        if v > 0:
            bg = "rgba(0, 128, 0, 0.18)"   # medium green
        elif v < 0:
            bg = "rgba(200, 0, 0, 0.20)"   # medium red
        else:
            bg = ""
        return [f"background-color: {bg};"] * len(row)

    styler = styler.apply(_row_color, axis=1)
    st.markdown(f"##### {title}")
    st.markdown(styler.set_table_attributes('class="compact-table"').to_html(),
                unsafe_allow_html=True)

def render_user_card_with_toggles(user: str, df_user: pd.DataFrame) -> list[str]:
    # overall
    prem_sum = float(pd.to_numeric(df_user.get("TotalPremium", 0), errors="coerce").sum())
    pnl_sum  = float(pd.to_numeric(df_user.get("ProfitLoss", 0), errors="coerce").sum())
    overall  = _pcr(pnl_sum, prem_sum)

    st.markdown(f"### {user}")
    st.caption(f"{overall:+.1f}% overall")

    # Header aligned with our compact rows (checkbox | % | strategy)
    h0, h1, h2 = st.columns([0.07, 0.14, 0.79])
    with h0: st.write("")
    with h1: st.caption("PCR")
    with h2: st.caption("Strategy")

    selected: list[str] = []
    pcr_df = _user_strategy_pcr(df_user)

    # rows (wrap in a class so our compact CSS can target it)
    st.markdown("<div class='dc-rows'>", unsafe_allow_html=True)

    for _, r in pcr_df.iterrows():
        sname = str(r["Strategy"])
        pct   = float(r["PCR"]) if pd.notna(r["PCR"]) else 0.0
        color = "#32CD32" if pct > 0 else ("#CC3333" if pct < 0 else "#AAAAAA")

        # checkbox | % (fixed width) | strategy (flex)
        c0, c1, c2 = st.columns([0.07, 0.14, 0.79])

        # give a real label but hide it -> fixes accessibility warnings
        chk = c0.checkbox(
            f"Select {user} {sname}",
            key=f"dc_row_{user}_{sname}",
            value=False,
            label_visibility="collapsed",
        )

        c1.markdown(f"<span class='dc-pct' style='color:{color};'>{pct:+.1f}%</span>",
                    unsafe_allow_html=True)
        c2.write(ALIAS.get(sname, sname))

        if chk:
            selected.append(sname)

    st.markdown("</div>", unsafe_allow_html=True)
    return selected

# ---------- main tab ----------

def daily_compare_tab():
    _inject_dc_compact_css()
    st.subheader("Daily Compare (Google Sheets)")

    # Refresh button
    if st.button("Refresh data", key="dc_btn_refresh"):
        load_sheets_data.clear()
        _rerun()

    df = load_sheets_data()
    if df.empty:
        st.info("No data found yet. Start your watchers (Raw_* tabs) and click Refresh.")
        return

    # --- date range quick buttons (ordered) ---
    min_date, max_date = df["Date"].min(), df["Date"].max()

    if "dc_date_range" not in st.session_state or st.session_state.dc_date_range is None:
        st.session_state.dc_date_range = (max_date, max_date)

    def clamp(d: date) -> date:
        return max(min(d, max_date), min_date)

    def set_range(start: date, end: date):
        st.session_state.dc_date_range = (clamp(start), clamp(end))
        _rerun()

    # Row 1: Today, Yesterday, This Week
    r1c1, r1c2, r1c3 = st.columns(3)
    if r1c1.button("Today", key="dc_btn_today", use_container_width=True):
        t = date.today(); set_range(t, t)
    if r1c2.button("Yesterday", key="dc_btn_yesterday", use_container_width=True):
        y = date.today() - timedelta(days=1); set_range(y, y)
    if r1c3.button("This Week", key="dc_btn_this_week", use_container_width=True):
        d0 = date.today(); start = d0 - timedelta(days=d0.weekday()); set_range(start, d0)

    # Row 2: Last Week, This Month, Last Month, YTD
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

    # ---- Per-user UI: card with inline toggles; below it, the trade log ----
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

            picked_strats = render_user_card_with_toggles(user, df_user)

            if picked_strats:
                trades = df_user[df_user["Strategy"].isin(picked_strats)].copy()
                render_trades_table(trades, title=f"{user} — selected strategy trades")

    # ---- Global Strategy breakdown (selected range) ----
    st.divider()
    stats = (
        view.groupby(["User", "Strategy"], dropna=False)
        .agg(Premium=("TotalPremium", "sum"), PnL=("ProfitLoss", "sum"))
        .reset_index()
    )
    stats["PCR %"] = np.where(stats["Premium"] != 0, (stats["PnL"] / stats["Premium"]) * 100.0, np.nan)
    stats = stats[["User", "Strategy", "PCR %", "Premium", "PnL"]]

    st.subheader("Strategy breakdown (selected range)")
    st.dataframe(stats, use_container_width=True)
