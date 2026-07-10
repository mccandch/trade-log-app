"""Trade Log Analyzer tab.

Turns an uploaded options trade-log CSV (OptionOmega/SPX-style) into a full
strategy research dashboard: summary metrics, daily analysis, strategy /
Volga / side / time-of-day breakdowns, risk & margin stats, concurrent
exposure, outliers, an interactive rule tester, and CSV / Google Sheets
exports.

Entry point: render_trade_log_analyzer_tab()
"""

from __future__ import annotations

import io
import re
from datetime import datetime, time as dtime

import gspread
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials as UserCredentials
from gspread_dataframe import set_with_dataframe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY = "trade_log_"  # prefix for every widget key in this tab

REQUIRED_COLUMNS = [
    "Date Opened", "Time Opened", "Date Closed", "Time Closed", "P/L", "Strategy",
]

# Alternate header spellings seen in real exports -> canonical name
COLUMN_ALIASES = {
    "Opening Commissions + Fees": "Opening Commission",
    "Closing Commissions + Fees": "Closing Commission",
    "Opening Commission + Fees": "Opening Commission",
    "Closing Commission + Fees": "Closing Commission",
    "Opening Short/Long Ratio": "Ratio",
}

NUMERIC_COLUMNS = [
    "P/L", "P/L %", "Premium", "Opening Price", "Closing Price",
    "Avg. Closing Cost", "No. of Contracts", "Funds at Close", "Margin Req.",
    "Opening Commission", "Closing Commission", "Ratio", "Gap", "Movement",
    "Max Profit", "Max Loss", "Days in Trade",
]

TIME_BUCKETS = [
    ("09:30-10:00", dtime(9, 30), dtime(10, 0)),
    ("10:00-10:30", dtime(10, 0), dtime(10, 30)),
    ("10:30-11:00", dtime(10, 30), dtime(11, 0)),
    ("11:00-11:30", dtime(11, 0), dtime(11, 30)),
    ("11:30-12:00", dtime(11, 30), dtime(12, 0)),
    ("12:00-13:00", dtime(12, 0), dtime(13, 0)),
    ("13:00-14:00", dtime(13, 0), dtime(14, 0)),
    ("14:00-15:00", dtime(14, 0), dtime(15, 0)),
    ("15:00-16:00", dtime(15, 0), dtime(16, 0)),
]
TIME_BUCKET_ORDER = [b[0] for b in TIME_BUCKETS] + ["Outside RTH / Unknown"]

TRADE_COUNT_BUCKETS = [
    (1, 5), (6, 10), (11, 15), (16, 20), (21, 25),
    (26, 30), (31, 40), (41, 50), (51, 75), (76, np.inf),
]

ROLLING_WINDOWS_MONTHS = [1, 2, 3, 4, 6, 8, 9, 12]
ROLLING_WINDOW_LABELS = [f"{m} Mo" for m in ROLLING_WINDOWS_MONTHS]

# The rolling-stat metrics, each with how its values are formatted.
ROLLING_HISTORY_METRICS = [
    ("PCR %", "pct"),
    ("Win Rate %", "pct"),
    ("Total P/L", "money"),
    ("Max Drawdown $", "money"),
    ("Max Drawdown %", "pct"),
    ("MAR", "num"),
    ("CAGR %", "pct"),
    ("Profit Factor", "num"),
    ("Trades", "int"),
]

# Chart colors (validated reference palette; chart chrome comes from the
# Streamlit plotly theme so light/dark mode both work).
POS_COLOR = "#2a78d6"   # diverging cool pole -> gains
NEG_COLOR = "#e34948"   # diverging warm pole -> losses
LINE_COLOR = "#2a78d6"  # single-series lines/bars

# Fixed categorical slot order (validated set); each rolling window always
# keeps the same color even when only some windows are charted.
CATEGORICAL_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                      "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
ROLLING_WINDOW_COLORS = dict(zip(ROLLING_WINDOW_LABELS, CATEGORICAL_COLORS))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_currency(v, decimals: int = 2) -> str:
    """Format a number as currency, e.g. -1234.5 -> '-$1,234.50'."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "N/A"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.{decimals}f}"


def fmt_pct(v, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "N/A"
    return f"{v:,.{decimals}f}%"


def fmt_num(v, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "N/A"
    return f"{v:,.{decimals}f}"


def _safe_div(a, b):
    """Divide, returning NaN on zero/NaN denominator."""
    try:
        if b is None or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except (TypeError, ZeroDivisionError):
        return np.nan


def _pcr(g: pd.DataFrame):
    """Premium Capture Rate: total P/L as a % of total premium collected.

    Premium in the log is per contract (credit x 100) while P/L is for the
    whole position, so premium must be scaled by No. of Contracts.
    """
    if "Premium" not in g.columns:
        return np.nan
    prem = g["Premium"].abs()
    if "No. of Contracts" in g.columns:
        prem = prem * g["No. of Contracts"].fillna(1)
    return _safe_div(100 * g["P/L"].sum(), prem.sum())


def parse_numeric(value):
    """Robustly parse '$1,234.56', '-$1,234.56', '($1,234.56)', '12.5%',
    blanks, NaN and garbage into a float (NaN when unparseable)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "n/a", "-", "--"):
        return np.nan
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    if s.startswith("-"):
        negative = not negative if negative else True
        s = s.lstrip("-")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if negative else v


# ---------------------------------------------------------------------------
# Load / validate / clean
# ---------------------------------------------------------------------------

def load_trade_log(uploaded_file) -> pd.DataFrame:
    """Read the uploaded CSV into a raw DataFrame (no cleaning yet)."""
    raw = uploaded_file.getvalue()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file as UTF-8 or Latin-1.")


def validate_trade_log_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of missing required columns (empty list = valid)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def clean_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known header variants and parse all numeric columns."""
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(parse_numeric)
    return df


def _classify_side(strategy, legs) -> str:
    """'Put' / 'Call' / 'Mixed/Other' from the Strategy name, falling back
    to parsing option types out of the Legs string."""
    text = str(strategy).lower() if pd.notna(strategy) else ""
    has_put = "put" in text
    has_call = "call" in text
    if not has_put and not has_call and isinstance(legs, str):
        types = set(re.findall(r"\d+(?:\.\d+)?\s+([PC])\b", legs))
        has_put, has_call = "P" in types, "C" in types
    if has_put and has_call:
        return "Mixed/Other"
    if has_put:
        return "Put"
    if has_call:
        return "Call"
    return "Mixed/Other"


def _classify_volga(strategy) -> str:
    """'Volga' / 'Non-Volga' / 'Unknown' from the Strategy name."""
    if pd.isna(strategy):
        return "Unknown"
    normalized = " ".join(re.sub(r"[^a-z]+", " ", str(strategy).lower()).split())
    if "non volga" in normalized:
        return "Non-Volga"
    if "volga" in normalized:
        return "Volga"
    return "Unknown"


def _time_bucket(ts) -> str:
    if pd.isna(ts):
        return "Outside RTH / Unknown"
    t = ts.time()
    for label, start, end in TIME_BUCKETS:
        if start <= t < end:
            return label
    return "Outside RTH / Unknown"


def _fixed_pct_buckets(s: pd.Series) -> pd.Series:
    """Bucket a %-style series into the fixed <-1% ... >1% ranges."""
    edges = [-np.inf, -1, -0.5, 0, 0.5, 1, np.inf]
    labels = ["< -1%", "-1% to -0.5%", "-0.5% to 0%", "0% to 0.5%", "0.5% to 1%", "> 1%"]
    out = pd.cut(s, edges, labels=labels).astype(object)
    return out.where(s.notna(), "Unknown")


def _quantile_buckets(s: pd.Series, n: int = 5, prefix: str = "$") -> pd.Series:
    """Quantile-based buckets with readable range labels; 'Unknown' for NaN."""
    out = pd.Series("Unknown", index=s.index, dtype=object)
    valid = s.dropna()
    if valid.nunique() < 2:
        return out
    try:
        bins = pd.qcut(valid, min(n, valid.nunique()), duplicates="drop")
    except ValueError:
        return out
    out.loc[valid.index] = bins.map(
        lambda iv: f"{prefix}{iv.left:,.0f} to {prefix}{iv.right:,.0f}"
    ).astype(object)
    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add datetimes, calendar fields, hold time and classification columns."""
    df = df.copy()
    df["Open DateTime"] = pd.to_datetime(
        df["Date Opened"].astype(str) + " " + df["Time Opened"].astype(str),
        errors="coerce",
    )
    df["Close DateTime"] = pd.to_datetime(
        df["Date Closed"].astype(str) + " " + df["Time Closed"].astype(str),
        errors="coerce",
    )
    df["Open Date"] = df["Open DateTime"].dt.date
    df["Month"] = df["Open DateTime"].dt.to_period("M").astype(str)
    df["Year"] = df["Open DateTime"].dt.year
    df["Day of Week"] = df["Open DateTime"].dt.day_name()
    df["Open Hour"] = df["Open DateTime"].dt.hour
    df["Hold Minutes"] = (df["Close DateTime"] - df["Open DateTime"]).dt.total_seconds() / 60

    df["Trade Side"] = [
        _classify_side(s, l)
        for s, l in zip(df["Strategy"], df.get("Legs", pd.Series(index=df.index, dtype=object)))
    ]
    df["Volga Type"] = df["Strategy"].map(_classify_volga)
    df["Win/Loss"] = np.select(
        [df["P/L"] > 0, df["P/L"] < 0], ["Win", "Loss"], default="Breakeven"
    )
    df["Time Bucket"] = df["Open DateTime"].map(_time_bucket)

    if "Premium" in df.columns:
        df["Premium Bucket"] = _quantile_buckets(df["Premium"])
    if "Margin Req." in df.columns:
        df["Margin Bucket"] = _quantile_buckets(df["Margin Req."])
    if "Gap" in df.columns:
        df["Gap Bucket"] = _fixed_pct_buckets(df["Gap"])
    if "Movement" in df.columns:
        df["Movement Bucket"] = _fixed_pct_buckets(df["Movement"])
    return df


@st.cache_data(show_spinner="Parsing trade log...")
def prepare_trade_log(file_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    """Full load -> clean -> derive pipeline, cached on the file contents.

    Returns (df, info) where info reports rows dropped during cleaning.
    """
    raw = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            raw = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise ValueError("Could not decode the file as UTF-8 or Latin-1.")

    df = clean_trade_log(raw)
    missing = validate_trade_log_columns(df)
    if missing:
        return raw, {"missing_required": missing, "raw_rows": len(raw)}

    df = add_derived_columns(df)
    before = len(df)
    df = df[df["P/L"].notna() & df["Open DateTime"].notna()].reset_index(drop=True)
    info = {
        "missing_required": [],
        "raw_rows": before,
        "dropped_rows": before - len(df),
        "missing_optional": [
            c for c in ("Premium", "Margin Req.", "Reason For Close", "Underlying",
                        "Opening Commission", "Closing Commission", "No. of Contracts")
            if c not in df.columns
        ],
    }
    return df, info


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def calculate_drawdown(df: pd.DataFrame, starting_capital: float) -> tuple[pd.DataFrame, float, float]:
    """Equity curve + drawdown from trades ordered by close time.

    Returns (equity_df, max_drawdown_dollars, max_drawdown_pct).
    """
    order_col = "Close DateTime" if df["Close DateTime"].notna().any() else "Open DateTime"
    eq = df.sort_values(order_col).copy()
    eq["Cumulative P/L"] = eq["P/L"].cumsum()
    eq["Equity"] = starting_capital + eq["Cumulative P/L"]
    eq["Peak"] = eq["Equity"].cummax()
    eq["Drawdown $"] = eq["Equity"] - eq["Peak"]
    eq["Drawdown %"] = 100 * eq["Drawdown $"] / eq["Peak"]
    max_dd = eq["Drawdown $"].min() if len(eq) else np.nan
    max_dd_pct = eq["Drawdown %"].min() if len(eq) else np.nan
    return eq[[order_col, "Cumulative P/L", "Equity", "Drawdown $", "Drawdown %"]].rename(
        columns={order_col: "DateTime"}
    ), max_dd, max_dd_pct


def calculate_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day P/L, counts, win rate, margin stats and day result."""
    rows = []
    for day, g in df.groupby("Open Date"):
        pl = g["P/L"]
        wins, losses = (pl > 0).sum(), (pl < 0).sum()
        row = {
            "Open Date": day,
            "Trades": len(g),
            "Wins": wins,
            "Losses": losses,
            "Win Rate %": _safe_div(100 * wins, len(g)),
            "PCR %": _pcr(g),
            "Daily P/L": pl.sum(),
            "Avg Trade P/L": pl.mean(),
            "Best Trade": pl.max(),
            "Worst Trade": pl.min(),
        }
        if "Margin Req." in g.columns:
            row["Total Margin"] = g["Margin Req."].sum()
            row["Max Margin"] = g["Margin Req."].max()
            row["Avg Margin"] = g["Margin Req."].mean()
        total = pl.sum()
        row["Day Result"] = "Win" if total > 0 else ("Loss" if total < 0 else "Breakeven")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Open Date").reset_index(drop=True)


def _max_consecutive(results: pd.Series, target: str) -> int:
    best = run = 0
    for r in results:
        run = run + 1 if r == target else 0
        best = max(best, run)
    return best


def calculate_summary_metrics(df: pd.DataFrame, starting_capital: float) -> list[tuple[str, str]]:
    """Headline metrics for the executive summary, as (label, value) pairs."""
    pl = df["P/L"]
    wins, losses = df[pl > 0], df[pl < 0]
    gross_win, gross_loss = wins["P/L"].sum(), abs(losses["P/L"].sum())
    daily = calculate_daily_summary(df)
    _, max_dd, max_dd_pct = calculate_drawdown(df, starting_capital)

    total_comm = np.nan
    if "Opening Commission" in df.columns or "Closing Commission" in df.columns:
        total_comm = (
            df.get("Opening Commission", pd.Series(dtype=float)).sum()
            + df.get("Closing Commission", pd.Series(dtype=float)).sum()
        )

    # Annualized return / MAR from the calendar span of the log
    cagr = np.nan
    first = df["Open DateTime"].min()
    last = df["Close DateTime"].max() if df["Close DateTime"].notna().any() else df["Open DateTime"].max()
    if pd.notna(first) and pd.notna(last) and starting_capital > 0:
        days = (last - first).days
        end_equity = starting_capital + pl.sum()
        if days > 0 and end_equity > 0:
            cagr = 100 * ((end_equity / starting_capital) ** (365.0 / days) - 1)
    mar = _safe_div(cagr, abs(max_dd_pct)) if pd.notna(max_dd_pct) else np.nan

    avg_margin = df["Margin Req."].mean() if "Margin Req." in df.columns else np.nan

    win_days = (daily["Day Result"] == "Win").sum()
    loss_days = (daily["Day Result"] == "Loss").sum()

    metrics = [
        ("Total P/L", fmt_currency(pl.sum())),
        ("Total Trades", f"{len(df):,}"),
        ("Winning Trades", f"{len(wins):,}"),
        ("Losing Trades", f"{len(losses):,}"),
        ("Win Rate", fmt_pct(_safe_div(100 * len(wins), len(df)))),
        ("PCR (Premium Capture)", fmt_pct(_pcr(df))),
        ("Avg Trade P/L", fmt_currency(pl.mean())),
        ("Median Trade P/L", fmt_currency(pl.median())),
        ("Avg Winning Trade", fmt_currency(wins["P/L"].mean() if len(wins) else np.nan)),
        ("Avg Losing Trade", fmt_currency(losses["P/L"].mean() if len(losses) else np.nan)),
        ("Profit Factor", fmt_num(_safe_div(gross_win, gross_loss), 2)),
        ("Expectancy / Trade", fmt_currency(pl.mean())),
        ("Best Trade", fmt_currency(pl.max())),
        ("Worst Trade", fmt_currency(pl.min())),
        ("Total Commissions", fmt_currency(total_comm)),
        ("P/L After Commissions", fmt_currency(pl.sum() - total_comm) if pd.notna(total_comm) else "N/A"),
        ("Avg Margin Req.", fmt_currency(avg_margin, 0)),
        ("Avg Return on Margin / Trade", fmt_pct(_safe_div(100 * pl.mean(), avg_margin), 2)),
        ("Max Drawdown", f"{fmt_currency(max_dd)} ({fmt_pct(max_dd_pct)})"),
        ("CAGR (annualized)", fmt_pct(cagr)),
        ("MAR Ratio", fmt_num(mar, 2)),
        ("Avg Trades / Day", fmt_num(daily["Trades"].mean())),
        ("Median Trades / Day", fmt_num(daily["Trades"].median())),
        ("Max Trades / Day", f"{daily['Trades'].max():,}"),
        ("Winning Day %", fmt_pct(_safe_div(100 * win_days, len(daily)))),
        ("Losing Day %", fmt_pct(_safe_div(100 * loss_days, len(daily)))),
        ("Best Day", fmt_currency(daily["Daily P/L"].max())),
        ("Worst Day", fmt_currency(daily["Daily P/L"].min())),
    ]
    return metrics


def _trade_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Generic per-group trade statistics used by all breakdown sections."""
    rows = []
    for name, g in df.groupby(group_col, dropna=False):
        pl = g["P/L"]
        wins, losses = g[pl > 0], g[pl < 0]
        row = {
            group_col: name,
            "Trades": len(g),
            "Total P/L": pl.sum(),
            "Win Rate %": _safe_div(100 * len(wins), len(g)),
            "PCR %": _pcr(g),
            "Avg P/L": pl.mean(),
            "Median P/L": pl.median(),
            "Avg Win": wins["P/L"].mean() if len(wins) else np.nan,
            "Avg Loss": losses["P/L"].mean() if len(losses) else np.nan,
            "Profit Factor": _safe_div(wins["P/L"].sum(), abs(losses["P/L"].sum())),
            "Best Trade": pl.max(),
            "Worst Trade": pl.min(),
        }
        if "Reason For Close" in g.columns:
            reason = g["Reason For Close"].astype(str)
            row["Stop Losses"] = reason.str.contains("stop", case=False, na=False).sum()
            row["Expirations"] = reason.str.contains("expir", case=False, na=False).sum()
        if "Premium" in g.columns:
            row["Avg Premium"] = g["Premium"].mean()
        if "Margin Req." in g.columns:
            row["Avg Margin"] = g["Margin Req."].mean()
            row["Return on Margin %"] = _safe_div(100 * pl.sum(), g["Margin Req."].sum())
        if "Hold Minutes" in g.columns:
            row["Avg Hold (min)"] = g["Hold Minutes"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("Total P/L", ascending=False).reset_index(drop=True) if len(out) else out


def _daily_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-group day-level statistics (winning day %, best/worst day, ...)."""
    rows = []
    for name, g in df.groupby(group_col, dropna=False):
        daily = g.groupby("Open Date")["P/L"].agg(["sum", "count"])
        win_days = (daily["sum"] > 0).sum()
        loss_days = (daily["sum"] < 0).sum()
        rows.append({
            group_col: name,
            "Winning Day %": _safe_div(100 * win_days, len(daily)),
            "Losing Day %": _safe_div(100 * loss_days, len(daily)),
            "Avg Daily P/L": daily["sum"].mean(),
            "Best Day": daily["sum"].max(),
            "Worst Day": daily["sum"].min(),
            "Avg Trades/Day": daily["count"].mean(),
            "Max Trades/Day": daily["count"].max(),
        })
    return pd.DataFrame(rows)


def calculate_strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _trade_group_stats(df, "Strategy")


def calculate_volga_summary(df: pd.DataFrame) -> pd.DataFrame:
    trade = _trade_group_stats(df, "Volga Type")
    daily = _daily_group_stats(df, "Volga Type")
    return trade.merge(daily, on="Volga Type", how="left") if len(trade) else trade


def calculate_side_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _trade_group_stats(df, "Trade Side")


def calculate_time_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = _trade_group_stats(df, "Time Bucket")
    if len(out):
        out["__order"] = out["Time Bucket"].map(
            {b: i for i, b in enumerate(TIME_BUCKET_ORDER)}
        )
        out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return out


def calculate_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _trade_group_stats(df, "Reason For Close") if "Reason For Close" in df.columns else pd.DataFrame()


def calculate_trade_count_bucket_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Day-level stats bucketed by how many trades were taken that day."""
    rows = []
    for lo, hi in TRADE_COUNT_BUCKETS:
        label = f"{lo}-{int(hi)}" if np.isfinite(hi) else f"{lo}+"
        g = daily[(daily["Trades"] >= lo) & (daily["Trades"] <= hi)]
        if g.empty:
            continue
        win_days = (g["Day Result"] == "Win").sum()
        rows.append({
            "Trades/Day Bucket": label,
            "Days": len(g),
            "Avg Daily P/L": g["Daily P/L"].mean(),
            "Total P/L": g["Daily P/L"].sum(),
            "Winning Day %": _safe_div(100 * win_days, len(g)),
            "Avg Trades/Day": g["Trades"].mean(),
            "Best Day": g["Daily P/L"].max(),
            "Worst Day": g["Daily P/L"].min(),
        })
    return pd.DataFrame(rows)


def calculate_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, g in df.groupby("Month"):
        pl = g["P/L"]
        rows.append({
            "Month": month,
            "Total P/L": pl.sum(),
            "Trades": len(g),
            "Win Rate %": _safe_div(100 * (pl > 0).sum(), len(g)),
            "PCR %": _pcr(g),
        })
    return pd.DataFrame(rows).sort_values("Month").reset_index(drop=True)


def calculate_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby("Year"):
        pl = g["P/L"]
        rows.append({
            "Year": int(year),
            "Total P/L": pl.sum(),
            "Trades": len(g),
            "Win Rate %": _safe_div(100 * (pl > 0).sum(), len(g)),
            "PCR %": _pcr(g),
        })
    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)


def _window_stat_values(g: pd.DataFrame, starting_capital: float, days: int) -> dict:
    """Stat set for one rolling window's trades, annualized over `days`."""
    pl = g["P/L"]
    wins, losses = g[pl > 0], g[pl < 0]
    _, max_dd, max_dd_pct = calculate_drawdown(g, starting_capital)
    cagr = np.nan
    end_equity = starting_capital + pl.sum()
    if starting_capital > 0 and end_equity > 0 and days > 0:
        cagr = 100 * ((end_equity / starting_capital) ** (365.0 / days) - 1)
    return {
        "Total P/L": pl.sum(),
        "Win Rate %": _safe_div(100 * len(wins), len(g)),
        "PCR %": _pcr(g),
        "Avg P/L": pl.mean(),
        "Profit Factor": _safe_div(wins["P/L"].sum(), abs(losses["P/L"].sum())),
        "Max Drawdown $": max_dd,
        "Max Drawdown %": max_dd_pct,
        "CAGR %": cagr,
        "MAR": _safe_div(cagr, abs(max_dd_pct)) if pd.notna(max_dd_pct) else np.nan,
    }


def calculate_rolling_stats(df: pd.DataFrame, starting_capital: float,
                            end: pd.Timestamp | None = None,
                            data_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Trailing-window stats for each ROLLING_WINDOWS_MONTHS window.

    Windows are anchored to `end` (default: the last close/open time in the
    data) and include trades *opened* within the window. CAGR annualizes over
    the part of the window actually covered by the log, so a 12-month window
    on a 6-month log isn't diluted by empty months.
    """
    if df.empty:
        return pd.DataFrame()
    if end is None:
        end = (df["Close DateTime"].max() if df["Close DateTime"].notna().any()
               else df["Open DateTime"].max())
    if data_start is None:
        data_start = df["Open DateTime"].min()

    rows = []
    for months, label in zip(ROLLING_WINDOWS_MONTHS, ROLLING_WINDOW_LABELS):
        start = end - pd.DateOffset(months=months)
        span_start = max(start, data_start)
        days = max((end - span_start).days, 1)
        g = df[df["Open DateTime"] > start]
        row = {"Window": label, "From": span_start.date(), "Trades": len(g)}
        if len(g):
            row.update(_window_stat_values(g, starting_capital, days))
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_anchor_dates(data_start: pd.Timestamp, data_end: pd.Timestamp) -> list:
    """Month-start anchor dates covering the log, oldest to newest.

    Each anchor 'sees' only trades opened before it, so the newest anchor
    (the month start after the last trade) reflects the full latest data.
    """
    first = (data_start.to_period("M") + 1).to_timestamp()
    last = (data_end.to_period("M") + 1).to_timestamp()
    return list(pd.date_range(first, last, freq="MS"))


def calculate_rolling_history(df: pd.DataFrame, starting_capital: float,
                              anchors: list, data_start: pd.Timestamp,
                              data_end: pd.Timestamp) -> pd.DataFrame:
    """Rolling stats recomputed at every anchor date (long form).

    For each anchor A and window of N months, includes trades opened in
    (A - N months, A). Returns one row per (Anchor, Window) with the
    _window_stat_values metrics; windows with no trades keep NaN stats.
    Pass the full log's anchors/data_start/data_end when computing a single
    strategy so all scopes share the same calendar grid.
    """
    rows = []
    for anchor in anchors:
        for months, label in zip(ROLLING_WINDOWS_MONTHS, ROLLING_WINDOW_LABELS):
            w_start = anchor - pd.DateOffset(months=months)
            g = df[(df["Open DateTime"] > w_start) & (df["Open DateTime"] < anchor)]
            span_start = max(w_start, data_start)
            span_end = min(anchor, data_end)
            days = max((span_end - span_start).days, 1)
            row = {"Anchor": anchor, "Window": label, "Trades": len(g)}
            if len(g):
                row.update(_window_stat_values(g, starting_capital, days))
            rows.append(row)
    return pd.DataFrame(rows)


def rolling_history_table(hist: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot the long-form history into windows-as-rows, anchors-as-columns
    (newest first), matching the spreadsheet-style trend layout."""
    if hist.empty or metric not in hist.columns:
        return pd.DataFrame()
    t = hist.pivot(index="Window", columns="Anchor", values=metric)
    t = t.reindex(ROLLING_WINDOW_LABELS)
    t = t[sorted(t.columns, reverse=True)]
    t.columns = [f"{a.month}/{a.day}/{a.year % 100:02d}" for a in t.columns]
    return t.rename_axis(None, axis=1).reset_index()


def calculate_concurrent_exposure(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Running concurrent margin / open-trade count from open+close events."""
    need = df[df["Open DateTime"].notna() & df["Close DateTime"].notna()].copy()
    if need.empty or "Margin Req." not in need.columns:
        return pd.DataFrame(), {}
    margin = need["Margin Req."].fillna(0)
    opens = pd.DataFrame({
        "DateTime": need["Open DateTime"], "Margin Delta": margin, "Count Delta": 1,
    })
    closes = pd.DataFrame({
        "DateTime": need["Close DateTime"], "Margin Delta": -margin, "Count Delta": -1,
    })
    events = pd.concat([opens, closes]).sort_values("DateTime").reset_index(drop=True)
    events["Concurrent Margin"] = events["Margin Delta"].cumsum()
    events["Concurrent Trades"] = events["Count Delta"].cumsum()
    peak_idx = events["Concurrent Margin"].idxmax()
    stats = {
        "Max Concurrent Trades": int(events["Concurrent Trades"].max()),
        "Max Concurrent Margin": events["Concurrent Margin"].max(),
        "Avg Concurrent Margin": events["Concurrent Margin"].mean(),
        "Time of Max Exposure": events.loc[peak_idx, "DateTime"],
    }
    return events, stats


def calculate_risk_margin_metrics(df: pd.DataFrame, daily: pd.DataFrame) -> list[tuple[str, str]]:
    """Margin-based risk metrics (requires Margin Req.)."""
    m = df["Margin Req."]
    pl = df["P/L"]
    losses, wins = df[pl < 0], df[pl > 0]
    loss_pct = (100 * losses["P/L"] / losses["Margin Req."]).replace([np.inf, -np.inf], np.nan)
    win_pct = (100 * wins["P/L"] / wins["Margin Req."]).replace([np.inf, -np.inf], np.nan)
    return [
        ("Avg Margin / Trade", fmt_currency(m.mean(), 0)),
        ("Median Margin / Trade", fmt_currency(m.median(), 0)),
        ("Max Margin / Trade", fmt_currency(m.max(), 0)),
        ("Total Margin (all trades)", fmt_currency(m.sum(), 0)),
        ("Avg Daily Total Margin", fmt_currency(daily["Total Margin"].mean(), 0) if "Total Margin" in daily.columns else "N/A"),
        ("Max Daily Total Margin", fmt_currency(daily["Total Margin"].max(), 0) if "Total Margin" in daily.columns else "N/A"),
        ("P/L per $1,000 Margin", fmt_currency(_safe_div(1000 * pl.sum(), m.sum()))),
        ("Avg Return on Margin / Trade", fmt_pct(_safe_div(100 * pl.mean(), m.mean()), 2)),
        ("Worst Loss % of Margin", fmt_pct(loss_pct.min())),
        ("Avg Loss % of Margin", fmt_pct(loss_pct.mean())),
        ("Best Gain % of Margin", fmt_pct(win_pct.max())),
    ]


# ---------------------------------------------------------------------------
# Rule tester
# ---------------------------------------------------------------------------

def _drop_after_daily_event(df: pd.DataFrame, cutoff_times: dict) -> pd.DataFrame:
    """Keep only trades opened at/before each day's cutoff time (NaN = keep all)."""
    if not cutoff_times:
        return df
    cut = df["Open Date"].map(cutoff_times)
    keep = cut.isna() | (df["Open DateTime"] <= pd.to_datetime(cut))
    return df[keep]


def apply_rule_filters(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Apply the rule-tester simulation rules and return the surviving trades."""
    out = df.copy()

    if rules.get("min_time"):
        out = out[out["Open DateTime"].dt.time >= rules["min_time"]]
    if rules.get("max_time"):
        out = out[out["Open DateTime"].dt.time <= rules["max_time"]]
    if rules.get("exclude_strategies"):
        out = out[~out["Strategy"].isin(rules["exclude_strategies"])]
    if rules.get("exclude_volga"):
        out = out[~out["Volga Type"].isin(rules["exclude_volga"])]
    if rules.get("exclude_sides"):
        out = out[~out["Trade Side"].isin(rules["exclude_sides"])]
    if rules.get("min_premium") is not None and "Premium" in out.columns:
        out = out[out["Premium"].isna() | (out["Premium"] >= rules["min_premium"])]
    if rules.get("max_margin") is not None and "Margin Req." in out.columns:
        out = out[out["Margin Req."].isna() | (out["Margin Req."] <= rules["max_margin"])]

    # Drop entire days with more than X trades
    if rules.get("max_trades_per_day"):
        counts = out.groupby("Open Date")["P/L"].transform("count")
        out = out[counts <= rules["max_trades_per_day"]]

    # Stop trading for the rest of the day after the first stop loss closes
    if rules.get("stop_after_first_stop") and "Reason For Close" in out.columns:
        cutoffs = {}
        stops = out[out["Reason For Close"].astype(str).str.contains("stop", case=False, na=False)]
        for day, g in stops.groupby("Open Date"):
            cutoffs[day] = g["Close DateTime"].min()
        out = _drop_after_daily_event(out, cutoffs)

    # Stop trading once realized daily P/L crosses a loss or profit threshold
    loss_cut = rules.get("daily_loss_cutoff")
    profit_cut = rules.get("daily_profit_cutoff")
    if loss_cut or profit_cut:
        cutoffs = {}
        for day, g in out.groupby("Open Date"):
            g2 = g.sort_values("Close DateTime")
            cum = g2["P/L"].cumsum()
            hit = pd.Series(False, index=g2.index)
            if loss_cut:
                hit |= cum <= -abs(loss_cut)
            if profit_cut:
                hit |= cum >= abs(profit_cut)
            if hit.any():
                cutoffs[day] = g2.loc[hit.idxmax(), "Close DateTime"]
        out = _drop_after_daily_event(out, cutoffs)

    return out


def comparison_stats(df: pd.DataFrame, starting_capital: float) -> dict:
    """Compact stat set used for the Original vs Filtered comparison."""
    if df.empty:
        return {k: "N/A" for k in (
            "Total P/L", "Trades", "Win Rate", "PCR", "Avg Trade P/L",
            "Profit Factor", "Max Drawdown", "Worst Day", "Best Day",
            "Winning Day %", "Avg Trades/Day", "Max Trades/Day")}
    pl = df["P/L"]
    wins, losses = df[pl > 0], df[pl < 0]
    daily = calculate_daily_summary(df)
    _, max_dd, max_dd_pct = calculate_drawdown(df, starting_capital)
    win_days = (daily["Day Result"] == "Win").sum()
    return {
        "Total P/L": fmt_currency(pl.sum()),
        "Trades": f"{len(df):,}",
        "Win Rate": fmt_pct(_safe_div(100 * len(wins), len(df))),
        "PCR": fmt_pct(_pcr(df)),
        "Avg Trade P/L": fmt_currency(pl.mean()),
        "Profit Factor": fmt_num(_safe_div(wins["P/L"].sum(), abs(losses["P/L"].sum())), 2),
        "Max Drawdown": f"{fmt_currency(max_dd)} ({fmt_pct(max_dd_pct)})",
        "Worst Day": fmt_currency(daily["Daily P/L"].min()),
        "Best Day": fmt_currency(daily["Daily P/L"].max()),
        "Winning Day %": fmt_pct(_safe_div(100 * win_days, len(daily))),
        "Avg Trades/Day": fmt_num(daily["Trades"].mean()),
        "Max Trades/Day": f"{daily['Trades'].max():,}",
    }


# ---------------------------------------------------------------------------
# Google Sheets export
# ---------------------------------------------------------------------------

# Least-privilege scope: drive.file only touches files this app creates
# (enough to create, populate and share the report), not the whole Drive.
# The Sheets API accepts drive.file for app-created files, so no separate
# spreadsheets scope is needed.
SHEETS_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def report_export_enabled() -> bool:
    """Whether this deployment may create Google Sheets reports.

    Requires BOTH an explicit enable flag AND stored OAuth user credentials.
    The report is created as — and lands in the Drive of — whoever minted
    those credentials (see docs/google_sheets_oauth_setup.md). Keep both out
    of any public/shared deployment's secrets so the button never appears
    there and no other visitor can write to that Drive.
    """
    try:
        return bool(st.secrets.get("enable_sheets_export", False)) and "gcp_oauth" in st.secrets
    except Exception:
        return False


REPORT_FOLDER_NAME = "Trade Log Reports"
_DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"


def _oauth_credentials() -> UserCredentials:
    """Build & refresh OAuth user credentials for the report owner.

    Unlike a service account (which has zero Drive storage and fails with a
    quota error on the first create), an OAuth user creates files in their own
    Drive, counted against their own quota.
    """
    oauth = st.secrets["gcp_oauth"]
    creds = UserCredentials(
        None,  # no access token yet; refreshed below
        refresh_token=oauth["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=oauth["client_id"],
        client_secret=oauth["client_secret"],
        scopes=SHEETS_SCOPES,
    )
    creds.refresh(Request())
    return creds


def _get_or_create_report_folder(creds: UserCredentials) -> str | None:
    """Return the Drive folder id for REPORT_FOLDER_NAME, creating it if needed.

    With the drive.file scope this only sees folders the app itself created, so
    it reuses its own folder across runs (a same-named folder you made by hand
    is invisible to the app and left untouched). Returns None on any Drive error
    so the report still lands in Drive root rather than failing outright.
    """
    try:
        session = AuthorizedSession(creds)
        q = (
            f"name = '{REPORT_FOLDER_NAME}' "
            "and mimeType = 'application/vnd.google-apps.folder' "
            "and trashed = false"
        )
        resp = session.get(
            _DRIVE_FILES_URL,
            params={"q": q, "fields": "files(id,name)", "spaces": "drive"},
        )
        resp.raise_for_status()
        files = resp.json().get("files", [])
        if files:
            return files[0]["id"]
        resp = session.post(
            _DRIVE_FILES_URL,
            json={"name": REPORT_FOLDER_NAME,
                  "mimeType": "application/vnd.google-apps.folder"},
        )
        resp.raise_for_status()
        return resp.json()["id"]
    except Exception:
        return None


def _sheets_safe(table: pd.DataFrame) -> pd.DataFrame:
    """Convert datetimes/dates/objects to Sheets-serializable values."""
    t = table.copy()
    for c in t.columns:
        if pd.api.types.is_datetime64_any_dtype(t[c]):
            t[c] = t[c].astype(str)
        elif t[c].dtype == object:
            t[c] = t[c].map(
                lambda v: v if (v is None or isinstance(v, (str, int, float, bool)))
                else str(v)
            )
    return t


def build_google_sheets_report(raw_df: pd.DataFrame | None,
                               summary_tables: dict[str, pd.DataFrame],
                               share_email: str,
                               report_name: str | None = None) -> str:
    """Create a new Google Sheets spreadsheet with one worksheet per summary
    table (plus optionally the raw filtered trades), share it with the given
    email, and return its URL.

    The file is named `report_name` (falling back to a timestamp) and placed in
    the REPORT_FOLDER_NAME folder in the owner's Drive. Authenticates as the
    OAuth report owner (st.secrets["gcp_oauth"]). See report_export_enabled().
    """
    creds = _oauth_credentials()
    gc = gspread.authorize(creds)
    folder_id = _get_or_create_report_folder(creds)

    title = report_name or f"Trade Log Report {datetime.now():%Y-%m-%d %H.%M}"
    sh = gc.create(title, folder_id=folder_id) if folder_id else gc.create(title)
    if share_email:
        # The owner already has access; sharing a file with its own owner
        # errors, so don't let that abort the whole report.
        try:
            sh.share(share_email, perm_type="user", role="writer", notify=False)
        except gspread.exceptions.APIError:
            pass

    tables: dict[str, pd.DataFrame] = {}
    if raw_df is not None and len(raw_df):
        tables["Raw Filtered Trades"] = raw_df
    tables.update(summary_tables)

    first = True
    for name, table in tables.items():
        if table is None or table.empty:
            continue
        t = _sheets_safe(table)
        n_rows, n_cols = t.shape
        if first:
            ws = sh.sheet1
            ws.update_title(name[:100])
            ws.resize(rows=n_rows + 1, cols=max(n_cols, 1))
            first = False
        else:
            ws = sh.add_worksheet(title=name[:100], rows=n_rows + 1,
                                  cols=max(n_cols, 1))
        set_with_dataframe(ws, t, include_index=False, include_column_header=True)
    return sh.url


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _money_fmt(v):
    return fmt_currency(v) if pd.notna(v) else "—"


def _show_df(df: pd.DataFrame, height: int | None = None):
    """Render a dataframe with currency/percent formatting inferred from
    column names."""
    if df is None or df.empty:
        st.info("No data for this table with the current filters.")
        return
    money_terms = ("P/L", "Margin", "Premium", "Win", "Loss", "Trade P/L",
                   "Best", "Worst", "Day", "Equity", "Drawdown $")
    skip_money = ("Win Rate %", "Winning Day %", "Losing Day %", "Day Result",
                  "Win/Loss", "Trades/Day Bucket", "Day of Week", "Days",
                  "Wins", "Losses", "Return on Margin %", "Drawdown %",
                  "Avg Trades/Day", "Max Trades/Day", "Trades")
    fmt_map = {}
    for c in df.columns:
        if df[c].dtype.kind not in "fi":
            continue
        if c.endswith("%") or "Rate" in c or c in ("Drawdown %",):
            fmt_map[c] = lambda v: fmt_pct(v) if pd.notna(v) else "—"
        elif c in ("Profit Factor", "Avg Hold (min)", "Avg Trades/Day", "MAR"):
            fmt_map[c] = lambda v: fmt_num(v, 2) if pd.notna(v) else "—"
        elif c in ("Trades", "Wins", "Losses", "Days", "Stop Losses",
                   "Expirations", "Max Trades/Day", "Year", "Concurrent Trades"):
            fmt_map[c] = lambda v: f"{int(v):,}" if pd.notna(v) else "—"
        elif any(t in c for t in money_terms) and c not in skip_money:
            fmt_map[c] = _money_fmt
    kwargs = {"use_container_width": True}
    if height is not None:
        kwargs["height"] = height
    try:
        st.dataframe(df.style.format(fmt_map), **kwargs)
    except Exception:
        st.dataframe(df, **kwargs)


def _metric_grid(metrics: list[tuple[str, str]], per_row: int = 4):
    for i in range(0, len(metrics), per_row):
        cols = st.columns(per_row)
        for col, (label, value) in zip(cols, metrics[i:i + per_row]):
            col.metric(label, value)


def _bar_by_sign(x, y, title, x_title, y_title="P/L ($)"):
    """Bar chart where positive bars are blue and negative bars are red."""
    colors = [POS_COLOR if (pd.notna(v) and v >= 0) else NEG_COLOR for v in y]
    fig = go.Figure(go.Bar(
        x=list(x), y=list(y), marker_color=colors,
        hovertemplate="%{x}<br>P/L: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      showlegend=False)
    return fig


def _line_chart(x, y, title, x_title, y_title, fill=False, color=LINE_COLOR):
    fig = go.Figure(go.Scatter(
        x=list(x), y=list(y), mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy" if fill else None,
        hovertemplate="%{x}<br>%{y:$,.0f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      showlegend=False)
    return fig


def _plot(fig):
    st.plotly_chart(fig, use_container_width=True)


_ROLLING_FMTS = {
    "money": _money_fmt,
    "pct": lambda v: fmt_pct(v) if pd.notna(v) else "—",
    "num": lambda v: fmt_num(v, 2) if pd.notna(v) else "—",
    "int": lambda v: f"{int(v):,}" if pd.notna(v) else "—",
}
_ROLLING_HOVERS = {
    "money": "$%{y:,.0f}", "pct": "%{y:.1f}%", "num": "%{y:.2f}", "int": "%{y:,.0f}",
}


def _rolling_history_chart(hist: pd.DataFrame, metric: str, kind: str,
                           windows: list[str]):
    """One line per selected rolling window across the anchor dates."""
    hover = _ROLLING_HOVERS[kind]
    fig = go.Figure()
    for label in ROLLING_WINDOW_LABELS:  # fixed order = fixed colors
        if label not in windows:
            continue
        d = hist[hist["Window"] == label].sort_values("Anchor")
        fig.add_trace(go.Scatter(
            x=list(d["Anchor"]), y=list(d[metric]), mode="lines+markers",
            name=label, connectgaps=False,
            line=dict(color=ROLLING_WINDOW_COLORS[label], width=2),
            marker=dict(size=8),
            hovertemplate="%{x|%b %Y}<br>" + label + ": " + hover + "<extra></extra>",
        ))
    fig.update_layout(title=f"Rolling {metric} Over Time", xaxis_title="As-of date",
                      yaxis_title=metric, legend_title_text="Window")
    return fig


def _show_rolling_table(table: pd.DataFrame, kind: str):
    """Render a windows-x-anchors table with the metric's own formatting."""
    if table.empty:
        st.info("No data for this metric with the current scope.")
        return
    fmt = _ROLLING_FMTS[kind]
    date_cols = [c for c in table.columns if c != "Window"]
    try:
        st.dataframe(table.style.format({c: fmt for c in date_cols}),
                     use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_trade_log_analyzer_tab():
    """Render the full Trade Log Analyzer tab (call from the main app)."""
    st.subheader("Trade Log Analyzer")

    uploaded = st.file_uploader(
        "Upload trade log CSV", type=["csv"], key=KEY + "uploader",
        help="Options trade log export with Date/Time Opened & Closed, P/L and Strategy columns.",
    )
    if uploaded is None:
        st.info("Upload a trade log CSV to begin. Required columns: "
                + ", ".join(REQUIRED_COLUMNS))
        return

    try:
        df, info = prepare_trade_log(uploaded.getvalue())
    except Exception as exc:  # malformed file, encoding, etc.
        st.error(f"Could not read this file as a trade log CSV: {exc}")
        return

    # ---- Upload & Validation -------------------------------------------
    with st.expander("Upload & Validation", expanded=bool(info.get("missing_required"))):
        st.write(f"**Rows in file:** {info.get('raw_rows', len(df)):,}")
        if info.get("missing_required"):
            st.error("Missing required columns: **"
                     + ", ".join(info["missing_required"])
                     + "**. Fix the CSV headers and re-upload.")
            st.dataframe(df.head(20), use_container_width=True)
            return
        if info.get("dropped_rows"):
            st.warning(f"Dropped {info['dropped_rows']:,} row(s) with missing/unparseable "
                       "P/L or open date-time.")
        if info.get("missing_optional"):
            st.info("Optional columns not found (related metrics will be skipped): "
                    + ", ".join(info["missing_optional"]))
        st.caption("First 20 rows as parsed:")
        st.dataframe(df.head(20), use_container_width=True)

    if df.empty:
        st.error("No valid trade rows found after cleaning.")
        return

    # ---- Filters & settings --------------------------------------------
    with st.expander("Filters & Settings", expanded=True):
        c1, c2 = st.columns(2)
        starting_capital = c1.number_input(
            "Starting capital / buying power", min_value=1.0, value=1_100_000.0,
            step=50_000.0, format="%.0f", key=KEY + "capital",
        )
        min_d, max_d = df["Open Date"].min(), df["Open Date"].max()
        date_range = c2.date_input(
            "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
            key=KEY + "dates",
        )

        c5, c6, c7, c8 = st.columns(4)
        strategies = sorted(df["Strategy"].dropna().unique())
        sel_strategies = c5.multiselect("Strategy", strategies, default=strategies,
                                        key=KEY + "strategy")
        volga_opts = sorted(df["Volga Type"].unique())
        sel_volga = c6.multiselect("Volga Type", volga_opts, default=volga_opts,
                                   key=KEY + "volga")
        side_opts = sorted(df["Trade Side"].unique())
        sel_side = c7.multiselect("Trade Side", side_opts, default=side_opts,
                                  key=KEY + "side")
        sel_underlying = None
        if "Underlying" in df.columns:
            und_opts = sorted(df["Underlying"].dropna().unique())
            sel_underlying = c8.multiselect("Underlying", und_opts, default=und_opts,
                                            key=KEY + "underlying")

        c9, c10, c11 = st.columns([2, 1, 1])
        sel_reason = None
        if "Reason For Close" in df.columns:
            reason_opts = sorted(df["Reason For Close"].dropna().unique())
            sel_reason = c9.multiselect("Reason For Close", reason_opts,
                                        default=reason_opts, key=KEY + "reason")
        excl_be = c10.checkbox("Exclude breakeven trades", key=KEY + "excl_be")
        show_raw = c11.checkbox("Show raw data", key=KEY + "show_raw")

    # Apply filters
    fdf = df.copy()
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        fdf = fdf[(fdf["Open Date"] >= date_range[0]) & (fdf["Open Date"] <= date_range[1])]
    fdf = fdf[fdf["Strategy"].isin(sel_strategies)]
    fdf = fdf[fdf["Volga Type"].isin(sel_volga)]
    fdf = fdf[fdf["Trade Side"].isin(sel_side)]
    if sel_underlying is not None:
        fdf = fdf[fdf["Underlying"].isin(sel_underlying) | fdf["Underlying"].isna()]
    if sel_reason is not None:
        fdf = fdf[fdf["Reason For Close"].isin(sel_reason) | fdf["Reason For Close"].isna()]
    if excl_be:
        fdf = fdf[fdf["P/L"] != 0]

    if fdf.empty:
        st.warning("No trades match the current filters. Loosen the filters above.")
        return

    st.caption(f"Analyzing **{len(fdf):,}** trades from "
               f"**{fdf['Open Date'].min()}** to **{fdf['Open Date'].max()}**.")

    if show_raw:
        with st.expander("Raw (filtered) data", expanded=False):
            st.dataframe(fdf, use_container_width=True, height=400)

    # Shared computations
    daily = calculate_daily_summary(fdf)
    equity, max_dd, max_dd_pct = calculate_drawdown(fdf, starting_capital)

    # ---- Rule Tester --------------------------------------------------------
    with st.expander("Rule Tester / Filter Simulator", expanded=True):
        st.caption("Simulate trading rules on top of the filtered data and compare results.")
        r1, r2, r3, r4 = st.columns(4)
        use_min_t = r1.checkbox("Exclude trades before…", key=KEY + "rt_use_min_t")
        min_t = r1.time_input("Earliest entry", dtime(9, 30), key=KEY + "rt_min_t",
                              disabled=not use_min_t)
        use_max_t = r2.checkbox("Exclude trades after…", key=KEY + "rt_use_max_t")
        max_t = r2.time_input("Latest entry", dtime(15, 0), key=KEY + "rt_max_t",
                              disabled=not use_max_t)
        rt_strats = r3.multiselect("Exclude strategies", sorted(fdf["Strategy"].dropna().unique()),
                                   key=KEY + "rt_strats")
        rt_volga = r4.multiselect("Exclude Volga Type", sorted(fdf["Volga Type"].unique()),
                                  key=KEY + "rt_volga")

        r5, r6, r7, r8 = st.columns(4)
        rt_sides = r5.multiselect("Exclude Trade Side", sorted(fdf["Trade Side"].unique()),
                                  key=KEY + "rt_sides")
        rt_min_prem = r6.number_input("Min premium (0 = off)", min_value=0.0, value=0.0,
                                      key=KEY + "rt_min_prem")
        rt_max_margin = r7.number_input("Max margin (0 = off)", min_value=0.0, value=0.0,
                                        key=KEY + "rt_max_margin")
        rt_max_tpd = r8.number_input("Max trades/day (0 = off)", min_value=0, value=0,
                                     key=KEY + "rt_max_tpd")

        r9, r10, r11 = st.columns(3)
        rt_stop_after = r9.checkbox("Stop day after first stop loss", key=KEY + "rt_stop_after")
        rt_loss_cut = r10.number_input("Daily loss cutoff $ (0 = off)", min_value=0.0,
                                       value=0.0, key=KEY + "rt_loss_cut")
        rt_profit_cut = r11.number_input("Daily profit target $ (0 = off)", min_value=0.0,
                                         value=0.0, key=KEY + "rt_profit_cut")

        rules = {
            "min_time": min_t if use_min_t else None,
            "max_time": max_t if use_max_t else None,
            "exclude_strategies": rt_strats,
            "exclude_volga": rt_volga,
            "exclude_sides": rt_sides,
            "min_premium": rt_min_prem if rt_min_prem > 0 else None,
            "max_margin": rt_max_margin if rt_max_margin > 0 else None,
            "max_trades_per_day": rt_max_tpd if rt_max_tpd > 0 else None,
            "stop_after_first_stop": rt_stop_after,
            "daily_loss_cutoff": rt_loss_cut if rt_loss_cut > 0 else None,
            "daily_profit_cutoff": rt_profit_cut if rt_profit_cut > 0 else None,
        }
        sim = apply_rule_filters(fdf, rules)
        st.caption(f"Rules keep **{len(sim):,}** of **{len(fdf):,}** trades.")
        comp = pd.DataFrame({
            "Original": comparison_stats(fdf, starting_capital),
            "Filtered": comparison_stats(sim, starting_capital),
        })
        comp.index.name = "Metric"
        st.dataframe(comp, use_container_width=True)

    # ---- Executive Summary ---------------------------------------------
    with st.expander("Executive Summary", expanded=True):
        _metric_grid(calculate_summary_metrics(fdf, starting_capital))

        _plot(_line_chart(equity["DateTime"], equity["Cumulative P/L"],
                          "Cumulative P/L (Equity Curve)", "Date", "Cumulative P/L ($)"))
        _plot(_line_chart(equity["DateTime"], equity["Drawdown $"],
                          "Drawdown", "Date", "Drawdown ($)", fill=True, color=NEG_COLOR))

        monthly = calculate_monthly_summary(fdf)
        _plot(_bar_by_sign(monthly["Month"], monthly["Total P/L"],
                           "Monthly P/L", "Month"))

        fig = go.Figure(go.Histogram(
            x=daily["Daily P/L"], nbinsx=60, marker_color=LINE_COLOR,
            hovertemplate="Daily P/L: %{x}<br>Days: %{y}<extra></extra>",
        ))
        fig.update_layout(title="Daily P/L Distribution", xaxis_title="Daily P/L ($)",
                          yaxis_title="Number of Days", showlegend=False)
        _plot(fig)

    # ---- Rolling Stats Over Time -----------------------------------------
    roll_end = (fdf["Close DateTime"].max() if fdf["Close DateTime"].notna().any()
                else fdf["Open DateTime"].max())
    roll_data_start = fdf["Open DateTime"].min()
    roll_anchors = rolling_anchor_dates(roll_data_start, roll_end)
    roll_all = calculate_rolling_stats(fdf, starting_capital, roll_end, roll_data_start)
    hist_all = calculate_rolling_history(fdf, starting_capital, roll_anchors,
                                         roll_data_start, roll_end)

    st.markdown("#### Rolling Stats Over Time")
    st.caption("Each column is an as-of date (newest first); each row is the "
               "trailing 1-12 month window ending on that date, so reading "
               "across a row shows whether a stat is improving or declining. "
               "Drawdown, CAGR and MAR use the starting capital set above.")
    rc1, rc2 = st.columns(2)
    roll_scope = rc1.selectbox(
        "Scope", ["All Strategies (combined)"] + sorted(fdf["Strategy"].dropna().unique()),
        key=KEY + "roll_scope",
    )
    roll_chart_windows = rc2.multiselect(
        "Windows to chart", ROLLING_WINDOW_LABELS,
        default=["1 Mo", "3 Mo", "6 Mo", "12 Mo"], key=KEY + "roll_windows",
    )
    if roll_scope == "All Strategies (combined)":
        hist = hist_all
    else:
        hist = calculate_rolling_history(
            fdf[fdf["Strategy"] == roll_scope], starting_capital, roll_anchors,
            roll_data_start, roll_end,
        )

    for roll_metric, roll_kind in ROLLING_HISTORY_METRICS:
        with st.expander(f"Rolling {roll_metric}"):
            if roll_metric not in hist.columns:
                st.info("No trades in scope for this metric.")
                continue
            if roll_chart_windows:
                _plot(_rolling_history_chart(hist, roll_metric, roll_kind,
                                             roll_chart_windows))
            _show_rolling_table(rolling_history_table(hist, roll_metric), roll_kind)

    # ---- Daily Analysis --------------------------------------------------
    with st.expander("Daily Analysis"):
        win_days = (daily["Day Result"] == "Win").sum()
        loss_days = (daily["Day Result"] == "Loss").sum()
        win_day_pl = daily.loc[daily["Day Result"] == "Win", "Daily P/L"]
        loss_day_pl = daily.loc[daily["Day Result"] == "Loss", "Daily P/L"]
        _metric_grid([
            ("Trading Days", f"{len(daily):,}"),
            ("Winning Day %", fmt_pct(_safe_div(100 * win_days, len(daily)))),
            ("Losing Day %", fmt_pct(_safe_div(100 * loss_days, len(daily)))),
            ("Avg Winning Day", fmt_currency(win_day_pl.mean() if len(win_day_pl) else np.nan)),
            ("Avg Losing Day", fmt_currency(loss_day_pl.mean() if len(loss_day_pl) else np.nan)),
            ("Best Day", fmt_currency(daily["Daily P/L"].max())),
            ("Worst Day", fmt_currency(daily["Daily P/L"].min())),
            ("Max Consecutive Win Days", str(_max_consecutive(daily["Day Result"], "Win"))),
            ("Max Consecutive Loss Days", str(_max_consecutive(daily["Day Result"], "Loss"))),
        ])
        _plot(_bar_by_sign(daily["Open Date"], daily["Daily P/L"],
                           "Daily P/L", "Date"))
        fig = go.Figure(go.Bar(
            x=list(daily["Open Date"]), y=list(daily["Trades"]),
            marker_color=LINE_COLOR,
            hovertemplate="%{x}<br>Trades: %{y}<extra></extra>",
        ))
        fig.update_layout(title="Trade Count by Day", xaxis_title="Date",
                          yaxis_title="Trades", showlegend=False)
        _plot(fig)
        _show_df(daily, height=350)

    # ---- Trade Count Buckets ---------------------------------------------
    tc_buckets = calculate_trade_count_bucket_summary(daily)
    with st.expander("Trades-per-Day Buckets"):
        _show_df(tc_buckets)
        if len(tc_buckets):
            _plot(_bar_by_sign(tc_buckets["Trades/Day Bucket"], tc_buckets["Avg Daily P/L"],
                               "Average Daily P/L by Trades-per-Day Bucket",
                               "Trades per Day", "Avg Daily P/L ($)"))

    # ---- Strategy Breakdown ----------------------------------------------
    strat = calculate_strategy_summary(fdf)
    with st.expander("Strategy Breakdown"):
        _show_df(strat)
        if len(strat):
            _plot(_bar_by_sign(strat["Strategy"], strat["Total P/L"],
                               "Total P/L by Strategy", "Strategy"))

    # ---- Volga vs Non-Volga ----------------------------------------------
    volga = calculate_volga_summary(fdf)
    with st.expander("Volga vs Non-Volga"):
        _show_df(volga)
        if len(volga):
            _plot(_bar_by_sign(volga["Volga Type"], volga["Total P/L"],
                               "Total P/L by Volga Type", "Volga Type"))

    # ---- Calls vs Puts -----------------------------------------------------
    side = calculate_side_summary(fdf)
    with st.expander("Calls vs Puts"):
        _show_df(side)
        if len(side):
            _plot(_bar_by_sign(side["Trade Side"], side["Total P/L"],
                               "Total P/L by Trade Side", "Trade Side"))

    # ---- Time Analysis -----------------------------------------------------
    tb = calculate_time_bucket_summary(fdf)
    with st.expander("Time-of-Day & Day-of-Week Analysis"):
        _show_df(tb)
        if len(tb):
            _plot(_bar_by_sign(tb["Time Bucket"], tb["Total P/L"],
                               "Total P/L by Entry Time Bucket", "Entry Time"))
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        dow = fdf.groupby("Day of Week")["P/L"].sum()
        dow = dow.reindex([d for d in dow_order if d in dow.index])
        _plot(_bar_by_sign(dow.index, dow.values, "Total P/L by Day of Week",
                           "Day of Week"))

    # ---- Reason For Close --------------------------------------------------
    reason = calculate_reason_summary(fdf)
    if len(reason):
        with st.expander("Reason For Close Analysis"):
            _show_df(reason)
            reasons = fdf["Reason For Close"].astype(str)
            stop_losses = fdf[reasons.str.contains("stop", case=False, na=False) & (fdf["P/L"] < 0)]
            total_wins = fdf.loc[fdf["P/L"] > 0, "P/L"].sum()
            damage = _safe_div(abs(stop_losses["P/L"].sum()), total_wins)
            st.metric("Stop-Loss Damage Ratio",
                      fmt_pct(100 * damage) if pd.notna(damage) else "N/A",
                      help="Total losses from stop-loss trades ÷ total gains from winning trades.")
            _plot(_bar_by_sign(reason["Reason For Close"], reason["Total P/L"],
                               "Total P/L by Reason For Close", "Reason"))

    # ---- Risk & Margin -----------------------------------------------------
    with st.expander("Risk & Margin"):
        if "Margin Req." in fdf.columns and fdf["Margin Req."].notna().any():
            _metric_grid(calculate_risk_margin_metrics(fdf, daily))
        else:
            st.info("Margin Req. column not present — margin analysis skipped.")

        events, exp_stats = calculate_concurrent_exposure(fdf)
        if exp_stats:
            st.markdown("**Concurrent Exposure Estimate**")
            _metric_grid([
                ("Max Concurrent Open Trades", f"{exp_stats['Max Concurrent Trades']:,}"),
                ("Max Concurrent Margin", fmt_currency(exp_stats["Max Concurrent Margin"], 0)),
                ("Avg Concurrent Margin", fmt_currency(exp_stats["Avg Concurrent Margin"], 0)),
                ("Time of Max Exposure", str(exp_stats["Time of Max Exposure"])),
            ])
            _plot(_line_chart(events["DateTime"], events["Concurrent Margin"],
                              "Concurrent Margin Over Time", "Date", "Concurrent Margin ($)"))

    # ---- Monthly & Yearly ----------------------------------------------------
    monthly = calculate_monthly_summary(fdf)
    yearly = calculate_yearly_summary(fdf)
    with st.expander("Monthly & Yearly Reports"):
        st.markdown("**Monthly**")
        _show_df(monthly)
        st.markdown("**Yearly**")
        _show_df(yearly)

    # ---- Outliers ---------------------------------------------------------
    with st.expander("Outlier Analysis"):
        show_cols = [c for c in ("Open Date", "Time Opened", "Strategy", "Trade Side",
                                 "Volga Type", "Reason For Close", "Premium",
                                 "Margin Req.", "P/L") if c in fdf.columns]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 10 Winning Trades**")
            _show_df(fdf.nlargest(10, "P/L")[show_cols].reset_index(drop=True))
            st.markdown("**Top 10 Winning Days**")
            _show_df(daily.nlargest(10, "Daily P/L")[["Open Date", "Trades", "Daily P/L"]].reset_index(drop=True))
        with c2:
            st.markdown("**Top 10 Losing Trades**")
            _show_df(fdf.nsmallest(10, "P/L")[show_cols].reset_index(drop=True))
            st.markdown("**Top 10 Losing Days**")
            _show_df(daily.nsmallest(10, "Daily P/L")[["Open Date", "Trades", "Daily P/L"]].reset_index(drop=True))

        total = fdf["P/L"].sum()
        top5 = fdf.nlargest(5, "P/L")["P/L"].sum()
        worst5 = fdf.nsmallest(5, "P/L")["P/L"].sum()
        top5d = daily.nlargest(5, "Daily P/L")["Daily P/L"].sum()
        worst5d = daily.nsmallest(5, "Daily P/L")["Daily P/L"].sum()
        _metric_grid([
            ("P/L excl. Top 5 Trades", fmt_currency(total - top5)),
            ("P/L excl. Worst 5 Trades", fmt_currency(total - worst5)),
            ("P/L excl. Top 5 Days", fmt_currency(total - top5d)),
            ("P/L excl. Worst 5 Days", fmt_currency(total - worst5d)),
        ])

    # ---- Export -------------------------------------------------------------
    with st.expander("Export"):
        summary_metrics_df = pd.DataFrame(
            calculate_summary_metrics(fdf, starting_capital), columns=["Metric", "Value"]
        )
        outliers = pd.concat([
            fdf.nlargest(10, "P/L")[show_cols].assign(Outlier="Top 10 Win"),
            fdf.nsmallest(10, "P/L")[show_cols].assign(Outlier="Top 10 Loss"),
        ])
        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "Filtered trades (CSV)", fdf.to_csv(index=False).encode("utf-8"),
            "filtered_trades.csv", "text/csv", key=KEY + "dl_trades",
        )
        c2.download_button(
            "Daily summary (CSV)", daily.to_csv(index=False).encode("utf-8"),
            "daily_summary.csv", "text/csv", key=KEY + "dl_daily",
        )
        c3.download_button(
            "Strategy summary (CSV)", strat.to_csv(index=False).encode("utf-8"),
            "strategy_summary.csv", "text/csv", key=KEY + "dl_strat",
        )

        st.markdown("**Google Sheets report**")
        if not report_export_enabled():
            st.caption("Google Sheets export is disabled on this deployment. "
                       "Use the CSV downloads above.")
        else:
            g1, g2 = st.columns([2, 1])
            share_email = g1.text_input(
                "Share report with (email)", value="chad.mccandless@gmail.com",
                key=KEY + "gs_email",
            )
            include_raw = g2.checkbox("Include raw trades sheet", value=True,
                                      key=KEY + "gs_raw")
            if st.button("Create Google Sheets report", key=KEY + "gs_build"):
                try:
                    with st.spinner("Creating Google Sheets report (this can take a minute)..."):
                        url = build_google_sheets_report(
                            fdf if include_raw else None,
                            {
                                "Summary Metrics": summary_metrics_df,
                                "Rolling Stats (All Strategies)": roll_all,
                                "Rolling History (All Strategies)": hist_all,
                                "Daily Summary": daily,
                                "Strategy Summary": strat,
                                "Volga Summary": volga,
                                "Side Summary": side,
                                "Time Bucket Summary": tb,
                                "Trade Count Buckets": tc_buckets,
                                "Monthly Summary": monthly,
                                "Yearly Summary": yearly,
                                "Outliers": outliers,
                            },
                            share_email.strip(),
                            report_name=uploaded.name.rsplit(".", 1)[0].strip() or None,
                        )
                    st.session_state[KEY + "gs_url"] = url
                except Exception as exc:
                    st.error(f"Could not create the Google Sheets report: {exc}")
            if st.session_state.get(KEY + "gs_url"):
                st.success(f"Report created: {st.session_state[KEY + 'gs_url']}")
