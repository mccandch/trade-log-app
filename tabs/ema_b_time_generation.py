"""
EMA-B Time Generation (Phase 1)

Goal: replicate EMA-B with indicator values.py logic EXACTLY (no compression in UI).
- Upload 1-minute SPX CSV
- Configure parameters (defaults match script)
- Generate 2 CSV outputs (bullish puts / bearish calls) INCLUDING indicator columns
"""

from dataclasses import dataclass
import io
from typing import Tuple
from datetime import date, timedelta

import pandas as pd
import streamlit as st


# ----------------------------
# Params (defaults match script)
# ----------------------------

@dataclass
class Params:
    # EMA periods
    low_ema_period: int = 20
    high_ema_period: int = 40

    # Timing
    close_offset_minutes: int = 30
    check_interval: int = 5
    start_time: str = "9:30"   # allow "9:30" or "09:30"
    end_time: str = "15:55"

    # Days (Mon-Fri)
    include_days = ("Mon", "Tue", "Wed", "Thu", "Fri")

    # Date filter defaults (inclusive)
    date_start: str = "2022-05-16"
    date_end: str = "2026-01-24"

    # Feature settings
    atr_len: int = 14
    slope_lookback_min: int = 5  # bars

    # Compression exists in script but Phase 1 ignores it
    apply_compression_filters: bool = False
    call_compression_cutoff: float = -0.28
    put_compression_cutoff: float = 0.25

    # VWAP trend filters — BOTH SIDES (defaults match script)
    apply_put_vwap_filter: bool = True
    put_vwap_dist_atr_cutoff: float = 2.4674  # 60% retention example

    apply_call_vwap_filter: bool = True
    call_vwap_dist_atr_cutoff: float = -1.8409  # 60% retention example


# ----------------------------
# Helpers (match script)
# ----------------------------

def hhmm_to_minutes(hhmm: str) -> int:
    """Convert 'H:MM' or 'HH:MM' to minutes since midnight."""
    h_str, m_str = hhmm.strip().split(":")
    return int(h_str) * 60 + int(m_str)


def compute_atr(df_in: pd.DataFrame, length: int) -> pd.Series:
    """
    Wilder-style ATR (matches script):
    - If High/Low exist: True Range
    - Else: close-to-close abs move
    - Wilder smoothing: EMA with alpha = 1/length
    """
    df = df_in
    has_hl = ("High" in df.columns) and ("Low" in df.columns)

    close = df["close"]
    prev_close = close.shift(1)

    if has_hl:
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
    else:
        tr = (close - prev_close).abs()

    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _load_and_normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Match script expectations:
    - Time -> date
    - Last -> close
    """
    df = df_raw.copy()

    if "Time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Time": "date"})
    if "Last" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"Last": "close"})
    if "Close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"Close": "close"})

    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'Time' column (or 'date').")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Ensure numerics exist (needed for VWAP TP calc)
    for col in ["Open", "High", "Low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def _apply_filters(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    # Date filter defaults (inclusive): [DATE_START, DATE_END + 1 day)
    start_dt = pd.to_datetime(p.date_start)
    end_dt_exclusive = pd.to_datetime(p.date_end) + pd.Timedelta(days=1)
    df = df[(df["date"] >= start_dt) & (df["date"] < end_dt_exclusive)].copy()

    # DOW filter
    df["day_of_week"] = df["date"].dt.strftime("%a")
    df = df[df["day_of_week"].isin(p.include_days)].copy()

    # Time filters
    df["minutes_since_midnight"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    start_minutes = hhmm_to_minutes(p.start_time)
    end_minutes = hhmm_to_minutes(p.end_time)

    df = df[
        (df["minutes_since_midnight"] >= start_minutes)
        & (df["minutes_since_midnight"] <= end_minutes)
    ].copy()

    # Check interval mask (IMPORTANT: offset by start_minutes, matches script)
    df["is_check_time"] = ((df["minutes_since_midnight"] - start_minutes) % p.check_interval) == 0

    return df


def _compute_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    # EMA
    df["LOW_EMA"] = df["close"].ewm(span=p.low_ema_period, adjust=False).mean()
    df["HIGH_EMA"] = df["close"].ewm(span=p.high_ema_period, adjust=False).mean()

    # ATR_14
    df["ATR_14"] = compute_atr(df, p.atr_len)

    # Diff
    df["EMA_DIFF"] = df["LOW_EMA"] - df["HIGH_EMA"]
    df["EMA_DIFF_ATR"] = df["EMA_DIFF"] / df["ATR_14"].replace(0, pd.NA)

    # Slopes (MATCH SCRIPT: raw difference over lb bars, not divided by lb)
    lb = p.slope_lookback_min
    df["LOW_EMA_SLOPE_5"] = df["LOW_EMA"] - df["LOW_EMA"].shift(lb)
    df["LOW_EMA_SLOPE_5_ATR"] = df["LOW_EMA_SLOPE_5"] / df["ATR_14"].replace(0, pd.NA)

    df["EMA_DIFF_SLOPE_5"] = df["EMA_DIFF"] - df["EMA_DIFF"].shift(lb)
    df["EMA_DIFF_SLOPE_5_ATR"] = df["EMA_DIFF_SLOPE_5"] / df["ATR_14"].replace(0, pd.NA)

    # VWAP (price-only, session anchored) and signed VWAP_DIST_ATR (MATCH SCRIPT)
    if ("High" in df.columns) and ("Low" in df.columns):
        df["TP"] = (df["High"] + df["Low"] + df["close"]) / 3.0
    else:
        df["TP"] = df["close"]

    df["session"] = df["date"].dt.date
    df["VWAP"] = (
        df.groupby("session")["TP"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["VWAP_DIST_ATR"] = (df["close"] - df["VWAP"]) / df["ATR_14"].replace(0, pd.NA)

    return df


def build_output(df: pd.DataFrame, mask: pd.Series, p: Params) -> pd.DataFrame:
    rows = df.loc[mask, [
        "date",
        "close",
        "LOW_EMA", "HIGH_EMA",
        "EMA_DIFF", "ATR_14",
        "EMA_DIFF_ATR",
        "LOW_EMA_SLOPE_5_ATR",
        "EMA_DIFF_SLOPE_5_ATR",
        "VWAP",
        "VWAP_DIST_ATR",
    ]].copy()

    # Trade timestamps (MATCH SCRIPT):
    # enter next minute; close fixed offset from the check bar timestamp
    rows["OPEN_DATETIME"] = (rows["date"] + pd.Timedelta(minutes=1)).dt.strftime("%Y-%m-%d %H:%M")
    rows["CLOSE_DATETIME"] = (rows["date"] + pd.Timedelta(minutes=p.close_offset_minutes)).dt.strftime("%Y-%m-%d %H:%M")

    rows = rows.rename(columns={"close": "CLOSE"})

    return rows[[
        "OPEN_DATETIME", "CLOSE_DATETIME",
        "CLOSE",
        "LOW_EMA", "HIGH_EMA",
        "EMA_DIFF", "ATR_14",
        "EMA_DIFF_ATR",
        "LOW_EMA_SLOPE_5_ATR",
        "EMA_DIFF_SLOPE_5_ATR",
        "VWAP",
        "VWAP_DIST_ATR",
    ]].reset_index(drop=True)


def generate_entries(df_raw: pd.DataFrame, p: Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _load_and_normalize(df_raw)
    df = _apply_filters(df, p)
    df = _compute_indicators(df, p)

    bullish_mask = (df["LOW_EMA"] > df["HIGH_EMA"]) & df["is_check_time"]
    bearish_mask = (df["LOW_EMA"] < df["HIGH_EMA"]) & df["is_check_time"]

    # Compression filters exist but are off in Phase 1 (kept for parity)
    if p.apply_compression_filters:
        bullish_mask = bullish_mask & (df["EMA_DIFF_ATR"] >= p.put_compression_cutoff)
        bearish_mask = bearish_mask & (df["EMA_DIFF_ATR"] <= p.call_compression_cutoff)

    # VWAP filters (MATCH SCRIPT: PUT >= cutoff, CALL <= cutoff)
    if p.apply_put_vwap_filter:
        bullish_mask = bullish_mask & (df["VWAP_DIST_ATR"] >= p.put_vwap_dist_atr_cutoff)
    if p.apply_call_vwap_filter:
        bearish_mask = bearish_mask & (df["VWAP_DIST_ATR"] <= p.call_vwap_dist_atr_cutoff)

    bullish_df = build_output(df, bullish_mask, p)
    bearish_df = build_output(df, bearish_mask, p)
    return bullish_df, bearish_df



# ----------------------------
# Phase 2 — Backtest bucket summary (mirrors EMA-B step2.py)
# ----------------------------

# Trading session buckets: 09:30 to 15:45, 15-minute floor (fixed; market hours)
_PHASE2_TIME_INDEX = pd.date_range("1900-01-01 09:30", "1900-01-01 15:45", freq="15min").strftime("%H:%M")
_PHASE2_WEEKDAY_COLS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _phase2_detect_schema(df: pd.DataFrame) -> dict:
    cols = set(df.columns)

    if {"Date Opened", "Time Opened", "P/L"}.issubset(cols):
        return {"date_col": "Date Opened", "time_col": "Time Opened", "pl_col": "P/L"}

    if {"OpenDate", "OpenTime", "ProfitLoss"}.issubset(cols):
        return {"date_col": "OpenDate", "time_col": "OpenTime", "pl_col": "ProfitLoss"}

    raise ValueError(
        "Unrecognized CSV format.\n"
        "Expected either:\n"
        "  - Date Opened, Time Opened, P/L\n"
        "  - OpenDate, OpenTime, ProfitLoss\n"
        f"Found columns: {list(df.columns)}"
    )


def _phase2_weekday_abbrev(series: pd.Series) -> pd.Series:
    mapping = {
        "Monday": "Mon",
        "Tuesday": "Tue",
        "Wednesday": "Wed",
        "Thursday": "Thu",
        "Friday": "Fri",
    }
    return series.map(mapping)


def _phase2_prepare_trades_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    schema = _phase2_detect_schema(df)
    date_col, time_col, pl_col = schema["date_col"], schema["time_col"], schema["pl_col"]

    df[pl_col] = pd.to_numeric(df[pl_col], errors="coerce")
    df = df.dropna(subset=[pl_col]).copy()

    df["OPEN_DT"] = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str),
        errors="coerce",
    )
    df = df.dropna(subset=["OPEN_DT"]).copy()

    df["WEEKDAY"] = _phase2_weekday_abbrev(df["OPEN_DT"].dt.day_name())
    df["TIME_BUCKET"] = df["OPEN_DT"].dt.floor("15min").dt.strftime("%H:%M")
    df["WIN"] = df[pl_col] > 0

    # Keep only Mon–Fri (in case file contains weekend timestamps)
    df = df[df["WEEKDAY"].isin(_PHASE2_WEEKDAY_COLS)].copy()

    return df


def _phase2_window_start_from_latest(df: pd.DataFrame, lookback_days: int) -> pd.Timestamp:
    latest = df["OPEN_DT"].max()
    return latest - pd.Timedelta(days=lookback_days)


def _phase2_build_pivots_for_window(df: pd.DataFrame, start_dt: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    wdf = df[df["OPEN_DT"] >= start_dt].copy()

    grp = (
        wdf.groupby(["TIME_BUCKET", "WEEKDAY"])
           .agg(TRADES=("WIN", "size"), WINS=("WIN", "sum"))
           .reset_index()
    )
    grp["WIN_PCT"] = grp["WINS"] / grp["TRADES"]

    full_index = pd.MultiIndex.from_product([_PHASE2_TIME_INDEX, _PHASE2_WEEKDAY_COLS], names=["TIME_BUCKET", "WEEKDAY"])
    grp_idx = grp.set_index(["TIME_BUCKET", "WEEKDAY"]).reindex(full_index)

    win_pivot = grp_idx["WIN_PCT"].unstack("WEEKDAY").reindex(columns=_PHASE2_WEEKDAY_COLS)
    trades_pivot = grp_idx["TRADES"].unstack("WEEKDAY").reindex(columns=_PHASE2_WEEKDAY_COLS)

    return win_pivot, trades_pivot, len(wdf)


def _phase2_write_blank_rows(f, n: int):
    if n > 0:
        f.write("\n" * n)


def _phase2_tabs(n: int) -> str:
    return "\t" * max(0, n)


def _phase2_write_window_section_side_by_side(
    f,
    days: int,
    cutoff: pd.Timestamp,
    rows_used: int,
    win_pivot: pd.DataFrame,
    trades_pivot: pd.DataFrame | None,
    blank_rows_after: int,
):
    """
    Writes ONE window block.
    If trades_pivot is provided, WIN_PCT table is on the left and TRADES table is on the right (aligned rows).
    """
    # Layout sizes
    # Left table columns: TIME_BUCKET + 5 weekdays = 6
    left_cols = 1 + len(_PHASE2_WEEKDAY_COLS)
    gap_cols = 1  # blank separator column
    right_start_col = left_cols + gap_cols + 1  # 1-based column number where right title should appear

    # Title row(s)
    if trades_pivot is None:
        f.write(f"WIN_PCT — Last {days} days\n")
        f.write(f"Cutoff:\t{cutoff}\n")
        f.write(f"Rows used:\t{rows_used}\n")
    else:
        f.write(f"WIN_PCT — Last {days} days{_phase2_tabs(right_start_col - 1)}TRADES — Last {days} days\n")
        f.write(f"Cutoff:\t{cutoff}{_phase2_tabs(right_start_col - 2)}Cutoff:\t{cutoff}\n")
        f.write(f"Rows used:\t{rows_used}{_phase2_tabs(right_start_col - 2)}Rows used:\t{rows_used}\n")

    # Data tables
    win_tbl = win_pivot.copy()
    win_tbl.insert(0, "TIME_BUCKET", win_tbl.index)

    if trades_pivot is None:
        win_tbl.to_csv(f, sep="\t", index=False)
        _phase2_write_blank_rows(f, blank_rows_after)
        return

    trades_tbl = trades_pivot.copy()

    # Prefix headers so sheets doesn't collide names when pasted
    trades_tbl = trades_tbl.rename(columns={c: f"T_{c}" for c in trades_tbl.columns})
    win_tbl = win_tbl.rename(columns={c: f"W_{c}" for c in win_tbl.columns if c != "TIME_BUCKET"})

    combined = pd.concat(
        [
            win_tbl.reset_index(drop=True),
            pd.DataFrame({"": [""] * len(win_tbl)}),  # blank separator column
            trades_tbl.reset_index(drop=True),
        ],
        axis=1,
    )
    combined.to_csv(f, sep="\t", index=False)
    _phase2_write_blank_rows(f, blank_rows_after)


def _phase2_build_paste_ready_tsv(
    df: pd.DataFrame,
    lookback_windows_days: list[int],
    output_trades_pivot: bool,
    blank_rows_between_windows: int,
) -> Tuple[bytes, pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Returns:
      - TSV bytes (paste-ready)
      - First WIN_PCT pivot (for preview) for the smallest window in lookback_windows_days
      - Dict of WIN_PCT pivots for each window_days
    """
    if df.empty:
        raise ValueError("No valid rows after parsing. Check the input file format/content.")

    latest_trade = df["OPEN_DT"].max()
    earliest_trade = df["OPEN_DT"].min()

    # Deterministic ordering (script iterates LOOKBACK_WINDOWS_DAYS as provided)
    windows = list(lookback_windows_days)

    buf = io.StringIO()
    buf.write("Latest trade:\t" + str(latest_trade) + "\n")
    buf.write("Earliest trade:\t" + str(earliest_trade) + "\n\n")

    preview_win_pivot = None
    win_pivots_by_days: dict[int, pd.DataFrame] = {}

    for days in windows:
        start_dt = _phase2_window_start_from_latest(df, days)
        win_pivot, trades_pivot, rows_used = _phase2_build_pivots_for_window(df, start_dt)
        cutoff = start_dt

        if preview_win_pivot is None:
            preview_win_pivot = win_pivot.copy()

        win_pivots_by_days[days] = win_pivot.copy()

        _phase2_write_window_section_side_by_side(
            f=buf,
            days=days,
            cutoff=cutoff,
            rows_used=rows_used,
            win_pivot=win_pivot,
            trades_pivot=(trades_pivot if output_trades_pivot else None),
            blank_rows_after=blank_rows_between_windows,
        )

    if preview_win_pivot is None:
        preview_win_pivot = pd.DataFrame(index=_PHASE2_TIME_INDEX, columns=_PHASE2_WEEKDAY_COLS)

    return buf.getvalue().encode("utf-8"), preview_win_pivot, win_pivots_by_days


# ----------------------------
# Streamlit UI (Tab)
# ----------------------------

def ema_b_time_generation_tab():
    st.subheader("EMA-B Time Generation")

    with st.expander("Phase 1 — Generate entry-time CSVs", expanded=True):
        st.caption(
            "Replicates EMA-B with indicator values.py output (including indicators). "
            "Compression ignored in Phase 1 UI."
        )

        if "ema_b_phase1_bullish" not in st.session_state:
            st.session_state.ema_b_phase1_bullish = None
        if "ema_b_phase1_bearish" not in st.session_state:
            st.session_state.ema_b_phase1_bearish = None

        st.markdown("### 1) Upload 1-minute SPX data")
        up = st.file_uploader("SPX 1-minute CSV", type=["csv"], accept_multiple_files=False)

        st.markdown("### 2) Parameters (defaults match the script)")
        colA, colB, colC = st.columns(3)

        with colA:
            low_ema_period = st.number_input("LOW_EMA_PERIOD", min_value=2, max_value=500, value=20, step=1)
            high_ema_period = st.number_input("HIGH_EMA_PERIOD", min_value=2, max_value=500, value=40, step=1)
            atr_len = st.number_input("ATR_LEN", min_value=2, max_value=200, value=14, step=1)

        with colB:
            close_offset_minutes = st.number_input("CLOSE_OFFSET_MINUTES", min_value=1, max_value=240, value=30, step=1)
            check_interval = st.number_input("CHECK_INTERVAL (minutes)", min_value=1, max_value=60, value=5, step=1)
            slope_lookback_min = st.number_input("SLOPE_LOOKBACK_MIN", min_value=1, max_value=240, value=5, step=1)

        with colC:
            start_time = st.text_input("START_TIME (H:MM or HH:MM)", value="9:30")
            end_time = st.text_input("END_TIME (H:MM or HH:MM)", value="15:55")
            st.write("")

        yesterday = date.today() - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        use_date_filter = st.checkbox("Enable date range filter", value=True)

        if use_date_filter:
            c1, c2 = st.columns(2)
            with c1:
                date_start = st.text_input("DATE_START (YYYY-MM-DD)", value="2022-05-16")

            # session_state init so it defaults correctly even after reruns
            if "ema_b_date_end_str" not in st.session_state:
                st.session_state["ema_b_date_end_str"] = yesterday_str

            with c2:
                date_end = st.text_input("DATE_END (YYYY-MM-DD)", key="ema_b_date_end_str")
        else:
            date_start = "2022-05-16"
            date_end = yesterday_str

        st.markdown("### 3) VWAP filters (defaults match the script)")
        v1, v2 = st.columns(2)
        with v1:
            apply_put_vwap_filter = st.checkbox("APPLY_PUT_VWAP_FILTER",value=True,key="apply_put_vwap_filter_v1") 
            put_vwap_dist_atr_cutoff = st.number_input("PUT_VWAP_DIST_ATR_CUTOFF", value=2.4674, step=0.0001, format="%.4f")
        with v2:
            apply_call_vwap_filter = st.checkbox("APPLY_CALL_VWAP_FILTER",value=True,key="apply_call_vwap_filter_v1")
            call_vwap_dist_atr_cutoff = st.number_input("CALL_VWAP_DIST_ATR_CUTOFF", value=-1.8409, step=0.0001, format="%.4f")

        st.markdown("### 4) Run")
        run = st.button("Generate files", type="primary", disabled=(up is None), key="ema_b_phase1_generate")

        if run and up is not None:
            try:
                df_raw = pd.read_csv(up)

                p = Params(
                    low_ema_period=int(low_ema_period),
                    high_ema_period=int(high_ema_period),
                    close_offset_minutes=int(close_offset_minutes),
                    check_interval=int(check_interval),
                    start_time=str(start_time).strip(),
                    end_time=str(end_time).strip(),
                    date_start=str(date_start).strip(),
                    date_end=str(date_end).strip(),
                    atr_len=int(atr_len),
                    slope_lookback_min=int(slope_lookback_min),
                    apply_compression_filters=False,  # Phase 1 requirement
                    apply_put_vwap_filter=bool(apply_put_vwap_filter),
                    put_vwap_dist_atr_cutoff=float(put_vwap_dist_atr_cutoff),
                    apply_call_vwap_filter=bool(apply_call_vwap_filter),
                    call_vwap_dist_atr_cutoff=float(call_vwap_dist_atr_cutoff),
                )

                bullish_df, bearish_df = generate_entries(df_raw, p)
                st.session_state.ema_b_phase1_bullish = bullish_df
                st.session_state.ema_b_phase1_bearish = bearish_df

                st.success(f"Done. Bullish rows: {len(bullish_df):,} | Bearish rows: {len(bearish_df):,}")

            except Exception as e:
                st.error(f"Failed: {e}")

        bullish_df = st.session_state.ema_b_phase1_bullish
        bearish_df = st.session_state.ema_b_phase1_bearish

        if bullish_df is not None and bearish_df is not None:
            st.markdown("### 5) Preview")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.write("Preview — Bullish (PUTs)")
                st.dataframe(bullish_df.head(20), use_container_width=True)
            with pcol2:
                st.write("Preview — Bearish (CALLs)")
                st.dataframe(bearish_df.head(20), use_container_width=True)

            st.markdown("### 6) Download")
            d1, d2 = st.columns(2)
            bull_bytes = bullish_df.to_csv(index=False).encode("utf-8")
            bear_bytes = bearish_df.to_csv(index=False).encode("utf-8")

            with d1:
                st.download_button(
                    "Download PUT file (Bullish)",
                    data=bull_bytes,
                    file_name="bullish_entries_with_features_v3_vwap.csv",
                    mime="text/csv",
                    key="ema_b_phase1_download_put",
                )
            with d2:
                st.download_button(
                    "Download CALL file (Bearish)",
                    data=bear_bytes,
                    file_name="bearish_entries_with_features_v3_vwap.csv",
                    mime="text/csv",
                    key="ema_b_phase1_download_call",
                )

            if st.button("Clear outputs", key="ema_b_phase1_clear"):
                st.session_state.ema_b_phase1_bullish = None
                st.session_state.ema_b_phase1_bearish = None
                st.rerun()


    # ----------------------------
    # Phase 2 — Backtest bucket summary (upload backtester outputs)
    # ----------------------------
    with st.expander("Phase 2 — Backtest time-bucket win% (TSV)", expanded=False):
        st.caption(
            "Upload backtester trade logs (PUTs and/or CALLs). "
            "Generates paste-ready TSV blocks with WIN% (and optional TRADES) for multiple lookback windows."
        )

        # Session state (persist downloads)
        if "ema_b_phase2_put_tsv" not in st.session_state:
            st.session_state.ema_b_phase2_put_tsv = None
        if "ema_b_phase2_call_tsv" not in st.session_state:
            st.session_state.ema_b_phase2_call_tsv = None
        if "ema_b_phase2_put_preview" not in st.session_state:
            st.session_state.ema_b_phase2_put_preview = None
        if "ema_b_phase2_call_preview" not in st.session_state:
            st.session_state.ema_b_phase2_call_preview = None

        if "ema_b_phase2_put_win_pivots" not in st.session_state:
            st.session_state.ema_b_phase2_put_win_pivots = None
        if "ema_b_phase2_call_win_pivots" not in st.session_state:
            st.session_state.ema_b_phase2_call_win_pivots = None
        if "ema_b_phase2_windows" not in st.session_state:
            st.session_state.ema_b_phase2_windows = None

        st.markdown("### 1) Upload backtester trade logs")
        ucol1, ucol2 = st.columns(2)
        with ucol1:
            put_up = st.file_uploader("PUT trade log CSV", type=["csv"], key="ema_b_phase2_put_upload")
        with ucol2:
            call_up = st.file_uploader("CALL trade log CSV", type=["csv"], key="ema_b_phase2_call_upload")

        st.markdown("### 2) Settings (defaults match the script)")

        # Defaults from EMA-B step2.py
        default_windows = "30,90,180,270,365"
        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            windows_str = st.text_input(
                "LOOKBACK_WINDOWS_DAYS (comma-separated)",
                value=default_windows,
                help="Example: 30,90,180,270,365",
                key="ema_b_phase2_windows_str",
            )
        with c2:
            output_trades_pivot = st.checkbox(
                "OUTPUT_TRADES_PIVOT",
                value=True,
                help="If enabled, the TSV includes TRADES pivots side-by-side with WIN%.",
                key="ema_b_phase2_output_trades",
            )
        with c3:
            blank_rows_between = st.number_input(
                "BLANK_ROWS_BETWEEN_WINDOWS",
                min_value=0,
                max_value=20,
                value=3,
                step=1,
                help="Only affects spacing/readability in the TSV output.",
                key="ema_b_phase2_blank_rows",
            )

        # Parse windows deterministically
        def _parse_windows(s: str) -> list[int]:
            parts = [p.strip() for p in (s or "").split(",") if p.strip()]
            if not parts:
                raise ValueError("LOOKBACK_WINDOWS_DAYS cannot be empty.")
            out: list[int] = []
            for p in parts:
                v = int(p)
                if v <= 0:
                    raise ValueError("LOOKBACK_WINDOWS_DAYS must all be positive integers.")
                out.append(v)
            return out

        st.markdown("### 3) Run")
        run2 = st.button(
            "Generate bucket TSV(s)",
            type="primary",
            disabled=(put_up is None and call_up is None),
            key="ema_b_phase2_generate",
        )

        if run2:
            try:
                windows = _parse_windows(windows_str)

                st.session_state.ema_b_phase2_windows = windows

                if put_up is not None:
                    put_df_raw = pd.read_csv(put_up)
                    put_df = _phase2_prepare_trades_df(put_df_raw)
                    put_tsv, put_preview, put_win_pivots = _phase2_build_paste_ready_tsv(
                        put_df,
                        lookback_windows_days=windows,
                        output_trades_pivot=bool(output_trades_pivot),
                        blank_rows_between_windows=int(blank_rows_between),
                    )
                    st.session_state.ema_b_phase2_put_tsv = put_tsv
                    st.session_state.ema_b_phase2_put_preview = put_preview
                    st.session_state.ema_b_phase2_put_win_pivots = put_win_pivots

                if call_up is not None:
                    call_df_raw = pd.read_csv(call_up)
                    call_df = _phase2_prepare_trades_df(call_df_raw)
                    call_tsv, call_preview, call_win_pivots = _phase2_build_paste_ready_tsv(
                        call_df,
                        lookback_windows_days=windows,
                        output_trades_pivot=bool(output_trades_pivot),
                        blank_rows_between_windows=int(blank_rows_between),
                    )
                    st.session_state.ema_b_phase2_call_tsv = call_tsv
                    st.session_state.ema_b_phase2_call_preview = call_preview
                    st.session_state.ema_b_phase2_call_win_pivots = call_win_pivots

                st.success("Done. Scroll down to preview + download.")

            except Exception as e:
                st.error(f"Failed: {e}")

        # Preview (first window only)
        st.markdown("### 4) Preview (WIN% pivot for the first window)")
        pprev = st.session_state.ema_b_phase2_put_preview
        cprev = st.session_state.ema_b_phase2_call_preview

        if pprev is None and cprev is None:
            st.info("Upload trade logs and click Generate to see a preview.")
        else:
            pcol, ccol = st.columns(2)
            with pcol:
                if pprev is not None:
                    st.write("PUTs — WIN% pivot (first window)")
                    st.dataframe(pprev, use_container_width=True)
            with ccol:
                if cprev is not None:
                    st.write("CALLs — WIN% pivot (first window)")
                    st.dataframe(cprev, use_container_width=True)

        st.markdown("### 5) Download")
        dcol1, dcol2 = st.columns(2)

        put_bytes = st.session_state.ema_b_phase2_put_tsv
        call_bytes = st.session_state.ema_b_phase2_call_tsv

        with dcol1:
            if put_bytes is not None:
                st.download_button(
                    "Download PUT bucket TSV",
                    data=put_bytes,
                    file_name="put_time_bucket_win_pct.tsv",
                    mime="text/tab-separated-values",
                    key="ema_b_phase2_download_put",
                )
        with dcol2:
            if call_bytes is not None:
                st.download_button(
                    "Download CALL bucket TSV",
                    data=call_bytes,
                    file_name="call_time_bucket_win_pct.tsv",
                    mime="text/tab-separated-values",
                    key="ema_b_phase2_download_call",
                )

        if st.button("Clear Phase 2 outputs", key="ema_b_phase2_clear"):
            st.session_state.ema_b_phase2_put_tsv = None
            st.session_state.ema_b_phase2_call_tsv = None
            st.session_state.ema_b_phase2_put_preview = None
            st.session_state.ema_b_phase2_call_preview = None
            st.session_state.ema_b_phase2_put_win_pivots = None
            st.session_state.ema_b_phase2_call_win_pivots = None
            st.session_state.ema_b_phase2_windows = None
            st.rerun()


    # ----------------------------
    # Phase 3 — Score buckets (in-app replacement for Google Sheets scoring)
    # ----------------------------
    with st.expander("Phase 3 — Score buckets (keep / drop)", expanded=False):
        st.caption(
            "Uses Phase 2 in-memory WIN% pivots to score each (TIME_BUCKET × WEEKDAY) bucket. "
            "Buckets earn points per lookback window where WIN% meets the minimum threshold. "
            "Buckets must meet the minimum WIN% in at least MatchCount windows to be eligible for scoring."
        )

        # Require Phase 2 outputs (in-memory)
        phase2_windows = st.session_state.get("ema_b_phase2_windows")
        put_win_pivots = st.session_state.get("ema_b_phase2_put_win_pivots")
        call_win_pivots = st.session_state.get("ema_b_phase2_call_win_pivots")

        if not phase2_windows or (put_win_pivots is None and call_win_pivots is None):
            st.info("Run Phase 2 first (generate bucket TSVs) so Phase 3 has WIN% pivots to score.")
        else:
            st.markdown("### 1) Scoring inputs")

            c1, c2, c3 = st.columns(3)
            with c1:
                min_win_pct_input = st.number_input(
                    "Win percentage (min WIN% to earn points)",
                    min_value=0.0,
                    max_value=100.0,
                    value=65.0,
                    step=0.5,
                    help="Example: 65 means WIN% must be 65% or higher for that window to count.",
                    key="ema_b_phase3_min_win_pct",
                )
                min_win_pct = float(min_win_pct_input) / 100.0

            with c2:
                min_match_count = st.number_input(
                    "MatchCount (min windows that must meet Win%)",
                    min_value=1,
                    max_value=max(1, len(phase2_windows)),
                    value=3,
                    step=1,
                    key="ema_b_phase3_min_match_count",
                )

            with c3:
                min_total_points = st.number_input(
                    "Total Points (min score to KEEP bucket)",
                    min_value=0,
                    max_value=10_000,
                    value=10,
                    step=1,
                    key="ema_b_phase3_min_total_points",
                )

            st.markdown("### 2) Points per lookback window")

            # Default points mapping (matches your Google Sheets defaults when windows are the standard set)
            default_points_map = {30: 5, 90: 4, 180: 3, 270: 2, 365: 4}

            # Build/editable points table keyed off Phase 2 windows
            points_df_key = "ema_b_phase3_points_df"
            windows_tuple_key = "ema_b_phase3_windows_tuple"

            current_windows_tuple = tuple(phase2_windows)

            if st.session_state.get(windows_tuple_key) != current_windows_tuple or st.session_state.get(points_df_key) is None:
                pts = []
                for w in phase2_windows:
                    pts.append({"LOOKBACK_DAYS": int(w), "POINTS": int(default_points_map.get(int(w), 0))})
                st.session_state[points_df_key] = pd.DataFrame(pts)
                st.session_state[windows_tuple_key] = current_windows_tuple

            points_df = st.session_state[points_df_key]

            edited_points_df = st.data_editor(
                points_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "LOOKBACK_DAYS": st.column_config.NumberColumn("LOOKBACK_DAYS", disabled=True),
                    "POINTS": st.column_config.NumberColumn("POINTS", min_value=0, max_value=10_000, step=1),
                },
                key="ema_b_phase3_points_editor",
            )

            # Normalize edited points into a dict
            points_map: dict[int, int] = {
                int(r["LOOKBACK_DAYS"]): int(r["POINTS"])
                for _, r in edited_points_df.iterrows()
            }

            st.markdown("### 3) Run scoring")

            show_details = st.checkbox(
                "Show per-window WIN% and point contributions",
                value=False,
                key="ema_b_phase3_show_details",
            )

            run3 = st.button(
                "Score buckets",
                type="primary",
                key="ema_b_phase3_run",
            )

            def _score_from_win_pivots(win_pivots_by_days: dict[int, pd.DataFrame]) -> pd.DataFrame:
                # Build base index of all buckets
                full_index = pd.MultiIndex.from_product(
                    [_PHASE2_TIME_INDEX, _PHASE2_WEEKDAY_COLS],
                    names=["TIME_BUCKET", "WEEKDAY"],
                )
                out = pd.DataFrame(index=full_index).reset_index()

                # For each window, pull WIN% into a column and compute pass/points
                pass_cols = []
                win_cols = []
                pts_cols = []

                for days in phase2_windows:
                    days = int(days)
                    pivot = win_pivots_by_days.get(days)

                    col_win = f"WIN_{days}D"
                    col_pass = f"PASS_{days}D"
                    col_pts = f"PTS_{days}D"

                    if pivot is None:
                        # If window missing, treat as all NaN
                        out[col_win] = pd.NA
                        out[col_pass] = False
                        out[col_pts] = 0
                    else:
                        s = pivot.stack(dropna=False)
                        # s index: (TIME_BUCKET, WEEKDAY)
                        s.name = col_win
                        tmp = s.reset_index()
                        tmp.columns = ["TIME_BUCKET", "WEEKDAY", col_win]
                        out = out.merge(tmp, on=["TIME_BUCKET", "WEEKDAY"], how="left")

                        out[col_pass] = (out[col_win].astype("float") >= min_win_pct).fillna(False)
                        out[col_pts] = out[col_pass].astype(int) * int(points_map.get(days, 0))

                    win_cols.append(col_win)
                    pass_cols.append(col_pass)
                    pts_cols.append(col_pts)

                out["MatchCount"] = out[pass_cols].sum(axis=1).astype(int)
                out["RawScore"] = out[pts_cols].sum(axis=1).astype(int)

                out["Eligible"] = out["MatchCount"] >= int(min_match_count)
                out["Score"] = out["RawScore"].where(out["Eligible"], 0).astype(int)
                out["KEEP"] = out["Eligible"] & (out["Score"] >= int(min_total_points))

                # Order columns
                base_cols = ["TIME_BUCKET", "WEEKDAY", "MatchCount", "Score", "KEEP"]
                if show_details:
                    # include win/passes/points
                    detail_cols = []
                    for days in phase2_windows:
                        detail_cols.extend([f"WIN_{int(days)}D", f"PASS_{int(days)}D", f"PTS_{int(days)}D"])
                    out = out[base_cols + detail_cols]
                else:
                    out = out[base_cols]

                # Nice sorting
                out["TIME_SORT"] = pd.to_datetime("1900-01-01 " + out["TIME_BUCKET"])
                weekday_order = {d: i for i, d in enumerate(_PHASE2_WEEKDAY_COLS)}
                out["WD_SORT"] = out["WEEKDAY"].map(weekday_order).astype(int)
                out = out.sort_values(["TIME_SORT", "WD_SORT"]).drop(columns=["TIME_SORT", "WD_SORT"]).reset_index(drop=True)
                return out

            if run3:
                try:
                    if put_win_pivots is not None:
                        st.session_state.ema_b_phase3_put_scores = _score_from_win_pivots(put_win_pivots)
                    else:
                        st.session_state.ema_b_phase3_put_scores = None

                    if call_win_pivots is not None:
                        st.session_state.ema_b_phase3_call_scores = _score_from_win_pivots(call_win_pivots)
                    else:
                        st.session_state.ema_b_phase3_call_scores = None

                    st.success("Done. See tables below.")

                except Exception as e:
                    st.error(f"Failed: {e}")

            # Ensure session keys exist
            if "ema_b_phase3_put_scores" not in st.session_state:
                st.session_state.ema_b_phase3_put_scores = None
            if "ema_b_phase3_call_scores" not in st.session_state:
                st.session_state.ema_b_phase3_call_scores = None

            st.markdown("### 4) Results")

            put_scores = st.session_state.ema_b_phase3_put_scores
            call_scores = st.session_state.ema_b_phase3_call_scores

            view_mode = st.radio(
                "Display mode",
                options=["Compact grid", "Long table"],
                horizontal=True,
                index=0,
                key="ema_b_phase3_view_mode",
                help="Compact grid matches the Google Sheets layout (time rows × weekday columns). Long table is useful for exports/debugging.",
            )

            show_match_in_cells = False
            if view_mode == "Compact grid":
                show_match_in_cells = st.checkbox(
                    "Show MatchCount in cells (Score (MatchCount))",
                    value=False,
                    key="ema_b_phase3_show_match_in_cells",
                )

            def _build_grids(scores_df: pd.DataFrame):
                # Ensure deterministic ordering
                weekday_order = list(_PHASE2_WEEKDAY_COLS)
                time_order = list(_PHASE2_TIME_INDEX)

                # Pivot to grids
                score_grid = (
                    scores_df.pivot(index="TIME_BUCKET", columns="WEEKDAY", values="Score")
                            .reindex(index=time_order, columns=weekday_order)
                )
                match_grid = (
                    scores_df.pivot(index="TIME_BUCKET", columns="WEEKDAY", values="MatchCount")
                            .reindex(index=time_order, columns=weekday_order)
                )
                keep_grid = (
                    scores_df.pivot(index="TIME_BUCKET", columns="WEEKDAY", values="KEEP")
                            .reindex(index=time_order, columns=weekday_order)
                            .fillna(False)
                )
                eligible_grid = (match_grid.fillna(0).astype(int) >= int(min_match_count))

                if show_match_in_cells:
                    # Render "Score (MatchCount)" for readability
                    display = score_grid.copy()
                    for r in display.index:
                        for c in display.columns:
                            s = display.loc[r, c]
                            m = match_grid.loc[r, c]
                            if pd.isna(s) and pd.isna(m):
                                display.loc[r, c] = ""
                            else:
                                s_int = int(0 if pd.isna(s) else s)
                                m_int = int(0 if pd.isna(m) else m)
                                display.loc[r, c] = f"{s_int} ({m_int})"
                    return display, keep_grid, eligible_grid, score_grid, match_grid
                else:
                    return score_grid, keep_grid, eligible_grid, score_grid, match_grid

            def _style_compact(display_df: pd.DataFrame, keep_mask: pd.DataFrame, eligible_mask: pd.DataFrame):
                # High-contrast colors that work in dark mode.
                KEEP_BG = "#1f6f3a"       # dark green
                KEEP_FG = "#ffffff"
                INELIG_BG = "#2b2b2b"     # dark gray
                INELIG_FG = "#bdbdbd"

                def apply_styles(data):
                    out = pd.DataFrame("", index=data.index, columns=data.columns)
                    for r in data.index:
                        for c in data.columns:
                            if r not in keep_mask.index or c not in keep_mask.columns:
                                continue
                            is_keep = bool(keep_mask.loc[r, c])
                            is_eligible = bool(eligible_mask.loc[r, c]) if (r in eligible_mask.index and c in eligible_mask.columns) else False

                            if is_keep:
                                out.loc[r, c] = f"background-color: {KEEP_BG}; color: {KEEP_FG}; font-weight: 700;"
                            elif not is_eligible:
                                out.loc[r, c] = f"background-color: {INELIG_BG}; color: {INELIG_FG};"
                            else:
                                # Eligible but not keep: no background, readable text
                                out.loc[r, c] = "color: #e6e6e6;"
                    return out

                return display_df.style.apply(apply_styles, axis=None)

            r1, r2 = st.columns(2)

            with r1:
                if put_scores is not None:
                    st.write("PUT buckets — scored")
                    if view_mode == "Compact grid":
                        disp, keep_mask, elig_mask, score_grid, match_grid = _build_grids(put_scores)
                        st.dataframe(
                            _style_compact(disp, keep_mask, elig_mask),
                            use_container_width=True,
                            height=520,
                        )
                    else:
                        st.dataframe(put_scores, use_container_width=True, height=520)

                    keep_put = put_scores[put_scores["KEEP"]].copy()
                    keep_put_tsv = keep_put.to_csv(sep="\t", index=False).encode("utf-8")
                    st.download_button(
                        "Download PUT keep buckets (TSV)",
                        data=keep_put_tsv,
                        file_name="put_keep_buckets.tsv",
                        mime="text/tab-separated-values",
                        key="ema_b_phase3_download_put_keep",
                    )
                else:
                    st.info("No PUT pivots found (upload/generate PUTs in Phase 2).")

            with r2:
                if call_scores is not None:
                    st.write("CALL buckets — scored")
                    if view_mode == "Compact grid":
                        disp, keep_mask, elig_mask, score_grid, match_grid = _build_grids(call_scores)
                        st.dataframe(
                            _style_compact(disp, keep_mask, elig_mask),
                            use_container_width=True,
                            height=520,
                        )
                    else:
                        st.dataframe(call_scores, use_container_width=True, height=520)

                    keep_call = call_scores[call_scores["KEEP"]].copy()
                    keep_call_tsv = keep_call.to_csv(sep="\t", index=False).encode("utf-8")
                    st.download_button(
                        "Download CALL keep buckets (TSV)",
                        data=keep_call_tsv,
                        file_name="call_keep_buckets.tsv",
                        mime="text/tab-separated-values",
                        key="ema_b_phase3_download_call_keep",
                    )
                else:
                    st.info("No CALL pivots found (upload/generate CALLs in Phase 2).")
            if st.button("Clear Phase 3 outputs", key="ema_b_phase3_clear"):
                            st.session_state.ema_b_phase3_put_scores = None
                            st.session_state.ema_b_phase3_call_scores = None
                            st.rerun()


    # ----------------------------
    # Phase 4 — Filter Phase 1 entries to Phase 3 KEEP buckets
    # ----------------------------
    with st.expander("Phase 4 — Filter Phase 1 entries to KEEP buckets", expanded=False):
        st.caption(
            "Filters the Phase 1 PUT/CALL entry files down to ONLY rows whose OPEN_DATETIME falls inside "
            "a Phase 3 KEEP bucket (15-minute buckets)."
        )

        # Session state outputs
        if "ema_b_phase4_put_filtered" not in st.session_state:
            st.session_state.ema_b_phase4_put_filtered = None
        if "ema_b_phase4_call_filtered" not in st.session_state:
            st.session_state.ema_b_phase4_call_filtered = None

        # Inputs (in-memory)
        phase1_put_df = st.session_state.get("ema_b_phase1_bullish")   # PUT entries
        phase1_call_df = st.session_state.get("ema_b_phase1_bearish") # CALL entries

        put_scores = st.session_state.get("ema_b_phase3_put_scores")
        call_scores = st.session_state.get("ema_b_phase3_call_scores")

        # --- Logic mirrors EMA-B step 3.py ---
        BUCKET_LEN_MIN = 15  # [start, start+15)
        WEEKDAY_MAP = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
        WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

        def _hhmm_to_minutes(hhmm: str) -> int:
            h, m = hhmm.split(":")
            return int(h) * 60 + int(m)

        def build_bucket_lookup(bucket_list):
            # { "Mon": [(start,end), ...], ... }
            out = {d: [] for d in WEEKDAYS}
            for day, start in bucket_list:
                out[day].append((int(start), int(start) + BUCKET_LEN_MIN))
            for d in out:
                out[d].sort(key=lambda x: x[0])
            return out

        def in_any_bucket(day_abbrev: str, minute_of_day: int, bucket_lookup: dict) -> bool:
            intervals = bucket_lookup.get(day_abbrev, [])
            for start, end in intervals:
                if minute_of_day >= start and minute_of_day < end:
                    return True
            return False

        def _keep_buckets_from_scores(scores_df: pd.DataFrame) -> list[tuple[str, int]]:
            # Scores DF columns (Phase 3): TIME_BUCKET, WEEKDAY, KEEP
            keep_df = scores_df[scores_df["KEEP"]].copy()
            bucket_list = []
            for _, r in keep_df.iterrows():
                day = str(r["WEEKDAY"])
                tb = str(r["TIME_BUCKET"])
                if day in WEEKDAYS and ":" in tb:
                    bucket_list.append((day, _hhmm_to_minutes(tb)))
            # stable / deterministic
            return sorted(set(bucket_list), key=lambda x: (WEEKDAYS.index(x[0]), x[1]))

        def _filter_entries_to_keep_buckets(entries_df: pd.DataFrame, bucket_list: list[tuple[str, int]]) -> tuple[pd.DataFrame, dict]:
            if entries_df is None or len(entries_df) == 0:
                return entries_df, {"input_rows": 0, "bad_open_datetime": 0, "output_rows": 0}

            if "OPEN_DATETIME" not in entries_df.columns:
                raise ValueError(f"Missing required column OPEN_DATETIME. Found columns: {list(entries_df.columns)}")

            df = entries_df.copy()

            # Parse OPEN_DATETIME exactly as script does
            open_dt = pd.to_datetime(df["OPEN_DATETIME"], format="%Y-%m-%d %H:%M", errors="coerce")
            bad = int(open_dt.isna().sum())
            if bad:
                df = df.loc[~open_dt.isna()].copy()
                open_dt = open_dt.loc[~open_dt.isna()]

            bucket_lookup = build_bucket_lookup(bucket_list)

            weekday_abbrev = open_dt.dt.weekday.map(WEEKDAY_MAP)
            minute_of_day = open_dt.dt.hour * 60 + open_dt.dt.minute

            keep_mask = [
                (d in bucket_lookup) and in_any_bucket(d, int(m), bucket_lookup)
                for d, m in zip(weekday_abbrev.fillna(""), minute_of_day)
            ]

            out_df = df.loc[keep_mask].copy()

            # Enforce exact output formatting (YYYY-MM-DD HH:MM)
            out_df["OPEN_DATETIME"] = pd.to_datetime(out_df["OPEN_DATETIME"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            if "CLOSE_DATETIME" in out_df.columns:
                out_df["CLOSE_DATETIME"] = pd.to_datetime(out_df["CLOSE_DATETIME"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

            stats = {"input_rows": int(len(df)), "bad_open_datetime": bad, "output_rows": int(len(out_df))}
            return out_df.reset_index(drop=True), stats

        # --- Availability checks ---
        c1, c2 = st.columns(2)
        with c1:
            st.write("**PUT side**")
            st.write("✅ Phase 1 PUT entries" if phase1_put_df is not None else "❌ Phase 1 PUT entries")
            st.write("✅ Phase 3 PUT scores" if put_scores is not None else "❌ Phase 3 PUT scores")
        with c2:
            st.write("**CALL side**")
            st.write("✅ Phase 1 CALL entries" if phase1_call_df is not None else "❌ Phase 1 CALL entries")
            st.write("✅ Phase 3 CALL scores" if call_scores is not None else "❌ Phase 3 CALL scores")

        can_run_put = (phase1_put_df is not None) and (put_scores is not None)
        can_run_call = (phase1_call_df is not None) and (call_scores is not None)

        run4 = st.button(
            "Generate filtered PUT + CALL files",
            type="primary",
            disabled=(not can_run_put and not can_run_call),
            key="ema_b_phase4_run",
        )

        if run4:
            try:
                if can_run_put:
                    put_bucket_list = _keep_buckets_from_scores(put_scores)
                    filtered_put, put_stats = _filter_entries_to_keep_buckets(phase1_put_df, put_bucket_list)
                    st.session_state.ema_b_phase4_put_filtered = filtered_put
                    st.write(f"PUT: {put_stats['input_rows']} → {put_stats['output_rows']} rows (dropped {put_stats['bad_open_datetime']} bad OPEN_DATETIME)")

                if can_run_call:
                    call_bucket_list = _keep_buckets_from_scores(call_scores)
                    filtered_call, call_stats = _filter_entries_to_keep_buckets(phase1_call_df, call_bucket_list)
                    st.session_state.ema_b_phase4_call_filtered = filtered_call
                    st.write(f"CALL: {call_stats['input_rows']} → {call_stats['output_rows']} rows (dropped {call_stats['bad_open_datetime']} bad OPEN_DATETIME)")

                st.success("Phase 4 outputs generated.")
            except Exception as e:
                st.error(f"Failed: {e}")

        # Downloads
        st.markdown("### Download")
        d1, d2 = st.columns(2)

        put_filtered = st.session_state.get("ema_b_phase4_put_filtered")
        call_filtered = st.session_state.get("ema_b_phase4_call_filtered")

        with d1:
            if put_filtered is not None:
                st.download_button(
                    "Download PUT filtered file",
                    data=put_filtered.to_csv(index=False).encode("utf-8"),
                    file_name="bullish_entries_KEEP_BUCKETS.csv",
                    mime="text/csv",
                    key="ema_b_phase4_download_put",
                )
        with d2:
            if call_filtered is not None:
                st.download_button(
                    "Download CALL filtered file",
                    data=call_filtered.to_csv(index=False).encode("utf-8"),
                    file_name="bearish_entries_KEEP_BUCKETS.csv",
                    mime="text/csv",
                    key="ema_b_phase4_download_call",
                )

        if st.button("Clear Phase 4 outputs", key="ema_b_phase4_clear"):
            st.session_state.ema_b_phase4_put_filtered = None
            st.session_state.ema_b_phase4_call_filtered = None
            st.rerun()
