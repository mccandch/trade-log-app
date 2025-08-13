import io, re
from pathlib import Path
from datetime import date, timedelta
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# YAML is optional; app still runs without it
try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def _rerun():
    try:
        st.experimental_rerun()
    except Exception:
        st.rerun()


# ---------------- Strategy mapping (YAML + fallback) ----------------
@st.cache_resource(show_spinner=False)
def lvb_load_strategy_rules() -> Dict[str, Dict]:
    data = None
    if yaml:
        for p in (Path(__file__).parent.parent / "strategy_mapping.yaml",
                  Path.cwd() / "strategy_mapping.yaml"):
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                break
    if data is None:
        data = {"rules": [
            {"source":"backtest","match":"EH: 10:04 call $3 target, 2x stop","canonical":"Early Hour"},
            {"source":"backtest","match":"EH: 10:04 put $3 target, 2x stop","canonical":"Early Hour"},
            {"source":"backtest","match":"EH: 12:48 put $3 target, 2x stop","canonical":"Early Hour"},
            {"source":"backtest","match":"EMA-B:  5 minute calls","canonical":"EMA-B"},
            {"source":"backtest","match":"EMA-B: 5 minute puts","canonical":"EMA-B"},
            {"source":"backtest","match":"EMA-T: bearish","canonical":"EMA-T"},
            {"source":"backtest","match":"EMA-T: bullish","canonical":"EMA-T"},
            {"source":"backtest","match":"EMA:  CCS all tranche","canonical":"EMA Credit Spread"},
            {"source":"backtest","match":"EMA: PCS all tranches","canonical":"EMA Credit Spread"},
            {"source":"backtest","match":"PH: DI Call 15:25 ","canonical":"PH"},
            {"source":"backtest","match":"PH: DI Put 15:25","canonical":"PH"},
            {"source":"backtest","match":"ema-1x:  CCS 20/40","canonical":"EMA-1x"},
            {"source":"backtest","match":"ema-1x:  PCS 20/40","canonical":"EMA-1x"},
        ]}
    exact = {"live": {}, "backtest": {}}
    patterns = {"live": [], "backtest": []}
    for r in data.get("rules", []):
        src = "live" if str(r.get("source","backtest")).lower().startswith("live") else "backtest"
        canon = r.get("canonical")
        match_val = r.get("match")
        ignore = bool(r.get("ignore", False))
        if not match_val:
            continue
        if r.get("regex", False):
            try:
                patterns[src].append((re.compile(match_val), canon, ignore))
            except re.error:
                pass
        else:
            exact[src][str(match_val)] = (canon, ignore)
    return {"exact": exact, "patterns": patterns}


def lvb_map_strategy_name(source: str, raw_name: str) -> Tuple[str, bool]:
    rules = lvb_load_strategy_rules()
    s = "live" if str(source).lower().startswith("live") else "backtest"
    raw = "" if raw_name is None else str(raw_name)
    maybe = rules["exact"].get(s, {}).get(raw)
    if maybe:
        canon, ign = maybe
        return (raw if canon is None else canon, ign)
    for patt, canon, ign in rules["patterns"].get(s, []):
        try:
            if patt.search(raw):
                return (raw if canon is None else canon, ign)
        except re.error:
            pass
    return raw, False


# ---------------- IO helpers ----------------
@st.cache_data(show_spinner=False)
def lvb_read_csv_file(uploaded) -> pd.DataFrame:
    if hasattr(uploaded, "read"):
        return pd.read_csv(uploaded)
    return pd.read_csv(uploaded)


def lvb_persist_upload_bytes(file_obj, state_key: str):
    if file_obj is not None:
        try:
            st.session_state[state_key] = file_obj.getvalue()
        except Exception:
            file_obj.seek(0)
            st.session_state[state_key] = file_obj.read()
        return io.BytesIO(st.session_state[state_key])
    if state_key in st.session_state:
        return io.BytesIO(st.session_state[state_key])
    return None


def lvb_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ---------------- Normalization + stats ----------------
def lvb_live_side_from_tradetype(trade_type: str) -> Optional[str]:
    if pd.isna(trade_type): return None
    return "Call" if "Call" in str(trade_type) else "Put"


def lvb_robust_side_from_text(s: str) -> Optional[str]:
    s0 = str(s).strip().lower()
    s0 = re.sub(r"\s+", " ", s0)
    if ("ccs" in s0) or ("bearish" in s0) or re.search(r"\bcall(s)?\b", s0):
        return "Call"
    if ("pcs" in s0) or ("bullish" in s0) or re.search(r"\bput(s)?\b", s0):
        return "Put"
    return None


def lvb_normalize_live(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df[df.get("Strategy","") != "Megatrend"].copy()
    df["StrategyMapped"] = df["Strategy"].apply(lambda x: lvb_map_strategy_name("live", x)[0])
    df["PremiumSold"] = pd.to_numeric(df.get("TotalPremium", 0), errors="coerce").abs().fillna(0).round(2)
    df["PnL"] = pd.to_numeric(df.get("ProfitLoss", 0), errors="coerce").fillna(0.0)
    df["PnL_rounded"] = df["PnL"].round(2)
    df["PnL_stats"] = df["PnL_rounded"]
    if "TradeType" in df.columns:
        df["Side"] = df["TradeType"].apply(lvb_live_side_from_tradetype)
    else:
        df["Side"] = None
    # "Date" + optional "TimeOpened"
    if "TimeOpened" in df.columns:
        df["OpenDT"] = pd.to_datetime(df["Date"].astype(str) + " " + df["TimeOpened"].astype(str), errors="coerce")
    else:
        df["OpenDT"] = pd.to_datetime(df.get("Date"), errors="coerce")
    return df


def lvb_normalize_backtest(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    mapped = df["Strategy"].apply(lambda x: lvb_map_strategy_name("backtest", x))
    df["StrategyMapped"] = mapped.apply(lambda t: t[0])
    df["IgnoreFlag"] = mapped.apply(lambda t: t[1])
    df = df[~df["IgnoreFlag"]].copy()
    df["Side"] = df["Strategy"].apply(lvb_robust_side_from_text)

    def _num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(
            s.astype(str)
             .str.replace(r"[\$,]", "", regex=True)
             .str.replace(r"\s*(cr|db)\s*$", "", regex=True)
             .str.strip(),
            errors="coerce",
        )
    df["Premium"] = _num(df.get("Premium"))
    df["Avg. Closing Cost"] = _num(df.get("Avg. Closing Cost"))
    contracts = pd.to_numeric(df.get("No. of Contracts"), errors="coerce").fillna(1).clip(lower=1)

    # Premium total heuristic using "Legs" if available
    def _credit_per_contract(legs: str) -> float:
        s = str(legs)
        sells = re.findall(r"(?:STO|SELL TO OPEN)\s*([0-9]*\.?[0-9]+)", s, flags=re.I)
        buys  = re.findall(r"(?:BTO|BUY TO OPEN)\s*([0-9]*\.?[0-9]+)", s, flags=re.I)
        if not sells and not buys:
            return np.nan
        return sum(map(float, sells)) - sum(map(float, buys))
    df["_credit_per"] = df["Legs"].apply(_credit_per_contract) if "Legs" in df.columns else np.nan

    def _premium_total(row) -> float:
        prem = row.get("Premium", np.nan)
        cpc  = row.get("_credit_per", np.nan)
        n    = row.get("No. of Contracts", 1)
        if pd.isna(prem): return np.nan
        if pd.notna(cpc) and pd.notna(n):
            expected_total = cpc * 100 * n
            expected_one   = cpc * 100
            return float(prem) if abs(prem - expected_total) <= abs(prem - expected_one) else float(prem) * float(n)
        if prem < 75:
            return float(prem) * float(n)
        return float(prem)

    df["PremiumSold"] = df.apply(_premium_total, axis=1).round(2)

    close_is_price = df["Avg. Closing Cost"] < 50
    close_total = df["Avg. Closing Cost"].where(~close_is_price, df["Avg. Closing Cost"] * 100 * contracts).fillna(0)
    df["PnL_gross"] = (df["PremiumSold"] - close_total).round(2)
    df["_PnL_net"] = _num(df.get("P/L")).round(2) if "P/L" in df.columns else np.nan
    df["PnL_stats"] = df["_PnL_net"].where(df["_PnL_net"].notna(), df["PnL_gross"]).fillna(0)

    # "Date Opened" + optional "Time Opened"
    if "Time Opened" in df.columns:
        df["OpenDT"] = pd.to_datetime(df["Date Opened"].astype(str) + " " + df["Time Opened"].astype(str), errors="coerce")
    else:
        df["OpenDT"] = pd.to_datetime(df.get("Date Opened"), errors="coerce")

    df["PnL_rounded"] = df["PnL_gross"]
    return df


def lvb_prepare_for_stats(df: pd.DataFrame, view: str) -> Tuple[pd.DataFrame, str]:
    if view == "Combined":
        out = df.copy(); out["StrategyKey"] = out["StrategyMapped"]; return out, "Strategy"
    if view == "Split Calls/Puts by Strategy":
        out = df.copy(); out["StrategyKey"] = out["StrategyMapped"] + " / " + out["Side"].fillna(""); return out, "Strategy / Side"
    if view == "Calls only":
        out = df[df["Side"] == "Call"].copy(); out["StrategyKey"] = out["StrategyMapped"] + " / Call"; return out, "Strategy (Calls)"
    if view == "Puts only":
        out = df[df["Side"] == "Put"].copy(); out["StrategyKey"] = out["StrategyMapped"] + " / Put"; return out, "Strategy (Puts)"
    out = df.copy(); out["StrategyKey"] = out["StrategyMapped"]; return out, "Strategy"


def lvb_compute_strategy_stats(df: pd.DataFrame, strat_col: str = "StrategyKey", pnl_col: str = "PnL_stats") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[strat_col, "Trades", "Win Rate %", "Premium Sold", "Premium Captured", "PCR %"])
    g = df.groupby(strat_col).agg(
        Trades=(pnl_col, "size"),
        **{"Win Rate %": (pnl_col, lambda x: (pd.to_numeric(x, errors="coerce") > 0).mean() * 100.0)},
        **{"Premium Sold": ("PremiumSold", lambda x: pd.to_numeric(x, errors="coerce").sum())},
        **{"Premium Captured": (pnl_col, lambda x: pd.to_numeric(x, errors="coerce").sum())},
    ).reset_index()
    g["PCR %"] = (g["Premium Captured"] / g["Premium Sold"].where(pd.to_numeric(g["Premium Sold"], errors="coerce") != 0, np.nan)) * 100.0
    return g


def lvb_greedy_match_with_tolerance(live_df: pd.DataFrame, back_df: pd.DataFrame, tol_minutes: int) -> pd.DataFrame:
    from datetime import timedelta
    tol = timedelta(minutes=tol_minutes)
    matches = []
    keys = sorted(set(zip(live_df["StrategyMapped"], live_df["Side"])))
    back_unused = set(back_df.index)
    for strat, side in keys:
        L = live_df[(live_df["StrategyMapped"] == strat) & (live_df["Side"] == side)].sort_values("OpenDT")
        B = back_df[(back_df["StrategyMapped"] == strat) & (back_df["Side"] == side)].sort_values("OpenDT")
        if L.empty or B.empty:
            continue
        i = j = 0
        L_idx, B_idx = L.index.to_list(), B.index.to_list()
        while i < len(L_idx) and j < len(B_idx):
            li, bj = L_idx[i], B_idx[j]
            if bj not in back_unused:
                j += 1; continue
            dt_l, dt_b = L.at[li, "OpenDT"], B.at[bj, "OpenDT"]
            if abs(dt_l - dt_b) <= tol:
                matches.append((li, bj)); back_unused.remove(bj); i += 1; j += 1
            else:
                i += (dt_l < dt_b); j += (dt_b <= dt_l)
    return pd.DataFrame(matches, columns=["LiveIdx", "BackIdx"]) if matches else pd.DataFrame(columns=["LiveIdx", "BackIdx"])


# ---------------- Styling helpers ----------------
def lvb_render_table_html(
    df,
    currency_cols,
    percent_cols,
    posneg_cols,
    pair_band: bool,
    band_stride: int,
) -> str:
    styler = lvb_style_table_center_currency_percent(
        df, currency_cols, percent_cols, posneg_cols, pair_band, band_stride
    ).set_table_attributes('class="compact-table"')
    return styler.to_html()


def lvb_style_table_center_currency_percent(
    df: pd.DataFrame,
    currency_cols: List[str],
    percent_cols: List[str],
    posneg_cols: List[str],
    pair_band: bool = False,
    band_stride: int = 2,
):
    fmt = {}
    for c in currency_cols:
        if c in df.columns: fmt[c] = "${:,.2f}"
    for c in percent_cols:
        if c in df.columns: fmt[c] = "{:,.2f}%"
    if "Trades" in df.columns:
        fmt["Trades"] = "{:,.0f}"

    styler = df.style.format(fmt)
    if hasattr(styler, "hide_index"): styler = styler.hide_index()
    else: styler = styler.hide(axis="index")

    if pair_band and not df.empty:
        stride = max(int(band_stride), 1)
        def _band_rows(data: pd.DataFrame):
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            for i in range(len(data.index)):
                band = (i // stride) % 2
                color = "#141414" if band == 0 else "#0d0d0d"
                styles.iloc[i, :] = f"background-color: {color};"
            return styles
        styler = styler.apply(_band_rows, axis=None)
        if stride == 2:
            def _pair_border_thick(data: pd.DataFrame):
                styles = pd.DataFrame("", index=data.index, columns=data.columns)
                for i in range(0, len(data.index), 2):
                    if i == 0: continue
                    styles.iloc[i, :] = "border-top: 2px solid #444;"
                return styles
            styler = styler.apply(_pair_border_thick, axis=None)

    # Only color rows if a 'Source' column exists
    if "Source" in df.columns:
        def _source_color(row: pd.Series):
            if "Source" not in row.index:
                return [""] * len(row)
            is_live = str(row["Source"]).strip().lower().startswith("live")
            color = "#FFFFFF" if is_live else "#B8A15A"  # muted gold for Backtest
            return [f"color: {color};"] * len(row)
        styler = styler.apply(_source_color, axis=1)

    def _color_posneg(v):
        if pd.isna(v): return ""
        try: v = float(v)
        except Exception: return ""
        if v > 0: return "background-color: rgba(0, 128, 0, 0.18);"
        if v < 0: return "background-color: rgba(200, 0, 0, 0.20);"
        return ""
    for c in posneg_cols:
        if c in df.columns:
            styler = styler.applymap(_color_posneg, subset=pd.IndexSlice[:, [c]])

    styler = styler.set_table_styles([{"selector": "th, td", "props": [("text-align", "center")]}], overwrite=False)
    return styler

def lvb_render_table_html(
    df: pd.DataFrame,
    currency_cols: List[str],
    percent_cols: List[str],
    posneg_cols: List[str],
    pair_band: bool,
    band_stride: int,
) -> str:
    styler = lvb_style_table_center_currency_percent(
        df, currency_cols, percent_cols, posneg_cols, pair_band, band_stride
    ).set_table_attributes('class="compact-table"')
    return styler.to_html()

# ---------------- UI ----------------
def main():
    st.subheader("Compare Logs — Live vs Backtest")

    # Inject CSS for Backtest KPI colors
    st.markdown("""
    <style>
    .backtest-kpis [data-testid="stMetric"],
    .backtest-kpis [data-testid="stMetric"] * {
        color: #B8A15A !important; /* muted gold */
    }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([0.32, 0.68], gap="large")
    with left:
        if "lv_start_date" not in st.session_state:
            st.session_state.lv_start_date = date.today() - timedelta(days=60)
        if "lv_end_date" not in st.session_state:
            st.session_state.lv_end_date = date.today()

        st.markdown("##### Filters")
        range_val = st.date_input(
            "Date range",
            value=(st.session_state.lv_start_date, st.session_state.lv_end_date),
            key="lvb_date_range",
        )
        if isinstance(range_val, tuple) and len(range_val) == 2:
            st.session_state.lv_start_date, st.session_state.lv_end_date = range_val

        c1, c2, c3 = st.columns(3)
        if c1.button("This Week", key="lvb_btn_this_week", use_container_width=True):
            d0 = date.today()
            s = d0 - timedelta(days=d0.weekday())
            st.session_state.lv_start_date, st.session_state.lv_end_date = s, d0
        if c2.button("Last Week", key="lvb_btn_last_week", use_container_width=True):
            d0 = date.today()
            this_mon = d0 - timedelta(days=d0.weekday())
            last_mon = this_mon - timedelta(days=7)
            st.session_state.lv_start_date, st.session_state.lv_end_date = last_mon, last_mon + timedelta(days=6)
        if c3.button("This Month", key="lvb_btn_this_month", use_container_width=True):
            d0 = date.today()
            st.session_state.lv_start_date, st.session_state.lv_end_date = d0.replace(day=1), d0

        d1c, d2c, d3c = st.columns(3)
        if d1c.button("Last Month", key="lvb_btn_last_month", use_container_width=True):
            d0 = date.today()
            first_this = d0.replace(day=1)
            last_prev = first_this - timedelta(days=1)
            first_prev = last_prev.replace(day=1)
            st.session_state.lv_start_date, st.session_state.lv_end_date = first_prev, last_prev
        if d2c.button("YTD", key="lvb_btn_ytd", use_container_width=True):
            d0 = date.today()
            st.session_state.lv_start_date, st.session_state.lv_end_date = date(d0.year,1,1), d0
        if d3c.button("Today", key="lvb_btn_today", use_container_width=True):
            d0 = date.today()
            st.session_state.lv_start_date, st.session_state.lv_end_date = d0, d0

        st.divider()
        tolerance = st.number_input("Time tolerance (minutes)", min_value=0, max_value=60, value=2, step=1, key="tol_minutes")

        st.markdown("##### Uploads")
        live_upload = st.file_uploader("Live results CSV", type=["csv"], key="live_csv")
        back_upload = st.file_uploader("Backtest results CSV", type=["csv"], key="back_csv")

        st.markdown("##### View options")
        view_mode = st.radio("View mode", ["Combined", "Calls only", "Puts only", "Split Calls/Puts by Strategy"], horizontal=False, key="view_mode")
        source_filter = st.radio("Show in comparison table", ["Both", "Live only", "Backtest only"], horizontal=False, key="source_filter")

    with right:
        live_file = lvb_persist_upload_bytes(st.session_state.get("live_csv"), "_live_bytes")
        back_file = lvb_persist_upload_bytes(st.session_state.get("back_csv"), "_back_bytes")

        if (live_file is None) or (back_file is None):
            st.info("Upload both CSVs to begin.")
            return

        live_raw = lvb_read_csv_file(live_file)
        back_raw = lvb_read_csv_file(back_file)

        live_norm = lvb_normalize_live(live_raw)
        back_norm = lvb_normalize_backtest(back_raw)

        start_date, end_date = st.session_state.lv_start_date, st.session_state.lv_end_date

        def _within_window(ts: pd.Series, s: date, e: date) -> pd.Series:
            return (ts >= pd.Timestamp(s)) & (ts <= pd.Timestamp(e) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

        live_sel_full = live_norm[_within_window(live_norm["OpenDT"], start_date, end_date)].copy()
        back_sel_full = back_norm[_within_window(back_norm["OpenDT"], start_date, end_date)].copy()

        all_strats = sorted(set(live_sel_full["StrategyMapped"]).union(back_sel_full["StrategyMapped"]))
        selected_strats = st.multiselect("Filter strategies", all_strats, default=all_strats, key="strat_picker")

        def _keep(df: pd.DataFrame) -> pd.DataFrame:
            return df[df["StrategyMapped"].isin(selected_strats)].copy()

        live_sel = _keep(live_sel_full)
        back_sel = _keep(back_sel_full)

        live_prepped, _ = lvb_prepare_for_stats(live_sel, st.session_state.view_mode)
        back_prepped, _ = lvb_prepare_for_stats(back_sel, st.session_state.view_mode)

        live_stats = lvb_compute_strategy_stats(live_prepped, "StrategyKey", pnl_col="PnL_stats")
        back_stats = lvb_compute_strategy_stats(back_prepped, "StrategyKey", pnl_col="PnL_stats")

        merged = pd.merge(live_stats, back_stats, on="StrategyKey", how="outer",
                          suffixes=(" (Live)", " (Backtest)")).fillna(0)

        # Stack for Live vs Backtest display
        stack_rows = []
        for _, r in merged.iterrows():
            strat = r["StrategyKey"]
            stack_rows.append({"Strategy": strat, "Source": "Live",
                               "Trades": r["Trades (Live)"], "Win Rate %": r["Win Rate % (Live)"],
                               "Premium Sold": r["Premium Sold (Live)"], "Premium Captured": r["Premium Captured (Live)"],
                               "PCR %": r["PCR % (Live)"]})
            stack_rows.append({"Strategy": strat, "Source": "Backtest",
                               "Trades": r["Trades (Backtest)"], "Win Rate %": r["Win Rate % (Backtest)"],
                               "Premium Sold": r["Premium Sold (Backtest)"], "Premium Captured": r["Premium Captured (Backtest)"],
                               "PCR %": r["PCR % (Backtest)"]})
        comparison_stacked = pd.DataFrame(stack_rows)
        if not comparison_stacked.empty:
            comparison_stacked["Source"] = pd.Categorical(comparison_stacked["Source"], categories=["Live", "Backtest"], ordered=True)
            comparison_stacked = comparison_stacked.sort_values(["Strategy", "Source"]).reset_index(drop=True)

        if st.session_state.source_filter != "Both":
            allowed = ["Live"] if st.session_state.source_filter == "Live only" else ["Backtest"]
            comparison_stacked = comparison_stacked[comparison_stacked["Source"].isin(allowed)].reset_index(drop=True)

        # Totals rows
        def _summary_row(df: pd.DataFrame, src: str) -> Optional[dict]:
            sub = df[df["Source"] == src]
            if sub.empty: return None
            trades = pd.to_numeric(sub["Trades"], errors="coerce").fillna(0)
            wr     = pd.to_numeric(sub["Win Rate %"], errors="coerce").fillna(0)
            sold   = pd.to_numeric(sub["Premium Sold"], errors="coerce").fillna(0)
            cap    = pd.to_numeric(sub["Premium Captured"], errors="coerce").fillna(0)
            t_sum = float(trades.sum())
            w_wr  = float((wr * trades).sum() / t_sum) if t_sum > 0 else 0.0
            s_sum = float(sold.sum())
            c_sum = float(cap.sum())
            pcr   = float((c_sum / s_sum) * 100.0) if s_sum else 0.0
            return {"Strategy": "All Selected (Total)", "Source": src, "Trades": t_sum,
                    "Win Rate %": w_wr, "Premium Sold": s_sum, "Premium Captured": c_sum, "PCR %": pcr}

        totals = []
        if st.session_state.source_filter in ("Both", "Live only"):
            r = _summary_row(comparison_stacked, "Live");  totals += [r] if r else []
        if st.session_state.source_filter in ("Both", "Backtest only"):
            r = _summary_row(comparison_stacked, "Backtest"); totals += [r] if r else []
        if totals:
            comparison_stacked = pd.concat([comparison_stacked, pd.DataFrame(totals)], ignore_index=True)

        st.markdown("### Adjusted Live vs Backtest — Strategy Comparison")
        table_col, chart_col = st.columns([0.66, 0.34], gap="small")

        band_stride = 2 if st.session_state.source_filter == "Both" else 1
        comp_styler = lvb_style_table_center_currency_percent(
            comparison_stacked,
            currency_cols=["Premium Sold", "Premium Captured"],
            percent_cols=["Win Rate %", "PCR %"],
            posneg_cols=["Premium Captured", "PCR %"],
            pair_band=True,
            band_stride=band_stride,
        )
        def _mark_totals(data: pd.DataFrame):
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            mask = data["Strategy"].astype(str).str.contains(r"All Selected \(Total\)", regex=True)
            styles.loc[mask, :] = "font-weight: 700; border-top: 2px solid #444;"
            return styles
        comp_styler = comp_styler.apply(_mark_totals, axis=None).set_table_attributes('class="compact-table"')

        with table_col:
            st.markdown(comp_styler.to_html(), unsafe_allow_html=True)

        with chart_col:
            st.markdown("#### Quick visuals")
            metric = st.selectbox("Metric", ["Premium Captured", "PCR %", "Win Rate %", "Trades"], index=0,
                                  help="Bar chart by strategy (Live vs Backtest).")

            bar_df = comparison_stacked[~comparison_stacked["Strategy"].astype(str)
                                        .str.contains(r"All Selected \(Total\)", regex=True)].copy()

            if not bar_df.empty:
                strat_order = bar_df["Strategy"].drop_duplicates().tolist()
                def tooltip_for(col):
                    if col in ["PCR %", "Win Rate %"]:
                        return alt.Tooltip(f"{col}:Q", title=col, format=".2f")
                    if col in ["Premium Sold", "Premium Captured"]:
                        return alt.Tooltip(f"{col}:Q", title=col, format=",.2f")
                    return alt.Tooltip(f"{col}:Q", title=col, format=",.0f")

                bar = (
                    alt.Chart(bar_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Strategy:N", sort=strat_order, title=None, axis=alt.Axis(labelAngle=-30)),
                        xOffset=alt.XOffset("Source:N"),
                        y=alt.Y(f"{metric}:Q", title=metric),
                        color=alt.Color("Source:N", legend=alt.Legend(orient="bottom")),
                        tooltip=["Strategy", "Source", tooltip_for(metric)],
                    )
                    .properties(height=250)
                    .interactive()
                )
                bar = bar.configure_view(stroke=None).configure_axis(labelPadding=2, titlePadding=4)\
                         .properties(padding={"left": 4, "right": 4, "top": 4, "bottom": 4})
                st.altair_chart(bar, use_container_width=True)
            else:
                st.info("No rows to chart for the current filters.")

            # Line chart synced to metric
            line_map = {
                "Premium Captured": ("cum_pnl", "Cumulative Premium Captured ($)", "$,.0f"),
                "Trades":           ("daily_trades", "Trades", ",.0f"),
                "PCR %":            ("daily_pcr", "PCR %", ".2f"),
                "Win Rate %":       ("daily_wr", "Win Rate %", ".2f"),
            }
            series_key, y_title, y_fmt = line_map.get(metric, ("cum_pnl", "Cumulative Premium Captured ($)", "$,.0f"))

            ts_live = live_sel.assign(Source="Live", Date=pd.to_datetime(live_sel["OpenDT"]).dt.date)
            ts_back = back_sel.assign(Source="Backtest", Date=pd.to_datetime(back_sel["OpenDT"]).dt.date)
            ts = pd.concat([ts_live, ts_back], ignore_index=True)

            if not ts.empty:
                ts["PnL_stats"]   = pd.to_numeric(ts["PnL_stats"], errors="coerce").fillna(0)
                ts["PremiumSold"] = pd.to_numeric(ts.get("PremiumSold", 0), errors="coerce").fillna(0)
                ts["Win"]         = pd.to_numeric(ts.get("PnL_rounded", 0), errors="coerce").fillna(0) > 0

                g = ts.groupby(["Source", "Date"])

                if series_key == "cum_pnl":
                    daily = g["PnL_stats"].sum().reset_index(name="Value")
                    daily["Value"] = daily.groupby("Source")["Value"].cumsum()
                elif series_key == "daily_trades":
                    daily = g.size().reset_index(name="Value")
                elif series_key == "daily_pcr":
                    tmp = g.agg(PnL=("PnL_stats", "sum"), Sold=("PremiumSold", "sum")).reset_index()
                    tmp["Value"] = np.where(tmp["Sold"] != 0, (tmp["PnL"] / tmp["Sold"]) * 100.0, 0.0)
                    daily = tmp[["Source", "Date", "Value"]]
                else:  # daily_wr
                    tmp = g["Win"].mean().reset_index(name="Value")
                    tmp["Value"] = tmp["Value"] * 100.0
                    daily = tmp

                line = (
                    alt.Chart(daily)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Date:T", title=None),
                        y=alt.Y("Value:Q", title=y_title),
                        color=alt.Color("Source:N", legend=alt.Legend(orient="bottom")),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            "Source:N",
                            alt.Tooltip("Value:Q", title=y_title, format=y_fmt),
                        ],
                    )
                    .properties(height=250)
                    .interactive()
                ).configure_view(stroke=None).configure_axis(labelPadding=2, titlePadding=4)\
                 .properties(padding={"left": 4, "right": 4, "top": 4, "bottom": 4})

                st.altair_chart(line, use_container_width=True)
            else:
                st.info("No trades in the selected window for the line chart.")

        # KPIs
        # Inject CSS to color Backtest KPI labels & values
        st.markdown("""
            <style>
            div[data-testid="stMetric"] p:contains("Backtest") {
                color: #B8A15A !important;
            }
            div[data-testid="stMetric"] div:has(p:contains("Backtest")) {
                color: #B8A15A !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("### Summary KPIs (for selected strategies)")
        def agg_summary(df_stats: pd.DataFrame) -> Tuple[int, float, float, float, float]:
            if df_stats.empty: return 0, 0.0, 0.0, 0.0, 0.0
            trades = int(df_stats["Trades"].sum())
            winrate = float(np.average(df_stats["Win Rate %"], weights=df_stats["Trades"])) if trades > 0 else 0.0
            prem_sold = float(df_stats["Premium Sold"].sum())
            prem_cap = float(df_stats["Premium Captured"].sum())
            pcr = (prem_cap / prem_sold * 100.0) if prem_sold else 0.0
            return trades, winrate, prem_sold, prem_cap, pcr
        live_trades, live_wr, live_sold, live_cap, live_pcr = agg_summary(lvb_compute_strategy_stats(lvb_prepare_for_stats(live_sel, st.session_state.view_mode)[0], "StrategyKey"))
        back_trades, back_wr, back_sold, back_cap, back_pcr = agg_summary(lvb_compute_strategy_stats(lvb_prepare_for_stats(back_sel, st.session_state.view_mode)[0], "StrategyKey"))
        l1, l2, l3, l4, l5 = st.columns(5)
        l1.metric("Live Trades", f"{live_trades:,}")
        l2.metric("Live Win Rate", f"{live_wr:.1f}%")
        l3.metric("Live Premium Sold", f"${live_sold:,.0f}")
        l4.metric("Live Premium Captured", f"${live_cap:,.0f}")
        l5.metric("Live PCR", f"{live_pcr:.1f}%")
        st.markdown('<div class="backtest-kpis">', unsafe_allow_html=True)
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Backtest Trades", f"{back_trades:,}")
        b2.metric("Backtest Win Rate", f"{back_wr:.1f}%")
        b3.metric("Backtest Premium Sold", f"${back_sold:,.0f}")
        b4.metric("Backtest Premium Captured", f"${back_cap:,.0f}")
        b5.metric("Backtest PCR", f"{back_pcr:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        # Tolerance matching + detail tables
        matches = lvb_greedy_match_with_tolerance(live_sel, back_sel, st.session_state.tol_minutes)
        if matches.empty:
            matched_pairs = pd.DataFrame(columns=["Strategy","Side","OpenDT_Live","OpenDT_Back","LivePnL","BackPnL"])
            live_only = live_sel.copy(); back_only = back_sel.copy()
        else:
            live_matched = set(matches["LiveIdx"].tolist())
            back_matched = set(matches["BackIdx"].tolist())
            live_only = live_sel.loc[~live_sel.index.isin(live_matched), ["OpenDT","StrategyMapped","Side","PremiumSold","PnL_rounded"]].sort_values("OpenDT")
            back_only = back_sel.loc[~back_sel.index.isin(back_matched), ["OpenDT","StrategyMapped","Side","PremiumSold","PnL_rounded"]].sort_values("OpenDT")
            matched_pairs = matches.copy()
            matched_pairs["Strategy"] = matched_pairs["LiveIdx"].map(live_sel["StrategyMapped"])
            matched_pairs["Side"] = matched_pairs["LiveIdx"].map(live_sel["Side"])
            matched_pairs["OpenDT_Live"] = matched_pairs["LiveIdx"].map(live_sel["OpenDT"])
            matched_pairs["OpenDT_Back"] = matched_pairs["BackIdx"].map(back_sel["OpenDT"])
            matched_pairs["LivePnL"] = matched_pairs["LiveIdx"].map(live_sel["PnL_rounded"])
            matched_pairs["BackPnL"] = matched_pairs["BackIdx"].map(back_sel["PnL_rounded"])

        if not matches.empty:
            opposite = matched_pairs[
                ((matched_pairs["LivePnL"] > 0) & (matched_pairs["BackPnL"] <= 0)) |
                ((matched_pairs["LivePnL"] <= 0) & (matched_pairs["BackPnL"] > 0))
            ].copy()
        else:
            opposite = pd.DataFrame(columns=["Strategy","Side","OpenDT_Live","OpenDT_Back","LivePnL","BackPnL"])

        st.markdown("### Detail Tables")
        with st.expander("Trades only in Live (after tolerance match)"):
            st.markdown(lvb_render_table_html(live_only, ["PremiumSold"], [], ["PnL_rounded"], True, 1), unsafe_allow_html=True)
        with st.expander("Trades only in Backtest (after tolerance match)"):
            st.markdown(lvb_render_table_html(back_only, ["PremiumSold"], [], ["PnL_rounded"], True, 1), unsafe_allow_html=True)
        with st.expander("Matched pairs with opposite outcomes"):
            opp_display = opposite.drop(columns=[c for c in ["LiveIdx","BackIdx"] if c in opposite.columns])
            st.markdown(lvb_render_table_html(opp_display, [], [], ["LivePnL","BackPnL"], True, 1), unsafe_allow_html=True)
        with st.expander("All matched pairs (details)"):
            pairs_display = matched_pairs.drop(columns=[c for c in ["LiveIdx","BackIdx"] if c in matched_pairs.columns])
            st.markdown(lvb_render_table_html(pairs_display, [], [], ["LivePnL","BackPnL"], True, 1), unsafe_allow_html=True)

        st.markdown("### Downloads")
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("Download Comparison CSV", lvb_df_to_csv_bytes(comparison_stacked), file_name="comparison.csv")
        c2.download_button("Download Trades Only in Live CSV", lvb_df_to_csv_bytes(live_only), file_name="trades_only_live_tol.csv")
        c3.download_button("Download Trades Only in Backtest CSV", lvb_df_to_csv_bytes(back_only), file_name="trades_only_backtest_tol.csv")
        c4.download_button("Download Opposite Outcomes CSV", lvb_df_to_csv_bytes(opp_display if 'opp_display' in locals() else opposite), file_name="opposite_outcomes_tol.csv")


# --- TAB ENTRY-POINT ---
def live_vs_backtest_tab():
    return main()


if __name__ == "__main__":
    main()
