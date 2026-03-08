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


color_scale = alt.Scale(
    domain=["Live", "Backtest"],  # Match exactly how 'Source' appears in your data
    range=["#FFFFFF", "#B8A15A"]   # Live=white, Backtest=muted gold
)


# ---------------- Strategy mapping (YAML + fallback) ----------------
@st.cache_resource(show_spinner=False)
def lvb_load_strategy_rules(_cache_bust: int = 0) -> Dict[str, Dict]:
    """
    Loads mapping rules from strategy_mapping.yaml if present.

    Debug info (visible via lvb_debug_strategy_mapping_panel) is stored in st.session_state:
      __lvb_yaml_loaded, __lvb_yaml_path, __lvb_yaml_error, __lvb_rule_counts
    """
    data = None
    yaml_path = None
    yaml_error = None

    if yaml:
        candidates = [
            Path(__file__).parent.parent / "strategy_mapping.yaml",
            Path.cwd() / "strategy_mapping.yaml",
        ]
        for p in candidates:
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    yaml_path = str(p)
                except Exception as e:
                    yaml_error = f"{type(e).__name__}: {e}"
                    data = None
                break

    if data is None:
        # fallback defaults (keeps app usable if YAML missing)
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
        st.session_state["__lvb_yaml_loaded"] = False
        st.session_state["__lvb_yaml_path"] = None
    else:
        st.session_state["__lvb_yaml_loaded"] = True
        st.session_state["__lvb_yaml_path"] = yaml_path

    st.session_state["__lvb_yaml_error"] = yaml_error

    exact: Dict[str, Dict[str, tuple]] = {"live": {}, "backtest": {}}
    patterns: Dict[str, list] = {"live": [], "backtest": []}

    exact_ct = {"live": 0, "backtest": 0}
    patt_ct  = {"live": 0, "backtest": 0}
    bad_regex = []

    for r in (data or {}).get("rules", []) or []:
        src = "live" if str(r.get("source", "backtest")).lower().startswith("live") else "backtest"
        canon = r.get("canonical")
        match_val = r.get("match")
        ignore = bool(r.get("ignore", False))
        if not match_val:
            continue

        if r.get("regex", False):
            try:
                comp = re.compile(str(match_val))
                patterns[src].append((comp, canon, ignore))
                patt_ct[src] += 1
            except re.error as e:
                bad_regex.append({"source": src, "match": match_val, "error": str(e)})
        else:
            exact[src][str(match_val)] = (canon, ignore)
            exact_ct[src] += 1

    st.session_state["__lvb_rule_counts"] = {
        "exact_live": exact_ct["live"],
        "exact_backtest": exact_ct["backtest"],
        "regex_live": patt_ct["live"],
        "regex_backtest": patt_ct["backtest"],
        "bad_regex": bad_regex,
        "cache_bust": int(_cache_bust),
    }

    return {"exact": exact, "patterns": patterns}



def lvb_map_strategy_name(source: str, raw_name: str) -> Tuple[str, bool]:
    """
    Returns (canonical_name, ignore_flag).

    Debug info:
      __lvb_last_map, __lvb_last_map_result
    """
    cache_bust = int(st.session_state.get("__lvb_cache_bust", 0))
    rules = lvb_load_strategy_rules(cache_bust)

    s = "live" if str(source).lower().startswith("live") else "backtest"
    raw = "" if raw_name is None else str(raw_name)

    st.session_state["__lvb_last_map"] = {"source": s, "raw": raw}

    maybe = rules["exact"].get(s, {}).get(raw)
    if maybe:
        canon, ign = maybe
        st.session_state["__lvb_last_map_result"] = {"type": "exact", "canonical": canon, "ignore": ign}
        return (raw if canon is None else canon, ign)

    for patt, canon, ign in rules["patterns"].get(s, []):
        try:
            if patt.search(raw):
                st.session_state["__lvb_last_map_result"] = {"type": "regex", "pattern": patt.pattern, "canonical": canon, "ignore": ign}
                return (raw if canon is None else canon, ign)
        except re.error:
            pass

    st.session_state["__lvb_last_map_result"] = {"type": "none", "canonical": raw, "ignore": False}
    return raw, False


# ---------------- IO helpers ----------------
@st.cache_data(show_spinner=False)
def lvb_read_csv_file(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()

    def _read(source):
        # Peek at header to learn expected column count
        header_line = pd.read_csv(source, nrows=0)
        ncols = len(header_line.columns)
        # Reset source for full read
        if hasattr(source, "seek"):
            source.seek(0)
        # Truncate rows that have extra fields instead of skipping them
        return pd.read_csv(
            source,
            engine="python",
            on_bad_lines=lambda cols: cols[:ncols],
        )

    # ✅ handle cached bytes
    if isinstance(uploaded, (bytes, bytearray, memoryview)):
        return _read(io.BytesIO(bytes(uploaded)))

    # ✅ handle UploadedFile / file-like
    if hasattr(uploaded, "seek"):
        try:
            uploaded.seek(0)
        except Exception:
            pass
    return _read(uploaded)

def lvb_persist_upload_bytes(file_obj, state_key: str):
    """
    Persist uploaded file bytes into st.session_state[state_key].

    Returns:
        bytes | None
    """

    # Treat "no file" as no bytes
    if file_obj is None:
        st.session_state.pop(state_key, None)
        return None

    # Streamlit sometimes provides a DeletedFile sentinel when the user clears/replaces the upload.
    if type(file_obj).__name__ == "DeletedFile":
        st.session_state.pop(state_key, None)
        return None

    # If we already have bytes cached, return them
    cached = st.session_state.get(state_key)
    if isinstance(cached, (bytes, bytearray)) and len(cached) > 0:
        return cached

    # Only read from real UploadedFile objects
    try:
        data = file_obj.getvalue()  # UploadedFile supports this
    except Exception:
        # Some file-like objects support read()
        try:
            data = file_obj.read()
        except Exception:
            # Not readable (or was deleted mid-run)
            st.session_state.pop(state_key, None)
            return None

    st.session_state[state_key] = data
    return data

def lvb_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def _num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    # Always return a float Series; if column is missing, return a default-valued Series
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


# ---------------- Normalization + stats ----------------
def lvb_live_side_from_tradetype(trade_type: str) -> Optional[str]:
    if pd.isna(trade_type): return None
    return "Call" if "Call" in str(trade_type) else "Put"


def lvb_robust_side_from_text(s: str) -> Optional[str]:
    s0 = str(s).strip().lower()
    s0 = re.sub(r"\s+", " ", s0)
    # Handle "calls w put hedge" / "puts w call hedge" patterns:
    # the first option type before "w" is the primary direction.
    m = re.search(r"\b(call|put)s?\s+w\b", s0)
    if m:
        return "Call" if m.group(1) == "call" else "Put"
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
    df["PnL"] = _num_col(df, "ProfitLoss", 0.0)
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

    def _side_from_legs(legs: str) -> Optional[str]:
        """Derive Side from the Legs column (e.g. 'P STO' → Put, 'C STO' → Call).
        Returns None when both sides are present (combined legs like iron condors)
        so the row can be split later."""
        s = str(legs)
        has_call = bool(re.search(r"\bC\s+STO\b", s))
        has_put = bool(re.search(r"\bP\s+STO\b", s))
        if has_call and not has_put:
            return "Call"
        if has_put and not has_call:
            return "Put"
        return None

    if "Legs" in df.columns:
        df["Side"] = df["Legs"].apply(_side_from_legs)
        # Fall back to strategy name where Legs didn't resolve
        mask = df["Side"].isna()
        df.loc[mask, "Side"] = df.loc[mask, "Strategy"].apply(lvb_robust_side_from_text)
    else:
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

    # Split combined-leg rows (e.g. iron condors with both C STO and P STO)
    # into separate Call and Put rows so each side can match independently.
    if "Legs" in df.columns:
        _legs = df["Legs"].astype(str)
        _has_c = _legs.str.contains(r"\bC\s+STO\b", regex=True, na=False)
        _has_p = _legs.str.contains(r"\bP\s+STO\b", regex=True, na=False)
        _combined = _has_c & _has_p
        if _combined.any():
            new_rows = []
            for idx in df.index[_combined]:
                row = df.loc[idx]
                parts = [l.strip() for l in str(row["Legs"]).split("|")]
                call_parts = [l for l in parts if re.search(r"\bC\s+(STO|BTO)\b", l)]
                put_parts  = [l for l in parts if re.search(r"\bP\s+(STO|BTO)\b", l)]

                def _side_prem(legs):
                    cr, q = 0.0, 1
                    for lg in legs:
                        m = re.match(r"(\d+)\s+", lg)
                        if m:
                            q = max(q, int(m.group(1)))
                        for p in re.findall(r"STO\s+([0-9]*\.?[0-9]+)", lg):
                            cr += float(p)
                        for p in re.findall(r"BTO\s+([0-9]*\.?[0-9]+)", lg):
                            cr -= float(p)
                    return cr * q * 100

                cp, pp = _side_prem(call_parts), _side_prem(put_parts)
                tot = cp + pp
                cr = (cp / tot) if tot else 0.5
                pr = (pp / tot) if tot else 0.5

                for side, legs_list, ratio in [("Call", call_parts, cr), ("Put", put_parts, pr)]:
                    nr = row.copy()
                    nr["Side"] = side
                    nr["Legs"] = " | ".join(legs_list)
                    nr["PremiumSold"] = round(float(row.get("PremiumSold", 0)) * ratio, 2)
                    for c in ("PnL_stats", "PnL_gross", "PnL_rounded"):
                        if c in nr.index and pd.notna(row.get(c)):
                            nr[c] = round(float(row[c]) * ratio, 2)
                    new_rows.append(nr)
            df = pd.concat([df[~_combined], pd.DataFrame(new_rows)], ignore_index=True)

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
    matches = []          # list of (live_orig_idx, back_orig_idx)
    live_used = set()     # original live indices already consumed
    back_used = set()     # original back indices already consumed

    def _greedy(l_rows, b_rows):
        """Greedy nearest-match: for each live row (sorted), find closest
        unused backtest row within tolerance. Both inputs are lists of
        (orig_idx, OpenDT) tuples, pre-sorted by OpenDT."""
        b_start = 0
        for l_idx, dt_l in l_rows:
            if l_idx in live_used or pd.isna(dt_l):
                continue
            best_b = None
            best_delta = None
            for k in range(b_start, len(b_rows)):
                b_idx, dt_b = b_rows[k]
                if b_idx in back_used or pd.isna(dt_b):
                    continue
                delta = abs(dt_l - dt_b)
                if delta <= tol:
                    if best_delta is None or delta < best_delta:
                        best_b = b_idx
                        best_delta = delta
                elif dt_b > dt_l + tol:
                    break
            if best_b is not None:
                matches.append((l_idx, best_b))
                live_used.add(l_idx)
                back_used.add(best_b)
                # slide window past consumed entries at front
                while b_start < len(b_rows) and b_rows[b_start][0] in back_used:
                    b_start += 1

    def _to_pairs(df):
        """Convert df to list of (orig_index, OpenDT) sorted by OpenDT."""
        sub = df[["OpenDT"]].copy()
        sub = sub.sort_values("OpenDT")
        return list(zip(sub.index, sub["OpenDT"]))

    # ── Pass 1: exact (Strategy, Side) match ──
    live_keys = set(zip(live_df["StrategyMapped"], live_df["Side"]))
    back_keys = set(zip(back_df["StrategyMapped"], back_df["Side"]))
    keys = sorted((s, sd) for s, sd in live_keys | back_keys if pd.notna(sd))

    for strat, side in keys:
        L = live_df[(live_df["StrategyMapped"] == strat) & (live_df["Side"] == side)]
        B = back_df[(back_df["StrategyMapped"] == strat) & (back_df["Side"] == side)]
        if not L.empty and not B.empty:
            _greedy(_to_pairs(L), _to_pairs(B))

    # ── Pass 2: strategy-only fallback for remaining unmatched trades ──
    remaining = live_df[~live_df.index.isin(live_used)]
    if not remaining.empty:
        for strat in sorted(remaining["StrategyMapped"].dropna().unique()):
            L = remaining[remaining["StrategyMapped"] == strat]
            B = back_df[(back_df["StrategyMapped"] == strat) & ~back_df.index.isin(back_used)]
            if not L.empty and not B.empty:
                _greedy(_to_pairs(L), _to_pairs(B))

    return pd.DataFrame(matches, columns=["LiveIdx", "BackIdx"]) if matches else pd.DataFrame(columns=["LiveIdx", "BackIdx"])


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
            styler = styler.map(_color_posneg, subset=pd.IndexSlice[:, [c]])

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

    st.markdown("""
    <style>
    /* Target the selected items (tags) in multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #1E90FF !important;  /* DodgerBlue */
        color: white !important;               /* Text color */
    }

    /* Optional: Change hover color of the 'x' remove icon */
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


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

        if live_sel.empty and back_sel.empty:
            st.info(f"No trades found between {start_date} and {end_date}. Try expanding the date range.")

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

        expected_cols = ["Strategy","Source","Trades","Win Rate %","Premium Sold","Premium Captured","PCR %"]
        if stack_rows:
            comparison_stacked = pd.DataFrame(stack_rows)
        else:
            # ensure downstream code always sees the expected schema
            comparison_stacked = pd.DataFrame(columns=expected_cols)

        if not comparison_stacked.empty:
            comparison_stacked["Source"] = pd.Categorical(
                comparison_stacked["Source"], categories=["Live", "Backtest"], ordered=True
            )
            comparison_stacked = comparison_stacked.sort_values(["Strategy", "Source"]).reset_index(drop=True)


        # Totals rows
        def _summary_row(df: pd.DataFrame, src: str) -> Optional[dict]:
            # If there's no data or no 'Source' column, there's nothing to summarize
            if df is None or df.empty or "Source" not in df.columns:
                return None

            sub = df[df["Source"] == src]
            if sub.empty:
                return None

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
                        color=alt.Color("Source:N",scale = color_scale, legend=alt.Legend(orient="bottom")),
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
                        color=alt.Color("Source:N",scale=color_scale, legend=alt.Legend(orient="bottom")),
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
        # Scoped KPI colors: Backtest = muted gold, Live = white
        # 1) CSS (once)
        st.markdown("""
        <style>
        .metric-card { margin: 0 0 1rem 0; }
        .metric-card .metric-label {
        font-size: .9rem; line-height: 1.1; margin-bottom: .35rem;
        color: #FFFFFF; opacity: .85;
        }
        .metric-card .metric-value {
        font-size: 2.2rem; font-weight: 700; letter-spacing: .2px;
        }
        .metric-card.gold .metric-label,
        .metric-card.gold .metric-value { color: #B8A15A !important; } /* muted gold */
        </style>
        """, unsafe_allow_html=True)

        # 2) Helper
        def metric_html(label: str, value: str, gold: bool = False) -> str:
            cls = "metric-card gold" if gold else "metric-card"
            return f"""
        <div class="{cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        </div>
        """


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

        # Live KPIs (wrap in .live-kpis)
        l1, l2, l3, l4, l5 = st.columns(5)
        l1.metric("Live Trades", f"{live_trades:,}")
        l2.metric("Live Win Rate", f"{live_wr:.1f}%")
        l3.metric("Live Premium Sold", f"${live_sold:,.0f}")
        l4.metric("Live Premium Captured", f"${live_cap:,.0f}")
        l5.metric("Live PCR", f"{live_pcr:.1f}%")

        # Backtest KPIs (HTML so we can color them)
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.markdown(metric_html("Backtest Trades", f"{back_trades:,}", gold=True), unsafe_allow_html=True)
        b2.markdown(metric_html("Backtest Win Rate", f"{back_wr:.1f}%", gold=True), unsafe_allow_html=True)
        b3.markdown(metric_html("Backtest Premium Sold", f"${back_sold:,.0f}", gold=True), unsafe_allow_html=True)
        b4.markdown(metric_html("Backtest Premium Captured", f"${back_cap:,.0f}", gold=True), unsafe_allow_html=True)
        b5.markdown(metric_html("Backtest PCR", f"{back_pcr:.1f}%", gold=True), unsafe_allow_html=True)

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
            # Use unified stats PnL (net when available, else gross), then round once
            matched_pairs["LivePnL"] = matched_pairs["LiveIdx"].map(live_sel["PnL_stats"]).round(2)
            matched_pairs["BackPnL"] = matched_pairs["BackIdx"].map(back_sel["PnL_stats"]).round(2)
            
        if not matches.empty:
            EPS = 0.01  # treat +/- 1 cent as flat
            def _sign(s):
                return np.where(s > EPS, 1, np.where(s < -EPS, -1, 0))

            matched_pairs["LiveSign"] = _sign(matched_pairs["LivePnL"])
            matched_pairs["BackSign"] = _sign(matched_pairs["BackPnL"])
            opposite = matched_pairs[(matched_pairs["LiveSign"] * matched_pairs["BackSign"]) == -1].copy()
        else:
            opposite = pd.DataFrame(columns=["Strategy","Side","OpenDT_Live","OpenDT_Back","LivePnL","BackPnL"])

        st.markdown("### Detail Tables")
        live_only_pnl = pd.to_numeric(live_only.get("PnL_rounded", 0), errors="coerce").sum()
        back_only_pnl = pd.to_numeric(back_only.get("PnL_rounded", 0), errors="coerce").sum()
        with st.expander(f"Trades only in Live (after tolerance match) — PnL: ${live_only_pnl:,.2f}"):
            st.markdown(lvb_render_table_html(live_only, ["PremiumSold"], [], ["PnL_rounded"], True, 1), unsafe_allow_html=True)
        with st.expander(f"Trades only in Backtest (after tolerance match) — PnL: ${back_only_pnl:,.2f}"):
            st.markdown(lvb_render_table_html(back_only, ["PremiumSold"], [], ["PnL_rounded"], True, 1), unsafe_allow_html=True)
        with st.expander("Matched pairs with opposite outcomes"):
            opp_display = opposite.drop(columns=[c for c in ["LiveIdx","BackIdx"] if c in opposite.columns])
            st.markdown(lvb_render_table_html(opp_display, [], [], ["LivePnL","BackPnL"], True, 1), unsafe_allow_html=True)
        with st.expander("All matched pairs (details)"):
            pairs_display = matched_pairs.drop(columns=[c for c in ["LiveIdx","BackIdx","LiveSign","BackSign"] if c in matched_pairs.columns])
            pairs_display = pairs_display.sort_values(["Strategy", "OpenDT_Live", "Side"]).reset_index(drop=True)
            st.markdown(lvb_render_table_html(pairs_display, [], [], ["LivePnL","BackPnL"], True, 2), unsafe_allow_html=True)

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
