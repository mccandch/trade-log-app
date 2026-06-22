# watcher_db.py — efficient incremental sync to Google Sheets
#
# Fast path  (every POLL_SECONDS):  appends new rows, patches changed recent rows only.
#   → Sheet is NEVER cleared during fast-path; no flicker, no race window.
#
# Full sync  (startup + every FULL_SYNC_INTERVAL_SECS):  full reconciliation of the
#   last LOOKBACK_DAYS, preserving older history rows already on the sheet.
#
# Run directly (unbuffered):
#   python -u watcher_db.py

import os
import time
import sqlite3
import warnings
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials

# ================== CONFIG ==================
DB_PATH = r"C:\Users\Administrator\AppData\Local\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState\data.db3"
GOOGLE_SA_JSON = r"C:\Users\Administrator\AppData\Local\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState\service_account.json"
SHEET_NAME = "TradeLog"
TAB_PREFIX = "Raw_"
USER_NAME = "Chad"                        # <-- CHANGE per machine
POLL_SECONDS = 20                         # seconds between incremental syncs
LOOKBACK_DAYS = 60                        # full-sync window (days)
INCREMENTAL_DAYS = 2                      # fast-path looks back this many days
FULL_SYNC_INTERVAL_SECS = 86_400         # redo full sync every 24 h
ACCOUNTS_INCLUDE: List[str] = []
ACCOUNTS_EXCLUDE: List[str] = ["IB:U16631465", "IB:U2604407"]
# ============================================

warnings.filterwarnings("ignore", message=r".*utcnow\(\) is deprecated.*")
warnings.filterwarnings("ignore", message=r".*utcfromtimestamp\(\) is deprecated.*")

HEADER = [
    "User", "DateTime", "Source", "TradeID", "Account",
    "Strategy", "Right", "Strike", "Premium", "PnL", "BatchID",
]
_COL_END = chr(ord("A") + len(HEADER) - 1)  # "K"
_TID_IDX = HEADER.index("TradeID")           # 3 (0-based)

# ---- Module-level sync state (reset on process restart → forces full sync at startup) ----
_row_index: Dict[str, int] = {}    # TradeID (str) → 1-based sheet row number
_val_cache: Dict[str, tuple] = {}  # TradeID (str) → tuple[str, ...] for change detection
_next_data_row: int = 2            # next sheet row available for append
_last_full_sync: Optional[datetime] = None


# ================== HELPERS ==================

def ticks_to_dt_utc(ticks: Optional[int]) -> Optional[datetime]:
    """Convert .NET ticks to UTC datetime.

    Pending/unfilled trades carry .NET DateTime.MaxValue as a sentinel
    (ticks == 3155378975999999999). Python's datetime can represent year 9999
    so without this guard the sentinel would pass through, overflow pandas
    Timestamp range downstream, silently become NaT, and drop the trade.
    """
    if ticks in (None, ""):
        return None
    try:
        ticks = int(ticks)
        if ticks >= 3155378975000000000:   # DateTime.MaxValue sentinel
            return None
        sec = (ticks - 621355968000000000) / 10_000_000
        return datetime.fromtimestamp(sec, timezone.utc)
    except Exception:
        return None


def normalize_right(val) -> Optional[str]:
    s = "" if val is None else str(val).strip().upper()
    if s in {"C", "CALL", "CALLS", "1"}:
        return "C"
    if s in {"P", "PUT", "PUTS", "2"}:
        return "P"
    return None


def accounts_ok(acct: Optional[str]) -> bool:
    if ACCOUNTS_INCLUDE:
        return acct in ACCOUNTS_INCLUDE
    if ACCOUNTS_EXCLUDE:
        return acct not in ACCOUNTS_EXCLUDE
    return True


# ================== SHEET AUTH / OPEN ==================

def open_sheet_and_tab() -> gspread.Worksheet:
    creds = Credentials.from_service_account_file(
        GOOGLE_SA_JSON,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    gc = gspread.authorize(creds)
    sh = gc.open(SHEET_NAME)
    tab_name = f"{TAB_PREFIX}{USER_NAME}"
    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=len(HEADER))
        ws.update(values=[HEADER], range_name="A1")
        try:
            ws.freeze(rows=1)
        except Exception:
            pass
    return ws


def ensure_header(ws) -> None:
    try:
        first = ws.row_values(1)
    except Exception:
        return
    want = [h.strip() for h in HEADER]
    got = [c.strip() for c in first][: len(want)]
    if got != want:
        ws.update(values=[HEADER], range_name="A1")
        try:
            ws.freeze(rows=1)
        except Exception:
            pass


# ================== DB PULL ==================

def pick_short_leg(
    cur: sqlite3.Cursor, order_id: Optional[int], hint_right: Optional[str] = None
) -> Tuple[Optional[str], Optional[float]]:
    """Return (right, strike) of the short leg for this order.

    hint_right: if 'C' or 'P', restrict search to legs of that type first.
    This matters for iron condors where both spread trades share one OrderIDOpen
    but one is a call spread and the other is a put spread.
    """
    if not order_id:
        return (None, None)
    rows = cur.execute(
        "SELECT PutCall, Strike, Qty FROM OrderLeg WHERE OrderID=?", (order_id,)
    ).fetchall()
    if not rows:
        return (None, None)
    # Restrict to matching legs when we have a type hint; fall back to all legs.
    if hint_right in ("C", "P"):
        candidates = [r for r in rows if normalize_right(r[0]) == hint_right] or rows
    else:
        candidates = rows
    short = [r for r in candidates if r[2] is not None and float(r[2]) < 0]
    pick = (
        sorted(short, key=lambda r: abs(float(r[2])), reverse=True)[0]
        if short
        else candidates[0]
    )
    right = normalize_right(pick[0])
    try:
        strike = float(pick[1]) if pick[1] is not None else None
    except Exception:
        strike = None
    return right, strike


def pull_trades_df(con: sqlite3.Connection, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Pull trades from the last `days` days (with Year/Month/Day sentinel fallback)."""
    cur = con.cursor()
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - pd.Timedelta(days=days)
    cutoff_ticks = 621355968000000000 + int(cutoff.timestamp() * 10_000_000)
    cutoff_ymd = int(cutoff.strftime("%Y%m%d"))

    rows = cur.execute(
        """
        SELECT TradeID, Account, Strategy, DateOpened,
               TotalPremium, ProfitLoss, OrderIDOpen, Year, Month, Day,
               ShortCall, ShortPut
        FROM Trade
        WHERE DateOpened >= ?
           OR (Year IS NOT NULL AND Year * 10000 + Month * 100 + Day >= ?)
        ORDER BY TradeID ASC
        """,
        (cutoff_ticks, cutoff_ymd),
    ).fetchall()

    src = os.path.basename(DB_PATH)
    items = []
    for TradeID, Account, Strategy, DateOpened, TotalPremium, ProfitLoss, OrderIDOpen, Year, Month, Day, ShortCall, ShortPut in rows:
        if not accounts_ok(Account):
            continue
        dt = ticks_to_dt_utc(DateOpened)
        if dt is None and Year and Month and Day:
            try:
                dt = datetime(int(Year), int(Month), int(Day), 9, 30, 0, tzinfo=timezone.utc)
            except Exception:
                pass

        # Use ShortCall / ShortPut directly — these are always correct per trade,
        # even when two trades (call spread + put spread) share the same OrderIDOpen.
        if ShortCall is not None:
            right, strike = "C", float(ShortCall)
        elif ShortPut is not None:
            right, strike = "P", float(ShortPut)
        else:
            right, strike = pick_short_leg(cur, OrderIDOpen)

        batch_id = f"db-{USER_NAME}-{dt.date().isoformat()}" if dt else None
        items.append({
            "User":     USER_NAME,
            "DateTime": dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if dt else "",
            "Source":   src,
            "TradeID":  TradeID,
            "Account":  Account,
            "Strategy": Strategy or "",
            "Right":    right or "",
            "Strike":   strike,
            "Premium":  float(TotalPremium) if TotalPremium is not None else None,
            "PnL":      float(ProfitLoss)   if ProfitLoss   is not None else None,
            "BatchID":  batch_id,
        })

    df = pd.DataFrame(items)
    for c in HEADER:
        if c not in df.columns:
            df[c] = pd.NA
    return df[HEADER].copy()


# ================== SHEET READ ==================

def read_sheet_df(ws) -> pd.DataFrame:
    try:
        df = get_as_dataframe(ws, evaluate_formulas=True, header=0).dropna(how="all")
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=HEADER)
    lower_map = {h.lower(): h for h in HEADER}
    rename = {}
    for c in list(df.columns):
        name = str(c).strip()
        if name in HEADER:
            rename[c] = name
        elif name.lower() in lower_map:
            rename[c] = lower_map[name.lower()]
    if rename:
        df = df.rename(columns=rename)
    for c in HEADER:
        if c not in df.columns:
            df[c] = pd.NA
    # Normalize TradeID: Google Sheets returns integers as floats ("12345.0").
    # Strip the trailing ".0" so IDs match the DB's integer strings ("12345").
    if "TradeID" in df.columns:
        def _norm_tid(v):
            s = str(v).strip() if v is not None and not (isinstance(v, float) and pd.isna(v)) else ""
            return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s
        df["TradeID"] = df["TradeID"].apply(_norm_tid)
    return df.reindex(columns=HEADER, fill_value=pd.NA)


def history_preserving_merge(ws, db_df: pd.DataFrame) -> pd.DataFrame:
    """Keep sheet rows older than LOOKBACK_DAYS; replace the in-window portion with DB data."""
    sheet_df = read_sheet_df(ws)
    if "Account" in sheet_df.columns:
        sheet_df = sheet_df[sheet_df["Account"].apply(accounts_ok)]

    cutoff_naive = (datetime.now(timezone.utc) - pd.Timedelta(days=LOOKBACK_DAYS)).replace(tzinfo=None)
    sheet_dt = pd.to_datetime(sheet_df.get("DateTime", pd.Series(dtype="object")), errors="coerce")
    old_mask = sheet_dt.isna() | (sheet_dt < cutoff_naive)
    sheet_old = sheet_df[old_mask].copy()
    # Drop no-TradeID rows — they are junk/phantom rows, not real history.
    # All legitimate trades have auto-increment TradeIDs from the DB.
    if "TradeID" in sheet_old.columns and not sheet_old.empty:
        keep = sheet_old["TradeID"].notna() & (sheet_old["TradeID"].astype(str).str.strip() != "")
        sheet_old = sheet_old[keep].copy()

    for c in HEADER:
        if c not in db_df.columns:
            db_df[c] = pd.NA
    db_df = db_df.reindex(columns=HEADER, fill_value=pd.NA)

    combined = pd.concat([sheet_old, db_df], ignore_index=True)
    has_id = combined["TradeID"].notna() & (combined["TradeID"] != "")
    with_id = combined[has_id].drop_duplicates(subset=["TradeID"], keep="last")
    final = with_id.copy()   # no_id rows discarded — all real trades have IDs

    final_dt = pd.to_datetime(final.get("DateTime", pd.Series(dtype="object")), errors="coerce")
    final = (
        final.assign(_dt=final_dt)
             .sort_values(by=["_dt", "TradeID"], kind="stable", na_position="last")
             .drop(columns=["_dt"])
    )
    return final.reindex(columns=HEADER, fill_value=pd.NA)


# ================== SYNC STATE HELPERS ==================

def _val_key(vals: list) -> tuple:
    """Stable string tuple used for change detection."""
    return tuple(
        "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
        for v in vals
    )


def _row_vals(item: dict) -> list:
    """Convert a trade dict to an ordered HEADER value list for the Sheets API."""
    out = []
    for col in HEADER:
        v = item.get(col, "")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v = ""
        out.append(v)
    return out


def _init_state_from_df(final: pd.DataFrame) -> None:
    """Rebuild _row_index, _val_cache, and _next_data_row from a freshly written DataFrame."""
    global _row_index, _val_cache, _next_data_row
    _row_index = {}
    _val_cache = {}
    for i, row in enumerate(final.itertuples(index=False)):
        tid = str(getattr(row, "TradeID", "") or "")
        if tid:
            rnum = i + 2  # row 1 = header; first data row = 2
            vals = [getattr(row, col, "") for col in HEADER]
            _row_index[tid] = rnum
            _val_cache[tid] = _val_key(vals)
    _next_data_row = len(final) + 2


# ================== SYNC FUNCTIONS ==================

def full_sync(ws) -> None:
    """Append any DB rows that are missing from the sheet. Never bulk-overwrites.

    Google Sheets auto-reverts any large write operation (update, clear, batch_clear)
    when the sheet is open in a browser — it restores the previous version within
    seconds. The only safe write operation is append_rows, which the browser handles
    as a normal incremental change. So full_sync now works like a catch-up incremental:
    find which TradeIDs are absent from the sheet, append only those rows.
    """
    global _last_full_sync
    with sqlite3.connect(DB_PATH) as con:
        db_df = pull_trades_df(con, days=LOOKBACK_DAYS)

    # Read the sheet as it actually exists right now.
    sheet_df = read_sheet_df(ws)

    # Ensure the sheet has a header row.
    first_row = ws.row_values(1) if ws.row_count > 0 else []
    if not first_row or first_row[0] != HEADER[0]:
        ws.update(values=[HEADER], range_name="A1", value_input_option="USER_ENTERED")
        try:
            ws.freeze(rows=1)
        except Exception:
            pass
        sheet_df = pd.DataFrame(columns=HEADER)

    # Determine which TradeIDs are already in the sheet.
    if "TradeID" in sheet_df.columns and not sheet_df.empty:
        existing_ids = set(
            sheet_df["TradeID"].dropna().astype(str).str.strip()
        ) - {""}
    else:
        existing_ids = set()

    # Find DB rows whose TradeID is not yet in the sheet.
    db_id_col = db_df["TradeID"].astype(str).str.strip()
    missing_mask = (
        db_df["TradeID"].notna()
        & (db_id_col != "")
        & (~db_id_col.isin(existing_ids))
    )
    missing_df = db_df[missing_mask].copy()

    print(
        f"  full_sync: sheet={len(existing_ids)} known TradeIDs, "
        f"db={len(db_df)}, appending {len(missing_df)} missing rows",
        flush=True,
    )

    if not missing_df.empty:
        to_append = [_row_vals(r.to_dict()) for _, r in missing_df.iterrows()]
        # append_rows uses INSERT_ROWS mode and is not bounded by grid size.
        _CHUNK = 500
        for i in range(0, len(to_append), _CHUNK):
            chunk = to_append[i : i + _CHUNK]
            ws.append_rows(chunk, value_input_option="USER_ENTERED")
            print(f"    appended rows {i+1}-{i+len(chunk)}", flush=True)

    # Rebuild in-memory state from what the sheet actually contains now.
    updated_sheet_df = read_sheet_df(ws)

    # Revert detection: if the sheet has fewer rows than before we started,
    # an external process cleared our writes. Don't commit this broken state —
    # leave _last_full_sync = None so full_sync retries on the next cycle.
    if len(updated_sheet_df) < len(existing_ids):
        print(
            f"  full_sync WARN: sheet reverted during write "
            f"(expected >={len(existing_ids)}, got {len(updated_sheet_df)}) "
            f"— will retry next cycle",
            flush=True,
        )
        return

    _init_state_from_df(updated_sheet_df)
    _last_full_sync = datetime.now(timezone.utc)
    print(f"Full sync complete: {len(updated_sheet_df)} rows in '{ws.title}'.", flush=True)


def incremental_sync(ws) -> None:
    """Fast path: append new rows and patch changed recent rows only. Never clears the sheet."""
    global _next_data_row
    with sqlite3.connect(DB_PATH) as con:
        recent = pull_trades_df(con, days=INCREMENTAL_DAYS)

    to_append: list = []
    to_update: list = []   # (row_num, vals)

    for _, row in recent.iterrows():
        tid = str(row.get("TradeID") or "")
        if not tid:
            continue
        vals = _row_vals(row.to_dict())
        key = _val_key(vals)
        if tid not in _row_index:
            to_append.append(vals)
        elif _val_cache.get(tid) != key:
            to_update.append((_row_index[tid], vals))

    if to_append:
        resp = ws.append_rows(to_append, value_input_option="USER_ENTERED")
        # Use the API response to get the actual row where appending landed,
        # rather than trusting _next_data_row (which can be wrong after a revert).
        try:
            updated_range = resp["updates"]["updatedRange"]  # e.g. "Raw_Chad!A2740:K3239"
            range_part = updated_range.split("!")[1].split(":")[0]  # "A2740"
            first_row = int("".join(c for c in range_part if c.isdigit()))
        except Exception:
            first_row = _next_data_row  # fallback
        for i, vals in enumerate(to_append):
            tid = str(vals[_TID_IDX])
            if tid:
                _row_index[tid] = first_row + i
                _val_cache[tid] = _val_key(vals)
        _next_data_row = first_row + len(to_append)

    if to_update:
        ws.batch_update(
            [
                {"range": f"A{rnum}:{_COL_END}{rnum}", "values": [vals]}
                for rnum, vals in to_update
            ],
            value_input_option="USER_ENTERED",
        )
        for _, vals in to_update:
            tid = str(vals[_TID_IDX])
            _val_cache[tid] = _val_key(vals)

    if to_append or to_update:
        print(
            f"Incremental: +{len(to_append)} new, ~{len(to_update)} updated in '{ws.title}'.",
            flush=True,
        )
    else:
        print(f"Incremental: no changes in '{ws.title}'.", flush=True)


def sync_tab(ws) -> None:
    now = datetime.now(timezone.utc)
    if (
        _last_full_sync is None
        or (now - _last_full_sync).total_seconds() >= FULL_SYNC_INTERVAL_SECS
    ):
        full_sync(ws)
    else:
        incremental_sync(ws)


# ================== RUNNER ==================

def main():
    tab_name = f"{TAB_PREFIX}{USER_NAME}"
    print(f"Starting sync for {USER_NAME} -> '{tab_name}'. Ctrl+C to stop.", flush=True)
    ws = None
    while True:
        try:
            if ws is None:
                ws = open_sheet_and_tab()
                # Force a full sync on (re)connect so _row_index is always valid.
                global _last_full_sync
                _last_full_sync = None
            sync_tab(ws)
        except KeyboardInterrupt:
            print("Stopped.", flush=True)
            break
        except Exception as e:
            print(f"Error: {e} — will reconnect on next cycle.", flush=True)
            traceback.print_exc()
            ws = None   # force open_sheet_and_tab() + full sync on next iteration
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
