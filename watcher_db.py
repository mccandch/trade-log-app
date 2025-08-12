# watcher_db.py — history-preserving TRUE SYNC to Google Sheets
# Keeps all historical rows on the sheet and true-syncs only the last LOOKBACK_DAYS
# (updates, inserts, and deletions) from your local DB window.
#
# Run directly (unbuffered is nice for logs):
#   python -u watcher_db.py
# Or via your batch script.

import os
import time
import sqlite3
import warnings
import traceback
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
from google.oauth2.service_account import Credentials

# ================== CONFIG (leave this block as-is) ==================
DB_PATH = r"C:\Users\Administrator\AppData\Local\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState\data.db3"  # <-- CHANGE if needed
GOOGLE_SA_JSON = r"C:\Users\Administrator\AppData\Local\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState\service_account.json"  # <-- CHANGE
SHEET_NAME = "TradeLog"                 # Google Sheet file name
TAB_PREFIX = "Raw_"                     # tab will be f"{TAB_PREFIX}{USER_NAME}"
USER_NAME = "Chad"                      # <-- CHANGE per machine
POLL_SECONDS = 20                       # how often to re-scan
LOOKBACK_DAYS = 60                      # how many days back to pull
# Accounts filter (optional). If INCLUDE is non-empty, only those are synced.
ACCOUNTS_INCLUDE: List[str] = []
ACCOUNTS_EXCLUDE: List[str] = ["IB:U16631465", "IB:U2604407"]
# ====================================================================

warnings.filterwarnings("ignore", message=r".*utcnow\(\) is deprecated.*")
warnings.filterwarnings("ignore", message=r".*utcfromtimestamp\(\) is deprecated.*")

HEADER = [
    "User", "DateTime", "Source", "TradeID", "Account",
    "Strategy", "Right", "Strike", "Premium", "PnL", "BatchID",
]

# ----------------- Helpers -----------------
def ticks_to_dt_utc(ticks: Optional[int]) -> Optional[datetime]:
    """Convert .NET ticks to aware UTC datetime (or None)."""
    if ticks in (None, ""):
        return None
    try:
        sec = (int(ticks) - 621355968000000000) / 10_000_000
        return datetime.fromtimestamp(sec, timezone.utc)
    except Exception:
        return None

def normalize_right(val) -> Optional[str]:
    s = "" if val is None else str(val).strip().upper()
    if s in {"C","CALL","CALLS","1"}: return "C"
    if s in {"P","PUT","PUTS","2"}:   return "P"
    if s.isdigit(): return "C" if s == "1" else ("P" if s == "2" else None)
    return None

def accounts_ok(acct: Optional[str]) -> bool:
    if ACCOUNTS_INCLUDE:
        return acct in ACCOUNTS_INCLUDE
    if ACCOUNTS_EXCLUDE:
        return acct not in ACCOUNTS_EXCLUDE
    return True

def auth_client() -> gspread.Client:
    creds = Credentials.from_service_account_file(
        GOOGLE_SA_JSON,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)

def open_sheet_and_tab():
    gc = auth_client()
    sh = gc.open(SHEET_NAME)
    tab_name = f"{TAB_PREFIX}{USER_NAME}"
    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=20)
        ws.update("A1", [HEADER])
        try:
            ws.freeze(rows=1)
        except Exception:
            pass
    return sh, ws

def ensure_header(ws) -> None:
    """Ensure row 1 matches HEADER exactly (does not clear data)."""
    try:
        first = ws.row_values(1)
    except Exception:
        return
    want = [h.strip() for h in HEADER]
    got = [c.strip() for c in first][:len(want)]
    if got != want:
        ws.update("A1", [HEADER])
        try:
            ws.freeze(rows=1)
        except Exception:
            pass

# ----------------- DB Pull & Transform -----------------
def pick_short_leg(cur: sqlite3.Cursor, order_id: Optional[int]) -> Tuple[Optional[str], Optional[float]]:
    """
    Choose a representative short leg from the opening order:
      1) prefer Qty < 0, largest |Qty|
      2) else first leg
    Returns (Right, Strike)
    """
    if not order_id:
        return (None, None)
    rows = cur.execute(
        'SELECT PutCall, Strike, Qty FROM OrderLeg WHERE OrderID=?', (order_id,)
    ).fetchall()
    if not rows:
        return (None, None)
    short = [r for r in rows if (r[2] is not None and float(r[2]) < 0)]
    if short:
        short.sort(key=lambda r: abs(float(r[2])), reverse=True)
        pick = short[0]
    else:
        pick = rows[0]
    right = normalize_right(pick[0])
    try:
        strike = float(pick[1]) if pick[1] is not None else None
    except Exception:
        strike = None
    return (right, strike)

def pull_trades_df(con: sqlite3.Connection) -> pd.DataFrame:
    """
    Pull trades within LOOKBACK_DAYS from DB for USER_NAME.
    Expected tables/columns:
      Trade(TradeID, Account, Strategy, DateOpened, TotalPremium, ProfitLoss, OrderIDOpen)
      OrderLeg(OrderID, PutCall, Strike, Qty)
    """
    cur = con.cursor()

    # Lookback cutoff in ticks (UTC-aware)
    now_utc = datetime.now(timezone.utc)
    cutoff_dt = now_utc - pd.Timedelta(days=LOOKBACK_DAYS)
    cutoff_ticks = 621355968000000000 + int(cutoff_dt.timestamp() * 10_000_000)

    q = (
        """
        SELECT TradeID, Account, Strategy, DateOpened, TotalPremium, ProfitLoss, OrderIDOpen
        FROM Trade
        WHERE DateOpened >= ?
        ORDER BY TradeID ASC
        """
    )
    rows = cur.execute(q, (cutoff_ticks,)).fetchall()

    items = []
    src = os.path.basename(DB_PATH)
    for TradeID, Account, Strategy, DateOpened, TotalPremium, ProfitLoss, OrderIDOpen in rows:
        if not accounts_ok(Account):
            continue
        dt = ticks_to_dt_utc(DateOpened)
        right, strike = pick_short_leg(cur, OrderIDOpen)
        batch_id = f"db-{USER_NAME}-{dt.date().isoformat()}" if dt else None

        items.append({
            "User": USER_NAME,
            "DateTime": dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if dt else "",
            "Source": src,
            "TradeID": TradeID,
            "Account": Account,
            "Strategy": (Strategy or ""),
            "Right": (right or ""),
            "Strike": strike,
            "Premium": float(TotalPremium) if TotalPremium is not None else None,
            "PnL": float(ProfitLoss) if ProfitLoss is not None else None,
            "BatchID": batch_id,
        })

    df = pd.DataFrame(items)
    # Ensure required columns/order even if no rows
    for c in HEADER:
        if c not in df.columns:
            df[c] = pd.NA
    return df[HEADER].copy()

# ----------------- Sheet Read & History-Preserving Merge -----------------
def read_sheet_df(ws) -> pd.DataFrame:
    """
    Read entire tab into a DataFrame that ALWAYS has HEADER columns.
    Handles odd/missing headers gracefully.
    """
    try:
        df = get_as_dataframe(ws, evaluate_formulas=True, header=0).dropna(how="all")
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        # Return an empty frame with the exact schema expected downstream
        return pd.DataFrame(columns=HEADER)

    # Normalize column names: strip, case-insensitive
    normalized = {}
    lower_map = {h.lower(): h for h in HEADER}
    for c in list(df.columns):
        name = str(c).strip()
        if name in HEADER:
            normalized[c] = name
        else:
            low = name.lower()
            if low in lower_map:
                normalized[c] = lower_map[low]
    if normalized:
        df = df.rename(columns=normalized)

    # Ensure all required columns exist, *then* force order with reindex
    for c in HEADER:
        if c not in df.columns:
            df[c] = pd.NA

    # Reindex guarantees the columns are present and in order
    df = df.reindex(columns=HEADER, fill_value=pd.NA)
    return df

def history_preserving_merge(ws, db_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep sheet history older than LOOKBACK_DAYS; true-sync the window with DB.
    Returns final DataFrame with columns = HEADER.
    """
    sheet_df = read_sheet_df(ws)

    # DEBUG: if columns come back odd, log once
    try:
        print("Sheet cols seen:", list(sheet_df.columns), flush=True)
    except Exception:
        pass

    # Apply account filters to sheet history too
    if "Account" in sheet_df.columns:
        sheet_df = sheet_df[sheet_df["Account"].apply(accounts_ok)]

    # Cutoff naive for pandas comparison
    cutoff_aware = datetime.now(timezone.utc) - pd.Timedelta(days=LOOKBACK_DAYS)
    cutoff_naive = cutoff_aware.replace(tzinfo=None)

    # Treat blank/NaT DateTime as old → preserve (use .get to avoid KeyError)
    dt_series = sheet_df.get("DateTime", pd.Series(dtype="object"))
    sheet_dt = pd.to_datetime(dt_series, errors="coerce")
    old_mask = sheet_dt.isna() | (sheet_dt < cutoff_naive)
    sheet_old = sheet_df[old_mask].copy()

    # Ensure DB df has required columns/order
    for c in HEADER:
        if c not in db_df.columns:
            db_df[c] = pd.NA
    db_df = db_df.reindex(columns=HEADER, fill_value=pd.NA)

    # Combine: history + DB window, then de-dup by TradeID (DB wins)
    combined = pd.concat([sheet_old, db_df], ignore_index=True)

    has_id = combined["TradeID"].notna() & (combined["TradeID"] != "")
    with_id = combined[has_id].drop_duplicates(subset=["TradeID"], keep="last")
    no_id = combined[~has_id]
    final = pd.concat([with_id, no_id], ignore_index=True)

    # Stable sort by DateTime then TradeID
    final_dt = pd.to_datetime(final.get("DateTime", pd.Series(dtype="object")), errors="coerce")
    final = (
        final.assign(_dt=final_dt)
             .sort_values(by=["_dt", "TradeID"], kind="stable", na_position="last")
             .drop(columns=["_dt"])
    )

    # Force final schema
    final = final.reindex(columns=HEADER, fill_value=pd.NA)
    return final

# ----------------- TRUE SYNC writer -----------------
def sync_tab(ws) -> None:
    """Read sheet history, merge with DB window, and write back (keeps history)."""
    with sqlite3.connect(DB_PATH) as con:
        db_df = pull_trades_df(con)

    ensure_header(ws)  # ensure header row is correct
    final = history_preserving_merge(ws, db_df)

    # Clear + rewrite (history preserved via the merge)
    ws.clear()
    set_with_dataframe(
        ws, final.fillna(""),
        include_index=False,
        include_column_header=True,
        resize=True,
    )
    try:
        ws.freeze(rows=1)
    except Exception:
        pass
    print(f"Synced {len(final)} rows to '{ws.title}' (history preserved).", flush=True)

# ----------------- Runner -----------------
def run_once():
    try:
        if not os.path.exists(DB_PATH):
            print(f"DB not found: {DB_PATH}", flush=True)
            return
        _, ws = open_sheet_and_tab()
        ensure_header(ws)
        sync_tab(ws)
    except Exception as e:
        print("Sync error:", e, flush=True)
        traceback.print_exc()

def main():
    print(f"History-preserving true sync for {USER_NAME} → tab '{TAB_PREFIX}{USER_NAME}'. Ctrl+C to stop.", flush=True)
    if POLL_SECONDS and POLL_SECONDS > 0:
        while True:
            run_once()
            time.sleep(POLL_SECONDS)
    else:
        run_once()

if __name__ == "__main__":
    main()
