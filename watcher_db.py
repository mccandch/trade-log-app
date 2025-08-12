# watcher_db.py - Google Sheets sync from trading SQLite DB, including Right/Strike
# Run:  python watcher_db.py

import os
import time
import sqlite3
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ================== CONFIG ==================
DB_PATH = r"C:\Users\Administrator\AppData\Local\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState\data.db3"  # <-- CHANGE if needed
GOOGLE_SA_JSON = r"C:\Users\Administrator\AppData\Local\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState\service_account.json"  # <-- CHANGE
SHEET_NAME = "TradeLog"                 # Google Sheet file name
TAB_PREFIX = "Raw_"                     # tab will be f"{TAB_PREFIX}{USER_NAME}"
USER_NAME = "Chad"                      # <-- CHANGE per machine
POLL_SECONDS = 20                       # how often to re-scan
LOOKBACK_DAYS = 60                      # how many days back to pull
# Accounts filter (optional). If INCLUDE is non-empty, only those are synced.
ACCOUNTS_INCLUDE: List[str] = []
ACCOUNTS_EXCLUDE: List[str] = []
# ============================================


def ticks_to_dt_utc(ticks: int) -> Optional[datetime]:
    """
    Convert .NET ticks to UTC datetime (naive).
    .NET ticks: 100ns intervals since 0001-01-01.
    Unix epoch ticks = 621355968000000000
    """
    if ticks is None:
        return None
    try:
        sec = (int(ticks) - 621355968000000000) / 10_000_000
        return datetime.utcfromtimestamp(sec)
    except Exception:
        return None


def normalize_right(val):
    """Map various encodings to 'C' or 'P'. Returns None if unknown."""
    if val is None:
        return None
    s = str(val).strip().upper()
    if s in {"C", "CALL", "CALLS", "1"}:
        return "C"
    if s in {"P", "PUT", "PUTS", "2"}:
        return "P"
    if s.isdigit():
        return "C" if s == "1" else ("P" if s == "2" else None)
    return None


def pick_short_leg(cur: sqlite3.Cursor, order_id: Optional[int]) -> Tuple[Optional[str], Optional[float]]:
    """
    Return (Right, Strike) for the most representative short leg from the opening order.
    Heuristic:
      1) Prefer legs with Qty < 0 (short).
      2) If multiple, choose the one with largest abs(Qty).
      3) If none are short, pick the first leg.
    """
    if not order_id:
        return (None, None)
    rows = cur.execute(
        'SELECT PutCall, Strike, Qty FROM OrderLeg WHERE OrderID=?',
        (order_id,)
    ).fetchall()
    if not rows:
        return (None, None)

    short = [r for r in rows if (r[2] is not None and float(r[2]) < 0)]
    pick = short[0] if short else rows[0]
    if short:
        short.sort(key=lambda r: abs(float(r[2])), reverse=True)
        pick = short[0]

    right = normalize_right(pick[0])
    try:
        strike = float(pick[1]) if pick[1] is not None else None
    except Exception:
        strike = None
    return (right, strike)


def open_sheet_and_tab():
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
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=20)
        # initialize header
        header = ["User","DateTime","Source","TradeID","Account","Strategy","Right","Strike","Premium","PnL","BatchID"]
        ws.update("A1", [header])
    return sh, ws


def get_existing_trade_ids(ws) -> set:
    """Return set of existing TradeID values in the sheet (as strings)."""
    try:
        records = ws.get_all_records()
        return {str(r.get("TradeID")) for r in records if r.get("TradeID") not in (None, "")}
    except Exception:
        return set()


def accounts_ok(acct: Optional[str]) -> bool:
    if acct is None:
        return True
    if ACCOUNTS_INCLUDE:
        return acct in ACCOUNTS_INCLUDE
    if ACCOUNTS_EXCLUDE:
        return acct not in ACCOUNTS_EXCLUDE
    return True


def pull_trades_df(con: sqlite3.Connection) -> pd.DataFrame:
    """
    Pull recent trades and build a dataframe with the columns we need.
    """
    cur = con.cursor()
    q = """
    SELECT TradeID, Account, Strategy, DateOpened, TotalPremium, ProfitLoss, OrderIDOpen
    FROM Trade
    WHERE DateOpened >= ?
    ORDER BY TradeID ASC
    """
    now_utc = datetime.utcnow()
    cutoff_dt = now_utc - pd.Timedelta(days=LOOKBACK_DAYS)
    cutoff_ticks = 621355968000000000 + int(cutoff_dt.timestamp() * 10_000_000)

    rows = cur.execute(q, (cutoff_ticks,)).fetchall()

    items = []
    src = os.path.basename(DB_PATH)

    for TradeID, Account, Strategy, DateOpened, TotalPremium, ProfitLoss, OrderIDOpen in rows:
        if not accounts_ok(Account):
            continue

        dt = ticks_to_dt_utc(DateOpened)
        right, strike = pick_short_leg(cur, OrderIDOpen)

        # Batch by calendar date (helps if you want to track/replace per-day later)
        batch_id = f"db-{USER_NAME}-{dt.date().isoformat()}" if dt else None

        items.append({
            "User": USER_NAME,
            "DateTime": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "",
            "Source": src,
            "TradeID": TradeID,
            "Account": Account,
            "Strategy": Strategy or "",
            "Right": right or "",
            "Strike": strike,
            "Premium": float(TotalPremium) if TotalPremium is not None else None,
            "PnL": float(ProfitLoss) if ProfitLoss is not None else None,
            "BatchID": batch_id,
        })

    return pd.DataFrame(items)


def append_new_rows(ws, df: pd.DataFrame):
    """
    Append rows that have not yet been written (by TradeID).
    If the header row is missing new columns, we ensure it's present.
    """
    if df.empty:
        print("No rows found in DB for the lookback window.")
        return

    header = ["User","DateTime","Source","TradeID","Account","Strategy","Right","Strike","Premium","PnL","BatchID"]
    try:
        first_row = ws.row_values(1)
    except Exception:
        first_row = []
    if [h.strip() for h in first_row] != header:
        ws.update("A1", [header])  # rewrite header if needed

    existing_ids = get_existing_trade_ids(ws)
    new_df = df[~df["TradeID"].astype(str).isin(existing_ids)].copy()

    if new_df.empty:
        print("No new trades to append (all TradeIDs already exist).")
        return

    new_df = new_df.fillna("")
    values = new_df[header].values.tolist()

    CHUNK = 500
    for start in range(0, len(values), CHUNK):
        ws.append_rows(values[start:start+CHUNK], value_input_option="USER_ENTERED", table_range="A1")

    print(f"Appended {len(new_df)} new rows.")


def run_once():
    try:
        if not os.path.exists(DB_PATH):
            print(f"DB not found: {DB_PATH}")
            return
        _, ws = open_sheet_and_tab()
        con = sqlite3.connect(DB_PATH)
        try:
            df = pull_trades_df(con)
        finally:
            con.close()
        append_new_rows(ws, df)
    except Exception as e:
        print("Scan error:", e)


def main():
    print(f"Starting DB watcher for {USER_NAME} -> tab '{TAB_PREFIX}{USER_NAME}'. Ctrl+C to stop.")
    while True:
        run_once()
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
