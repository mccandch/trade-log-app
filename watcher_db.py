#!/usr/bin/env python3
"""
Watcher (DB → Google Sheets) — per-user tab + account filter + skip-if-unchanged
- Reads local SQLite: data.db3 (read-only)
- Publishes sanitized rows to Google Sheets
- One tab per user: Raw_<USER_NAME>  (e.g., Raw_Chad)
- Idempotent per (User, Day): FileName = "db-YYYY-MM-DD"
- Keeps full timestamp in DateTime; Date is date-only for filters
- Account filtering:
    * By default exports ALL accounts
    * To restrict, set ACCOUNTS_INCLUDE = ["acctA","acctB"]
    * Or exclude a set with ACCOUNTS_EXCLUDE = ["acctX", ...]
"""

import os
import time
import hashlib
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
from google.oauth2.service_account import Credentials

# ================== CONFIG ==================
USER_NAME      = "Chad"   # Change to "Kelly" on Kelly's machine
DB_PATH        = r"C:\Users\Administrator\AppData\Local\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState\data.db3"

GOOGLE_SA_JSON = r"C:\Users\Administrator\AppData\Local\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState\service_account.json"
SHEET_NAME     = "TradeLog"
TAB_NAME       = None      # None -> use Raw_<USER_NAME>; or set a fixed name if you prefer

# Accounts filtering
ACCOUNTS_INCLUDE = []      # [] or None -> ALL accounts; else list exact Account values to include
ACCOUNTS_EXCLUDE = ["IB:U2604407", "IB:U16631465"]      # optional: accounts to exclude (ignored if INCLUDE is non-empty)

# Only read recent rows (set to 0/None for ALL)
DAYS_BACK      = 1

# Scan interval
POLL_SECONDS   = 15
# ============================================

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

BASE_COLS = ["User","Date","DateTime","FileName","BatchId","Strategy","TotalPremium","ProfitLoss"]

def tab_name():
    return TAB_NAME or f"Raw_{USER_NAME}"

def open_ws():
    creds = Credentials.from_service_account_file(GOOGLE_SA_JSON, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open(SHEET_NAME)
    name = tab_name()
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=200, cols=20)
        set_with_dataframe(ws, pd.DataFrame(columns=BASE_COLS))
    return sh, ws

def ensure_base_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in BASE_COLS:
        if c not in out.columns:
            out[c] = pd.Series(dtype="object")
    out["DateTime"] = pd.to_datetime(out.get("DateTime"), errors="coerce")
    out["Date"]     = pd.to_datetime(out.get("Date"), errors="coerce")
    return out[BASE_COLS]

# ----- Helpers -----
# .NET ticks → naive UTC datetime
def ticks_to_dt(ticks):
    if pd.isna(ticks): return pd.NaT
    try:
        return datetime(1,1,1) + timedelta(microseconds=int(ticks)/10)
    except Exception:
        return pd.NaT

def list_accounts(db_path: str):
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        acc = pd.read_sql_query('SELECT DISTINCT Account FROM "Trade" WHERE Account IS NOT NULL ORDER BY Account;', conn)
        conn.close()
        return acc["Account"].dropna().astype(str).tolist()
    except Exception:
        return []

def read_from_sqlite(db_path: str, days_back=None) -> pd.DataFrame:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)

    base_q = '''
      SELECT Account, Strategy, TotalPremium, ProfitLoss, DateOpened
      FROM "Trade"
      WHERE Strategy IS NOT NULL
    '''
    params = []
    q = base_q

    # Account include/exclude
    if ACCOUNTS_INCLUDE:
        placeholders = ",".join(["?"] * len(ACCOUNTS_INCLUDE))
        q += f" AND Account IN ({placeholders})"
        params.extend(ACCOUNTS_INCLUDE)
    elif ACCOUNTS_EXCLUDE:
        placeholders = ",".join(["?"] * len(ACCOUNTS_EXCLUDE))
        q += f" AND Account NOT IN ({placeholders})"
        params.extend(ACCOUNTS_EXCLUDE)

    df = pd.read_sql_query(q, conn, params=params)
    conn.close()

    # All rows in this watcher belong to USER_NAME
    df["User"] = USER_NAME

    # Time fields
    dt = df["DateOpened"].apply(ticks_to_dt)
    df["DateTime"] = pd.to_datetime(dt)
    df["Date"]     = df["DateTime"].dt.normalize()

    # Clean fields
    df["Strategy"]     = df["Strategy"].astype(str).str.strip()
    df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce").fillna(0.0)
    df["ProfitLoss"]   = pd.to_numeric(df["ProfitLoss"],  errors="coerce").fillna(0.0)

    # Optional lookback
    if days_back and days_back > 0:
        cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).tz_localize(None)
        df = df[df["DateTime"] >= cutoff]

    return df[["User","Date","DateTime","Strategy","TotalPremium","ProfitLoss"]].copy()

def day_batch_id(day_df: pd.DataFrame) -> str:
    h = hashlib.md5()
    for row in day_df[["DateTime","Strategy","TotalPremium","ProfitLoss"]].astype(str).itertuples(index=False):
        h.update("|".join(row).encode("utf-8"))
    return h.hexdigest()

def replace_day(ws, df_current: pd.DataFrame, user: str, day_ts: pd.Timestamp, day_df: pd.DataFrame) -> pd.DataFrame:
    df_current = ensure_base_cols(df_current)

    file_name = f"db-{day_ts.date().isoformat()}"
    batch_id  = day_batch_id(day_df)

    # ---------- OPTIMIZATION: skip write if BatchId already matches ----------
    existing_subset = df_current[(df_current["User"] == user) & (df_current["FileName"] == file_name)]
    if not existing_subset.empty:
        existing_ids = existing_subset["BatchId"].dropna().unique().tolist()
        if len(existing_ids) == 1 and existing_ids[0] == batch_id:
            print(f"No changes for {user} {file_name}; skipping write.")
            return df_current
    # ------------------------------------------------------------------------

    # Remove only this (User, FileName) then append new rows
    keep = ~((df_current["User"] == user) & (df_current["FileName"] == file_name))
    df_current = df_current.loc[keep].copy()

    new_rows = day_df.copy()
    new_rows["User"]     = user
    new_rows["FileName"] = file_name
    new_rows["BatchId"]  = batch_id
    new_rows = ensure_base_cols(new_rows)

    final = pd.concat([df_current, new_rows], ignore_index=True)
    set_with_dataframe(ws, final, include_index=False)
    return final

def scan_and_sync():
    if not os.path.exists(DB_PATH):
        print(f"DB not found: {DB_PATH}")
        return

    _, ws = open_ws()

    # One-time: list accounts so you know what to put in ACCOUNTS_INCLUDE
    accs = list_accounts(DB_PATH)
    if accs:
        print(f"Available accounts in DB: {', '.join(accs)}")

    df_existing = get_as_dataframe(ws, evaluate_formulas=True, dtype={"Date": str}).dropna(how="all")
    if df_existing.empty:
        df_existing = pd.DataFrame(columns=BASE_COLS)
    else:
        df_existing["Date"]     = pd.to_datetime(df_existing.get("Date"), errors="coerce")
        df_existing["DateTime"] = pd.to_datetime(df_existing.get("DateTime"), errors="coerce")
        df_existing["User"]     = df_existing.get("User","").astype(str).str.strip()
        df_existing["Strategy"] = df_existing.get("Strategy","").astype(str).str.strip()

    df = read_from_sqlite(DB_PATH, days_back=DAYS_BACK)
    if df.empty:
        print("No rows found in DB for the selected window.")
        return

    for day, day_df in df.groupby("Date"):
        try:
            df_existing = replace_day(ws, df_existing, USER_NAME, pd.to_datetime(day), day_df)
            print(f"Synced {USER_NAME} {day}: {len(day_df)} rows")
        except Exception as e:
            print(f"Skipping {day}: {e}")

def main():
    print(f"Starting DB watcher for {USER_NAME} → tab '{tab_name()}' ... Ctrl+C to stop.")
    while True:
        try:
            scan_and_sync()
        except Exception as e:
            print("Scan error:", e)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
