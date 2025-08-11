#!/usr/bin/env python3
"""
Watcher script
- Monitors a folder for latest daily log export (CSV or TXT)
- Parses file into DataFrame
- Replaces rows in Google Sheet for that file (idempotent "overwrite this day")
Setup:
1) Put your Google service account JSON on this machine (e.g., service_account.json).
2) Share your Google Sheet with the service account email.
3) Fill the CONFIG section below.
Run:
python watcher.py
"""
import os
import time
import hashlib
from datetime import datetime
from typing import Optional

import pandas as pd
from dateutil import parser as dtparser

import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
from google.oauth2.service_account import Credentials

# ================== CONFIG ==================
WATCH_FOLDER = r"C:\Path\To\Your\Logs"     # <-- CHANGE: folder with log-YYYY-M-D.txt and CSVs
GOOGLE_SA_JSON = r"C:\Path\To\service_account.json"  # <-- CHANGE
SHEET_NAME = "TradeLog"                    # <-- CHANGE: visible name of the Google Sheet file
TAB_NAME = "Raw"                           # we write everything here
FILE_GLOB_PREFIX = "log-"                  # files look like log-2025-8-11.txt
POLL_SECONDS = 15                          # scan interval
USER_NAME = "TraderA"                      # <-- CHANGE: identify whose machine this is
# ============================================

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

def open_gsheet():
    creds = Credentials.from_service_account_file(GOOGLE_SA_JSON, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open(SHEET_NAME)
    try:
        ws = sh.worksheet(TAB_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB_NAME, rows=100, cols=20)
        # create header
        set_with_dataframe(ws, pd.DataFrame(columns=[
            "User","Date","FileName","BatchId","Strategy","TotalPremium","ProfitLoss"
        ]))
    return sh, ws

def smart_read(path: str) -> pd.DataFrame:
    # Try CSV first with automatic separator
    try:
        df = pd.read_csv(path, engine="python")
        return df
    except Exception:
        pass
    # Try tab-delimited
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
        return df
    except Exception:
        pass
    # Try pipe-delimited
    try:
        df = pd.read_csv(path, sep="|", engine="python")
        return df
    except Exception:
        raise RuntimeError(f"Could not parse file: {path}")

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect at minimum columns for date, strategy, TotalPremium, ProfitLoss.
    We'll try common variants and coerce types.
    """
    # Copy to avoid warnings
    df = df.copy()

    # Date column: try 'Date' else 'OpenDate' else parse from text fields
    date_col = None
    for c in ["Date","TradeDate","OpenDate","CloseDate"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # give up and use today's date
        df["Date"] = pd.Timestamp.today().normalize()
    else:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        df["Date"] = pd.to_datetime(df["Date"])

    # Strategy column
    strat_col = None
    for c in ["Strategy","Template","TradeType","Label"]:
        if c in df.columns:
            strat_col = c
            break
    if strat_col is None:
        df["Strategy"] = "Unknown"
    else:
        df["Strategy"] = df[strat_col].astype(str)

    # Premium / PnL columns
    prem_col = None
    for c in ["TotalPremium","Premium","Credit","CollectedPremium"]:
        if c in df.columns:
            prem_col = c
            break
    pnl_col = None
    for c in ["ProfitLoss","PnL","NetPnL","Net_PnL"]:
        if c in df.columns:
            pnl_col = c
            break

    if prem_col is None or pnl_col is None:
        raise RuntimeError("Could not find premium/pnl columns. Found columns: " + ", ".join(df.columns))

    # Coerce numeric
    df["TotalPremium"] = pd.to_numeric(df[prem_col], errors="coerce").fillna(0.0)
    df["ProfitLoss"] = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)

    # Keep only needed columns
    keep = ["Date","Strategy","TotalPremium","ProfitLoss"]
    out = df[keep].copy()
    return out

def file_batch_id(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def replace_file_rows(ws, df_current: pd.DataFrame, file_name: str, batch_id: str, user: str, date_guess: Optional[pd.Timestamp]) -> None:
    # Drop previous rows for this file (by FileName)
    if not df_current.empty and "FileName" in df_current.columns:
        df_current = df_current[df_current["FileName"] != file_name].copy()

    # Append new rows
    df_to_add = df_current.copy()

    # Ensure columns
    base_cols = ["User","Date","FileName","BatchId","Strategy","TotalPremium","ProfitLoss"]
    for c in base_cols:
        if c not in df_to_add.columns:
            df_to_add[c] = pd.Series(dtype="object")

    # Prepare new data
    new_rows = normalized.copy()
    new_rows.insert(0, "User", user)
    new_rows.insert(1, "Date", new_rows["Date"] if "Date" in new_rows.columns else date_guess)
    new_rows.insert(2, "FileName", file_name)
    new_rows.insert(3, "BatchId", batch_id)

    # Concatenate and write back
    final = pd.concat([df_current, new_rows[base_cols]], ignore_index=True)
    set_with_dataframe(ws, final, include_index=False)

def scan_and_sync():
    sh, ws = open_gsheet()

    # Load existing
    df_existing = get_as_dataframe(ws, evaluate_formulas=True, dtype={"Date":str})
    if df_existing.shape[0] <= 1:
        df_existing = pd.DataFrame(columns=["User","Date","FileName","BatchId","Strategy","TotalPremium","ProfitLoss"])
    else:
        # Coerce types
        df_existing = df_existing.dropna(how="all")
        if "Date" in df_existing.columns:
            df_existing["Date"] = pd.to_datetime(df_existing["Date"], errors="coerce")

    # Find candidate files
    files = [f for f in os.listdir(WATCH_FOLDER) if f.startswith(FILE_GLOB_PREFIX) and (f.endswith(".txt") or f.endswith(".csv"))]
    # Process newest first
    files.sort(key=lambda x: os.path.getmtime(os.path.join(WATCH_FOLDER, x)), reverse=True)

    for fname in files:
        path = os.path.join(WATCH_FOLDER, fname)
        try:
            df = smart_read(path)
            normalized = normalize(df)
            batch_id = file_batch_id(path)
            replace_file_rows(ws, df_existing, fname, batch_id, USER_NAME, normalized["Date"].max() if "Date" in normalized.columns else None)
            print(f"Synced {fname} -> {len(normalized)} rows")
        except Exception as e:
            print(f"Skipping {fname}: {e}")

def main():
    print("Starting watcher... Ctrl+C to stop.")
    while True:
        try:
            scan_and_sync()
        except Exception as e:
            print("Scan error:", e)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
