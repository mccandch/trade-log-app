# watcher_db.py — New Machine Setup

## What you need

- Python 3.10+ installed (https://www.python.org/downloads/)
- TradeAutomationToolbox installed and run at least once (so `data.db3` exists)
- The files from the zip: `watcher_db.py`, `service_account.json`, `run_watcher.bat`, `requirements_watcher.txt`

---

## Step 1 — Choose an install folder

Pick a folder, e.g. `C:\Users\YourName\Documents\TradeWatcher\`

Put these four files in that folder:
```
TradeWatcher\
  watcher_db.py
  service_account.json
  run_watcher.bat
  requirements_watcher.txt
```

---

## Step 2 — Create a Python virtual environment

Open a **Command Prompt** in that folder and run:

```cmd
python -m venv .venv
.venv\Scripts\pip install -r requirements_watcher.txt
```

---

## Step 3 — Edit watcher_db.py

Open `watcher_db.py` in Notepad or any editor. Find the **CONFIG** section near the top (lines 26–36) and update:

### `DB_PATH` (line 26)
Point this to your `data.db3`. On a fresh install of TradeAutomationToolbox it will be:
```python
DB_PATH = r"C:\Users\YOUR_USERNAME\AppData\Local\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState\data.db3"
```
Replace `YOUR_USERNAME` with your Windows username. To find the exact path, open File Explorer, paste this into the address bar and press Enter:
```
%LOCALAPPDATA%\Packages\TradeAutomationToolbox_f46cr67q31chc\LocalState
```
Then copy the full path and paste it into `DB_PATH`.

> `GOOGLE_SA_JSON` is already set to look for `service_account.json` in the same folder as `watcher_db.py` — no change needed.

### `USER_NAME` (line 30)
```python
USER_NAME = "YourName"   # controls which tab gets written: Raw_YourName
```
Each machine must have a **unique** name so they write to different tabs.

### `ACCOUNTS_EXCLUDE` (line 36)
Update this list if the new machine's brokerage accounts differ from Chad's:
```python
ACCOUNTS_EXCLUDE: List[str] = ["IB:U16631465", "IB:U2604407"]
```
Leave it empty (`[]`) to sync all accounts.

---

## Step 4 — Edit run_watcher.bat

Open `run_watcher.bat` and update the `BASE` variable to match your install folder:
```bat
set "BASE=C:\Users\YOUR_USERNAME\Documents\TradeWatcher"
```

---

## Step 5 — Run it

Double-click `run_watcher.bat`, or from a Command Prompt:
```cmd
.venv\Scripts\python -u watcher_db.py
```

On first run it does a full sync — this may take 10–30 seconds. After that it polls every 20 seconds and only pushes changes.

---

## Step 6 (optional) — Run on startup

To have the watcher start automatically with Windows:

1. Press `Win+R`, type `shell:startup`, press Enter
2. Create a shortcut to `run_watcher.bat` in that folder

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `data.db3` not found | Run TradeAutomationToolbox first; confirm the path in `DB_PATH` |
| `service_account.json` error | Make sure the file is in the same folder as `watcher_db.py` |
| Wrong tab in Google Sheets | Check `USER_NAME` — each machine needs a unique value |
| `ModuleNotFoundError` | Re-run `.venv\Scripts\pip install -r requirements_watcher.txt` |
| Watcher already running error | Check Task Manager for an existing `python` process running `watcher_db.py` |
