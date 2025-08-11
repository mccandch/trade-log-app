# Streamlit Trade Stats MVP

## What you get
- `watcher.py`: runs on each PC, uploads your daily log to Google Sheets and **replaces** that day's rows when the file changes.
- `app.py`: Streamlit dashboard that reads the sheet and shows your daily % metrics (overall + by strategy) with date filtering.
- `requirements.txt`: packages for both.

## Google Sheets setup
1. Create a Google Cloud service account and download JSON credentials.
2. Create a Google Sheet named e.g. **TradeLog**.
3. Share that sheet with your service account's **client_email**.
4. In Streamlit Community Cloud, set **Secrets** like this:
```
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@...gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
[sheets]
sheet_name = "TradeLog"
tab_name = "Raw"
```

## Watcher config
Edit the CONFIG block at the top of `watcher.py`:
- `WATCH_FOLDER`: folder containing your logs (e.g., `C:\Users\you\Documents\TradeLogs`)
- `GOOGLE_SA_JSON`: path to your service account file on that PC.
- `SHEET_NAME`: your Google Sheet name (e.g., `TradeLog`)
- `USER_NAME`: `TraderA` or `TraderB`

Run: `python watcher.py` (it scans every 15s). It will create the header row on first run.

## Streamlit local run
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to a GitHub repo.
2. Deploy app from that repo.
3. Add the **Secrets** (same as above) in the Streamlit app settings.

## Notes
- The watcher performs **simple overwrite** per file: it deletes rows where `FileName==log-YYYY-M-D.txt` and re-uploads parsed rows for that file.
- The parser is tolerant of CSV/TXT with commas, tabs, or pipes. It expects at least columns for **Strategy**, **TotalPremium**, **ProfitLoss**, and a **Date** (or it will default to today's date).
- Later we can switch the storage to Supabase for UPSERTs and real-time.
