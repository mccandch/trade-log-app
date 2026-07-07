# Trade Log Analyzer

Documentation for the **Trade Log Analyzer** tab ([tabs/trade_log_analyzer.py](../tabs/trade_log_analyzer.py)),
added 2026-07-07. Upload an options trade-log CSV (OptionOmega/backtester-style
export for SPX/RUT strategies) and get a full strategy research dashboard.

## Integration

- `app.py` renders it as the third `st.tabs()` tab via
  `tabs.trade_log_analyzer.render_trade_log_analyzer_tab()`.
- All widget keys are prefixed `trade_log_` to avoid collisions with the other tabs.
- All filter controls live **inside the tab** (not the sidebar) so they don't
  show while using Daily Compare / Live vs Backtest.
- Parsing is cached with `@st.cache_data` keyed on the uploaded file bytes
  (`prepare_trade_log`).

## Expected CSV format

Required columns (upload is rejected with an error listing what's missing):
`Date Opened`, `Time Opened`, `Date Closed`, `Time Closed`, `P/L`, `Strategy`.

Optional (metrics that need them are skipped when absent): `Premium`,
`Margin Req.`, `Reason For Close`, `Underlying`, `No. of Contracts`,
`Opening/Closing Commission`, `Legs`, `Gap`, `Movement`, `Max Profit`,
`Max Loss`, and more.

Header aliases are normalized in `COLUMN_ALIASES`, e.g. real exports say
`Opening Commissions + Fees` → `Opening Commission`,
`Opening Short/Long Ratio` → `Ratio`.

Numeric parsing (`parse_numeric`) tolerates `$1,234.56`, `-$1,234.56`,
`($1,234.56)`, `12.5%`, blanks, and garbage (→ NaN). Rows missing P/L or an
unparseable open date-time are dropped, with a count shown under
Upload & Validation.

### Data-format quirks (important)

- **`Premium` is per contract** (net credit × 100), while **`P/L` is for the
  whole position**. Nearly all trades in these logs are 2-lots. Any metric
  relating P/L to premium must scale premium by `No. of Contracts`.
  Verified against the trades: for expired trades,
  `P/L ≈ Premium × No. of Contracts − commissions`.
- `Legs` strings look like `2 Jul 8 5560 P STO 2.00 | 2 Jul 8 5460 P BTO 0.05`
  (strike then `P`/`C`), used as fallback for Put/Call classification.

## Derived classifications

- **Trade Side**: "Put"/"Call" from Strategy name, falling back to parsing
  `P`/`C` out of Legs; both types present → "Mixed/Other".
- **Volga Type**: Strategy containing "non volga" (any punctuation/case) →
  "Non-Volga"; otherwise containing "volga" → "Volga"; else "Unknown".
- **Time Bucket**: entry-time buckets, 30-min from 09:30–12:00 then hourly to
  16:00; anything else → "Outside RTH / Unknown".
- Also: Win/Loss, Open/Close DateTime, Open Date, Month, Year, Day of Week,
  Open Hour, Hold Minutes, quantile-based Premium/Margin buckets, fixed
  Gap/Movement % buckets.

## Key metric definitions

- **PCR (Premium Capture Rate)** — the user's primary stat:
  `100 × Σ P/L ÷ Σ (|Premium| × No. of Contracts)`.
  Shown immediately to the right of Win Rate % in every table that has one
  (daily, strategy, Volga, side, time-bucket, reason, monthly, yearly) plus
  the executive summary and rule-tester comparison. Sanity check vs the
  user's backtester: VOLGA: Puts ≈ 30% (a naive P/L ÷ Premium gave ~60% —
  that was a bug, fixed 2026-07-07).
- **Return on margin**: reported *per trade* (`avg P/L ÷ avg margin`), not
  total-P/L ÷ avg-margin (which produced a meaningless 4,757% — deliberate
  deviation from the original spec).
- **Max drawdown**: from the equity curve of trades ordered by close time,
  seeded with starting capital (default $1,100,000).
- **CAGR**: annualized from first open to last close; **MAR** = CAGR ÷ |max DD %|.
- **Stop-loss damage ratio**: |losses on stop-loss trades| ÷ total winning-trade gains.

## Section layout (expanders, in order)

Upload & Validation → Filters & Settings → **Rule Tester (expanded, at top —
user preference)** → Executive Summary (expanded) → Daily Analysis →
Trades-per-Day Buckets → Strategy Breakdown → Volga vs Non-Volga →
Calls vs Puts → Time-of-Day & Day-of-Week → Reason For Close → Risk & Margin
(incl. concurrent-exposure estimate from open/close margin events) →
Monthly & Yearly → Outliers → Export.

Filters: starting capital, date range, Strategy/Volga/Side/Underlying/Reason
multiselects, exclude-breakeven, show-raw. **Min/Max P/L filters were removed
on user request** (2026-07-07) — don't re-add them.

## Rule Tester semantics

Simulates entry-time window, strategy/Volga/side exclusions, min premium,
max margin, drop-whole-days-over-X-trades, and two intraday cutoffs
("stop after first stop loss", daily loss/profit cutoffs). The cutoffs find
the first *close* that crosses the threshold each day and drop trades
**opened after** that moment — trades already open at the cutoff still count
(mirrors live behavior). Results shown as an Original vs Filtered table
(`comparison_stats`).

## Export

- CSV downloads: filtered trades, daily summary, strategy summary.
- **Google Sheets report** (replaced the original Excel export on user
  request): `build_google_sheets_report` creates a spreadsheet named
  `Trade Log Report <timestamp>` with one worksheet per summary table
  (raw trades optional via checkbox), shares it as writer with the email in
  the text box (defaults to chad.mccandless@gmail.com), and displays the URL.
  Auth reuses the app's service account from
  `st.secrets["gcp_service_account"]` (present in the global
  `~/.streamlit/secrets.toml`).

## Charts

Plotly `graph_objects` with Streamlit's plotly theme handling light/dark
chrome. Colors from the validated dataviz reference palette: single-series
blue `#2a78d6`; sign-colored bars blue (gain) / red `#e34948` (loss).

## Testing

No committed test suite; verification was done with two scratchpad scripts
(recreate as needed):
1. **Smoke test** — runs every calculation function against a real log CSV and
   asserts totals reconcile across all breakdowns and PCR sits next to Win Rate %.
2. **Full-render test** — bare-mode Streamlit run that monkeypatches
   `st.file_uploader` to return the sample CSV so every section executes
   (this caught a real `height=None` dataframe bug).

Reference data: `C:\Users\Administrator\Downloads\VOLGA 2 year trade log.csv`
(13,401 trades, 2024-07 → 2026-06, total P/L $935,900.08, overall PCR 21.5%).

## Gotchas

- Installed Streamlit is newer than the `streamlit==1.37.0` pin in
  requirements.txt: it rejects `height=None` in `st.dataframe` and deprecates
  `use_container_width` (still functional). Avoid new-API-only params if the
  pin must keep working.
- `plotly` and `openpyxl` were added to requirements.txt for this tab
  (openpyxl now unused after the Excel→Sheets switch, but harmless).
