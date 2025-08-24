# Futures Compression + Accumulation (RR=3) — Render Cron + Google Sheets

## What this does
- Scans **BTCUSDT, ETHUSDT, SOLUSDT** on **15m / 1h / 4h**.
- Uses **prev-bar box lock** + RR=3 TP logic.
- Sends **Telegram alerts** for valid compression/breakout signals.
- **Appends scan results to Google Sheets** every run.

## Files
- `futures_compression_accumulation_scanner_v2.py` — the scanner (RR=3 hotfix).
- `run_scan_and_log.py` — wrapper: runs scanner, saves CSV, logs to Google Sheets.
- `render.yaml` — Render Cron config (every 15 minutes).
- `requirements.txt` — Python deps.

## Setup steps
1. Copy your RR=3 scanner into this folder.
2. Set up a Google service account, enable Sheets API, and share your Sheet with that account.
3. On Render, set `GCP_SA_JSON` (contents of the JSON key) as a secret.
4. Also set `TG_TOKEN` and `TG_CHAT` for Telegram alerts.
5. Deploy this repo to Render Cron. Logs go to Google Sheets + Telegram.

## Local test
```bash
pip install -r requirements.txt
$env:GCP_SA_JSON = Get-Content -Raw .\sa.json   # PowerShell
python run_scan_and_log.py
```
