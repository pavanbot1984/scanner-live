#!/usr/bin/env bash
set -euo pipefail

# Ensure unbuffered Python so logs flush immediately
export PYTHONUNBUFFERED=1

# Decode Google Sheets credentials if provided
if [[ -n "${GSHEET_CREDS_B64:-}" ]]; then
  echo "$GSHEET_CREDS_B64" | base64 -d > /app/creds.json
  export GSHEET_CREDS="/app/creds.json"
elif [[ -n "${GCP_SA_JSON:-}" ]]; then
  # direct JSON string passed as env
  echo "$GCP_SA_JSON" > /app/creds.json
  export GSHEET_CREDS="/app/creds.json"
else
  echo "[GSHEET] No credentials provided, skipping Sheets append" >&2
fi

mkdir -p /data
cd /app

while true; do
  echo "[$(date -u +'%F %T')] running scanner…" >&1
  python -u futures_compression_accumulation_scanner_v2.py \
    --fast --bb-window 60 --bb-percentile 40 --alert-thresh 50 \
    --tg --tg-token "${TG_TOKEN}" --tg-chat "${TG_CHAT}" \
    --csv "/data/scanner_signals_$(date +%F).csv" \
    ${GSHEET_ID:+--gsheet-id "$GSHEET_ID"} \
    ${GSHEET_TAB:+--gsheet-tab "$GSHEET_TAB"} \
    ${GSHEET_CREDS:+--gsheet-creds "$GSHEET_CREDS"} \
    || echo "[WARN] scanner exited non-zero, continuing…" >&2

  echo "[$(date -u +'%F %T')] sleeping 15m…" >&1
  sleep 900
done
