#!/usr/bin/env python3
import os
import sys
import csv
import json
import shlex
import tempfile
import datetime
import subprocess

# -------- helpers --------
def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.lower() in ("1", "true", "yes", "on")

def now_utc_date_str() -> str:
    # timezone-aware UTC (avoids DeprecationWarning for utcnow)
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

# -------- Google Sheets append --------
def append_csv_to_gsheet(csv_path: str, sheet_id: str | None, sheet_name: str | None, creds_json: str | None) -> None:
    """
    Append rows from csv_path to a Google Sheet.
    Prefers sheet_id (open_by_key). If not provided but sheet_name is, opens by title.
    Requires service account JSON (creds_json).
    """
    if not os.path.exists(csv_path):
        print(f"[GS] CSV not found, skip: {csv_path}", flush=True)
        return
    if not creds_json:
        print("[GS] GCP_SA_JSON missing → Sheets append skipped", flush=True)
        return
    if not sheet_id and not sheet_name:
        print("[GS] No GSHEET_ID/GSHEET_NAME provided → Sheets append skipped", flush=True)
        return

    # Write creds to a temp file
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        tf.write(creds_json.encode("utf-8"))
        tf.flush()
        tf.close()
        creds_path = tf.name

        # Lazy import only if needed
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except Exception as e:
            print(f"[GS] gspread/google-auth not installed: {e}", flush=True)
            return

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(creds)

        # Open spreadsheet
        sh = None
        if sheet_id:
            sh = gc.open_by_key(sheet_id)
        else:
            sh = gc.open(sheet_name)

        # Choose/ensure worksheet
        tab = os.getenv("GSHEET_TAB", "signals")
        titles = [ws.title for ws in sh.worksheets()]
        if tab in titles:
            ws = sh.worksheet(tab)
        else:
            ws = sh.add_worksheet(tab, rows=1000, cols=26)

        # Read CSV
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            print("[GS] CSV has no rows, nothing to append", flush=True)
            return

        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []

        # Write header if sheet is empty (A1 empty)
        try:
            existing_header = ws.get_values("A1:Z1")
        except Exception:
            existing_header = []

        if not existing_header or (len(existing_header) > 0 and all(c == "" for c in existing_header[0])):
            ws.append_row(header)
            print(f"[GS] header written to tab '{tab}'", flush=True)

        if data_rows:
            ws.append_rows(data_rows, value_input_option="RAW")
            print(f"[GS] appended {len(data_rows)} rows -> tab '{tab}'", flush=True)
        else:
            print("[GS] no data rows to append (only header present)", flush=True)

    except Exception as e:
        print(f"[GS] append failed: {e}", flush=True)
    finally:
        try:
            os.remove(tf.name)
        except Exception:
            pass

# -------- main runner --------
def main() -> int:
    # Resolve scanner path
    scanner = os.getenv("SCANNER_PATH", "futures_compression_accumulation_scanner_v2.py")
    if not os.path.exists(scanner):
        print(f"[ERR] scanner not found: {scanner}", flush=True)
        return 1

    # Build dated CSV name for this run
    today = now_utc_date_str()
    csv_path = f"scanner_signals_{today}.csv"

    # Read knobs from env
    pad            = os.getenv("PAD", "0.001")
    alert_thresh   = os.getenv("ALERT_THRESH", "50")  # adjust in render.yaml if needed
    min_box_15m    = os.getenv("MIN_BOX_15M", "0.10")
    min_box_1h     = os.getenv("MIN_BOX_1H", "0.20")
    min_box_4h     = os.getenv("MIN_BOX_4H", "0.18")
    min_exp_x      = os.getenv("MIN_EXP_X", "3")
    target_r       = os.getenv("TARGET_R", "").strip()
    vburst_mult    = os.getenv("VBURST_MULT", "1.25")
    entry_retest   = env_bool("ENTRY_RETEST", True)

    # Telegram
    tg_on    = env_bool("TG", True)
    tg_token = os.getenv("TG_TOKEN", "")
    tg_chat  = os.getenv("TG_CHAT", "")

    # Google Sheets (cron flavor: append after run)
    gsheet_id   = os.getenv("GSHEET_ID", "").strip()
    gsheet_name = os.getenv("GSHEET_NAME", "").strip()  # fallback
    gcp_sa_json = os.getenv("GCP_SA_JSON", "").strip()

    if gsheet_id and not gcp_sa_json:
        print("[GSHEET] GCP_SA_JSON missing → Sheets append will be skipped", flush=True)

    # Build scanner command (NO '--fast' here)
    cmd = [
        sys.executable, "-u", scanner,
        "--bb-window", "60",
        "--bb-percentile", "40",
        "--alert-thresh", str(alert_thresh),
        "--csv", csv_path,
        "--pad", str(pad),
        "--min-box-pct-15m", str(min_box_15m),
        "--min-box-pct-1h",  str(min_box_1h),
        "--min-box-pct-4h",  str(min_box_4h),
        "--min-exp-move-x",  str(min_exp_x),
        "--vburst-mult",     str(vburst_mult),
    ]
    if target_r:
        cmd += ["--target-r", str(target_r)]
    if entry_retest:
        cmd.append("--entry-retest")
    if tg_on and tg_token and tg_chat:
        cmd += ["--tg", "--tg-token", tg_token, "--tg-chat", tg_chat]

    # Log the command
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd), flush=True)

    # Execute and stream logs
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line.rstrip(), flush=True)
        p.wait()
        print(f"[DONE] exit={p.returncode}", flush=True)
        rc = p.returncode
    except Exception as e:
        print(f"[ERR] failed to run: {e}", flush=True)
        return 1

    # After run: append to Google Sheets if configured
    try:
        if gcp_sa_json and (gsheet_id or gsheet_name):
            append_csv_to_gsheet(csv_path, gsheet_id if gsheet_id else None, gsheet_name if gsheet_name else None, gcp_sa_json)
    except Exception as e:
        print(f"[GS] post-run append error: {e}", flush=True)

    return rc

if __name__ == "__main__":
    sys.exit(main())
