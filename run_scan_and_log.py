#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, subprocess, sys, json, time
from datetime import datetime
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SHEET_NAME = os.environ.get("GSHEET_NAME", "Compression_Accum_Logs")
TABS = {"15m": "15m", "1h": "1h", "4h": "4h"}

PAD            = os.environ.get("PAD", "0.001")
ALERT_THRESH   = os.environ.get("ALERT_THRESH", "12")
MIN_BOX_15M    = os.environ.get("MIN_BOX_15M", "0.10")
MIN_BOX_1H     = os.environ.get("MIN_BOX_1H", "0.20")
MIN_BOX_4H     = os.environ.get("MIN_BOX_4H", "0.18")
MIN_EXP_X      = os.environ.get("MIN_EXP_X", "3")
TARGET_R       = os.environ.get("TARGET_R", "3")
VBURST_MULT    = os.environ.get("VBURST_MULT", "1.25")
VBURST_HIGHER  = os.environ.get("VBURST_MULT_HIGHER", "1.5")
ENTRY_RETEST   = os.environ.get("ENTRY_RETEST", "true").lower() in ("1","true","yes")

TG        = os.environ.get("TG", "true").lower() in ("1","true","yes")
TG_TOKEN  = os.environ.get("TG_TOKEN", "")
TG_CHAT   = os.environ.get("TG_CHAT", "")

SCANNER = os.environ.get("SCANNER_PATH", "futures_compression_accumulation_scanner_v2.py")

def get_gspread_client():
    sa_json = os.environ.get("GCP_SA_JSON", "")
    if not sa_json:
        print("[GSHEET] GCP_SA_JSON missing", file=sys.stderr)
        sys.exit(1)
    data = json.loads(sa_json)
    creds = Credentials.from_service_account_info(
        data, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return gspread.authorize(creds)

def append_df(ws, df: pd.DataFrame):
    if df.empty: return
    cols = ["time_ist","symbol","tf","is_compression","confidence","bias","last",
            "range_low","range_high","width","box_pct","seg","breakout",
            "vol_surge","rr_ok","entry","sl","tp1","tp2","tp3","risk","expected_move","run_ts"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    existing = ws.get_all_values()
    if not existing:
        ws.append_rows([df.columns.tolist()], value_input_option="USER_ENTERED")
    ws.append_rows(df.astype(object).values.tolist(), value_input_option="USER_ENTERED")

def run_one_tf(tf: str) -> pd.DataFrame:
    min_box = {"15m": MIN_BOX_15M, "1h": MIN_BOX_1H, "4h": MIN_BOX_4H}[tf]
    vburst  = VBURST_MULT if tf == "15m" else VBURST_HIGHER
    tmp_csv = f"/tmp/scan_{tf}_{int(time.time())}.csv"

    cmd = [
        sys.executable, SCANNER,
        "--breakout-watch",
        "--pad", PAD,
        "--min-exp-move-x", MIN_EXP_X,
        "--target-r", TARGET_R,
        "--vburst-mult", vburst,
        "--alert-thresh", ALERT_THRESH,
        "--csv", tmp_csv
    ]
    cmd += ["--min-box-pct-15m", MIN_BOX_15M, "--min-box-pct-1h", MIN_BOX_1H, "--min-box-pct-4h", MIN_BOX_4H]

    if ENTRY_RETEST:
        cmd.append("--entry-retest")
    if TG and TG_TOKEN and TG_CHAT:
        cmd += ["--tg", "--tg-token", TG_TOKEN, "--tg-chat", TG_CHAT]

    print(f"[RUN] {tf}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    try:
        df = pd.read_csv(tmp_csv)
    except Exception as e:
        print(f"[CSV] No data for {tf}: {e}", file=sys.stderr)
        return pd.DataFrame()
    if "tf" in df.columns:
        df = df[df["tf"].str.lower() == tf.lower()].copy()
    return df

def main():
    gclient = get_gspread_client()
    sh = gclient.open(SHEET_NAME)

    all_frames = []
    for tf in ("15m","1h","4h"):
        df = run_one_tf(tf)
        if df.empty: continue
        df["run_ts"] = datetime.utcnow().isoformat(timespec="seconds")
        all_frames.append(df)
        try:
            ws = sh.worksheet(TABS[tf])
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=TABS[tf], rows=2000, cols=30)
        append_df(ws, df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        try:
            ws_all = sh.worksheet("combined")
        except gspread.WorksheetNotFound:
            ws_all = sh.add_worksheet(title="combined", rows=5000, cols=30)
        append_df(ws_all, combined)

    print("[DONE] Logged to Google Sheets.")

if __name__ == "__main__":
    main()
