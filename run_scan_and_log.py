#!/usr/bin/env python3
import os, sys, json, shlex, subprocess, tempfile, datetime

def env_bool(name, default=False):
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.lower() in ("1", "true", "yes", "on")

def main():
    # 1) Resolve scanner path (env allows swapping script names)
    scanner = os.getenv("SCANNER_PATH", "futures_compression_accumulation_scanner_v2.py")
    if not os.path.exists(scanner):
        print(f"[ERR] scanner not found: {scanner}", flush=True)
        return 1

    # 2) Build a dated CSV filename (stateless container run)
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    csv_path = f"scanner_signals_{today}.csv"

    # 3) Telegram envs
    tg_on    = env_bool("TG", True)
    tg_token = os.getenv("TG_TOKEN", "")
    tg_chat  = os.getenv("TG_CHAT", "")

    # 4) Google Sheets creds: prefer raw JSON in GCP_SA_JSON (your render.yaml)
    #    We write it to a temp file and pass the path to the scanner flags you added.
    gsheet_id   = os.getenv("GSHEET_ID", os.getenv("GSHEET_NAME", ""))  # support either var
    gsheet_tab  = os.getenv("GSHEET_TAB", "signals")
    gcp_sa_json = os.getenv("GCP_SA_JSON", "")
    creds_path  = ""

    if gsheet_id and gcp_sa_json:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tf.write(gcp_sa_json.encode("utf-8"))
        tf.flush(); tf.close()
        creds_path = tf.name
        print("[GSHEET] service account json materialized", flush=True)
    elif gsheet_id and not gcp_sa_json:
        print("[GSHEET] GCP_SA_JSON missing → Sheets append will be skipped", flush=True)

    # 5) Pull the rest of your knobs from env (with sane defaults)
    #    These match keys shown in your render.yaml
    pad            = os.getenv("PAD", "0.001")
    alert_thresh   = os.getenv("ALERT_THRESH", "50")    # NOTE: your yaml has "12"; change there if intended
    min_box_15m    = os.getenv("MIN_BOX_15M", "0.10")
    min_box_1h     = os.getenv("MIN_BOX_1H", "0.20")
    min_box_4h     = os.getenv("MIN_BOX_4H", "0.18")
    min_exp_x      = os.getenv("MIN_EXP_X", "3")
    target_r       = os.getenv("TARGET_R", "3")
    vburst_mult    = os.getenv("VBURST_MULT", "1.25")
    vburst_mult_hi = os.getenv("VBURST_MULT_HIGHER", "1.5")
    entry_retest   = env_bool("ENTRY_RETEST", True)

    # 6) Build the command
    cmd = [
        sys.executable, "-u", scanner,
        "--fast",
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
    # Optional higher vburst multiplier if your script supports it:
    # (leave in cmd or remove if not used in your scanner args)
    # cmd += ["--vburst-mult-higher", str(vburst_mult_hi)]

    if entry_retest:
        cmd.append("--entry-retest")

    if tg_on and tg_token and tg_chat:
        cmd += ["--tg", "--tg-token", tg_token, "--tg-chat", tg_chat]

    if gsheet_id and creds_path:
        # these flags must exist in your scanner (we added earlier):
        cmd += ["--gsheet-id", gsheet_id, "--gsheet-tab", gsheet_tab, "--gsheet-creds", creds_path]

    print("[RUN]", " ".join(shlex.quote(c) for c in cmd), flush=True)

    # 7) Execute and stream output to stdout/stderr (so Render shows logs)
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line.rstrip(), flush=True)
        p.wait()
        print(f"[DONE] exit={p.returncode}", flush=True)
        return p.returncode
    except Exception as e:
        print(f"[ERR] failed to run: {e}", flush=True)
        return 1
    finally:
        if creds_path:
            try:
                os.remove(creds_path)
            except Exception:
                pass

if __name__ == "__main__":
    sys.exit(main())
