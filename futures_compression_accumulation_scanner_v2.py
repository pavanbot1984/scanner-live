#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Futures Compression + Accumulation Scanner v2 (RR=3, prev-bar box lock, hotfixes)

- Exchange: Binance USDT-M Perp (public REST)
- Assets: BTCUSDT, ETHUSDT, SOLUSDT (extendable)
- Timeframes: 15m, 1h, 4h
- CONFIRMATION: Box is locked on the PREVIOUS bar; breakout is evaluated on CURRENT bar
- ENTRY: Breakout close beyond box (±pad); optional retest entry to tighten risk
- SL: Breakout candle extreme
- TP: TP1=1R, TP2=2R, TP3=targetR (default 3R)
- FILTERS: min box % per TF, expected_move ≥ X×risk (default X=3), optional MTF agreement
- ALERTS: Compression heads-up and breakout alerts (only when RR gate passes)

Example run (Windows):
  python futures_compression_accumulation_scanner_v2.py --breakout-watch --entry-retest --pad 0.001 ^
    --min-box-pct-15m 0.10 --min-box-pct-1h 0.20 --min-box-pct-4h 0.18 --min-exp-move-x 3 --target-r 3 --alert-thresh 12 ^
    --tg --tg-token YOUR_TOKEN --tg-chat YOUR_CHAT_ID
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests

FAPI_BASE = "https://fapi.binance.com"
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TFS = ["15m", "1h", "4h"]

# ---------- time/alerts ----------
def ts_now_ms() -> int:
    import time
    return int(time.time() * 1000)

def to_ist(ts_ms: int) -> str:
    return pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S")

def send_tg(msg: str, token: str, chat_id: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": msg, "disable_web_page_preview": True}, timeout=12)
    except Exception as e:
        print(f"[TG] send error: {e}", file=sys.stderr)

# ---------- data ----------
def fetch_futures_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = f"{FAPI_BASE}/fapi/v1/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    data = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"]  = df["open_time"].astype(np.int64)
    df["close_time"] = df["close_time"].astype(np.int64)
    for c in ["open","high","low","close","volume","qav","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time","open","high","low","close","volume","close_time"]].reset_index(drop=True)

# ---------- indicators ----------
def bollinger_bands(close: pd.Series, length: int = 20, mult: float = 2.0):
    sma = close.rolling(length).mean()
    std = close.rolling(length).std(ddof=0)
    upper = sma + mult * std
    lower = sma - mult * std
    bbw = (upper - lower) / sma
    return sma, upper, lower, bbw

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    dirn = np.sign(close.diff().fillna(0))
    return (dirn * volume).fillna(0).cumsum()

def chaikin_ad(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series) -> pd.Series:
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    return (clv.fillna(0) * v).cumsum()

def atr_from_hlc(h: pd.Series, l: pd.Series, c: pd.Series, length: int = 14) -> pd.Series:
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def slope(series: pd.Series, lookback: int) -> float:
    if len(series) < lookback + 2:
        return np.nan
    y = series.iloc[-lookback:].astype(float).values
    x = np.arange(len(y))
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    return float(np.polyfit(x, y, 1)[0])

# ---------- helpers for prev-bar box lock ----------
def calc_bbw_series(close: pd.Series, bb_len=20, bb_mult=2.0) -> pd.Series:
    _, _, _, bbw = bollinger_bands(close, bb_len, bb_mult)
    return bbw

def detect_compression_slice_up_to(sub: pd.DataFrame, bbw: pd.Series,
                                   bb_percentile=0.20, bb_window=120) -> Dict[str, Any]:
    """
    Detect compression on LAST bar of `sub`, and LOCK box using the
    contiguous <=thresh segment that ENDS at that bar.
    """
    if len(bbw.dropna()) < max(20, 30):
        return {"is_compression": False}
    recent = bbw.tail(bb_window).dropna()
    if recent.empty: return {"is_compression": False}
    thresh = np.nanpercentile(recent.values, bb_percentile * 100.0)
    last_bbw = float(bbw.iloc[-1])
    if not (np.isfinite(last_bbw) and last_bbw <= thresh):
        return {"is_compression": False}

    seg_len = 0; idx = len(sub) - 1
    while idx >= 0 and pd.notna(bbw.iloc[idx]) and bbw.iloc[idx] <= thresh:
        seg_len += 1; idx -= 1
    if seg_len < 3:
        return {"is_compression": False}

    comp_slice = sub.iloc[len(sub)-seg_len:]
    h, l = sub["high"], sub["low"]
    comp_high = float(h.loc[comp_slice.index].max())
    comp_low  = float(l.loc[comp_slice.index].min())
    width     = comp_high - comp_low

    return {
        "is_compression": True,
        "bbw": last_bbw, "bbw_thresh": float(thresh),
        "range_high": comp_high, "range_low": comp_low, "range_width": width,
        "segment_len": int(seg_len)
    }

# ---------- 1R/2R/targetR TP/SL ----------
def compute_tp_sl(df: pd.DataFrame, side: str, entry: float, use_retest: bool, box_edge: float, target_r: float = 3.0) -> Dict[str, Any]:
    """
    SL at breakout candle extreme.
    TP1 = 1R, TP2 = 2R, TP3 = target_r * R (default 3R).
    Optional retest entry brings entry closer to box edge by 40% of bar range.
    """
    last = df.iloc[-1]
    brk_high, brk_low = float(last["high"]), float(last["low"])

    if use_retest:
        rng = max(1e-9, brk_high - brk_low)
        if side == "UP":
            entry = max(box_edge, entry - 0.40 * rng)
        else:
            entry = min(box_edge, entry + 0.40 * rng)

    if side == "UP":
        sl   = brk_low
        risk = max(0.0001, entry - sl)
        return {"entry": round(entry,6), "sl": round(sl,6),
                "tp1": round(entry + 1*risk,6),
                "tp2": round(entry + 2*risk,6),
                "tp3": round(entry + target_r*risk,6),
                "risk": round(risk,6)}
    else:
        sl   = brk_high
        risk = max(0.0001, sl - entry)
        return {"entry": round(entry,6), "sl": round(sl,6),
                "tp1": round(entry - 1*risk,6),
                "tp2": round(entry - 2*risk,6),
                "tp3": round(entry - target_r*risk,6),
                "risk": round(risk,6)}

def expected_move(df: pd.DataFrame, width: float, weight_width: float = 0.6, weight_atr: float = 0.4,
                  atr_len: int = 14, atr_mult: float = 1.2) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    atr = atr_from_hlc(h, l, c, length=atr_len).iloc[-1]
    atr_comp = float(atr) * atr_mult if np.isfinite(atr) else 0.0
    return float(max(0.0, weight_width * width + weight_atr * atr_comp))

def breakout_direction(price: float, rh: float, rl: float, pad: float = 0.002) -> Optional[str]:
    if price >= rh * (1.0 + pad): return "UP"
    if price <= rl * (1.0 - pad): return "DOWN"
    return None

def min_box_pct_for_tf(tf: str, args) -> float:
    tf = tf.lower()
    if tf == "15m": return args.min_box_pct_15m / 100.0
    if tf == "1h":  return args.min_box_pct_1h  / 100.0
    if tf == "4h":  return args.min_box_pct_4h  / 100.0
    return 0.0

def combine_scores(comp: Dict[str, Any], accu: Dict[str, Any]) -> Tuple[float, str]:
    score = (0.45 * comp.get("bbw_score", 0.0) +
             0.35 * ((accu.get("obv_score", 0.0) + accu.get("ad_score", 0.0)) / 2.0) +
             0.20 * accu.get("flat_score", 0.0))
    score = max(0.0, min(1.0, score)) * 100.0
    rh, rl, close = comp.get("range_high", np.nan), comp.get("range_low", np.nan), comp.get("last_close", np.nan)
    pos = (close - rl) / max(rh - rl, 1e-9) if (np.isfinite(rh) and np.isfinite(rl) and np.isfinite(close)) else 0.5
    mf = (accu.get("obv_score", 0.0) + accu.get("ad_score", 0.0)) / 2.0
    bias = "Bullish" if (mf >= 0 and pos > 0.6) else ("Bearish" if (mf <= 0 and pos < 0.4) else "Neutral")
    return round(score, 1), bias

def detect_accumulation(df: pd.DataFrame, obv_lb=40, ad_lb=40, flat_lb=40) -> Dict[str, Any]:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    series_obv = obv(c, v)
    series_ad  = chaikin_ad(h, l, c, v)
    obv_s, ad_s = slope(series_obv, obv_lb), slope(series_ad, ad_lb)

    atr = atr_from_hlc(h, l, c, length=14)
    pr  = c.rolling(flat_lb).max() - c.rolling(flat_lb).min()
    last_flat = (pr / (atr * 2.0)).iloc[-1]
    flat_score = float(max(0.0, min(1.0, 1.0 - (last_flat if np.isfinite(last_flat) else 1.0))))

    obv_score = float(np.tanh(obv_s)) if np.isfinite(obv_s) else 0.0
    ad_score  = float(np.tanh(ad_s))  if np.isfinite(ad_s)  else 0.0
    return {"obv_score": obv_score, "ad_score": ad_score, "flat_score": flat_score}

# ---------- scan per symbol/tf (prev-bar lock) ----------
def scan_symbol_tf(symbol: str, tf: str,
                   bb_len=20, bb_mult=2.0, bb_percentile=0.20, bb_window=120,
                   obv_lb=40, ad_lb=40, flat_lb=40,
                   vburst_lookback=30, vburst_mult=1.5,
                   pad=0.002, args=None) -> Optional[Dict[str, Any]]:
    try:
        df = fetch_futures_klines(symbol, tf, limit=max(600, bb_window + 80))
        if df.empty or len(df) < max(bb_window + bb_len + 10, 200):
            return None

        # ---- Prev-bar box lock: lock compression on i-1; evaluate breakout on i ----
        i = len(df) - 1  # last closed bar
        sub_prev = df.iloc[:i]          # up to i-1
        bbw_full = calc_bbw_series(df["close"], bb_len, bb_mult)
        bbw_prev = bbw_full.iloc[:i]

        comp = detect_compression_slice_up_to(sub_prev, bbw_prev, bb_percentile, bb_window)
        if not comp.get("is_compression"):
            # keep table populated even when no valid box
            last_close = float(df["close"].iloc[i])
            fallback = {
                "is_compression": False, "range_high": float(df["high"].iloc[i]),
                "range_low": float(df["low"].iloc[i]), "range_width": float(df["high"].iloc[i]-df["low"].iloc[i]),
                "segment_len": 0, "bbw": None, "bbw_thresh": None
            }
            comp = {**fallback}

        last_close = float(df["close"].iloc[i])
        rh, rl, width = comp.get("range_high"), comp.get("range_low"), comp.get("range_width")
        br = breakout_direction(last_close, rh, rl, pad=pad)

        # accumulation/money-flow scores on full df
        accu = detect_accumulation(df, obv_lb, ad_lb, flat_lb)

        # derive bbw-based compression score (hotfix)
        bbw_val = comp.get("bbw", None)
        bbw_thr = comp.get("bbw_thresh", None)
        if bbw_val is not None and bbw_thr is not None and bbw_thr > 0:
            bbw_score = max(0.0, min(1.0, (bbw_thr - bbw_val) / bbw_thr))
        else:
            bbw_score = 0.0

        score, bias = combine_scores({
            "bbw_score": bbw_score,
            "range_high": rh, "range_low": rl,
            "last_close": last_close, "segment_len": comp.get("segment_len", 0)
        }, accu)

        # volume burst on current bar
        v = df["volume"]; v_ema = v.ewm(span=vburst_lookback, adjust=False).mean()
        vol_surge = bool(v.iloc[-1] > vburst_mult * (v_ema.iloc[-2] if len(v_ema) > 1 else v_ema.iloc[-1]))

        # gates
        mid = (rh + rl) / 2.0 if (rh and rl) else last_close
        box_pct_frac = (width / max(1e-9, mid)) if width and mid else 0.0

        # Reject tiny boxes by TF
        if box_pct_frac < min_box_pct_for_tf(tf, args):
            br = None
            vol_surge = False

        entry = None; tpsl: Dict[str, Any] = {}
        exp_move = expected_move(df, width) if width else 0.0
        if br:
            if br == "UP":
                raw_entry = rh * (1.0 + pad)
                entry_hint = rh
            else:
                raw_entry = rl * (1.0 - pad)
                entry_hint = rl

            tpsl = compute_tp_sl(
                df, br,
                entry=raw_entry,
                use_retest=args.entry_retest,
                box_edge=entry_hint,
                target_r=args.target_r
            )
            entry = tpsl["entry"]

        rr_ok = False
        if br and entry is not None:
            risk = tpsl.get("risk", None)
            rr_ok = (risk is not None) and (exp_move >= args.min_exp_move_x * risk)

        return {
            "symbol": symbol, "tf": tf, "time_ist": to_ist(int(df["close_time"].iloc[-1])),
            "is_compression": bool(comp.get("segment_len", 0) >= 3), "confidence": score, "bias": bias,
            "last": round(last_close, 6), "range_low": round(rl, 6) if rl else None,
            "range_high": round(rh, 6) if rh else None, "width": round(width, 6) if width else None,
            "box_pct": round(box_pct_frac * 100, 3),
            "seg": comp.get("segment_len", 0),
            "breakout": br or "", "vol_surge": vol_surge, "rr_ok": rr_ok,
            "entry": round(entry, 6) if entry is not None else None,
            "sl": tpsl.get("sl"), "tp1": tpsl.get("tp1"), "tp2": tpsl.get("tp2"), "tp3": tpsl.get("tp3"),
            "risk": tpsl.get("risk"),
            "expected_move": round(exp_move, 6) if exp_move else None
        }

    except Exception as e:
        print(f"[Scan] {symbol} {tf} error: {e}", file=sys.stderr)
        return None

# ---------- formatting ----------
def print_blocks(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No data.")
        return
    by_tf: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_tf.setdefault(r["tf"], []).append(r)

    for tf in TFS:
        rows = by_tf.get(tf, [])
        if not rows: continue
        print(f"\n=== {tf.upper()} @ {rows[0]['time_ist']} ===")
        df = pd.DataFrame(rows)
        cols = ["symbol","is_compression","confidence","bias","last",
                "range_low","range_high","width","box_pct","seg",
                "breakout","vol_surge","rr_ok","entry","sl","tp1","tp2","tp3","risk","expected_move"]
        print(df[cols].sort_values(["is_compression","confidence"], ascending=[False, False]).to_string(index=False))

def format_alert_line(r: Dict[str, Any], target_r: float) -> str:
    em = f"{r['expected_move']}" if r.get("expected_move") is not None else "—"
    tps = ""
    if r.get("entry") is not None:
        tps = f"\n• Entry {r['entry']} | SL {r['sl']} | TP1/TP2/TP{int(target_r)}R: {r['tp1']}/{r['tp2']}/{r['tp3']}"
    return (
        f"⚡ {r['symbol']} {r['tf']} @ {r['time_ist']}\n"
        f"• Compression: {r['is_compression']} | Conf: {r['confidence']}/100 | Bias: {r['bias']}\n"
        f"• Box: {r['range_low']}→{r['range_high']} (w={r['width']}) | Box%: {r.get('box_pct','—')} | Seg: {r['seg']}\n"
        f"• Breakout: {r['breakout'] or '—'} | VolBurst: {bool(r['vol_surge'])} | RR_gate_ok: {r.get('rr_ok', False)} | ExpMove: {em}{tps}"
    )

# ---------- cli ----------
def main():
    ap = argparse.ArgumentParser(
        description="Binance USDT-M Futures Compression+Accum (15m/1h/4h) with prev-bar box lock, TP/SL, expected move, and RR gates"
    )
    ap.add_argument("--assets", nargs="+", default=ASSETS)
    ap.add_argument("--alert-thresh", type=float, default=12.0)

    # Compression knobs
    ap.add_argument("--bb-len", type=int, default=20)
    ap.add_argument("--bb-mult", type=float, default=2.0)
    ap.add_argument("--bb-percentile", type=float, default=0.20)
    ap.add_argument("--bb-window", type=int, default=120)

    # Accumulation knobs
    ap.add_argument("--obv-lb", type=int, default=40)
    ap.add_argument("--ad-lb", type=int, default=40)
    ap.add_argument("--flat-lb", type=int, default=40)

    # Breakout confirmation / volume
    ap.add_argument("--vburst-lookback", type=int, default=30)
    ap.add_argument("--vburst-mult", type=float, default=1.25)
    ap.add_argument("--pad", type=float, default=0.001, help="Breakout confirmation pad (e.g., 0.001 = 0.1%)")

    # RR=3 controls (defaults tuned for 15m)
    ap.add_argument("--min-box-pct-15m", type=float, default=0.10, help="Min box width as % of price for 15m")
    ap.add_argument("--min-box-pct-1h",  type=float, default=0.20, help="Min box width as % of price for 1h")
    ap.add_argument("--min-box-pct-4h",  type=float, default=0.18, help="Min box width as % of price for 4h")
    ap.add_argument("--min-exp-move-x",  type=float, default=3.0,   help="Require expected_move >= X * risk (RR gate)")
    ap.add_argument("--target-r",        type=float, default=3.0,   help="Take-profit target in R (TP3 = target_r * R)")
    ap.add_argument("--entry-retest",    action="store_true",        help="Use retest entry (tighter risk) after breakout")
    ap.add_argument("--mtf-agree",       action="store_true",        help="Require direction NOT opposed by higher TF bias")

    # Alerts / output
    ap.add_argument("--breakout-watch", action="store_true", help="Alert on volume-backed breakout that passes RR gate")
    ap.add_argument("--csv", default="", help="Save last run to CSV")

    # Telegram
    ap.add_argument("--tg", action="store_true")
    ap.add_argument("--tg-token", default="")
    ap.add_argument("--tg-chat", default="")
    args = ap.parse_args()

    assets = [a.upper().strip() for a in args.assets if a.upper().strip() in set(ASSETS)]
    if not assets:
        print("No valid assets.")
        sys.exit(0)

    print(f"🧭 Futures Compression+Accum v2 @ {to_ist(ts_now_ms())} | TFs={','.join([t.upper() for t in TFS])} | Assets={','.join(assets)}")

    results: List[Dict[str, Any]] = []
    for sym in assets:
        for tf in TFS:
            r = scan_symbol_tf(
                sym, tf,
                args.bb_len, args.bb_mult, args.bb_percentile, args.bb_window,
                args.obv_lb, args.ad_lb, args.flat_lb,
                args.vburst_lookback, args.vburst_mult,
                pad=args.pad, args=args
            )
            if r:
                results.append(r)

    # Optional: MTF agreement post-filter
    if args.mtf_agree and results:
        by_sym: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            by_sym.setdefault(r["symbol"], []).append(r)

        def opposite(b: str) -> str:
            return "Bearish" if b == "Bullish" else ("Bullish" if b == "Bearish" else "Neutral")

        filtered: List[Dict[str, Any]] = []
        for sym, rows in by_sym.items():
            bias_map = {row["tf"]: row["bias"] for row in rows}
            for row in rows:
                keep = True
                if row["tf"] in {"15m","1h"} and row.get("breakout"):
                    if bias_map.get("4h") == opposite(row["bias"]) or (row["tf"]=="15m" and bias_map.get("1h")==opposite(row["bias"])):
                        keep = False
                if keep: filtered.append(row)
        results = filtered

    print_blocks(results)

    # CSV save
    if args.csv:
        try:
            pd.DataFrame(results).to_csv(args.csv, index=False)
            print(f"\n[CSV] Saved -> {args.csv}")
        except Exception as e:
            print(f"[CSV] Save failed: {e}", file=sys.stderr)

    # Alerts
    if args.tg and args.tg_token and args.tg_chat:
        for r in results:
            fire = False
            strong_comp = r["is_compression"] and r["confidence"] >= args.alert_thresh
            if strong_comp:
                fire = True
            if args.breakout_watch and r.get("breakout") in {"UP", "DOWN"} and r.get("vol_surge") and r.get("rr_ok"):
                fire = True
            if not fire:
                continue
            send_tg(format_alert_line(r, args.target_r), args.tg_token, args.tg_chat)

if __name__ == "__main__":
    main()
