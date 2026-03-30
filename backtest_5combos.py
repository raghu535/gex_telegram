"""
5 MORNING TRADE COMBINATIONS BACKTEST
======================================
Each combo: entry 10:00 or 10:30 AM ET, exit 2:00 PM ET.
One trade per day max. P&L in SPX points.

Run: python backtest_5combos.py
"""

import json
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gex_data.db")
REPORTS = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────

def pt_min(t):
    p = t.split(":")
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 60.0


def et2pt(h, m=0):
    return (h - 3) * 60 + m


def nearest(snaps, target_min):
    best, bd = None, 1e9
    for s in snaps:
        d = abs(pt_min(s["time_pt"]) - target_min)
        if d < bd:
            bd, best = d, s
    return best


def simulate_trade(snaps_after_entry, direction, entry, target, stop, time_exit_pt_min):
    """Walk snapshots. Return (exit_price, exit_reason, pnl_pts)."""
    for s in snaps_after_entry:
        px = s["curr_price"]
        t = pt_min(s["time_pt"])

        # Stop
        if direction == 1 and px <= stop:
            return px, "stop", px - entry
        if direction == -1 and px >= stop:
            return px, "stop", entry - px

        # Target
        if direction == 1 and px >= target:
            return px, "target", px - entry
        if direction == -1 and px <= target:
            return px, "target", entry - px

        # Time exit
        if t >= time_exit_pt_min:
            pnl = (px - entry) * direction
            return px, "time_exit", pnl

    # EOD fallback
    if snaps_after_entry:
        px = snaps_after_entry[-1]["curr_price"]
        return px, "eod", (px - entry) * direction
    return entry, "no_data", 0.0


# ── data loading ─────────────────────────────────────────────

def load_all():
    """Load features CSV, DB snapshots index, and yfinance data."""
    import yfinance as yf

    # Features
    feat_df = pd.read_csv(os.path.join(REPORTS, "daily_features_full.csv"))
    feat = {r["date"]: r for _, r in feat_df.iterrows()}

    # Market data
    print("  Downloading SPX + VIX from yfinance...")
    spx = yf.download("^GSPC", start="2025-07-01", end="2026-02-25", progress=False)
    vix = yf.download("^VIX", start="2025-07-01", end="2026-02-25", progress=False)
    for df in [spx, vix]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

    mkt = pd.DataFrame(index=spx.index)
    mkt["spx_close"] = spx["Close"]
    mkt["spx_open"] = spx["Open"]
    mkt["spx_high"] = spx["High"]
    mkt["spx_low"] = spx["Low"]
    mkt["vix_close"] = vix["Close"].reindex(spx.index, method="ffill")
    mkt["vix_open"] = vix["Open"].reindex(spx.index, method="ffill")
    mkt.index = mkt.index.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    return feat, mkt, conn


def load_snaps(conn, date_str):
    rows = conn.execute(
        "SELECT time_pt, curr_price, net_gamma, call_wall, put_floor "
        "FROM gex_snapshots WHERE date_pt=? AND session_tag='RTH' ORDER BY time_pt",
        (date_str,),
    ).fetchall()
    return [dict(r) for r in rows]


def vix_regime(v):
    if v < 15:
        return "<15"
    elif v <= 20:
        return "15-20"
    else:
        return ">20"


# ── COMBO 1: VIX MOVE + EM POSITION ─────────────────────────

def combo1(feat, mkt, conn):
    dates = sorted(feat.keys())
    mkt_dates = sorted(mkt.index.tolist())
    trades = []

    for date_str in dates:
        ft = feat[date_str]
        em_pts = ft["em_pts"]
        anchor = ft["anchor"]
        if pd.isna(em_pts) or pd.isna(anchor) or em_pts == 0:
            continue

        # Find vix 2 days ago
        idx = None
        for j, md in enumerate(mkt_dates):
            if md == date_str:
                idx = j
                break
        if idx is None or idx < 2:
            continue

        vix_2d_ago_close = float(mkt.loc[mkt_dates[idx - 2], "vix_close"])
        if date_str not in mkt.index:
            continue
        vix_today_open = float(mkt.loc[date_str, "vix_open"])
        vix_prev_close = float(mkt.loc[mkt_dates[idx - 1], "vix_close"])

        if pd.isna(vix_2d_ago_close) or vix_2d_ago_close == 0:
            continue
        vix_2d_change = (vix_today_open - vix_2d_ago_close) / vix_2d_ago_close

        em_pos = ft["first_30_em_position"]
        if pd.isna(em_pos):
            continue

        snaps = load_snaps(conn, date_str)
        if len(snaps) < 10:
            continue

        entry_px = ft["price_at_10am"]
        if pd.isna(entry_px):
            continue

        em_pos_10am = (entry_px - anchor) / em_pts

        direction = 0
        target = anchor
        stop = 0

        if vix_2d_change > 0.05 and em_pos_10am < -0.25:
            direction = 1
            stop = anchor - em_pts
        elif vix_2d_change < -0.05 and em_pos_10am > 0.25:
            direction = -1
            stop = anchor + em_pts

        if direction == 0:
            continue

        snaps_after = [s for s in snaps if pt_min(s["time_pt"]) > et2pt(10, 0)]
        if not snaps_after:
            continue

        exit_px, reason, pnl = simulate_trade(
            snaps_after, direction, entry_px, target, stop, et2pt(14, 0))

        trades.append({
            "date": date_str, "combo": 1,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": round(entry_px, 2), "exit": round(exit_px, 2),
            "target": round(target, 2), "stop": round(stop, 2),
            "exit_reason": reason, "pnl_pts": round(pnl, 2),
            "win": int(pnl > 0),
            "vix_2d_chg": round(vix_2d_change * 100, 1),
            "em_pos_10am": round(em_pos_10am, 3),
            "vix_regime": vix_regime(vix_prev_close),
            "vix_prev": round(vix_prev_close, 1),
        })

    return pd.DataFrame(trades)


# ── COMBO 2: SWEEP PRIOR HIGH/LOW + RECLAIM ─────────────────

def combo2(feat, mkt, conn):
    dates = sorted(feat.keys())
    trades = []

    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        ft = feat[today]
        fy = feat[yesterday]

        snaps = load_snaps(conn, today)
        if len(snaps) < 10:
            continue

        prior_high = fy["day_high"]
        prior_low = fy["day_low"]
        prior_close = fy["day_close"]
        if pd.isna(prior_high) or pd.isna(prior_low):
            continue

        # First 30 min (6:30-7:00 PT = 9:30-10:00 ET)
        f30 = [s for s in snaps if pt_min(s["time_pt"]) <= et2pt(10, 0)]
        if not f30:
            continue

        f30_high = max(s["curr_price"] for s in f30)
        f30_low = min(s["curr_price"] for s in f30)

        snap_10am = nearest(snaps, et2pt(10, 0))
        if not snap_10am:
            continue
        px_10am = snap_10am["curr_price"]

        direction = 0
        target = prior_close
        stop = 0

        # LONG: swept below prior low then reclaimed
        if f30_low < prior_low and px_10am > prior_low:
            direction = 1
            stop = f30_low - 5
        # SHORT: swept above prior high then reclaimed
        elif f30_high > prior_high and px_10am < prior_high:
            direction = -1
            stop = f30_high + 5

        if direction == 0:
            continue

        snaps_after = [s for s in snaps if pt_min(s["time_pt"]) > et2pt(10, 0)]
        if not snaps_after:
            continue

        exit_px, reason, pnl = simulate_trade(
            snaps_after, direction, px_10am, target, stop, et2pt(14, 0))

        vix_prev = float(mkt.loc[yesterday, "vix_close"]) if yesterday in mkt.index else np.nan

        trades.append({
            "date": today, "combo": 2,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": round(px_10am, 2), "exit": round(exit_px, 2),
            "target": round(target, 2), "stop": round(stop, 2),
            "exit_reason": reason, "pnl_pts": round(pnl, 2),
            "win": int(pnl > 0),
            "prior_high": round(prior_high, 2),
            "prior_low": round(prior_low, 2),
            "f30_high": round(f30_high, 2),
            "f30_low": round(f30_low, 2),
            "vix_regime": vix_regime(vix_prev) if not pd.isna(vix_prev) else "?",
            "vix_prev": round(vix_prev, 1) if not pd.isna(vix_prev) else np.nan,
        })

    return pd.DataFrame(trades)


# ── COMBO 3: GAMMA + VIX + EM (triple filter) ───────────────

def combo3(feat, mkt, conn):
    dates = sorted(feat.keys())
    mkt_dates = sorted(mkt.index.tolist())
    trades = []

    for date_str in dates:
        ft = feat[date_str]
        em_pts = ft["em_pts"]
        anchor = ft["anchor"]
        og = ft["open_gamma"]
        f30_ratio = ft["first_30_ratio"]

        if pd.isna(em_pts) or pd.isna(anchor) or pd.isna(og) or pd.isna(f30_ratio):
            continue
        if em_pts == 0:
            continue

        # Prior day VIX
        idx = None
        for j, md in enumerate(mkt_dates):
            if md == date_str:
                idx = j
                break
        if idx is None or idx < 1:
            continue
        vix_prev = float(mkt.loc[mkt_dates[idx - 1], "vix_close"])
        if pd.isna(vix_prev):
            continue

        snaps = load_snaps(conn, date_str)
        if len(snaps) < 10:
            continue

        snap_10am = nearest(snaps, et2pt(10, 0))
        if not snap_10am:
            continue
        px_10am = snap_10am["curr_price"]
        em_pos_10am = (px_10am - anchor) / em_pts

        direction = 0
        target = anchor
        stop = 0

        # LONG
        if og < -3e9 and vix_prev > 20 and em_pos_10am < -0.5 and f30_ratio < 0.3:
            direction = 1
            stop = anchor - em_pts

        # SHORT
        if og > 3e9 and vix_prev < 15 and em_pos_10am > 0.5 and f30_ratio < 0.3:
            direction = -1
            stop = anchor + em_pts

        if direction == 0:
            continue

        snaps_after = [s for s in snaps if pt_min(s["time_pt"]) > et2pt(10, 0)]
        if not snaps_after:
            continue

        exit_px, reason, pnl = simulate_trade(
            snaps_after, direction, px_10am, target, stop, et2pt(14, 0))

        trades.append({
            "date": date_str, "combo": 3,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": round(px_10am, 2), "exit": round(exit_px, 2),
            "target": round(target, 2), "stop": round(stop, 2),
            "exit_reason": reason, "pnl_pts": round(pnl, 2),
            "win": int(pnl > 0),
            "open_gamma_bn": round(og / 1e9, 1),
            "vix_prev": round(vix_prev, 1),
            "em_pos_10am": round(em_pos_10am, 3),
            "f30_ratio": round(f30_ratio, 3),
            "vix_regime": vix_regime(vix_prev),
        })

    return pd.DataFrame(trades)


# ── COMBO 4: COMPRESSION THEN EXPANSION ──────────────────────

def combo4(feat, mkt, conn):
    dates = sorted(feat.keys())
    mkt_dates = sorted(mkt.index.tolist())
    trades = []

    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        ft = feat[today]
        fy = feat[yesterday]

        em_pts = ft["em_pts"]
        anchor = ft["anchor"]
        f30_ratio = ft["first_30_ratio"]
        f30_dir = ft["first_30_direction"]

        if pd.isna(em_pts) or pd.isna(anchor) or pd.isna(f30_ratio) or pd.isna(f30_dir):
            continue
        if em_pts == 0:
            continue

        prior_range = fy["day_range"]
        if pd.isna(prior_range):
            continue
        prior_range_ratio = prior_range / (em_pts * 2)

        if prior_range_ratio >= 0.5 or f30_ratio <= 0.3:
            continue

        snaps = load_snaps(conn, today)
        if len(snaps) < 10:
            continue

        snap_10am = nearest(snaps, et2pt(10, 0))
        if not snap_10am:
            continue
        px_10am = snap_10am["curr_price"]

        if abs(f30_dir) < 1:
            continue

        if f30_dir < 0:
            direction = -1
            target = ft["em_lower"]
            stop = px_10am + 15
        else:
            direction = 1
            target = ft["em_upper"]
            stop = px_10am - 15

        snaps_after = [s for s in snaps if pt_min(s["time_pt"]) > et2pt(10, 0)]
        if not snaps_after:
            continue

        exit_px, reason, pnl = simulate_trade(
            snaps_after, direction, px_10am, target, stop, et2pt(14, 0))

        # VIX
        idx = None
        for j, md in enumerate(mkt_dates):
            if md == today:
                idx = j
                break
        vix_prev = float(mkt.loc[mkt_dates[idx - 1], "vix_close"]) if idx and idx >= 1 else np.nan

        trades.append({
            "date": today, "combo": 4,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": round(px_10am, 2), "exit": round(exit_px, 2),
            "target": round(target, 2), "stop": round(stop, 2),
            "exit_reason": reason, "pnl_pts": round(pnl, 2),
            "win": int(pnl > 0),
            "prior_range_ratio": round(prior_range_ratio, 3),
            "f30_ratio": round(f30_ratio, 3),
            "vix_regime": vix_regime(vix_prev) if not pd.isna(vix_prev) else "?",
            "vix_prev": round(vix_prev, 1) if not pd.isna(vix_prev) else np.nan,
        })

    return pd.DataFrame(trades)


# ── COMBO 5: SWEEP PRIOR LOW + NEGATIVE GAMMA ───────────────

def combo5(feat, mkt, conn):
    dates = sorted(feat.keys())
    mkt_dates = sorted(mkt.index.tolist())
    trades = []

    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        ft = feat[today]
        fy = feat[yesterday]

        prior_high = fy["day_high"]
        prior_low = fy["day_low"]
        prior_close = fy["day_close"]
        if pd.isna(prior_high) or pd.isna(prior_low) or pd.isna(prior_close):
            continue

        snaps = load_snaps(conn, today)
        if len(snaps) < 10:
            continue

        # First 60 min (6:30-7:30 PT = 9:30-10:30 ET)
        f60 = [s for s in snaps if pt_min(s["time_pt"]) <= et2pt(10, 30)]
        if not f60:
            continue

        f60_high = max(s["curr_price"] for s in f60)
        f60_low = min(s["curr_price"] for s in f60)

        snap_1030 = nearest(snaps, et2pt(10, 30))
        if not snap_1030:
            continue
        px_1030 = snap_1030["curr_price"]

        direction = 0
        target = prior_close
        stop = 0

        # LONG: swept below prior low + neg gamma + reclaimed
        if f60_low < prior_low and px_1030 > prior_low:
            # Find gamma at time of sweep
            sweep_gamma = None
            for s in f60:
                if s["curr_price"] <= prior_low:
                    sweep_gamma = s["net_gamma"]
                    break
            if sweep_gamma is not None and sweep_gamma < -3e9:
                direction = 1
                stop = f60_low - 5

        # SHORT: swept above prior high + pos gamma + reclaimed
        if direction == 0 and f60_high > prior_high and px_1030 < prior_high:
            sweep_gamma = None
            for s in f60:
                if s["curr_price"] >= prior_high:
                    sweep_gamma = s["net_gamma"]
                    break
            if sweep_gamma is not None and sweep_gamma > 3e9:
                direction = -1
                stop = f60_high + 5

        if direction == 0:
            continue

        # Entry at 10:30 AM ET
        snaps_after = [s for s in snaps if pt_min(s["time_pt"]) > et2pt(10, 30)]
        if not snaps_after:
            continue

        exit_px, reason, pnl = simulate_trade(
            snaps_after, direction, px_1030, target, stop, et2pt(14, 0))

        idx = None
        for j, md in enumerate(mkt_dates):
            if md == today:
                idx = j
                break
        vix_prev = float(mkt.loc[mkt_dates[idx - 1], "vix_close"]) if idx and idx >= 1 else np.nan

        trades.append({
            "date": today, "combo": 5,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": round(px_1030, 2), "exit": round(exit_px, 2),
            "target": round(target, 2), "stop": round(stop, 2),
            "exit_reason": reason, "pnl_pts": round(pnl, 2),
            "win": int(pnl > 0),
            "prior_low": round(prior_low, 2),
            "prior_high": round(prior_high, 2),
            "f60_low": round(f60_low, 2),
            "f60_high": round(f60_high, 2),
            "vix_regime": vix_regime(vix_prev) if not pd.isna(vix_prev) else "?",
            "vix_prev": round(vix_prev, 1) if not pd.isna(vix_prev) else np.nan,
        })

    return pd.DataFrame(trades)


# ── PIN (from prior test) ───────────────────────────────────

def backtest_pin(feat, conn):
    """Butterfly at 2pm: gamma>+3Bn AND price within 15pts of call_wall.
    Win = close within 10pts of 2pm wall."""
    dates = sorted(feat.keys())
    trades = []

    for date_str in dates:
        snaps = load_snaps(conn, date_str)
        if len(snaps) < 10:
            continue

        snap_2pm = nearest(snaps, et2pt(14, 0))
        if not snap_2pm:
            continue

        gamma_2pm = snap_2pm["net_gamma"]
        if gamma_2pm <= 3e9:
            continue

        cw_2pm = snap_2pm["call_wall"]
        px_2pm = snap_2pm["curr_price"]
        if abs(px_2pm - cw_2pm) > 15:
            continue

        close_px = snaps[-1]["curr_price"]
        dist = abs(close_px - cw_2pm)
        win = dist <= 10
        pnl = 15.0 if win else -5.0  # normalized to pts

        trades.append({
            "date": date_str, "setup": "PIN",
            "direction": "NEUTRAL",
            "entry": round(px_2pm, 2), "exit": round(close_px, 2),
            "target": cw_2pm, "stop": 0,
            "exit_reason": "close",
            "pnl_pts": pnl, "win": int(win),
            "distance": round(dist, 1),
            "gamma_2pm_bn": round(gamma_2pm / 1e9, 1),
        })

    return pd.DataFrame(trades)


# ── REPORTING ────────────────────────────────────────────────

def report_combo(name, df):
    """Print stats for one combo."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    if len(df) == 0:
        print("  No trades.")
        return

    n = len(df)
    wins = df["win"].sum()
    pnl_arr = df["pnl_pts"].values
    avg_pnl = pnl_arr.mean()
    total_pnl = pnl_arr.sum()
    sharpe = (avg_pnl / pnl_arr.std()) if pnl_arr.std() > 0 else 0

    best_i = np.argmax(pnl_arr)
    worst_i = np.argmin(pnl_arr)

    # Max consec losses
    max_cl = 0
    cl = 0
    for p in pnl_arr:
        if p <= 0:
            cl += 1
            max_cl = max(max_cl, cl)
        else:
            cl = 0

    print(f"  Total trades: {n}  (L:{(df['direction']=='LONG').sum()} S:{(df['direction']=='SHORT').sum()})")
    print(f"  Win rate:     {wins}/{n} ({wins/n*100:.0f}%)")

    # Long/short split
    for d in ["LONG", "SHORT"]:
        sub = df[df["direction"] == d]
        if len(sub) > 0:
            print(f"    {d}: {sub['win'].sum()}/{len(sub)} ({sub['win'].mean()*100:.0f}%), "
                  f"avg {sub['pnl_pts'].mean():+.1f} pts")

    print(f"  Avg P&L:      {avg_pnl:+.1f} pts")
    print(f"  Total P&L:    {total_pnl:+.1f} pts")
    print(f"  Sharpe:       {sharpe:+.3f}")
    print(f"  Best:         {pnl_arr[best_i]:+.1f} pts on {df.iloc[best_i]['date']}")
    print(f"  Worst:        {pnl_arr[worst_i]:+.1f} pts on {df.iloc[worst_i]['date']}")
    print(f"  Max consec L: {max_cl}")

    # Exit reasons
    if "exit_reason" in df.columns:
        reasons = df["exit_reason"].value_counts()
        print(f"  Exits:        {dict(reasons)}")

    # VIX regime
    if "vix_regime" in df.columns:
        print(f"  By VIX regime:")
        for vr in ["<15", "15-20", ">20"]:
            sub = df[df["vix_regime"] == vr]
            if len(sub) > 0:
                print(f"    VIX {vr:>5}: {len(sub):>2} trades, "
                      f"win {sub['win'].mean()*100:.0f}%, "
                      f"avg {sub['pnl_pts'].mean():+.1f} pts")

    # Monthly
    df_m = df.copy()
    df_m["month"] = df_m["date"].str[:7]
    print(f"  Monthly:")
    for month in sorted(df_m["month"].unique()):
        m = df_m[df_m["month"] == month]
        print(f"    {month}: {len(m):>2} trades, "
              f"{m['win'].sum()}/{len(m)} wins, "
              f"{m['pnl_pts'].sum():>+7.1f} pts")

    # Trade list
    print()
    show_cols = [c for c in ["date", "direction", "entry", "exit", "exit_reason",
                              "pnl_pts", "win"] if c in df.columns]
    extra = [c for c in df.columns if c not in show_cols
             and c not in ["combo", "target", "stop", "vix_regime", "vix_prev"]
             and not c.startswith("prior_") and not c.startswith("f30_") and not c.startswith("f60_")]
    show = show_cols + extra[:2]
    print(df[show].to_string(index=False))


def combo_summary_row(name, df):
    """Return dict for comparison table."""
    if len(df) == 0:
        return {"combo": name, "trades": 0, "win_pct": 0, "avg_pnl": 0,
                "total_pnl": 0, "sharpe": 0, "best": 0, "worst": 0}
    pnl = df["pnl_pts"].values
    return {
        "combo": name,
        "trades": len(df),
        "win_pct": round(df["win"].mean() * 100, 1),
        "avg_pnl": round(pnl.mean(), 1),
        "total_pnl": round(pnl.sum(), 1),
        "sharpe": round(pnl.mean() / pnl.std(), 3) if pnl.std() > 0 else 0,
        "best": round(pnl.max(), 1),
        "worst": round(pnl.min(), 1),
    }


# ── MAIN ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  5 MORNING TRADE COMBINATIONS BACKTEST")
    print("=" * 70)

    feat, mkt, conn = load_all()
    print(f"  {len(feat)} trading days loaded\n")

    # Run all combos
    print("[1/5] COMBO 1: VIX MOVE + EM POSITION...")
    c1 = combo1(feat, mkt, conn)
    report_combo("COMBO 1: VIX MOVE + EM POSITION", c1)

    print("\n[2/5] COMBO 2: SWEEP PRIOR HIGH/LOW + RECLAIM...")
    c2 = combo2(feat, mkt, conn)
    report_combo("COMBO 2: SWEEP PRIOR HIGH/LOW + RECLAIM", c2)

    print("\n[3/5] COMBO 3: GAMMA + VIX + EM (triple filter)...")
    c3 = combo3(feat, mkt, conn)
    report_combo("COMBO 3: GAMMA + VIX + EM (triple filter)", c3)

    print("\n[4/5] COMBO 4: COMPRESSION THEN EXPANSION...")
    c4 = combo4(feat, mkt, conn)
    report_combo("COMBO 4: COMPRESSION THEN EXPANSION", c4)

    print("\n[5/5] COMBO 5: SWEEP + NEGATIVE GAMMA...")
    c5 = combo5(feat, mkt, conn)
    report_combo("COMBO 5: SWEEP PRIOR LOW + NEGATIVE GAMMA", c5)

    # ── COMPARISON TABLE ──
    combos = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    rows = [combo_summary_row(k, v) for k, v in combos.items()]
    comp_df = pd.DataFrame(rows)

    print(f"\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}\n")
    print(f"  {'COMBO':<6} {'TRADES':>7} {'WIN%':>6} {'AVG PNL':>9} "
          f"{'TOTAL PNL':>11} {'SHARPE':>8} {'BEST':>7} {'WORST':>7}")
    print("  " + "-" * 65)
    for _, r in comp_df.iterrows():
        print(f"  {r['combo']:<6} {r['trades']:>7} {r['win_pct']:>5.0f}% "
              f"{r['avg_pnl']:>+8.1f} {r['total_pnl']:>+10.1f} "
              f"{r['sharpe']:>+7.3f} {r['best']:>+6.1f} {r['worst']:>+6.1f}")

    # ── TOP 2 + PIN COMBINED ──
    print(f"\n{'='*70}")
    print("  COMBINE TOP 2 SHARPE + PIN")
    print(f"{'='*70}")

    # Sort by Sharpe
    ranked = sorted(rows, key=lambda x: x["sharpe"], reverse=True)
    top2_names = [r["combo"] for r in ranked[:2] if r["trades"] > 0]
    print(f"  Top 2 by Sharpe: {top2_names}")

    # PIN
    print("  Running PIN backtest...")
    pin_df = backtest_pin(feat, conn)
    if len(pin_df) > 0:
        print(f"  PIN: {len(pin_df)} trades, "
              f"{pin_df['win'].sum()}/{len(pin_df)} wins, "
              f"{pin_df['pnl_pts'].sum():+.1f} pts")

    # Combine: morning trade from best combo, then pin if no conflict
    best_combo_df = combos[top2_names[0]] if top2_names else pd.DataFrame()
    second_combo_df = combos[top2_names[1]] if len(top2_names) > 1 else pd.DataFrame()

    # Get all dates
    all_dates = set()
    for df in [best_combo_df, second_combo_df, pin_df]:
        if len(df) > 0:
            all_dates.update(df["date"].tolist())
    all_dates = sorted(all_dates)

    best_dates = set(best_combo_df["date"].tolist()) if len(best_combo_df) > 0 else set()
    second_dates = set(second_combo_df["date"].tolist()) if len(second_combo_df) > 0 else set()
    pin_dates = set(pin_df["date"].tolist()) if len(pin_df) > 0 else set()

    combined_rows = []
    for d in all_dates:
        # Morning: best combo first, then second
        morning_used = False
        if d in best_dates:
            row = best_combo_df[best_combo_df["date"] == d].iloc[0]
            combined_rows.append({
                "date": d, "setup": top2_names[0],
                "direction": row["direction"],
                "entry": row["entry"], "exit": row["exit"],
                "pnl_pts": row["pnl_pts"], "win": row["win"],
            })
            morning_used = True
        elif d in second_dates:
            row = second_combo_df[second_combo_df["date"] == d].iloc[0]
            combined_rows.append({
                "date": d, "setup": top2_names[1],
                "direction": row["direction"],
                "entry": row["entry"], "exit": row["exit"],
                "pnl_pts": row["pnl_pts"], "win": row["win"],
            })
            morning_used = True

        # PIN: can coexist (morning exits by 2pm, pin starts at 2pm)
        if d in pin_dates:
            row = pin_df[pin_df["date"] == d].iloc[0]
            combined_rows.append({
                "date": d, "setup": "PIN",
                "direction": "NEUTRAL",
                "entry": row["entry"], "exit": row["exit"],
                "pnl_pts": row["pnl_pts"], "win": row["win"],
            })

    combined = pd.DataFrame(combined_rows)
    if len(combined) > 0:
        combined["cumulative_pnl"] = combined["pnl_pts"].cumsum()

        n = len(combined)
        wins = combined["win"].sum()
        pnl_arr = combined["pnl_pts"].values
        total_pnl = pnl_arr.sum()

        # Max drawdown
        cum = np.cumsum(pnl_arr)
        peak = np.maximum.accumulate(cum)
        max_dd = (peak - cum).max()

        # Trades per week
        combined["week"] = pd.to_datetime(combined["date"]).dt.isocalendar().week
        trades_per_week = combined.groupby("week").size().mean()

        print(f"\n  Combined: {n} trades, {wins}/{n} wins ({wins/n*100:.0f}%)")
        print(f"  Total P&L: {total_pnl:+.1f} pts")
        print(f"  Max drawdown: {max_dd:.1f} pts")
        print(f"  Avg trades/week: {trades_per_week:.1f}")

        # By setup
        print(f"\n  {'SETUP':<8} {'TRADES':>7} {'WIN%':>6} {'TOTAL':>9}")
        print("  " + "-" * 35)
        for setup in combined["setup"].unique():
            sub = combined[combined["setup"] == setup]
            print(f"  {setup:<8} {len(sub):>7} {sub['win'].mean()*100:>5.0f}% "
                  f"{sub['pnl_pts'].sum():>+8.1f}")

        # Monthly
        combined["month"] = combined["date"].str[:7]
        print(f"\n  {'MONTH':<10} {'TRADES':>7} {'WINS':>6} {'P&L':>9} {'WIN%':>6}")
        print("  " + "-" * 42)
        for month in sorted(combined["month"].unique()):
            m = combined[combined["month"] == month]
            w = m["win"].sum()
            print(f"  {month:<10} {len(m):>7} {w:>6} {m['pnl_pts'].sum():>+8.1f} "
                  f"{w/len(m)*100:>5.0f}%")

    # ── SAVE ──
    print(f"\n  Saving reports...")

    c1.to_csv(os.path.join(REPORTS, "combo1_vix_em_trades.csv"), index=False)
    c2.to_csv(os.path.join(REPORTS, "combo2_sweep_reclaim_trades.csv"), index=False)
    c3.to_csv(os.path.join(REPORTS, "combo3_triple_filter_trades.csv"), index=False)
    c4.to_csv(os.path.join(REPORTS, "combo4_compression_trades.csv"), index=False)
    c5.to_csv(os.path.join(REPORTS, "combo5_sweep_gamma_trades.csv"), index=False)

    comp_df.to_csv(os.path.join(REPORTS, "combo_comparison_table.csv"), index=False)

    with open(os.path.join(REPORTS, "combo_comparison_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    if len(combined) > 0:
        combined.to_csv(os.path.join(REPORTS, "combo_plus_pin_combined.csv"), index=False)
        eq = combined[["date", "setup", "pnl_pts", "cumulative_pnl"]].copy()
        eq.to_csv(os.path.join(REPORTS, "combo_plus_pin_equity.csv"), index=False)

        summary = {
            "top2_combos": top2_names,
            "total_trades": int(n),
            "win_rate": round(wins / n * 100, 1),
            "total_pnl_pts": round(float(total_pnl), 1),
            "max_drawdown_pts": round(float(max_dd), 1),
            "avg_trades_per_week": round(float(trades_per_week), 1),
        }
        with open(os.path.join(REPORTS, "combo_plus_pin_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # Equity curve chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if len(combined) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                            gridspec_kw={"height_ratios": [3, 1]})
            fig.suptitle("Top 2 Combos + PIN - Equity Curve (pts)", fontsize=14, fontweight="bold")

            dates_dt = pd.to_datetime(combined["date"])
            cum = combined["cumulative_pnl"].values

            ax1.plot(dates_dt, cum, color="black", linewidth=1.5)
            ax1.fill_between(dates_dt, cum, where=(cum >= 0), alpha=0.2, color="green")
            ax1.fill_between(dates_dt, cum, where=(cum < 0), alpha=0.2, color="red")
            ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax1.set_ylabel("Cumulative P&L (pts)")
            ax1.grid(True, alpha=0.3)

            colors = {"PIN": "orange", top2_names[0]: "blue"}
            if len(top2_names) > 1:
                colors[top2_names[1]] = "green"
            for setup, color in colors.items():
                mask = combined["setup"] == setup
                if mask.any():
                    ax2.bar(dates_dt[mask], combined.loc[mask, "pnl_pts"],
                            color=color, alpha=0.7, width=1, label=setup)

            ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax2.set_ylabel("Trade P&L (pts)")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.tight_layout()
            plt.savefig(os.path.join(REPORTS, "combo_plus_pin_equity.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved equity curve chart")
    except Exception as e:
        print(f"  Chart error: {e}")

    for f in ["combo1_vix_em_trades.csv", "combo2_sweep_reclaim_trades.csv",
              "combo3_triple_filter_trades.csv", "combo4_compression_trades.csv",
              "combo5_sweep_gamma_trades.csv", "combo_comparison_summary.json",
              "combo_comparison_table.csv", "combo_plus_pin_combined.csv",
              "combo_plus_pin_equity.csv", "combo_plus_pin_summary.json"]:
        path = os.path.join(REPORTS, f)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"    [{status}] {f}")

    conn.close()
    print(f"\n  BACKTEST COMPLETE.\n")


if __name__ == "__main__":
    main()
