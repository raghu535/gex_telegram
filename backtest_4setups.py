"""
4-SETUP INDEPENDENT BACKTEST
=============================
Setup 1: COIL / GAP TRADE (overnight gamma flip)
Setup 2: MORNING BET (10am directional with gamma)
Setup 3: PIN TRADE (2pm butterfly on closing wall)
Setup 4: SWING TRADE (EOD gamma extreme -> next day)

Run: python backtest_4setups.py
"""

import json
import os
import sqlite3
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gex_data.db")
REPORTS = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────

def pt_min(t):
    """time_pt string -> minutes since midnight PT."""
    p = t.split(":")
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 60.0


def et2pt(h, m=0):
    """ET hour:min -> PT minutes since midnight (PT = ET - 3)."""
    return (h - 3) * 60 + m


def nearest(snaps, target_min):
    """Snapshot nearest to target PT minutes."""
    best, bd = None, 1e9
    for s in snaps:
        d = abs(pt_min(s["time_pt"]) - target_min)
        if d < bd:
            bd, best = d, s
    return best


def load_features():
    """Load daily_features_full.csv into a dict keyed by date."""
    df = pd.read_csv(os.path.join(REPORTS, "daily_features_full.csv"))
    return {row["date"]: row for _, row in df.iterrows()}


def load_day_snaps(conn, date_str):
    """Load RTH snapshots for a date, return list of dicts."""
    rows = conn.execute(
        "SELECT time_pt, curr_price, net_gamma, call_wall, put_floor "
        "FROM gex_snapshots WHERE date_pt=? AND session_tag='RTH' "
        "ORDER BY time_pt", (date_str,)
    ).fetchall()
    return [dict(r) for r in rows]


def snap_at_time(snaps, et_h, et_m=0):
    """Get snapshot nearest an ET time."""
    return nearest(snaps, et2pt(et_h, et_m))


# ── SETUP 1: COIL / GAP TRADE ───────────────────────────────

def backtest_coil(feat, conn):
    """Overnight gamma flip -> trade in gap direction."""
    dates = sorted(feat.keys())
    trades = []

    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        ft = feat[today]
        fy = feat[yesterday]

        snaps = load_day_snaps(conn, today)
        if len(snaps) < 10:
            continue

        snaps_y = load_day_snaps(conn, yesterday)
        if len(snaps_y) < 5:
            continue

        eod_gamma_y = snaps_y[-1]["net_gamma"]
        open_gamma = snaps[0]["net_gamma"]

        # Check coil: opposite signs AND >5Bn flip
        signs_opposite = (eod_gamma_y > 0 and open_gamma < 0) or (eod_gamma_y < 0 and open_gamma > 0)
        flip_size = abs(eod_gamma_y - open_gamma)

        if not (signs_opposite and flip_size > 5e9):
            continue

        prior_close = fy["day_close"]
        open_price = ft["open_price"]
        em_pts = ft["em_pts"]
        anchor = ft["anchor"]

        if pd.isna(em_pts) or pd.isna(anchor) or em_pts == 0:
            continue

        # Direction: gap direction
        gap = open_price - prior_close
        if abs(gap) < 1:
            continue

        if gap < 0:
            direction = -1  # gap down -> buy put
            target = anchor - 0.75 * em_pts
            stop = prior_close  # gap fill
        else:
            direction = 1  # gap up -> buy call
            target = anchor + 0.75 * em_pts
            stop = prior_close

        entry = open_price
        exit_price = entry
        exit_reason = "eod"
        time_exit_min = et2pt(14, 0)

        for s in snaps:
            px = s["curr_price"]
            t = pt_min(s["time_pt"])

            # Check stop (gap fill)
            if direction == 1 and px <= stop:
                exit_price = px
                exit_reason = "stop"
                break
            if direction == -1 and px >= stop:
                exit_price = px
                exit_reason = "stop"
                break

            # Check target
            if direction == 1 and px >= target:
                exit_price = px
                exit_reason = "target"
                break
            if direction == -1 and px <= target:
                exit_price = px
                exit_reason = "target"
                break

            # Time exit
            if t >= time_exit_min:
                exit_price = px
                exit_reason = "time_exit"
                break

        if exit_reason == "eod":
            exit_price = snaps[-1]["curr_price"]

        pnl_pts = (exit_price - entry) * direction
        win = pnl_pts > 0

        trades.append({
            "date": today,
            "setup": "COIL",
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "target": round(target, 2),
            "stop": round(stop, 2),
            "exit_reason": exit_reason,
            "pnl_pts": round(pnl_pts, 2),
            "win": int(win),
            "gap_pts": round(gap, 2),
            "eod_gamma_prior_bn": round(eod_gamma_y / 1e9, 1),
            "open_gamma_bn": round(open_gamma / 1e9, 1),
            "flip_bn": round(flip_size / 1e9, 1),
        })

    return pd.DataFrame(trades)


# ── SETUP 2: MORNING BET (10am) ─────────────────────────────

def backtest_morning(feat, conn, use_gamma_filter=True):
    """Directional bet at 10am on big-open days."""
    dates = sorted(feat.keys())
    trades = []

    for date_str in dates:
        ft = feat[date_str]
        f30_ratio = ft["first_30_ratio"]

        if pd.isna(f30_ratio) or f30_ratio <= 0.3:
            continue

        snaps = load_day_snaps(conn, date_str)
        if len(snaps) < 10:
            continue

        f30_dir = ft["first_30_direction"]
        og = ft["open_gamma"]
        em_pts = ft["em_pts"]
        anchor = ft["anchor"]

        if pd.isna(em_pts) or pd.isna(anchor) or pd.isna(og) or em_pts == 0:
            continue

        # Direction from first 30 min
        if abs(f30_dir) < 1:
            continue
        direction = 1 if f30_dir > 0 else -1

        # Gamma filter
        gamma_confirms = False
        if direction == -1 and og < -2e9:
            gamma_confirms = True
        elif direction == 1 and og > 2e9:
            gamma_confirms = True

        if use_gamma_filter and not gamma_confirms:
            continue

        entry = ft["price_at_10am"]
        if pd.isna(entry):
            continue

        # Targets
        if direction == 1:
            target = ft["em_upper"]
            stop = entry - 15
        else:
            target = ft["em_lower"]
            stop = entry + 15

        # Simulate through snapshots after 10am
        snaps_after = [s for s in snaps if pt_min(s["time_pt"]) > et2pt(10, 0)]
        if not snaps_after:
            continue

        exit_price = entry
        exit_reason = "eod"
        time_exit_min = et2pt(14, 0)

        for s in snaps_after:
            px = s["curr_price"]
            t = pt_min(s["time_pt"])

            # Stop
            if direction == 1 and px <= stop:
                exit_price = px
                exit_reason = "stop"
                break
            if direction == -1 and px >= stop:
                exit_price = px
                exit_reason = "stop"
                break

            # Target
            if direction == 1 and px >= target:
                exit_price = px
                exit_reason = "target"
                break
            if direction == -1 and px <= target:
                exit_price = px
                exit_reason = "target"
                break

            # Time exit at 2pm
            if t >= time_exit_min:
                exit_price = px
                exit_reason = "time_exit"
                break

        if exit_reason == "eod":
            exit_price = snaps_after[-1]["curr_price"]

        pnl_pts = (exit_price - entry) * direction
        win = pnl_pts > 0

        group = ft.get("group", "?")

        trades.append({
            "date": date_str,
            "setup": "MORNING_BET",
            "group": group,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "target": round(target, 2),
            "stop": round(stop, 2),
            "exit_reason": exit_reason,
            "pnl_pts": round(pnl_pts, 2),
            "win": int(win),
            "f30_ratio": round(f30_ratio, 3),
            "open_gamma_bn": round(og / 1e9, 1),
            "gamma_confirms": gamma_confirms,
        })

    return pd.DataFrame(trades)


# ── SETUP 3: PIN TRADE (2pm, closing wall) ──────────────────

def backtest_pin(feat, conn):
    """Butterfly at 2pm if gamma >+3Bn and price near wall.

    The butterfly is centered at the 2pm call_wall (that is where
    you actually place the trade).  Win/loss is measured against
    that SAME 2pm wall at market close.

    Returns (primary_df using 2pm wall with 10pt threshold,
             comparison_df using closing wall — circular but shown for contrast).
    """
    dates = sorted(feat.keys())
    trades_2pm = []
    trades_cw_close = []

    for date_str in dates:
        ft = feat[date_str]
        snaps = load_day_snaps(conn, date_str)
        if len(snaps) < 10:
            continue

        snap_2pm = snap_at_time(snaps, 14, 0)
        if not snap_2pm:
            continue

        gamma_2pm = snap_2pm["net_gamma"]
        if gamma_2pm <= 3e9:
            continue

        cw_2pm = snap_2pm["call_wall"]
        px_2pm = snap_2pm["curr_price"]
        dist_entry = abs(px_2pm - cw_2pm)

        if dist_entry > 15:
            continue

        close_px = snaps[-1]["curr_price"]
        cw_close = snaps[-1]["call_wall"]

        # PRIMARY: butterfly centered at 2pm wall, 10pt win zone
        dist_at_close = abs(close_px - cw_2pm)
        win = dist_at_close <= 10
        pnl = 1500 if win else -500

        trades_2pm.append({
            "date": date_str,
            "setup": "PIN",
            "call_wall_2pm": cw_2pm,
            "call_wall_close": cw_close,
            "wall_drift": cw_close - cw_2pm,
            "price_2pm": round(px_2pm, 2),
            "price_close": round(close_px, 2),
            "distance": round(dist_at_close, 1),
            "close_vs_wall": round(close_px - cw_2pm, 1),
            "gamma_2pm_bn": round(gamma_2pm / 1e9, 1),
            "pnl_dollars": pnl,
            "pnl_pts": round(pnl / 100, 1),
            "win": int(win),
        })

        # COMPARISON: closing wall (circular — wall follows price)
        dist_cw = abs(close_px - cw_close)
        trades_cw_close.append({
            "date": date_str,
            "setup": "PIN",
            "call_wall": cw_close,
            "price_close": round(close_px, 2),
            "distance": round(dist_cw, 1),
            "gamma_2pm_bn": round(gamma_2pm / 1e9, 1),
            "pnl_dollars": 1500 if dist_cw <= 10 else -500,
            "pnl_pts": round((1500 if dist_cw <= 10 else -500) / 100, 1),
            "win": int(dist_cw <= 10),
        })

    return pd.DataFrame(trades_2pm), pd.DataFrame(trades_cw_close)


# ── SETUP 4: SWING TRADE (EOD -> next day) ──────────────────

def backtest_swing(feat, conn):
    """EOD extreme gamma -> next day mean reversion trade."""
    dates = sorted(feat.keys())
    trades = []

    for i in range(len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i + 1]

        snaps_today = load_day_snaps(conn, today)
        snaps_tomorrow = load_day_snaps(conn, tomorrow)

        if len(snaps_today) < 5 or len(snaps_tomorrow) < 10:
            continue

        ft_today = feat[today]
        ft_tomorrow = feat[tomorrow]

        eod_gamma = snaps_today[-1]["net_gamma"]
        eod_gamma_bn = eod_gamma / 1e9
        avg_gamma = ft_today["avg_gamma"]

        # Check for 3+ consecutive neg gamma days
        consec_neg = False
        if i >= 2:
            d2 = feat[dates[i - 2]].get("avg_gamma", 0)
            d1 = feat[dates[i - 1]].get("avg_gamma", 0)
            d0 = avg_gamma
            if d2 < -2e9 and d1 < -2e9 and d0 < -2e9:
                consec_neg = True

        # Determine signal
        signal = None
        direction = 0
        if eod_gamma < -5e9:
            signal = "EOD_NEG"
            direction = 1  # LONG mean reversion
        elif consec_neg and eod_gamma < -2e9:
            signal = "CONSEC_NEG"
            direction = 1  # LONG
        elif eod_gamma > 5e9:
            signal = "EOD_POS"
            direction = -1  # SHORT (pin break)

        if signal is None:
            continue

        # Next day execution
        entry = snaps_tomorrow[0]["curr_price"]
        today_low = ft_today["day_low"]
        today_high = ft_today["day_high"]
        today_close = ft_today["day_close"]
        today_avg_price = (today_high + today_low + today_close) / 3.0

        if direction == 1:
            stop = today_low - 10
            # Target: prior day VWAP, but at least 15pts above entry
            target = max(today_avg_price, entry + 15)
        else:
            stop = today_high + 10
            # Target: prior day VWAP, but at least 15pts below entry
            target = min(today_avg_price, entry - 15)

        exit_price = entry
        exit_reason = "eod"
        time_exit_min = et2pt(14, 0)

        for s in snaps_tomorrow:
            px = s["curr_price"]
            t = pt_min(s["time_pt"])

            # Stop
            if direction == 1 and px <= stop:
                exit_price = px
                exit_reason = "stop"
                break
            if direction == -1 and px >= stop:
                exit_price = px
                exit_reason = "stop"
                break

            # Target
            if direction == 1 and px >= target:
                exit_price = px
                exit_reason = "target"
                break
            if direction == -1 and px <= target:
                exit_price = px
                exit_reason = "target"
                break

            # Time exit
            if t >= time_exit_min:
                exit_price = px
                exit_reason = "time_exit"
                break

        if exit_reason == "eod":
            exit_price = snaps_tomorrow[-1]["curr_price"]

        pnl_pts = (exit_price - entry) * direction
        win = pnl_pts > 0

        trades.append({
            "date": tomorrow,  # trade date is next day
            "signal_date": today,
            "setup": "SWING",
            "signal": signal,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "target": round(target, 2),
            "stop": round(stop, 2),
            "exit_reason": exit_reason,
            "pnl_pts": round(pnl_pts, 2),
            "win": int(win),
            "eod_gamma_bn": round(eod_gamma_bn, 1),
        })

    return pd.DataFrame(trades)


# ── COMBINED STRATEGY ────────────────────────────────────────

def combine_strategies(coil_df, morning_df, pin_df, swing_df):
    """Apply priority rules.

    Intraday: max 1 per day — COIL > MORNING > PIN.
    Swing: separate overnight book — never on same day as intraday.
    """
    # Collect all unique dates across all setups
    all_dates = set()
    for df in [coil_df, morning_df, pin_df, swing_df]:
        if len(df) > 0:
            all_dates.update(df["date"].tolist())
    all_dates = sorted(all_dates)

    # Build lookup sets
    coil_dates = set(coil_df["date"].tolist()) if len(coil_df) > 0 else set()
    morning_dates = set(morning_df["date"].tolist()) if len(morning_df) > 0 else set()
    pin_dates = set(pin_df["date"].tolist()) if len(pin_df) > 0 else set()
    swing_dates = set(swing_df["date"].tolist()) if len(swing_df) > 0 else set()

    combined = []

    for date_str in all_dates:
        intraday_used = False

        # Priority 1: COIL
        if date_str in coil_dates:
            row = coil_df[coil_df["date"] == date_str].iloc[0]
            combined.append({
                "date": date_str, "setup": "COIL",
                "direction": row["direction"],
                "entry": row["entry"], "exit": row["exit"],
                "exit_reason": row["exit_reason"],
                "pnl_pts": row["pnl_pts"], "win": row["win"],
            })
            intraday_used = True

        # Priority 2: MORNING (only if no coil)
        if not intraday_used and date_str in morning_dates:
            row = morning_df[morning_df["date"] == date_str].iloc[0]
            combined.append({
                "date": date_str, "setup": "MORNING",
                "direction": row["direction"],
                "entry": row["entry"], "exit": row["exit"],
                "exit_reason": row["exit_reason"],
                "pnl_pts": row["pnl_pts"], "win": row["win"],
            })
            intraday_used = True

        # Priority 3: PIN (only if no coil/morning)
        if not intraday_used and date_str in pin_dates:
            row = pin_df[pin_df["date"] == date_str].iloc[0]
            combined.append({
                "date": date_str, "setup": "PIN",
                "direction": "NEUTRAL",
                "entry": row["price_2pm"], "exit": row["price_close"],
                "exit_reason": "close",
                "pnl_pts": row["pnl_pts"], "win": row["win"],
            })
            intraday_used = True

        # SWING: separate book — only if NO intraday trade this day
        if not intraday_used and date_str in swing_dates:
            row = swing_df[swing_df["date"] == date_str].iloc[0]
            combined.append({
                "date": date_str, "setup": "SWING",
                "direction": row["direction"],
                "entry": row["entry"], "exit": row["exit"],
                "exit_reason": row["exit_reason"],
                "pnl_pts": row["pnl_pts"], "win": row["win"],
            })

    cdf = pd.DataFrame(combined)
    if len(cdf) > 0:
        cdf["cumulative_pnl"] = cdf["pnl_pts"].cumsum()
    return cdf


# ── REPORTING ────────────────────────────────────────────────

def report_setup(name, df, extra_fn=None):
    """Print stats for one setup."""
    print(f"\n{'='*70}")
    print(f"  SETUP: {name}")
    print(f"{'='*70}")

    if len(df) == 0:
        print("  No trades found.")
        return

    n = len(df)
    wins = df["win"].sum()
    wr = wins / n * 100
    avg_pnl = df["pnl_pts"].mean()
    total_pnl = df["pnl_pts"].sum()
    best = df.loc[df["pnl_pts"].idxmax()]
    worst = df.loc[df["pnl_pts"].idxmin()]

    print(f"  Instances: {n}")
    print(f"  Win rate:  {wins}/{n} ({wr:.0f}%)")
    print(f"  Avg P&L:   {avg_pnl:+.1f} pts")
    print(f"  Total P&L: {total_pnl:+.1f} pts")
    print(f"  Best:      {best['pnl_pts']:+.1f} pts on {best['date']}")
    print(f"  Worst:     {worst['pnl_pts']:+.1f} pts on {worst['date']}")

    # Exit reasons
    if "exit_reason" in df.columns:
        reasons = df["exit_reason"].value_counts()
        print(f"  Exit reasons: {dict(reasons)}")

    if extra_fn:
        extra_fn(df)

    # Trade list
    print()
    cols_to_show = [c for c in ["date", "direction", "entry", "exit", "exit_reason", "pnl_pts", "win"] if c in df.columns]
    extra_cols = [c for c in ["gap_pts", "flip_bn", "group", "f30_ratio", "open_gamma_bn",
                               "gamma_confirms", "signal", "eod_gamma_bn",
                               "distance", "gamma_2pm_bn", "wall_drift"] if c in df.columns]
    show = cols_to_show + extra_cols[:3]
    print(df[show].to_string(index=False))


def main():
    print("=" * 70)
    print("  4-SETUP INDEPENDENT BACKTEST")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("\nLoading daily features...")
    feat = load_features()
    print(f"  {len(feat)} trading days loaded")

    # ── SETUP 1: COIL ──
    print("\n[1/4] Running COIL backtest...")
    coil_df = backtest_coil(feat, conn)
    report_setup("COIL / GAP TRADE", coil_df)

    # ── SETUP 2: MORNING BET ──
    print("\n[2/4] Running MORNING BET backtest...")
    morning_df = backtest_morning(feat, conn, use_gamma_filter=True)

    def morning_extra(df):
        # By group
        for g in sorted(df["group"].unique()):
            sub = df[df["group"] == g]
            print(f"  Group {g}: {len(sub)} trades, "
                  f"win {sub['win'].mean()*100:.0f}%, "
                  f"avg {sub['pnl_pts'].mean():+.1f} pts")

        # Gamma regime
        for gc in [True, False]:
            sub = df[df["gamma_confirms"] == gc]
            if len(sub) > 0:
                label = "Gamma confirms" if gc else "No gamma confirm"
                print(f"  {label}: {len(sub)} trades, "
                      f"win {sub['win'].mean()*100:.0f}%, "
                      f"avg {sub['pnl_pts'].mean():+.1f} pts")

    report_setup("MORNING BET (with gamma filter)", morning_df, morning_extra)

    # Without gamma filter
    morning_no_gamma = backtest_morning(feat, conn, use_gamma_filter=False)
    print(f"\n  --- WITHOUT gamma filter ---")
    if len(morning_no_gamma) > 0:
        w = morning_no_gamma["win"].sum()
        n = len(morning_no_gamma)
        print(f"  Instances: {n}")
        print(f"  Win rate:  {w}/{n} ({w/n*100:.0f}%)")
        print(f"  Avg P&L:   {morning_no_gamma['pnl_pts'].mean():+.1f} pts")
        print(f"  Total P&L: {morning_no_gamma['pnl_pts'].sum():+.1f} pts")
        # Comparison
        gamma_yes = morning_no_gamma[morning_no_gamma["gamma_confirms"] == True]
        gamma_no = morning_no_gamma[morning_no_gamma["gamma_confirms"] == False]
        if len(gamma_yes) > 0:
            print(f"    With gamma:    {len(gamma_yes)} trades, "
                  f"win {gamma_yes['win'].mean()*100:.0f}%, "
                  f"avg {gamma_yes['pnl_pts'].mean():+.1f} pts")
        if len(gamma_no) > 0:
            print(f"    Without gamma: {len(gamma_no)} trades, "
                  f"win {gamma_no['win'].mean()*100:.0f}%, "
                  f"avg {gamma_no['pnl_pts'].mean():+.1f} pts")
        print(f"  GAMMA FILTER VERDICT: ", end="")
        if len(gamma_yes) > 0 and len(gamma_no) > 0:
            if gamma_yes["pnl_pts"].mean() > gamma_no["pnl_pts"].mean():
                print("HELPS (filtered trades outperform)")
            else:
                print("HURTS (filtered out trades were better)")
        else:
            print("Insufficient data to compare")

    # ── SETUP 3: PIN ──
    print("\n[3/4] Running PIN backtest...")
    pin_df, pin_cw_close_df = backtest_pin(feat, conn)

    def pin_extra(df):
        if "wall_drift" in df.columns:
            avg_drift = df["wall_drift"].mean()
            print(f"  Avg wall drift (2pm->close): {avg_drift:+.0f} pts")
        # Distance distribution
        if "distance" in df.columns and len(df) > 0:
            print(f"  Distance from 2pm wall at close:")
            for thresh in [5, 10, 15, 20]:
                ct = (df["distance"] <= thresh).sum()
                print(f"    within {thresh:>2}pts: {ct}/{len(df)} ({ct/len(df)*100:.0f}%)")

    report_setup("PIN TRADE (2pm wall, 10pt win zone)", pin_df, pin_extra)

    # Comparison with closing wall (circular)
    print(f"\n  --- Closing wall comparison (circular - wall follows price) ---")
    if len(pin_cw_close_df) > 0:
        w = pin_cw_close_df["win"].sum()
        n = len(pin_cw_close_df)
        pnl = pin_cw_close_df["pnl_dollars"].sum()
        print(f"  Instances: {n}, Win rate: {w}/{n} ({w/n*100:.0f}%), Total P&L: ${pnl:+,d}")
        print(f"  NOTE: Closing wall always near price — this is circular, not a real trade.")

    # ── SETUP 4: SWING ──
    print("\n[4/4] Running SWING backtest...")
    swing_df = backtest_swing(feat, conn)

    def swing_extra(df):
        # Long vs short
        for d in ["LONG", "SHORT"]:
            sub = df[df["direction"] == d]
            if len(sub) > 0:
                print(f"  {d}: {len(sub)} trades, "
                      f"win {sub['win'].mean()*100:.0f}%, "
                      f"avg {sub['pnl_pts'].mean():+.1f} pts, "
                      f"total {sub['pnl_pts'].sum():+.1f} pts")
        # By signal
        for sig in df["signal"].unique():
            sub = df[df["signal"] == sig]
            if len(sub) > 0:
                print(f"  {sig}: {len(sub)} trades, "
                      f"win {sub['win'].mean()*100:.0f}%, "
                      f"avg {sub['pnl_pts'].mean():+.1f} pts")

    report_setup("SWING TRADE (EOD gamma extreme)", swing_df, swing_extra)

    # ══════════════════════════════════════════════════════════
    #  COMBINED STRATEGY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  COMBINED 4-SETUP STRATEGY")
    print(f"{'='*70}")

    combined = combine_strategies(coil_df, morning_df, pin_df, swing_df)

    if len(combined) == 0:
        print("  No combined trades.")
        conn.close()
        return

    n = len(combined)
    wins = combined["win"].sum()
    total_pnl = combined["pnl_pts"].sum()
    pnl_arr = combined["pnl_pts"].values

    print(f"\n  Total trade days: {n}")
    print(f"  Win rate: {wins}/{n} ({wins/n*100:.0f}%)")
    print(f"  Total P&L: {total_pnl:+.1f} pts")
    print(f"  Avg P&L/trade: {pnl_arr.mean():+.1f} pts")

    # By setup
    print(f"\n  {'SETUP':<12} {'TRADES':>7} {'WIN%':>6} {'AVG':>8} {'TOTAL':>9}")
    print("  " + "-" * 45)
    for setup in ["COIL", "MORNING", "PIN", "SWING"]:
        sub = combined[combined["setup"] == setup]
        if len(sub) == 0:
            print(f"  {setup:<12} {'0':>7}")
            continue
        print(f"  {setup:<12} {len(sub):>7} {sub['win'].mean()*100:>5.0f}% "
              f"{sub['pnl_pts'].mean():>+7.1f} {sub['pnl_pts'].sum():>+8.1f}")

    # Monthly breakdown
    combined["month"] = combined["date"].str[:7]
    print(f"\n  {'MONTH':<10} {'TRADES':>7} {'WINS':>6} {'LOSSES':>7} "
          f"{'P&L':>9} {'WIN%':>6}")
    print("  " + "-" * 50)
    for month in sorted(combined["month"].unique()):
        m = combined[combined["month"] == month]
        w = m["win"].sum()
        l = len(m) - w
        print(f"  {month:<10} {len(m):>7} {w:>6} {l:>7} "
              f"{m['pnl_pts'].sum():>+8.1f} {w/len(m)*100:>5.0f}%")

    # Trades per week
    combined["week"] = pd.to_datetime(combined["date"]).dt.isocalendar().week
    weeks = combined.groupby("week").size()
    print(f"\n  Avg trades per week: {weeks.mean():.1f}")

    # Key metrics
    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_dd = drawdown.max()

    gross_wins = pnl_arr[pnl_arr > 0].sum()
    gross_losses = abs(pnl_arr[pnl_arr <= 0].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Consecutive streaks
    max_w, max_l, cur_w, cur_l = 0, 0, 0, 0
    for p in pnl_arr:
        if p > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)

    best_idx = np.argmax(pnl_arr)
    worst_idx = np.argmin(pnl_arr)

    # Total calendar days in range
    first_date = pd.to_datetime(combined["date"].iloc[0])
    last_date = pd.to_datetime(combined["date"].iloc[-1])
    total_cal_days = (last_date - first_date).days
    total_trading_days = len(feat)

    print(f"\n  KEY METRICS:")
    print(f"  {'='*45}")
    print(f"  Total trading days in data: {total_trading_days}")
    print(f"  Days with trades:           {n} ({n/total_trading_days*100:.0f}%)")
    print(f"  Win rate:                   {wins/n*100:.0f}%")
    print(f"  Total P&L:                  {total_pnl:+.1f} pts")
    print(f"  Avg P&L per trade:          {pnl_arr.mean():+.1f} pts")
    print(f"  Max consecutive wins:       {max_w}")
    print(f"  Max consecutive losses:     {max_l}")
    print(f"  Max drawdown:               {max_dd:.1f} pts")
    print(f"  Profit factor:              {profit_factor:.2f}")
    print(f"  Best trade:                 {pnl_arr[best_idx]:+.1f} pts on {combined.iloc[best_idx]['date']}")
    print(f"  Worst trade:                {pnl_arr[worst_idx]:+.1f} pts on {combined.iloc[worst_idx]['date']}")

    # ── SAVE FILES ──
    print(f"\n  Saving reports...")

    coil_df.to_csv(os.path.join(REPORTS, "setup1_coil_trades.csv"), index=False)
    morning_df.to_csv(os.path.join(REPORTS, "setup2_morning_trades.csv"), index=False)
    pin_df.to_csv(os.path.join(REPORTS, "setup3_pin_trades.csv"), index=False)
    swing_df.to_csv(os.path.join(REPORTS, "setup4_swing_trades.csv"), index=False)
    combined.to_csv(os.path.join(REPORTS, "combined_4setup_strategy.csv"), index=False)

    eq = combined[["date", "setup", "pnl_pts", "cumulative_pnl"]].copy()
    eq.to_csv(os.path.join(REPORTS, "combined_4setup_equity.csv"), index=False)

    summary = {
        "date_range": f"{combined['date'].iloc[0]} to {combined['date'].iloc[-1]}",
        "total_trading_days": int(total_trading_days),
        "trade_days": int(n),
        "trade_pct": round(n / total_trading_days * 100, 1),
        "win_rate": round(wins / n * 100, 1),
        "total_pnl_pts": round(float(total_pnl), 1),
        "avg_pnl_pts": round(float(pnl_arr.mean()), 1),
        "max_drawdown_pts": round(float(max_dd), 1),
        "profit_factor": round(float(profit_factor), 2),
        "max_consec_wins": int(max_w),
        "max_consec_losses": int(max_l),
        "by_setup": {},
    }
    for setup in ["COIL", "MORNING", "PIN", "SWING"]:
        sub = combined[combined["setup"] == setup]
        if len(sub) > 0:
            summary["by_setup"][setup] = {
                "trades": int(len(sub)),
                "win_rate": round(float(sub["win"].mean() * 100), 1),
                "total_pnl": round(float(sub["pnl_pts"].sum()), 1),
                "avg_pnl": round(float(sub["pnl_pts"].mean()), 1),
            }

    with open(os.path.join(REPORTS, "combined_4setup_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Equity curve chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle("4-Setup Combined Strategy - Equity Curve (pts)", fontsize=14, fontweight="bold")

        dates_dt = pd.to_datetime(combined["date"])
        cum_pnl = combined["cumulative_pnl"].values

        # Color line segments by setup
        setup_colors = {"COIL": "purple", "MORNING": "blue", "PIN": "orange", "SWING": "green"}

        ax1.plot(dates_dt, cum_pnl, color="black", linewidth=1.5)
        ax1.fill_between(dates_dt, cum_pnl, where=(cum_pnl >= 0), alpha=0.2, color="green")
        ax1.fill_between(dates_dt, cum_pnl, where=(cum_pnl < 0), alpha=0.2, color="red")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax1.set_ylabel("Cumulative P&L (pts)")
        ax1.grid(True, alpha=0.3)

        # Daily P&L bars colored by setup
        for setup, color in setup_colors.items():
            mask = combined["setup"] == setup
            if mask.any():
                ax2.bar(dates_dt[mask], combined.loc[mask, "pnl_pts"],
                        color=color, alpha=0.7, width=1, label=setup)

        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax2.set_ylabel("Daily P&L (pts)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS, "combined_4setup_equity.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved equity curve chart")
    except Exception as e:
        print(f"  Chart error: {e}")

    for f in ["setup1_coil_trades.csv", "setup2_morning_trades.csv",
              "setup3_pin_trades.csv", "setup4_swing_trades.csv",
              "combined_4setup_strategy.csv", "combined_4setup_equity.csv",
              "combined_4setup_summary.json", "combined_4setup_equity.png"]:
        path = os.path.join(REPORTS, f)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"    [{status}] {f}")

    conn.close()
    print(f"\n  BACKTEST COMPLETE.\n")


if __name__ == "__main__":
    main()
