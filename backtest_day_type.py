"""
Day-Type Backtest: Classify each RTH day at open, simulate ONE trade per day.
Usage: python backtest_day_type.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import sqlite3
from datetime import datetime, time
from collections import defaultdict
import pytz

PT = pytz.timezone("US/Pacific")
DB_PATH = "gex_data.db"

# ── Load all RTH snapshots grouped by date ──────────────────────────
conn = sqlite3.connect(DB_PATH)
rows = conn.execute("""
    SELECT timestamp_pt, date_pt, time_pt, curr_price, net_gamma,
           call_wall, put_floor
    FROM gex_snapshots
    WHERE session_tag = 'RTH'
    ORDER BY timestamp_pt
""").fetchall()
conn.close()

# Group by date
days = defaultdict(list)
for ts, date_pt, time_pt, price, gamma, cw, pf in rows:
    days[date_pt].append({
        "ts": ts,
        "time_pt": time_pt,
        "price": price,
        "gamma": gamma,
        "cw": cw,
        "pf": pf,
    })

print(f"Loaded {len(rows):,} RTH snapshots across {len(days)} trading days")
print()


# ── Day Type Classification ─────────────────────────────────────────
def classify_day(snaps):
    """Classify day type from first 5 snapshots (open window)."""
    if len(snaps) < 5:
        return "ROTATION"

    open_price = snaps[0]["price"]
    open_gamma = snaps[0]["gamma"] / 1e9
    cw = snaps[0]["cw"]
    pf = snaps[0]["pf"]
    spread = cw - pf if cw and pf else 999

    # Check first 5 snaps for early movement
    prices_5 = [s["price"] for s in snaps[:5]]
    early_range = max(prices_5) - min(prices_5)
    gamma_vals = [s["gamma"] / 1e9 for s in snaps[:5]]
    avg_gamma = sum(gamma_vals) / len(gamma_vals)

    # SWEEP DAY: negative gamma + tight spread (dealers amplify moves)
    if avg_gamma < -3 and spread < 40:
        return "SWEEP"

    # PIN DAY: strong positive gamma + tight spread (dealers pin price)
    if avg_gamma > 3 and spread < 30:
        return "PIN"

    # TREND DAY: strong early directional move + gamma supports direction
    if early_range > 8:
        if avg_gamma < -1:
            return "TREND"
        return "SWEEP"  # big moves in neg gamma = sweep territory

    # ROTATION DAY: low gamma, wide spread, no conviction
    return "ROTATION"


# ── Trade Simulations ───────────────────────────────────────────────
def sim_sweep(snaps):
    """Find first 15pt drop in 10 snaps, then 4pt reclaim. Buy at reclaim."""
    if len(snaps) < 12:
        return None

    # Find sweep window (first 2 hours of RTH = ~06:30-08:30 PT)
    sweep_snaps = [s for s in snaps if s["time_pt"] < "10:30:00"]
    if len(sweep_snaps) < 12:
        sweep_snaps = snaps[:min(60, len(snaps))]

    # Scan for 15pt drop in any 10-snap window
    for i in range(10, len(sweep_snaps)):
        window = sweep_snaps[i - 10:i + 1]
        high = max(s["price"] for s in window[:5])
        low = min(s["price"] for s in window[5:])
        drop = high - low

        if drop >= 15:
            # Now look for 4pt reclaim from the low
            low_price = low
            for j in range(i, len(sweep_snaps)):
                if sweep_snaps[j]["price"] >= low_price + 4:
                    # Entry at reclaim
                    entry = sweep_snaps[j]["price"]
                    stop = entry - 10
                    target = entry + 15

                    # Get VWAP (avg price up to this point)
                    prices_so_far = [s["price"] for s in snaps[:j + 1]]
                    vwap = sum(prices_so_far) / len(prices_so_far)
                    target = min(target, vwap) if vwap > entry else target

                    # Simulate forward from entry
                    for k in range(j + 1, len(snaps)):
                        p = snaps[k]["price"]
                        if p <= stop:
                            return {"entry": entry, "exit": stop, "pnl": -10, "reason": "stop"}
                        if p >= target:
                            return {"entry": entry, "exit": target, "pnl": target - entry, "reason": "target"}
                        # Time exit at end of sweep window
                        if snaps[k]["time_pt"] >= "10:30:00":
                            pnl = p - entry
                            return {"entry": entry, "exit": p, "pnl": pnl, "reason": "time"}

                    # If we run out of snaps
                    last_p = snaps[-1]["price"]
                    return {"entry": entry, "exit": last_p, "pnl": last_p - entry, "reason": "eod"}
            break  # Found drop but no reclaim

    return None  # No sweep setup found


def sim_pin(snaps):
    """At 2:00 PM PT (14:00), sell butterfly if price within 10pts of call_wall."""
    # Find snap closest to 14:00
    pm_snaps = [s for s in snaps if s["time_pt"] >= "12:00:00"]
    if not pm_snaps:
        return None

    # Use the snap closest to 12:00 PT (2PM ET equivalent in RTH context)
    # Actually RTH is 6:30-13:00 PT, so "2PM" in user's spec = 12:00 PT (last hour)
    late_snaps = [s for s in snaps if s["time_pt"] >= "12:00:00"]
    if not late_snaps:
        return None

    snap = late_snaps[0]
    price = snap["price"]
    cw = snap["cw"]

    if abs(price - cw) > 10:
        return None  # Not within 10pts of call wall

    # Simulate butterfly: wins $15 if close within 5pts of call_wall, loses $5 otherwise
    close_price = snaps[-1]["price"]
    if abs(close_price - cw) <= 5:
        return {"entry": price, "exit": close_price, "pnl": 15, "reason": "pin_win"}
    else:
        return {"entry": price, "exit": close_price, "pnl": -5, "reason": "pin_lose"}


def sim_trend(snaps):
    """Buy put at open. P&L = open_price - close_price, capped at -10 stop."""
    if len(snaps) < 10:
        return None

    open_price = snaps[0]["price"]
    # Track intraday for stop
    for i in range(1, len(snaps)):
        p = snaps[i]["price"]
        # Put loses when price goes UP
        if p - open_price >= 10:
            return {"entry": open_price, "exit": p, "pnl": -10, "reason": "stop"}

    close_price = snaps[-1]["price"]
    pnl = open_price - close_price
    pnl = max(pnl, -10)  # Cap loss at -10
    reason = "target" if pnl > 0 else "eod"
    return {"entry": open_price, "exit": close_price, "pnl": pnl, "reason": reason}


# ── Run Backtest ────────────────────────────────────────────────────
results = []
equity = [0]

for date in sorted(days.keys()):
    snaps = days[date]
    if len(snaps) < 5:
        continue

    day_type = classify_day(snaps)

    trade = None
    if day_type == "SWEEP":
        trade = sim_sweep(snaps)
    elif day_type == "PIN":
        trade = sim_pin(snaps)
    elif day_type == "TREND":
        trade = sim_trend(snaps)
    # ROTATION = no trade

    pnl = trade["pnl"] if trade else 0
    results.append({
        "date": date,
        "type": day_type,
        "traded": trade is not None,
        "pnl": round(pnl, 2),
        "reason": trade["reason"] if trade else "no_trade",
        "snaps": len(snaps),
    })
    equity.append(equity[-1] + pnl)


# ── Analysis ────────────────────────────────────────────────────────
print("=" * 70)
print("DAY-TYPE BACKTEST RESULTS")
print("=" * 70)
print()

# By day type
type_stats = defaultdict(lambda: {"count": 0, "trades": 0, "wins": 0, "pnls": []})
for r in results:
    t = type_stats[r["type"]]
    t["count"] += 1
    if r["traded"]:
        t["trades"] += 1
        t["pnls"].append(r["pnl"])
        if r["pnl"] > 0:
            t["wins"] += 1

print(f"{'Type':<12} {'Days':>5} {'Trades':>7} {'Win%':>6} {'Avg PnL':>8} {'Total':>8}")
print("-" * 50)
total_trades = 0
total_pnl = 0
total_wins = 0
for dtype in ["SWEEP", "PIN", "TREND", "ROTATION"]:
    t = type_stats[dtype]
    avg_pnl = sum(t["pnls"]) / len(t["pnls"]) if t["pnls"] else 0
    win_pct = (t["wins"] / t["trades"] * 100) if t["trades"] > 0 else 0
    total = sum(t["pnls"])
    print(f"{dtype:<12} {t['count']:>5} {t['trades']:>7} {win_pct:>5.0f}% {avg_pnl:>+7.1f} {total:>+7.1f}")
    total_trades += t["trades"]
    total_pnl += total
    total_wins += t["wins"]

print("-" * 50)
overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
avg_overall = total_pnl / total_trades if total_trades > 0 else 0
print(f"{'TOTAL':<12} {len(results):>5} {total_trades:>7} {overall_wr:>5.0f}% {avg_overall:>+7.1f} {total_pnl:>+7.1f}")
print()

# No-trade days
no_trade_days = sum(1 for r in results if not r["traded"])
print(f"Days with NO trade: {no_trade_days}/{len(results)} ({no_trade_days/len(results)*100:.0f}%)")
print()

# Max consecutive losers
max_consec_loss = 0
current_streak = 0
for r in results:
    if r["traded"] and r["pnl"] < 0:
        current_streak += 1
        max_consec_loss = max(max_consec_loss, current_streak)
    elif r["traded"] and r["pnl"] > 0:
        current_streak = 0
print(f"Max consecutive losers: {max_consec_loss}")

# Max drawdown
peak = 0
max_dd = 0
for e in equity:
    peak = max(peak, e)
    dd = peak - e
    max_dd = max(max_dd, dd)
print(f"Max drawdown: {max_dd:.1f} pts")
print(f"Final equity: {equity[-1]:+.1f} pts")
print()

# Day-by-day detail
print("=" * 70)
print("DAY-BY-DAY LOG")
print("=" * 70)
print(f"{'Date':<12} {'Type':<10} {'PnL':>7} {'Reason':<10} {'Equity':>8}")
print("-" * 50)
eq = 0
for r in results:
    eq += r["pnl"]
    marker = "***" if r["pnl"] > 10 else "  X" if r["pnl"] < -5 else "   "
    print(f"{r['date']:<12} {r['type']:<10} {r['pnl']:>+6.1f} {r['reason']:<10} {eq:>+7.1f} {marker}")

print()
print("=" * 70)
print("EQUITY CURVE (every 5 days)")
print("=" * 70)
bar_scale = max(abs(e) for e in equity) if equity else 1
bar_scale = max(bar_scale, 1)
for i in range(0, len(equity), 5):
    if i < len(results):
        date = results[min(i, len(results) - 1)]["date"]
    else:
        date = results[-1]["date"]
    val = equity[i]
    bar_len = int(abs(val) / bar_scale * 40)
    if val >= 0:
        bar = " " * 20 + "|" + "#" * bar_len
    else:
        bar = " " * max(0, 20 - bar_len) + "#" * bar_len + "|"
    print(f"{date} {val:>+7.1f} {bar}")

# Final point
val = equity[-1]
bar_len = int(abs(val) / bar_scale * 40)
if val >= 0:
    bar = " " * 20 + "|" + "#" * bar_len
else:
    bar = " " * max(0, 20 - bar_len) + "#" * bar_len + "|"
print(f"{results[-1]['date']} {val:>+7.1f} {bar}  << FINAL")
