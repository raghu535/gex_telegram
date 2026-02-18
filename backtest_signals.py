"""Backtest: What gamma conditions predict tradeable 30m moves (both longs and shorts)?"""

import sqlite3
from collections import defaultdict
from datetime import datetime
import pytz

PT = pytz.timezone("US/Pacific")

conn = sqlite3.connect("gex_data.db")
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT date_pt, timestamp_pt, curr_price, put_floor, call_wall, net_gamma
    FROM gex_snapshots
    WHERE session_tag='RTH'
    ORDER BY date_pt, timestamp_pt
""").fetchall()
conn.close()

by_date = defaultdict(list)
for r in rows:
    by_date[r["date_pt"]].append(dict(r))

# Build enriched snapshots with lookback + lookahead
all_snaps = []
for date, snaps in sorted(by_date.items()):
    for i, s in enumerate(snaps):
        price = s["curr_price"]
        gamma_bn = s["net_gamma"] / 1e9
        put_floor = s["put_floor"]
        call_wall = s["call_wall"]

        if not put_floor or not call_wall:
            continue
        spread = call_wall - put_floor
        if spread < 20:
            continue

        dist_put = price - put_floor
        dist_call = call_wall - price

        # Gamma velocity (5 snaps ~ 10min, 10 snaps ~ 20min)
        g_vel_10m = (gamma_bn - snaps[i - 5]["net_gamma"] / 1e9) if i >= 5 else 0
        g_vel_20m = (gamma_bn - snaps[i - 10]["net_gamma"] / 1e9) if i >= 10 else 0

        # Gamma acceleration
        g_accel = 0
        if i >= 10:
            vel_now = gamma_bn - snaps[i - 5]["net_gamma"] / 1e9
            vel_prev = snaps[i - 5]["net_gamma"] / 1e9 - snaps[i - 10]["net_gamma"] / 1e9
            g_accel = vel_now - vel_prev

        # Price velocity
        p_vel_20m = (price - snaps[i - 10]["curr_price"]) if i >= 10 else 0
        p_vel_10m = (price - snaps[i - 5]["curr_price"]) if i >= 5 else 0

        # Look ahead 30 min (~15 snaps)
        lookahead = snaps[i + 1 : i + 16]
        if len(lookahead) < 5:
            continue

        future_prices = [x["curr_price"] for x in lookahead]
        up_30m = max(future_prices) - price
        dn_30m = price - min(future_prices)
        close_30m = future_prices[-1] - price

        # Hour (PT)
        ts = s["timestamp_pt"]
        hour = 0
        if isinstance(ts, (int, float)):
            try:
                hour = datetime.fromtimestamp(ts, tz=PT).hour
            except Exception:
                pass

        all_snaps.append({
            "date": date, "price": price, "gamma_bn": gamma_bn,
            "g_vel_10m": g_vel_10m, "g_vel_20m": g_vel_20m,
            "g_accel": g_accel,
            "p_vel_20m": p_vel_20m, "p_vel_10m": p_vel_10m,
            "dist_put": dist_put, "dist_call": dist_call,
            "spread": spread, "hour": hour,
            "up_30m": up_30m, "dn_30m": dn_30m, "close_30m": close_30m,
        })

N = len(all_snaps)
print(f"Total enriched snapshots: {N}\n")

# =====================================================================
# 1. GAMMA VELOCITY -> 30m DIRECTIONAL MOVE
# =====================================================================
print("=" * 70)
print("1. GAMMA VELOCITY -> WHAT HAPPENS IN NEXT 30 MIN")
print("=" * 70)
print(f"{'Condition':<30} {'N':>5} {'Avg Up':>8} {'Avg Dn':>8} {'Net':>7} {'Big Up%':>8} {'Big Dn%':>8}")
print("-" * 75)

conditions = [
    ("dG < -3Bn/20m", lambda s: s["g_vel_20m"] < -3),
    ("dG -3 to -1Bn/20m", lambda s: -3 <= s["g_vel_20m"] < -1),
    ("dG -1 to 0", lambda s: -1 <= s["g_vel_20m"] < 0),
    ("dG 0 to +1", lambda s: 0 <= s["g_vel_20m"] < 1),
    ("dG +1 to +3Bn/20m", lambda s: 1 <= s["g_vel_20m"] < 3),
    ("dG > +3Bn/20m", lambda s: s["g_vel_20m"] > 3),
]

for label, filt in conditions:
    subset = [s for s in all_snaps if filt(s)]
    if len(subset) < 10:
        continue
    n = len(subset)
    avg_up = sum(s["up_30m"] for s in subset) / n
    avg_dn = sum(s["dn_30m"] for s in subset) / n
    net = sum(s["close_30m"] for s in subset) / n
    big_up = sum(1 for s in subset if s["up_30m"] > 10) / n * 100
    big_dn = sum(1 for s in subset if s["dn_30m"] > 10) / n * 100
    print(f"{label:<30} {n:>5} {avg_up:>7.1f} {avg_dn:>7.1f} {net:>+6.1f} {big_up:>7.0f}% {big_dn:>7.0f}%")

# =====================================================================
# 2. GAMMA LEVEL + VELOCITY COMBOS -> TRADEABLE SIGNALS
# =====================================================================
print(f"\n{'=' * 70}")
print("2. GAMMA-BASED TRADE SIGNALS (long/short)")
print("=" * 70)

signals = [
    # LONG signals (expect up move)
    ("LONG: G<-10Bn + dG turning up",
     lambda s: s["gamma_bn"] < -10 and s["g_vel_10m"] > 0 and s["g_vel_20m"] < 0,
     "up"),
    ("LONG: G<-5Bn + near put floor (<15)",
     lambda s: s["gamma_bn"] < -5 and s["dist_put"] < 15,
     "up"),
    ("LONG: G<-5Bn + price falling fast (>10pts/20m)",
     lambda s: s["gamma_bn"] < -5 and s["p_vel_20m"] < -10,
     "up"),
    ("LONG: G accel > +1 (gamma rising faster)",
     lambda s: s["g_accel"] > 1 and s["gamma_bn"] < 0,
     "up"),
    ("LONG: dG > +2Bn/20m (sharp gamma recovery)",
     lambda s: s["g_vel_20m"] > 2 and s["gamma_bn"] < 5,
     "up"),

    # SHORT signals (expect down move)
    ("SHORT: G flipping neg (was >0, now vel<-1)",
     lambda s: s["gamma_bn"] > -2 and s["g_vel_20m"] < -1,
     "dn"),
    ("SHORT: G<-5Bn + near call wall (<15)",
     lambda s: s["gamma_bn"] < -5 and s["dist_call"] < 15,
     "dn"),
    ("SHORT: G<-10Bn + dG still falling",
     lambda s: s["gamma_bn"] < -10 and s["g_vel_20m"] < -1,
     "dn"),
    ("SHORT: G accel < -1 (gamma falling faster)",
     lambda s: s["g_accel"] < -1 and s["gamma_bn"] < 5,
     "dn"),
    ("SHORT: Price rising into neg gamma",
     lambda s: s["gamma_bn"] < -5 and s["p_vel_20m"] > 10,
     "dn"),
]

print(f"{'Signal':<45} {'N':>5} {'AvgFav':>8} {'AvgAdv':>8} {'Win%':>6} {'AvgNet':>8}")
print("-" * 85)

for label, filt, direction in signals:
    subset = [s for s in all_snaps if filt(s)]
    if len(subset) < 10:
        # Still show with small N
        if len(subset) < 3:
            continue
    n = len(subset)
    if direction == "up":
        avg_fav = sum(s["up_30m"] for s in subset) / n
        avg_adv = sum(s["dn_30m"] for s in subset) / n
        wins = sum(1 for s in subset if s["close_30m"] > 3) / n * 100
        avg_net = sum(s["close_30m"] for s in subset) / n
    else:
        avg_fav = sum(s["dn_30m"] for s in subset) / n
        avg_adv = sum(s["up_30m"] for s in subset) / n
        wins = sum(1 for s in subset if s["close_30m"] < -3) / n * 100
        avg_net = sum(-s["close_30m"] for s in subset) / n
    print(f"{label:<45} {n:>5} {avg_fav:>7.1f} {avg_adv:>7.1f} {wins:>5.0f}% {avg_net:>+7.1f}")

# =====================================================================
# 3. TIME OF DAY VOLATILITY
# =====================================================================
print(f"\n{'=' * 70}")
print("3. VOLATILITY BY HOUR (PT) — is SPX more volatile at certain times?")
print("=" * 70)
print(f"{'Hour':<10} {'N':>6} {'Avg30mMove':>12} {'P50':>7} {'P75':>7} {'P90':>7} {'AvgGamma':>10}")
print("-" * 60)

hourly = defaultdict(list)
for s in all_snaps:
    if s["hour"] >= 6 and s["hour"] <= 13:
        hourly[s["hour"]].append(s)

for h in sorted(hourly.keys()):
    vals = hourly[h]
    n = len(vals)
    if n < 20:
        continue
    moves = sorted([max(s["up_30m"], s["dn_30m"]) for s in vals])
    avg_move = sum(moves) / n
    p50 = moves[n // 2]
    p75 = moves[int(n * 0.75)]
    p90 = moves[int(n * 0.90)]
    avg_g = sum(s["gamma_bn"] for s in vals) / n
    print(f"{h:02d}:00 PT   {n:>6} {avg_move:>11.1f} {p50:>6.1f} {p75:>6.1f} {p90:>6.1f} {avg_g:>+9.1f}")

# =====================================================================
# 4. BEST COMBINED SIGNALS - optimized for Telegram alerts
# =====================================================================
print(f"\n{'=' * 70}")
print("4. TOP TELEGRAM-WORTHY SIGNALS (high win%, low noise)")
print("=" * 70)

# Subsample: only check every 150 snaps (5min) to simulate cooldown
final_signals = [
    # Rubber band long: deep neg gamma + approaching floor + gamma starting to recover
    ("RUBBER BAND LONG",
     lambda s: s["gamma_bn"] < -8 and s["dist_put"] < 20 and s["g_vel_10m"] > -0.5,
     "up"),
    # Gamma squeeze short: neg gamma + near call wall
    ("SQUEEZE WARNING SHORT",
     lambda s: s["gamma_bn"] < -5 and s["dist_call"] < 20 and s["p_vel_20m"] > 5,
     "dn"),
    # Gamma crash: gamma falling fast + already negative
    ("GAMMA CRASH SHORT",
     lambda s: s["gamma_bn"] < -5 and s["g_vel_20m"] < -2,
     "dn"),
    # Recovery long: gamma was deeply negative, now snapping back
    ("GAMMA RECOVERY LONG",
     lambda s: s["gamma_bn"] < 0 and s["g_vel_20m"] > 2,
     "up"),
]

print(f"{'Signal':<30} {'N':>5} {'AvgFav':>8} {'AvgAdv':>8} {'Win%':>6} {'RR':>6} {'AvgNet':>8}")
print("-" * 75)

for label, filt, direction in final_signals:
    subset = [s for s in all_snaps if filt(s)]
    if len(subset) < 5:
        continue
    n = len(subset)
    if direction == "up":
        avg_fav = sum(s["up_30m"] for s in subset) / n
        avg_adv = sum(s["dn_30m"] for s in subset) / n
        wins = sum(1 for s in subset if s["close_30m"] > 3) / n * 100
        avg_net = sum(s["close_30m"] for s in subset) / n
    else:
        avg_fav = sum(s["dn_30m"] for s in subset) / n
        avg_adv = sum(s["up_30m"] for s in subset) / n
        wins = sum(1 for s in subset if s["close_30m"] < -3) / n * 100
        avg_net = sum(-s["close_30m"] for s in subset) / n
    rr = avg_fav / avg_adv if avg_adv > 0 else 99
    print(f"{label:<30} {n:>5} {avg_fav:>7.1f} {avg_adv:>7.1f} {wins:>5.0f}% {rr:>5.1f} {avg_net:>+7.1f}")
