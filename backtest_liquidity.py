"""Backtest: Liquidity grab patterns using GEX walls + price sweeps."""
import sqlite3
from collections import defaultdict

conn = sqlite3.connect("C:/Code/gex_telegram/gex_data.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("""
    SELECT date_pt, timestamp_pt, time_pt, curr_price, net_gamma/1e9 as gamma_bn,
           put_floor, call_wall
    FROM gex_snapshots WHERE session_tag='RTH'
    ORDER BY date_pt, timestamp_pt
""").fetchall()
conn.close()

by_date = defaultdict(list)
for r in rows:
    by_date[r["date_pt"]].append(dict(r))

grabs = []
for date, snaps in sorted(by_date.items()):
    if len(snaps) < 20:
        continue
    for i in range(10, len(snaps) - 15):
        s = snaps[i]
        price = s["curr_price"]
        gamma = s["gamma_bn"]
        cw = s["call_wall"]
        pf = s["put_floor"]
        if not cw or not pf or cw == 0 or pf == 0:
            continue

        p_3ago = snaps[i - 3]["curr_price"]
        p_5ago = snaps[i - 5]["curr_price"] if i >= 5 else p_3ago
        cw_5ago = snaps[i - 5]["call_wall"] if i >= 5 else cw
        pf_5ago = snaps[i - 5]["put_floor"] if i >= 5 else pf

        fwd_prices = [snaps[i + j]["curr_price"] for j in range(1, 16) if i + j < len(snaps)]
        if len(fwd_prices) < 10:
            continue
        close_30m = fwd_prices[-1] - price
        max_up = max(fwd_prices) - price
        max_dn = price - min(fwd_prices)

        # === GRAB ABOVE CALL WALL ===
        above_wall = price - cw
        was_below = p_5ago < cw_5ago
        if 0 < above_wall < 15 and was_below:
            grabs.append({
                "type": "ABOVE_WALL", "date": date, "time": s["time_pt"],
                "price": price, "level": cw, "pierce": above_wall,
                "gamma": gamma, "close_30m": close_30m,
                "max_up": max_up, "max_dn": max_dn,
            })

        # === GRAB BELOW PUT FLOOR ===
        below_floor = pf - price
        was_above = p_5ago > pf_5ago
        if 0 < below_floor < 15 and was_above:
            grabs.append({
                "type": "BELOW_FLOOR", "date": date, "time": s["time_pt"],
                "price": price, "level": pf, "pierce": below_floor,
                "gamma": gamma, "close_30m": close_30m,
                "max_up": max_up, "max_dn": max_dn,
            })

        # === NEW HIGH SWEEP ===
        if i >= 30:
            prev_high = max(snaps[j]["curr_price"] for j in range(max(0, i - 30), i - 3))
            if price > prev_high and 0 < price - prev_high < 10 and price > p_5ago + 8:
                grabs.append({
                    "type": "HIGH_SWEEP", "date": date, "time": s["time_pt"],
                    "price": price, "level": prev_high, "pierce": price - prev_high,
                    "gamma": gamma, "close_30m": close_30m,
                    "max_up": max_up, "max_dn": max_dn,
                })

        # === NEW LOW SWEEP ===
        if i >= 30:
            prev_low = min(snaps[j]["curr_price"] for j in range(max(0, i - 30), i - 3))
            if price < prev_low and 0 < prev_low - price < 10 and price < p_5ago - 8:
                grabs.append({
                    "type": "LOW_SWEEP", "date": date, "time": s["time_pt"],
                    "price": price, "level": prev_low, "pierce": prev_low - price,
                    "gamma": gamma, "close_30m": close_30m,
                    "max_up": max_up, "max_dn": max_dn,
                })

print("=" * 80)
print("LIQUIDITY GRAB ANALYSIS — GEX Walls + Price Sweeps")
print("=" * 80)

for grab_type in ["ABOVE_WALL", "BELOW_FLOOR", "HIGH_SWEEP", "LOW_SWEEP"]:
    sub = [g for g in grabs if g["type"] == grab_type]
    if not sub:
        print(f"\n--- {grab_type}: no signals ---")
        continue
    n = len(sub)

    if grab_type in ("ABOVE_WALL", "HIGH_SWEEP"):
        direction = "SHORT"
        wins = sum(1 for g in sub if g["close_30m"] < -3) / n * 100
        avg_fav = sum(g["max_dn"] for g in sub) / n
        avg_adv = sum(g["max_up"] for g in sub) / n
        avg_net = sum(-g["close_30m"] for g in sub) / n
    else:
        direction = "LONG"
        wins = sum(1 for g in sub if g["close_30m"] > 3) / n * 100
        avg_fav = sum(g["max_up"] for g in sub) / n
        avg_adv = sum(g["max_dn"] for g in sub) / n
        avg_net = sum(g["close_30m"] for g in sub) / n

    print(f"\n--- {grab_type} ({direction}) ---")
    print(f"  N={n}  Win%={wins:.0f}%  AvgFav={avg_fav:.1f}  AvgAdv={avg_adv:.1f}  AvgNet={avg_net:+.1f}")
    print(f"  Dates: {sorted(set(g['date'] for g in sub))}")

    # Break by gamma context
    if direction == "SHORT":
        combos = [
            ("G > +10Bn (strong pos)", lambda g: g["gamma"] > 10),
            ("G +5 to +10Bn", lambda g: 5 < g["gamma"] <= 10),
            ("G 0 to +5Bn", lambda g: 0 <= g["gamma"] <= 5),
            ("G < 0 (neg gamma)", lambda g: g["gamma"] < 0),
        ]
    else:
        combos = [
            ("G < -10Bn (strong neg)", lambda g: g["gamma"] < -10),
            ("G -10 to -5Bn", lambda g: -10 <= g["gamma"] < -5),
            ("G -5 to 0", lambda g: -5 <= g["gamma"] < 0),
            ("G > 0 (pos gamma)", lambda g: g["gamma"] > 0),
        ]

    print(f"  {'Gamma Context':<30} {'N':>5} {'Win%':>6} {'AvgNet':>8}")
    print(f"  {'-' * 50}")
    for label, filt in combos:
        ss = [g for g in sub if filt(g)]
        if len(ss) >= 3:
            nn = len(ss)
            if direction == "SHORT":
                w = sum(1 for g in ss if g["close_30m"] < -3) / nn * 100
                net = sum(-g["close_30m"] for g in ss) / nn
            else:
                w = sum(1 for g in ss if g["close_30m"] > 3) / nn * 100
                net = sum(g["close_30m"] for g in ss) / nn
            print(f"  {label:<30} {nn:>5} {w:>5.0f}% {net:>+7.1f}")

    # Break by pierce depth
    print(f"  {'Pierce Depth':<30} {'N':>5} {'Win%':>6} {'AvgNet':>8}")
    print(f"  {'-' * 50}")
    for label, filt in [
        ("0-3 pts (shallow)", lambda g: g["pierce"] <= 3),
        ("3-8 pts (moderate)", lambda g: 3 < g["pierce"] <= 8),
        ("8-15 pts (deep)", lambda g: g["pierce"] > 8),
    ]:
        ss = [g for g in sub if filt(g)]
        if len(ss) >= 3:
            nn = len(ss)
            if direction == "SHORT":
                w = sum(1 for g in ss if g["close_30m"] < -3) / nn * 100
                net = sum(-g["close_30m"] for g in ss) / nn
            else:
                w = sum(1 for g in ss if g["close_30m"] > 3) / nn * 100
                net = sum(g["close_30m"] for g in ss) / nn
            print(f"  {label:<30} {nn:>5} {w:>5.0f}% {net:>+7.1f}")

# Combined: wall grab + gamma context (best combos)
print(f"\n{'=' * 80}")
print("BEST COMBINATIONS — Wall Grab + Gamma")
print("=" * 80)

best = [
    ("WALL GRAB SHORT: above CW + G>+5",
     lambda g: g["type"] == "ABOVE_WALL" and g["gamma"] > 5, "SHORT"),
    ("WALL GRAB SHORT: above CW + G>+10",
     lambda g: g["type"] == "ABOVE_WALL" and g["gamma"] > 10, "SHORT"),
    ("FLOOR GRAB LONG: below PF + G<-5",
     lambda g: g["type"] == "BELOW_FLOOR" and g["gamma"] < -5, "LONG"),
    ("FLOOR GRAB LONG: below PF + G<-10",
     lambda g: g["type"] == "BELOW_FLOOR" and g["gamma"] < -10, "LONG"),
    ("HIGH SWEEP SHORT: new high + G>+5",
     lambda g: g["type"] == "HIGH_SWEEP" and g["gamma"] > 5, "SHORT"),
    ("LOW SWEEP LONG: new low + G<-5",
     lambda g: g["type"] == "LOW_SWEEP" and g["gamma"] < -5, "LONG"),
    # Contrarian: wall grab in wrong gamma
    ("WALL BREAK: above CW + G<0 (real break?)",
     lambda g: g["type"] == "ABOVE_WALL" and g["gamma"] < 0, "LONG"),
    ("FLOOR BREAK: below PF + G>0 (real break?)",
     lambda g: g["type"] == "BELOW_FLOOR" and g["gamma"] > 0, "SHORT"),
]

print(f"{'Signal':<45} {'N':>5} {'Win%':>6} {'AvgNet':>8}")
print("-" * 70)
for label, filt, direction in best:
    sub = [g for g in grabs if filt(g)]
    if len(sub) < 3:
        continue
    n = len(sub)
    if direction == "SHORT":
        w = sum(1 for g in sub if g["close_30m"] < -3) / n * 100
        net = sum(-g["close_30m"] for g in sub) / n
    else:
        w = sum(1 for g in sub if g["close_30m"] > 3) / n * 100
        net = sum(g["close_30m"] for g in sub) / n
    print(f"{label:<45} {n:>5} {w:>5.0f}% {net:>+7.1f}")
