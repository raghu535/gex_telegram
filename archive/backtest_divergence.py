"""Backtest: Does gamma lead price? Divergence analysis."""

import sqlite3
from collections import defaultdict

conn = sqlite3.connect("gex_data.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("""
    SELECT date_pt, timestamp_pt, curr_price, net_gamma, put_floor, call_wall
    FROM gex_snapshots WHERE session_tag='RTH'
    ORDER BY date_pt, timestamp_pt
""").fetchall()
conn.close()

by_date = defaultdict(list)
for r in rows:
    by_date[r["date_pt"]].append(dict(r))

all_snaps = []
for date, snaps in sorted(by_date.items()):
    if len(snaps) < 30:
        continue
    for i in range(15, len(snaps) - 15):
        s = snaps[i]
        price = s["curr_price"]
        gamma = s["net_gamma"] / 1e9

        p_5m = price - snaps[i - 3]["curr_price"]
        p_15m = price - snaps[i - 8]["curr_price"]
        g_5m = gamma - snaps[i - 3]["net_gamma"] / 1e9
        g_15m = gamma - snaps[i - 8]["net_gamma"] / 1e9

        p_fwd_15m = snaps[i + 8]["curr_price"] - price if i + 8 < len(snaps) else 0
        p_fwd_30m = snaps[i + 15]["curr_price"] - price if i + 15 < len(snaps) else 0

        all_snaps.append({
            "date": date, "price": price, "gamma": gamma,
            "p_5m": p_5m, "p_15m": p_15m,
            "g_5m": g_5m, "g_15m": g_15m,
            "p_fwd_15m": p_fwd_15m, "p_fwd_30m": p_fwd_30m,
        })

N = len(all_snaps)
print(f"Snapshots: {N}\n")

# =====================================================================
print("=" * 80)
print("1. WHEN GAMMA AND PRICE AGREE vs DIVERGE (15-min lookback)")
print("   Does price continue in ITS direction, or follow GAMMA direction?")
print("=" * 80)

scenarios = [
    ("CONFIRM UP:   price UP + gamma UP",
     lambda s: s["p_15m"] > 3 and s["g_15m"] > 0.3),
    ("CONFIRM DOWN: price DN + gamma DN",
     lambda s: s["p_15m"] < -3 and s["g_15m"] < -0.3),
    ("DIVERGE: price UP but gamma DN  (FAKE RALLY?)",
     lambda s: s["p_15m"] > 3 and s["g_15m"] < -0.3),
    ("DIVERGE: price DN but gamma UP  (FAKE DIP?)",
     lambda s: s["p_15m"] < -3 and s["g_15m"] > 0.3),
]

print(f"\n{'Scenario':<50} {'N':>5} {'Fwd15m':>8} {'Fwd30m':>8} {'Up%15':>7} {'Up%30':>7}")
print("-" * 82)

for label, filt in scenarios:
    subset = [s for s in all_snaps if filt(s)]
    if len(subset) < 20:
        continue
    n = len(subset)
    fwd15 = sum(s["p_fwd_15m"] for s in subset) / n
    fwd30 = sum(s["p_fwd_30m"] for s in subset) / n
    up15 = sum(1 for s in subset if s["p_fwd_15m"] > 3) / n * 100
    up30 = sum(1 for s in subset if s["p_fwd_30m"] > 3) / n * 100
    print(f"{label:<50} {n:>5} {fwd15:>+7.1f} {fwd30:>+7.1f} {up15:>6.0f}% {up30:>6.0f}%")

# =====================================================================
print(f"\n{'=' * 80}")
print("2. GAMMA TREND ALONE: Does gamma direction predict next 30m price?")
print("=" * 80)

print(f"\n--- 5-min gamma trend ---")
print(f"{'dG 5min':<35} {'N':>5} {'Fwd15m':>8} {'Fwd30m':>8} {'BigUp%':>8} {'BigDn%':>8}")
print("-" * 70)
for label, filt in [
    ("dG5m < -1Bn (sharp drop)", lambda s: s["g_5m"] < -1),
    ("dG5m -1 to -0.3Bn", lambda s: -1 <= s["g_5m"] < -0.3),
    ("dG5m -0.3 to +0.3 (flat)", lambda s: -0.3 <= s["g_5m"] <= 0.3),
    ("dG5m +0.3 to +1Bn", lambda s: 0.3 < s["g_5m"] <= 1),
    ("dG5m > +1Bn (sharp rise)", lambda s: s["g_5m"] > 1),
]:
    subset = [s for s in all_snaps if filt(s)]
    if len(subset) < 20:
        continue
    n = len(subset)
    fwd15 = sum(s["p_fwd_15m"] for s in subset) / n
    fwd30 = sum(s["p_fwd_30m"] for s in subset) / n
    bu = sum(1 for s in subset if s["p_fwd_30m"] > 10) / n * 100
    bd = sum(1 for s in subset if s["p_fwd_30m"] < -10) / n * 100
    print(f"{label:<35} {n:>5} {fwd15:>+7.1f} {fwd30:>+7.1f} {bu:>7.0f}% {bd:>7.0f}%")

print(f"\n--- 15-min gamma trend ---")
print(f"{'dG 15min':<35} {'N':>5} {'Fwd15m':>8} {'Fwd30m':>8} {'BigUp%':>8} {'BigDn%':>8}")
print("-" * 70)
for label, filt in [
    ("dG15m < -2Bn", lambda s: s["g_15m"] < -2),
    ("dG15m -2 to -0.5Bn", lambda s: -2 <= s["g_15m"] < -0.5),
    ("dG15m -0.5 to +0.5 (flat)", lambda s: -0.5 <= s["g_15m"] <= 0.5),
    ("dG15m +0.5 to +2Bn", lambda s: 0.5 < s["g_15m"] <= 2),
    ("dG15m > +2Bn", lambda s: s["g_15m"] > 2),
]:
    subset = [s for s in all_snaps if filt(s)]
    if len(subset) < 20:
        continue
    n = len(subset)
    fwd15 = sum(s["p_fwd_15m"] for s in subset) / n
    fwd30 = sum(s["p_fwd_30m"] for s in subset) / n
    bu = sum(1 for s in subset if s["p_fwd_30m"] > 10) / n * 100
    bd = sum(1 for s in subset if s["p_fwd_30m"] < -10) / n * 100
    print(f"{label:<35} {n:>5} {fwd15:>+7.1f} {fwd30:>+7.1f} {bu:>7.0f}% {bd:>7.0f}%")

# =====================================================================
print(f"\n{'=' * 80}")
print("3. DIVERGENCE DEEP DIVE: When gamma disagrees with price, who wins?")
print("=" * 80)

divs = [
    ("Rally + G falling: px>5 g<-0.5", "SHORT",
     lambda s: s["p_15m"] > 5 and s["g_15m"] < -0.5),
    ("Rally + G falling: px>10 g<-1", "SHORT",
     lambda s: s["p_15m"] > 10 and s["g_15m"] < -1),
    ("Rally + G falling fast: px>5 g<-2", "SHORT",
     lambda s: s["p_15m"] > 5 and s["g_15m"] < -2),
    ("Dip + G rising: px<-5 g>+0.5", "LONG",
     lambda s: s["p_15m"] < -5 and s["g_15m"] > 0.5),
    ("Dip + G rising: px<-10 g>+1", "LONG",
     lambda s: s["p_15m"] < -10 and s["g_15m"] > 1),
    ("Dip + G rising fast: px<-5 g>+2", "LONG",
     lambda s: s["p_15m"] < -5 and s["g_15m"] > 2),
    ("G leading UP: g15m>+1 but px flat(<3)", "LONG",
     lambda s: s["g_15m"] > 1 and abs(s["p_15m"]) < 3),
    ("G leading DN: g15m<-1 but px flat(<3)", "SHORT",
     lambda s: s["g_15m"] < -1 and abs(s["p_15m"]) < 3),
    ("G5m rising >0.5 + price not up yet", "LONG",
     lambda s: s["g_5m"] > 0.5 and s["p_5m"] < 1),
    ("G5m falling <-0.5 + price not dn yet", "SHORT",
     lambda s: s["g_5m"] < -0.5 and s["p_5m"] > -1),
]

print(f"\n{'Scenario':<45} {'Dir':<6} {'N':>5} {'Fwd15':>7} {'Fwd30':>7} {'Win%':>6}")
print("-" * 80)

for label, direction, filt in divs:
    subset = [s for s in all_snaps if filt(s)]
    if len(subset) < 10:
        continue
    n = len(subset)
    fwd15 = sum(s["p_fwd_15m"] for s in subset) / n
    fwd30 = sum(s["p_fwd_30m"] for s in subset) / n

    if direction == "LONG":
        win = sum(1 for s in subset if s["p_fwd_30m"] > 3) / n * 100
    else:
        win = sum(1 for s in subset if s["p_fwd_30m"] < -3) / n * 100

    print(f"{label:<45} {direction:<6} {n:>5} {fwd15:>+6.1f} {fwd30:>+6.1f} {win:>5.0f}%")

# =====================================================================
print(f"\n{'=' * 80}")
print("4. LEAD-LAG: When gamma moves big, how many minutes until price follows?")
print("=" * 80)

# Big gamma DROPS
print(f"\nBig gamma DROPS (dG5m < -1Bn): price response at each lag")
big_g_drops = []
for date, snaps in sorted(by_date.items()):
    for i in range(5, len(snaps) - 30):
        g_now = snaps[i]["net_gamma"] / 1e9
        g_5ago = snaps[i - 3]["net_gamma"] / 1e9
        dg = g_now - g_5ago
        if dg < -1:
            px = snaps[i]["curr_price"]
            lags = {}
            for lag_snaps, lag_label in [(1, "2m"), (3, "5m"), (5, "10m"), (8, "15m"), (15, "30m")]:
                if i + lag_snaps < len(snaps):
                    lags[lag_label] = snaps[i + lag_snaps]["curr_price"] - px
            if lags:
                big_g_drops.append(lags)

if big_g_drops:
    n = len(big_g_drops)
    print(f"  N = {n}")
    print(f"  {'Lag':>5} {'AvgMove':>10} {'Dn>3pts%':>10} {'Up>3pts%':>10}")
    print(f"  {'-'*38}")
    for lag in ["2m", "5m", "10m", "15m", "30m"]:
        vals = [d[lag] for d in big_g_drops if lag in d]
        if vals:
            avg = sum(vals) / len(vals)
            pct_dn = sum(1 for v in vals if v < -3) / len(vals) * 100
            pct_up = sum(1 for v in vals if v > 3) / len(vals) * 100
            print(f"  {lag:>5} {avg:>+9.1f} {pct_dn:>9.0f}% {pct_up:>9.0f}%")

# Big gamma RISES
print(f"\nBig gamma RISES (dG5m > +1Bn): price response at each lag")
big_g_rises = []
for date, snaps in sorted(by_date.items()):
    for i in range(5, len(snaps) - 30):
        g_now = snaps[i]["net_gamma"] / 1e9
        g_5ago = snaps[i - 3]["net_gamma"] / 1e9
        dg = g_now - g_5ago
        if dg > 1:
            px = snaps[i]["curr_price"]
            lags = {}
            for lag_snaps, lag_label in [(1, "2m"), (3, "5m"), (5, "10m"), (8, "15m"), (15, "30m")]:
                if i + lag_snaps < len(snaps):
                    lags[lag_label] = snaps[i + lag_snaps]["curr_price"] - px
            if lags:
                big_g_rises.append(lags)

if big_g_rises:
    n = len(big_g_rises)
    print(f"  N = {n}")
    print(f"  {'Lag':>5} {'AvgMove':>10} {'Up>3pts%':>10} {'Dn>3pts%':>10}")
    print(f"  {'-'*38}")
    for lag in ["2m", "5m", "10m", "15m", "30m"]:
        vals = [d[lag] for d in big_g_rises if lag in d]
        if vals:
            avg = sum(vals) / len(vals)
            pct_up = sum(1 for v in vals if v > 3) / len(vals) * 100
            pct_dn = sum(1 for v in vals if v < -3) / len(vals) * 100
            print(f"  {lag:>5} {avg:>+9.1f} {pct_up:>9.0f}% {pct_dn:>9.0f}%")

# =====================================================================
print(f"\n{'=' * 80}")
print("5. COMBINED: Divergence + gamma level (strongest signals)")
print("=" * 80)

combos = [
    # Fake rally: price up, gamma down, AND gamma is already negative
    ("FAKE RALLY: px up>5 + dG<-0.5 + G<0", "SHORT",
     lambda s: s["p_15m"] > 5 and s["g_15m"] < -0.5 and s["gamma"] < 0),
    ("FAKE RALLY: px up>5 + dG<-1 + G<-5", "SHORT",
     lambda s: s["p_15m"] > 5 and s["g_15m"] < -1 and s["gamma"] < -5),
    # Fake dip: price down, gamma up, AND gamma is negative (recovering)
    ("FAKE DIP: px dn>5 + dG>+0.5 + G<0", "LONG",
     lambda s: s["p_15m"] < -5 and s["g_15m"] > 0.5 and s["gamma"] < 0),
    ("FAKE DIP: px dn>5 + dG>+1 + G<-5", "LONG",
     lambda s: s["p_15m"] < -5 and s["g_15m"] > 1 and s["gamma"] < -5),
    # Gamma leading with level context
    ("G leading UP: dG>1 + px flat + G<0", "LONG",
     lambda s: s["g_15m"] > 1 and abs(s["p_15m"]) < 3 and s["gamma"] < 0),
    ("G leading DN: dG<-1 + px flat + G>0", "SHORT",
     lambda s: s["g_15m"] < -1 and abs(s["p_15m"]) < 3 and s["gamma"] > 0),
]

print(f"\n{'Signal':<45} {'Dir':<6} {'N':>5} {'Fwd15':>7} {'Fwd30':>7} {'Win%':>6}")
print("-" * 80)

for label, direction, filt in combos:
    subset = [s for s in all_snaps if filt(s)]
    if len(subset) < 5:
        continue
    n = len(subset)
    fwd15 = sum(s["p_fwd_15m"] for s in subset) / n
    fwd30 = sum(s["p_fwd_30m"] for s in subset) / n
    if direction == "LONG":
        win = sum(1 for s in subset if s["p_fwd_30m"] > 3) / n * 100
    else:
        win = sum(1 for s in subset if s["p_fwd_30m"] < -3) / n * 100
    print(f"{label:<45} {direction:<6} {n:>5} {fwd15:>+6.1f} {fwd30:>+6.1f} {win:>5.0f}%")
