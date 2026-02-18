"""Backtest: Swing trade signals — multi-day GEX patterns (2-5 day holds)."""
import sqlite3

conn = sqlite3.connect("C:/Code/gex_telegram/gex_data.db")
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT date_pt,
           MIN(curr_price) as low, MAX(curr_price) as high,
           MIN(net_gamma/1e9) as min_g, MAX(net_gamma/1e9) as max_g,
           AVG(net_gamma/1e9) as avg_g
    FROM gex_snapshots WHERE session_tag='RTH'
    GROUP BY date_pt ORDER BY date_pt
""").fetchall()

days = []
for r in rows:
    first = conn.execute(
        "SELECT curr_price, net_gamma/1e9 as g, put_floor, call_wall FROM gex_snapshots "
        "WHERE date_pt=? AND session_tag='RTH' ORDER BY timestamp_pt LIMIT 1",
        (r["date_pt"],),
    ).fetchone()
    last = conn.execute(
        "SELECT curr_price, net_gamma/1e9 as g, put_floor, call_wall FROM gex_snapshots "
        "WHERE date_pt=? AND session_tag='RTH' ORDER BY timestamp_pt DESC LIMIT 1",
        (r["date_pt"],),
    ).fetchone()
    days.append({
        "date": r["date_pt"], "open": first["curr_price"], "close": last["curr_price"],
        "low": r["low"], "high": r["high"], "range": r["high"] - r["low"],
        "g_open": first["g"], "g_close": last["g"], "g_avg": r["avg_g"],
        "g_min": r["min_g"], "g_max": r["max_g"],
        "pf": last["put_floor"], "cw": last["call_wall"],
        "change": last["curr_price"] - first["curr_price"],
    })
conn.close()

print("=" * 90)
print(f"SWING TRADE BACKTEST — Multi-day GEX Signals ({len(days)} trading days)")
print("=" * 90)

# Forward returns
for i in range(len(days)):
    for fwd, key in [(2, "fwd_2d"), (3, "fwd_3d"), (5, "fwd_5d")]:
        days[i][key] = days[min(i + fwd, len(days) - 1)]["close"] - days[i]["close"] if i + fwd < len(days) else None
    if i + 5 < len(days):
        fwd_highs = [days[i + j]["high"] for j in range(1, 6)]
        fwd_lows = [days[i + j]["low"] for j in range(1, 6)]
        days[i]["max_up_5d"] = max(fwd_highs) - days[i]["close"]
        days[i]["max_dn_5d"] = days[i]["close"] - min(fwd_lows)
    else:
        days[i]["max_up_5d"] = None
        days[i]["max_dn_5d"] = None


def analyze(label, subset, direction="LONG"):
    valid = [d for d in subset if d["fwd_5d"] is not None]
    if len(valid) < 5:
        print(f"  {label}: N={len(valid)} (insufficient)")
        return
    n = len(valid)
    if direction == "LONG":
        w2 = sum(1 for d in valid if d["fwd_2d"] > 5) / n * 100
        w3 = sum(1 for d in valid if d["fwd_3d"] > 5) / n * 100
        w5 = sum(1 for d in valid if d["fwd_5d"] > 5) / n * 100
        a2 = sum(d["fwd_2d"] for d in valid) / n
        a3 = sum(d["fwd_3d"] for d in valid) / n
        a5 = sum(d["fwd_5d"] for d in valid) / n
        mfe = sum(d["max_up_5d"] for d in valid) / n
        mae = sum(d["max_dn_5d"] for d in valid) / n
    else:
        w2 = sum(1 for d in valid if d["fwd_2d"] < -5) / n * 100
        w3 = sum(1 for d in valid if d["fwd_3d"] < -5) / n * 100
        w5 = sum(1 for d in valid if d["fwd_5d"] < -5) / n * 100
        a2 = sum(-d["fwd_2d"] for d in valid) / n
        a3 = sum(-d["fwd_3d"] for d in valid) / n
        a5 = sum(-d["fwd_5d"] for d in valid) / n
        mfe = sum(d["max_dn_5d"] for d in valid) / n
        mae = sum(d["max_up_5d"] for d in valid) / n
    tag = "PASS" if (w5 > 50 and a5 > 10) or (w5 <= 50 and a5 > 30) else ""
    marker = " <<<" if tag else ""
    print(f"  {label} ({direction})")
    print(f"    N={n}  2d: {w2:.0f}%/{a2:+.1f}  3d: {w3:.0f}%/{a3:+.1f}  5d: {w5:.0f}%/{a5:+.1f}  MFE5d={mfe:.0f}  MAE5d={mae:.0f}{marker}")


# === GAMMA CLOSE REGIME ===
print("\n--- GAMMA CLOSE REGIME -> Next 2/3/5 day returns ---")
analyze("Close G < -8Bn (deep neg)", [d for d in days if d["g_close"] < -8], "LONG")
analyze("Close G < -5Bn", [d for d in days if d["g_close"] < -5], "LONG")
analyze("Close G < -3Bn", [d for d in days if d["g_close"] < -3], "LONG")
analyze("Close G > +15Bn (very strong pos)", [d for d in days if d["g_close"] > 15], "SHORT")
analyze("Close G > +10Bn", [d for d in days if d["g_close"] > 10], "SHORT")
analyze("Close G > +5Bn", [d for d in days if d["g_close"] > 5], "SHORT")

# === GAMMA STREAKS ===
print("\n--- GAMMA STREAKS -> Buy/sell after N consecutive days ---")
for streak_len in [2, 3]:
    neg = []
    pos = []
    for i in range(streak_len, len(days)):
        if all(days[i - j]["g_close"] < -3 for j in range(streak_len)):
            neg.append(days[i])
        if all(days[i - j]["g_close"] > 5 for j in range(streak_len)):
            pos.append(days[i])
    analyze(f"{streak_len}-day neg gamma streak (G<-3Bn)", neg, "LONG")
    analyze(f"{streak_len}-day pos gamma streak (G>+5Bn)", pos, "SHORT")

# === WALL DISTANCE ===
print("\n--- PRICE vs WALLS at close -> Swing direction ---")
analyze("Close near put floor (<15pts)", [d for d in days if d["pf"] and d["close"] - d["pf"] < 15 and d["close"] - d["pf"] > 0], "LONG")
analyze("Close near call wall (<15pts)", [d for d in days if d["cw"] and d["cw"] - d["close"] < 15 and d["cw"] - d["close"] > 0], "SHORT")
analyze("Close BELOW put floor", [d for d in days if d["pf"] and d["close"] < d["pf"]], "LONG")
analyze("Close ABOVE call wall", [d for d in days if d["cw"] and d["close"] > d["cw"]], "SHORT")

# === RANGE SIGNALS ===
print("\n--- RANGE-BASED SIGNALS ---")
analyze("Wide range day (>70pts)", [d for d in days if d["range"] > 70], "LONG")
analyze("Narrow range (<25pts)", [d for d in days if d["range"] < 25], "LONG")
analyze("Down >30pts", [d for d in days if d["change"] < -30], "LONG")
analyze("Down >50pts", [d for d in days if d["change"] < -50], "LONG")
analyze("Up >30pts", [d for d in days if d["change"] > 30], "SHORT")
analyze("Up >50pts", [d for d in days if d["change"] > 50], "SHORT")

# === GAMMA FLIP ===
print("\n--- GAMMA FLIP DAYS ---")
flip_neg = [days[i] for i in range(1, len(days)) if days[i - 1]["g_close"] > 0 and days[i]["g_close"] < -3]
flip_pos = [days[i] for i in range(1, len(days)) if days[i - 1]["g_close"] < 0 and days[i]["g_close"] > 5]
analyze("Gamma flipped to neg (was pos, now <-3Bn)", flip_neg, "SHORT")
analyze("Gamma flipped to pos (was neg, now >+5Bn)", flip_pos, "LONG")

# === COMBINED ===
print("\n--- COMBINED (best potential combos) ---")
analyze("Down >20pts + G close < -5Bn", [d for d in days if d["change"] < -20 and d["g_close"] < -5], "LONG")
analyze("Down >20pts + G close < -8Bn", [d for d in days if d["change"] < -20 and d["g_close"] < -8], "LONG")
analyze("Close below PF + G < -5Bn", [d for d in days if d["pf"] and d["close"] < d["pf"] and d["g_close"] < -5], "LONG")
analyze("Up >20pts + G close > +10Bn", [d for d in days if d["change"] > 20 and d["g_close"] > 10], "SHORT")
analyze("Close above CW + G > +10Bn", [d for d in days if d["cw"] and d["close"] > d["cw"] and d["g_close"] > 10], "SHORT")
analyze("2 down days + G < -5Bn both", [days[i] for i in range(2, len(days)) if days[i-1]["change"] < -10 and days[i]["change"] < -10 and days[i]["g_close"] < -5], "LONG")
analyze("3+ down days in a row", [days[i] for i in range(3, len(days)) if all(days[i-j]["change"] < 0 for j in range(3))], "LONG")

# === TODAY's CONTEXT ===
print("\n" + "=" * 90)
print("TODAY's SETUP (Feb 18)")
print("=" * 90)
if days:
    today = days[-1]
    print(f"Close: {today['close']:.0f}  Gamma: {today['g_close']:+.1f}Bn  Change: {today['change']:+.0f}  Range: {today['range']:.0f}")
    print(f"Walls: PF={today['pf']}  CW={today['cw']}")
    print()
    # Which signals are active?
    active = []
    if today["g_close"] > 10:
        active.append("G > +10Bn -> SHORT bias next 2-5 days")
    if today["g_close"] > 15:
        active.append("G > +15Bn -> STRONG SHORT bias")
    if today["change"] > 30:
        active.append("Up >30pts -> potential SHORT swing")
    if today["change"] > 50:
        active.append("Up >50pts -> potential SHORT swing")
    if today["cw"] and today["close"] > today["cw"]:
        active.append(f"Close ABOVE call wall {today['cw']} -> SHORT bias")
    if today["g_close"] < -5:
        active.append("G < -5Bn -> LONG bias next 2-5 days")
    if today["change"] < -30:
        active.append("Down >30pts -> potential LONG swing")
    if not active:
        active.append("No strong swing signal active")
    for a in active:
        print(f"  -> {a}")
