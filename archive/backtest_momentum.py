"""Backtest: Bullish/Bearish momentum signals — early rally, dip-and-rip, wall break."""
import sqlite3
from collections import defaultdict

conn = sqlite3.connect("C:/Code/gex_telegram/gex_data.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("""
    SELECT date_pt, time_pt, timestamp_pt, curr_price, net_gamma/1e9 as gamma_bn,
           put_floor, call_wall
    FROM gex_snapshots WHERE session_tag='RTH'
    ORDER BY date_pt, timestamp_pt
""").fetchall()
conn.close()

by_date = defaultdict(list)
for r in rows:
    by_date[r["date_pt"]].append(dict(r))

print("=" * 90)
print("MOMENTUM SIGNAL BACKTEST — Early Rally, Dip&Rip, Wall Break, Early Selloff")
print(f"Days: {len(by_date)}  Snapshots: {len(rows)}")
print("=" * 90)

# === B1: EARLY RALLY — price >15pts above open within first 30min ===
print("\n--- B1: EARLY RALLY (price >15pts from open within first 30 min) ---")
b1_signals = []
for date, snaps in sorted(by_date.items()):
    if len(snaps) < 40:
        continue
    open_px = snaps[0]["curr_price"]
    triggered = None
    for i in range(1, min(15, len(snaps))):
        if snaps[i]["curr_price"] - open_px > 15:
            triggered = i
            break
    if triggered is None:
        continue
    s = snaps[triggered]
    price_at_trigger = s["curr_price"]
    gamma_at_trigger = s["gamma_bn"]
    fwd_30 = snaps[min(triggered + 10, len(snaps)-1)]["curr_price"] - price_at_trigger
    fwd_60 = snaps[min(triggered + 20, len(snaps)-1)]["curr_price"] - price_at_trigger
    fwd_120 = snaps[min(triggered + 40, len(snaps)-1)]["curr_price"] - price_at_trigger
    eod = snaps[-1]["curr_price"] - price_at_trigger
    max_up = max(snaps[j]["curr_price"] for j in range(triggered, len(snaps))) - price_at_trigger
    max_dn = price_at_trigger - min(snaps[j]["curr_price"] for j in range(triggered, len(snaps)))

    b1_signals.append({
        "date": date, "time": s["time_pt"], "price": price_at_trigger,
        "open": open_px, "move": price_at_trigger - open_px, "gamma": gamma_at_trigger,
        "fwd_30": fwd_30, "fwd_60": fwd_60, "fwd_120": fwd_120,
        "eod": eod, "max_up": max_up, "max_dn": max_dn,
    })

if b1_signals:
    n = len(b1_signals)
    print(f"  N={n}")
    for horizon, key in [("30min", "fwd_30"), ("60min", "fwd_60"), ("120min", "fwd_120"), ("EOD", "eod")]:
        w = sum(1 for s in b1_signals if s[key] > 3) / n * 100
        avg = sum(s[key] for s in b1_signals) / n
        print(f"  {horizon:>6}: Win%={w:.0f}%  AvgNet={avg:+.1f}")
    avg_mfe = sum(s["max_up"] for s in b1_signals) / n
    avg_mae = sum(s["max_dn"] for s in b1_signals) / n
    print(f"  Avg MFE (max up)={avg_mfe:.1f}  Avg MAE (max dn)={avg_mae:.1f}")

    print("\n  By gamma context:")
    for label, filt in [
        ("Gamma > +5Bn at trigger", lambda s: s["gamma"] > 5),
        ("Gamma 0 to +5Bn", lambda s: 0 <= s["gamma"] <= 5),
        ("Gamma < 0 at trigger", lambda s: s["gamma"] < 0),
    ]:
        sub = [s for s in b1_signals if filt(s)]
        if sub:
            nn = len(sub)
            w = sum(1 for s in sub if s["eod"] > 3) / nn * 100
            net = sum(s["eod"] for s in sub) / nn
            mfe = sum(s["max_up"] for s in sub) / nn
            print(f"    {label:<30} N={nn:>3}  EOD Win%={w:>5.0f}%  EOD Net={net:>+7.1f}  MFE={mfe:.1f}")

    print("\n  Per-day detail:")
    for s in b1_signals:
        g_tag = "POS" if s["gamma"] > 0 else "NEG"
        print(f"    {s['date']} {s['time'][:8]}  open={s['open']:.0f} trigger={s['price']:.0f} move=+{s['move']:.0f}  G={s['gamma']:+.1f}Bn({g_tag})  30m={s['fwd_30']:+.1f} 60m={s['fwd_60']:+.1f} EOD={s['eod']:+.1f}  MFE={s['max_up']:.0f} MAE={s['max_dn']:.0f}")
else:
    print("  No B1 signals found")

# === B2: DIP & RIP — drops >10pts from open, then rallies >15pts from low ===
print("\n--- B2: DIP & RIP (drops >10pts from open, then rallies >15pts from low within 1hr) ---")
b2_signals = []
for date, snaps in sorted(by_date.items()):
    if len(snaps) < 40:
        continue
    open_px = snaps[0]["curr_price"]
    first_30 = snaps[:15]
    low_idx = min(range(len(first_30)), key=lambda j: first_30[j]["curr_price"])
    low_px = first_30[low_idx]["curr_price"]
    dip = open_px - low_px
    if dip < 10:
        continue
    triggered = None
    for i in range(low_idx + 1, min(low_idx + 20, len(snaps))):
        if snaps[i]["curr_price"] - low_px > 15:
            triggered = i
            break
    if triggered is None:
        continue
    s = snaps[triggered]
    price_at_trigger = s["curr_price"]
    gamma_at_trigger = s["gamma_bn"]
    fwd_30 = snaps[min(triggered + 10, len(snaps)-1)]["curr_price"] - price_at_trigger
    fwd_60 = snaps[min(triggered + 20, len(snaps)-1)]["curr_price"] - price_at_trigger
    eod = snaps[-1]["curr_price"] - price_at_trigger
    max_up = max(snaps[j]["curr_price"] for j in range(triggered, len(snaps))) - price_at_trigger
    max_dn = price_at_trigger - min(snaps[j]["curr_price"] for j in range(triggered, len(snaps)))

    b2_signals.append({
        "date": date, "time": s["time_pt"], "price": price_at_trigger,
        "open": open_px, "low": low_px, "dip": dip, "gamma": gamma_at_trigger,
        "fwd_30": fwd_30, "fwd_60": fwd_60, "eod": eod,
        "max_up": max_up, "max_dn": max_dn,
    })

if b2_signals:
    n = len(b2_signals)
    w30 = sum(1 for s in b2_signals if s["fwd_30"] > 3) / n * 100
    weod = sum(1 for s in b2_signals if s["eod"] > 3) / n * 100
    avg30 = sum(s["fwd_30"] for s in b2_signals) / n
    avgeod = sum(s["eod"] for s in b2_signals) / n
    avg_mfe = sum(s["max_up"] for s in b2_signals) / n
    print(f"  N={n}  30m Win%={w30:.0f}% Net={avg30:+.1f}  EOD Win%={weod:.0f}% Net={avgeod:+.1f}  MFE={avg_mfe:.1f}")
    for s in b2_signals:
        print(f"    {s['date']} {s['time'][:8]}  open={s['open']:.0f} low={s['low']:.0f} dip={s['dip']:.0f} trigger={s['price']:.0f}  G={s['gamma']:+.1f}Bn  30m={s['fwd_30']:+.1f} EOD={s['eod']:+.1f}  MFE={s['max_up']:.0f}")
else:
    print("  No B2 signals found")

# === B3: CALL WALL BREAK — price breaks above call wall + gamma rising ===
print("\n--- B3: CALL WALL BREAK (price > call_wall + gamma rising vs 5 snaps ago) ---")
b3_signals = []
for date, snaps in sorted(by_date.items()):
    if len(snaps) < 30:
        continue
    found = False
    for i in range(10, len(snaps) - 15):
        if found:
            break
        s = snaps[i]
        cw = s["call_wall"]
        if not cw or cw == 0:
            continue
        price = s["curr_price"]
        gamma = s["gamma_bn"]
        if price > cw + 3 and gamma > 0:
            prev = snaps[i-5]
            if prev["curr_price"] <= prev["call_wall"]:
                g_5ago = snaps[i-5]["gamma_bn"]
                if gamma > g_5ago:
                    fwd = [snaps[i+j]["curr_price"] for j in range(1, 16) if i+j < len(snaps)]
                    if len(fwd) < 10:
                        continue
                    fwd_30 = fwd[-1] - price
                    eod = snaps[-1]["curr_price"] - price
                    max_up = max(fwd) - price
                    max_dn = price - min(fwd)
                    b3_signals.append({
                        "date": date, "time": s["time_pt"], "price": price,
                        "wall": cw, "gamma": gamma, "g_change": gamma - g_5ago,
                        "fwd_30": fwd_30, "eod": eod, "max_up": max_up, "max_dn": max_dn,
                    })
                    found = True

if b3_signals:
    n = len(b3_signals)
    w30 = sum(1 for s in b3_signals if s["fwd_30"] > 3) / n * 100
    weod = sum(1 for s in b3_signals if s["eod"] > 3) / n * 100
    avg30 = sum(s["fwd_30"] for s in b3_signals) / n
    avgeod = sum(s["eod"] for s in b3_signals) / n
    avg_mfe = sum(s["max_up"] for s in b3_signals) / n
    print(f"  N={n}  30m Win%={w30:.0f}% Net={avg30:+.1f}  EOD Win%={weod:.0f}% Net={avgeod:+.1f}  MFE={avg_mfe:.1f}")
    for s in b3_signals:
        print(f"    {s['date']} {s['time'][:8]}  px={s['price']:.0f} wall={s['wall']}  G={s['gamma']:+.1f}Bn(+{s['g_change']:.1f})  30m={s['fwd_30']:+.1f} EOD={s['eod']:+.1f}  MFE={s['max_up']:.0f}")
else:
    print("  No B3 signals found")

# === BEAR MIRROR: EARLY SELLOFF — price >15pts below open in first 30min ===
print("\n--- BEAR: EARLY SELLOFF (price >15pts BELOW open in first 30min) ---")
bear_signals = []
for date, snaps in sorted(by_date.items()):
    if len(snaps) < 40:
        continue
    open_px = snaps[0]["curr_price"]
    triggered = None
    for i in range(1, min(15, len(snaps))):
        if open_px - snaps[i]["curr_price"] > 15:
            triggered = i
            break
    if triggered is None:
        continue
    s = snaps[triggered]
    price_at_trigger = s["curr_price"]
    gamma_at_trigger = s["gamma_bn"]
    # For shorts: favorable = price goes down further
    fwd_30 = price_at_trigger - snaps[min(triggered + 10, len(snaps)-1)]["curr_price"]
    fwd_60 = price_at_trigger - snaps[min(triggered + 20, len(snaps)-1)]["curr_price"]
    eod = price_at_trigger - snaps[-1]["curr_price"]
    max_fav = price_at_trigger - min(snaps[j]["curr_price"] for j in range(triggered, len(snaps)))
    max_adv = max(snaps[j]["curr_price"] for j in range(triggered, len(snaps))) - price_at_trigger

    bear_signals.append({
        "date": date, "time": s["time_pt"], "price": price_at_trigger,
        "open": open_px, "drop": open_px - price_at_trigger, "gamma": gamma_at_trigger,
        "fwd_30": fwd_30, "fwd_60": fwd_60, "eod": eod,
        "max_fav": max_fav, "max_adv": max_adv,
    })

if bear_signals:
    n = len(bear_signals)
    print(f"  N={n}")
    for horizon, key in [("30min", "fwd_30"), ("60min", "fwd_60"), ("EOD", "eod")]:
        w = sum(1 for s in bear_signals if s[key] > 3) / n * 100
        avg = sum(s[key] for s in bear_signals) / n
        print(f"  {horizon:>6}: Win%={w:.0f}%  AvgNet={avg:+.1f}")
    avg_mfe = sum(s["max_fav"] for s in bear_signals) / n
    avg_mae = sum(s["max_adv"] for s in bear_signals) / n
    print(f"  Avg MFE (max favorable)={avg_mfe:.1f}  Avg MAE (max adverse)={avg_mae:.1f}")

    print("\n  By gamma context:")
    for label, filt in [
        ("Gamma < -5Bn (strong neg)", lambda s: s["gamma"] < -5),
        ("Gamma -5 to 0", lambda s: -5 <= s["gamma"] < 0),
        ("Gamma > 0 (positive)", lambda s: s["gamma"] >= 0),
    ]:
        sub = [s for s in bear_signals if filt(s)]
        if sub:
            nn = len(sub)
            w = sum(1 for s in sub if s["eod"] > 3) / nn * 100
            net = sum(s["eod"] for s in sub) / nn
            mfe = sum(s["max_fav"] for s in sub) / nn
            print(f"    {label:<30} N={nn:>3}  EOD Win%={w:>5.0f}%  EOD Net={net:>+7.1f}  MFE={mfe:.1f}")

    print("\n  Per-day detail:")
    for s in bear_signals:
        g_tag = "NEG" if s["gamma"] < 0 else "POS"
        print(f"    {s['date']} {s['time'][:8]}  open={s['open']:.0f} trigger={s['price']:.0f} drop={s['drop']:.0f}  G={s['gamma']:+.1f}Bn({g_tag})  30m={s['fwd_30']:+.1f} 60m={s['fwd_60']:+.1f} EOD={s['eod']:+.1f}  MFE={s['max_fav']:.0f} MAE={s['max_adv']:.0f}")
else:
    print("  No bear signals found")

# === SUMMARY ===
print("\n" + "=" * 90)
print("SUMMARY — Which momentum signals meet the bar?")
print("Bar: >50% win AND >10 pts net, OR <50% win with >30 pts net")
print("=" * 90)
for name, sigs, direction in [
    ("B1: Early Rally", b1_signals, "LONG"),
    ("B2: Dip & Rip", b2_signals, "LONG"),
    ("B3: Wall Break", b3_signals, "LONG"),
    ("BEAR: Early Selloff", bear_signals, "SHORT"),
]:
    if not sigs:
        print(f"  {name:<25} — NO DATA")
        continue
    n = len(sigs)
    w = sum(1 for s in sigs if s["eod"] > 3) / n * 100
    net = sum(s["eod"] for s in sigs) / n
    mfe = sum(s["max_up" if direction == "LONG" else "max_fav"] for s in sigs) / n
    passes = (w > 50 and net > 10) or (w <= 50 and net > 30)
    tag = "PASS" if passes else "FAIL"
    print(f"  {name:<25} N={n:>3}  Win%={w:>5.0f}%  Net={net:>+7.1f}  MFE={mfe:>6.1f}  [{tag}]")
