"""
GAMMA_SNAP Signal Simulator - Updated Telegram Rules
Simulates all signal tiers across GEX historical RTH data.
"""

import sqlite3
from datetime import datetime, timezone
from collections import defaultdict

DB_PATH = r"C:\Code\gex_telegram\gex_data.db"

COOLDOWNS = {
    "T1_LONG": 600,
    "T2_LONG": 600,
    "T3_LONG": 900,
    "T4_LONG": 600,
    "S1_SHORT": 900,
}

TELEGRAM_TIERS = {"T1_LONG", "T3_LONG", "S1_SHORT"}
DISCORD_ONLY = {"T2_LONG", "T4_LONG"}


def load_data():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT date_pt, timestamp_pt, curr_price, net_gamma, put_floor, call_wall "
        "FROM gex_snapshots WHERE session_tag = ? ORDER BY date_pt, timestamp_pt",
        ("RTH",)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def ts_to_str(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def run_simulation():
    rows = load_data()
    print(f"Loaded {len(rows)} RTH snapshots")
    print()

    by_date = defaultdict(list)
    for r in rows:
        by_date[r[0]].append(r)

    tier_stats = {t: {"total": 0, "telegram": 0, "wins": 0, "nets": []} for t in COOLDOWNS}
    day_stats = {}
    t1_day_events = defaultdict(list)

    for date_pt in sorted(by_date):
        snaps = by_date[date_pt]
        n = len(snaps)
        if n < 26:
            continue

        last_fire = {t: -1e9 for t in COOLDOWNS}
        day_rec = {"T1_tg": 0, "T3_tg": 0, "S1_tg": 0, "discord_total": 0, "T1_nets": []}

        for i in range(10, n - 15):
            _, ts, price, net_gamma, pf, cw = snaps[i]
            gamma_bn = net_gamma / 1e9

            _, _, p10, ng10, _, _ = snaps[i - 10]
            dP_20m = price - p10
            dG_20m = (net_gamma - ng10) / 1e9

            _, _, p8, ng8, _, _ = snaps[i - 8]
            dP_15m = price - p8
            dG_15m = (net_gamma - ng8) / 1e9

            _, _, p15a, _, _, _ = snaps[i + 15]
            close_30m = p15a - price

            fired_long = False
            signals = []

            if gamma_bn < -5 and dP_20m < -15:
                signals.append("T1_LONG")
                fired_long = True
            if not fired_long and gamma_bn < -5 and dP_20m < -10:
                signals.append("T2_LONG")
                fired_long = True
            if not fired_long and gamma_bn < -10 and dP_20m < -5:
                signals.append("T3_LONG")
                fired_long = True
            if not fired_long and gamma_bn < -15:
                signals.append("T4_LONG")
                fired_long = True
            if not fired_long and dP_15m >= 10 and dG_15m <= -1:
                signals.append("S1_SHORT")

            for sig in signals:
                cd = COOLDOWNS[sig]
                if ts - last_fire[sig] < cd:
                    continue
                last_fire[sig] = ts
                tier_stats[sig]["total"] += 1

                is_win = (close_30m > 3) if "LONG" in sig else (close_30m < -3)
                tier_stats[sig]["nets"].append(close_30m)
                if is_win:
                    tier_stats[sig]["wins"] += 1

                if sig in TELEGRAM_TIERS:
                    tier_stats[sig]["telegram"] += 1
                    if sig == "T1_LONG":
                        day_rec["T1_tg"] += 1
                        day_rec["T1_nets"].append(close_30m)
                        t1_day_events[date_pt].append((ts, price, gamma_bn, close_30m))
                    elif sig == "T3_LONG":
                        day_rec["T3_tg"] += 1
                    elif sig == "S1_SHORT":
                        day_rec["S1_tg"] += 1
                else:
                    day_rec["discord_total"] += 1

        day_stats[date_pt] = day_rec

    # OUTPUT
    sep = "=" * 80
    print(sep)
    print("1) PER-TIER SUMMARY")
    print(sep)
    # simpler header
    print(f"{'Tier':<12} {'Total':>6} {'Telegram':>9} {'Route':<10} {'Win%':>6} {'Avg Net':>8} {'Med Net':>8}")
    print("-" * 72)
    for t in ["T1_LONG", "T2_LONG", "T3_LONG", "T4_LONG", "S1_SHORT"]:
        s = tier_stats[t]
        total = s["total"]
        tg = s["telegram"]
        wins = s["wins"]
        nets = s["nets"]
        route = "Telegram" if t in TELEGRAM_TIERS else "Discord"
        if total > 0:
            wp = 100.0 * wins / total
            avg = sum(nets) / len(nets)
            sn = sorted(nets)
            med = sn[len(sn) // 2]
            print(f"{t:<12} {total:>6} {tg:>9} {route:<10} {wp:>5.1f}% {avg:>+8.2f} {med:>+8.2f}")
        else:
            print(f"{t:<12} {total:>6} {tg:>9} {route:<10}   N/A      N/A      N/A")

    print()
    print(sep)
    print("2) PER-DAY BREAKDOWN")
    print(sep)
    print(f"{'Date':<12} {'T1 TG':>6} {'T3 TG':>6} {'S1 TG':>6} {'Discord':>8} {'T1 AvgNet':>10}")
    print("-" * 52)
    for date_pt in sorted(day_stats):
        d = day_stats[date_pt]
        if d["T1_nets"]:
            t1a = f"{sum(d['T1_nets'])/len(d['T1_nets']):>+10.2f}"
        else:
            t1a = "       ---"
        print(f"{date_pt:<12} {d['T1_tg']:>6} {d['T3_tg']:>6} {d['S1_tg']:>6} {d['discord_total']:>8} {t1a}")

    print()
    print(sep)
    print("3) EXAMPLE T1 MULTI-FIRE DAYS")
    print(sep)
    multi = {d: e for d, e in t1_day_events.items() if len(e) >= 3}
    lbl = "3+ fires"
    if not multi:
        multi = {d: e for d, e in t1_day_events.items() if len(e) >= 2}
        lbl = "2+ fires"
        if not multi:
            multi = dict(list(t1_day_events.items())[:5])
            lbl = "all available"
    print(f"  Showing days with {lbl}")
    print()

    for date_pt in sorted(multi):
        evts = multi[date_pt]
        tnet = sum(e[3] for e in evts)
        print(f"  Date: {date_pt}  --  {len(evts)} T1 fires  --  cumulative net: {tnet:+.2f}")
        print(f"  {'#':<4} {'Time(UTC)':>10} {'Price':>10} {'GammaBn':>10} {'30mNet':>10} {'Result':>8}")
        print(f"  {'-'*56}")
        for idx, (ts2, pr, gb, c30) in enumerate(evts, 1):
            res = "WIN" if c30 > 3 else ("LOSS" if c30 < -3 else "FLAT")
            print(f"  {idx:<4} {ts_to_str(ts2):>10} {pr:>10.2f} {gb:>+10.2f} {c30:>+10.2f} {res:>8}")
        print()

    print(sep)
    tg_tot = sum(tier_stats[t]["telegram"] for t in TELEGRAM_TIERS)
    dc_tot = sum(tier_stats[t]["total"] for t in DISCORD_ONLY)
    al_tot = sum(tier_stats[t]["total"] for t in COOLDOWNS)
    print(f"TOTALS: {al_tot} signals fired  |  {tg_tot} to Telegram  |  {dc_tot} Discord-only")
    print(sep)


if __name__ == "__main__":
    run_simulation()
