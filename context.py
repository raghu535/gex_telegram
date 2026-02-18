"""Generate pre-market context report from backfilled data."""

import sys
import os
from datetime import datetime

import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gex_db import (
    get_latest_snapshot,
    get_previous_trading_day_eod,
    get_recent_snapshots,
    get_snapshot_count,
    get_today_snapshots,
    get_snapshots_range,
)
from trend_engine import (
    compute_fifteen_min_metrics,
    compute_one_hour_metrics,
    compute_overnight_drift,
    compute_advanced_metrics,
    find_pivot_strikes,
)

PT = pytz.timezone("US/Pacific")


def run():
    now = datetime.now(PT)
    print(f"=== PRE-MARKET CONTEXT === {now.strftime('%I:%M %p PT - %b %d, %Y')}")
    print(f"DB: {get_snapshot_count()} total snapshots")
    print()

    latest = get_latest_snapshot()
    if not latest:
        print("No snapshots in database.")
        return

    # Current state
    print("CURRENT STATE:")
    print(f"  Price: ${latest.curr_price:,.2f}")
    print(f"  Net Gamma: {latest.net_gamma_bn:+.2f} Bn")
    print(f"  Total Call: {latest.total_call_gamma / 1e9:+.2f} Bn")
    print(f"  Total Put: {latest.total_put_gamma / 1e9:+.2f} Bn")
    print(f"  Call Wall: {latest.call_wall}")
    print(f"  Put Floor: {latest.put_floor}")
    print(f"  Spread: {latest.spread} pts")
    print(f"  Gamma Ratio: {latest.gamma_ratio:.2f}")
    print(f"  Session: {latest.session_tag}")
    print(f"  As of: {latest.timestamp_pt.strftime('%I:%M %p PT')}")
    print()

    print("TOP 5 CALLS:")
    for s, g in latest.top5_calls:
        print(f"  {s}: {g / 1e9:+.2f} Bn")
    print("TOP 5 PUTS:")
    for s, g in latest.top5_puts:
        print(f"  {s}: {g / 1e9:+.2f} Bn")
    print()

    # Previous EOD
    eod = get_previous_trading_day_eod()
    if eod:
        print("PREVIOUS EOD (Last RTH Close):")
        print(f"  Price: ${eod.curr_price:,.2f}")
        print(f"  Net Gamma: {eod.net_gamma_bn:+.2f} Bn")
        print(f"  Call Wall: {eod.call_wall} | Put Floor: {eod.put_floor}")
        print(f"  Date: {eod.timestamp_pt.strftime('%b %d %I:%M %p PT')}")
        print()

        drift = compute_overnight_drift(eod, latest)
        if drift:
            print("OVERNIGHT DRIFT:")
            print(f"  Gamma: {drift.gamma_shift_pct:+.1f}%")
            print(f"    {drift.eod_net_gamma / 1e9:+.2f} Bn -> {drift.current_net_gamma / 1e9:+.2f} Bn")
            print(f"  Call Wall: {drift.eod_call_wall} -> {drift.current_call_wall} ({drift.call_wall_moved:+d})")
            print(f"  Put Floor: {drift.eod_put_floor} -> {drift.current_put_floor} ({drift.put_floor_moved:+d})")
            print(f"  Range Forecast: {drift.range_forecast}")
            print()

    # Regime from recent data
    recent_1h = get_recent_snapshots(60)
    recent_15m = get_recent_snapshots(15)

    if len(recent_1h) >= 3 and len(recent_15m) >= 2:
        m15 = compute_fifteen_min_metrics(recent_15m)
        if m15:
            m1h = compute_one_hour_metrics(recent_1h, m15)
            if m1h:
                print("REGIME:")
                print(f"  Classification: {m1h.regime.value}")
                print(f"  Confidence: {m1h.regime_confidence:.0f}%")
                print(f"  Gamma Slope: {m1h.net_gamma_slope:+.3f} Bn/hr")
                print(f"  Gamma Accel: {m1h.gamma_accel:+.3f}")
                print(f"  Price Range (1h): {m1h.price_range:.1f} pts")
                print()

    # Advanced metrics
    if len(recent_1h) >= 2:
        adv = compute_advanced_metrics(recent_1h, minutes_to_close=390)
        if adv:
            print("ADVANCED METRICS:")
            print(f"  GCI: {adv.gci:.1f}% ({adv.gci_label})")
            print(f"  Gamma Velocity: {adv.gamma_velocity:.4f} Bn/min ({adv.gamma_velocity_label})")
            print(f"  Asymmetry: {adv.gamma_asymmetry:.2f} ({adv.asymmetry_label})")
            print(f"  Squeeze Prob: {adv.squeeze_probability:.0f}%")
            print(f"  Pin Prob: {adv.pin_probability:.0f}%")
            if adv.pivot_strikes:
                print("  PIVOT STRIKES:")
                for p in adv.pivot_strikes:
                    print(f"    {p.strike}: {p.pivot_type.value} (put/call ratio {p.put_call_ratio:.2f})")
            print()

    # 5-day summary
    print("5-DAY SUMMARY:")
    from gex_db import get_conn
    conn = get_conn()
    rows = conn.execute(
        "SELECT date_pt, MIN(curr_price), MAX(curr_price), MIN(net_gamma), MAX(net_gamma), COUNT(*) "
        "FROM gex_snapshots WHERE session_tag = 'RTH' GROUP BY date_pt ORDER BY date_pt DESC LIMIT 5"
    ).fetchall()
    conn.close()

    if rows:
        for row in rows:
            date, lo_price, hi_price, lo_gamma, hi_gamma, count = row
            print(f"  {date}: Price {lo_price:,.0f}-{hi_price:,.0f} | "
                  f"Gamma {lo_gamma / 1e9:+.1f} to {hi_gamma / 1e9:+.1f} Bn | "
                  f"{count} snapshots")
    else:
        print("  No RTH data found in last 5 days")
    print()


if __name__ == "__main__":
    run()
