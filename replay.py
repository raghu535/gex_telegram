"""Replay a day's data through the engine and report all signals with timestamps."""

import sys
import os
from datetime import datetime, timedelta, time
from typing import Dict

import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gex_db import get_conn, get_snapshots_range, get_eod_snapshot
from gex_parser import GEXSnapshot
from trend_engine import (
    compute_fifteen_min_metrics,
    compute_one_hour_metrics,
    compute_overnight_drift,
    compute_advanced_metrics,
    compute_trend_dashboard,
    Regime,
)
from signal_interpreter import SignalInterpreter, SignalType
from trade_levels import compute_levels_for_regime, compute_long, compute_short

import yaml

PT = pytz.timezone("US/Pacific")

# Load cooldowns from config for replay simulation
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as _f:
    _CFG = yaml.safe_load(_f)
_COOLDOWNS = _CFG.get("cooldowns", {})


def get_snapshots_for_date(date_str: str):
    """Get all snapshots for a specific date."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM gex_snapshots WHERE date_pt = ? ORDER BY timestamp_pt ASC",
        (date_str,),
    ).fetchall()
    conn.close()

    snapshots = []
    for row in rows:
        ts = datetime.fromtimestamp(row["timestamp_pt"], tz=PT)
        import json
        snap = GEXSnapshot(
            timestamp_pt=ts,
            session_tag=row["session_tag"],
            curr_price=row["curr_price"] or 0.0,
            net_gamma=row["net_gamma"] or 0,
            total_call_gamma=row["total_call_gamma"] or 0,
            total_put_gamma=row["total_put_gamma"] or 0,
            call_wall=row["call_wall"] or 0,
            put_floor=row["put_floor"] or 0,
            top5_calls=json.loads(row["top5_calls"]) if row["top5_calls"] else [],
            top5_puts=json.loads(row["top5_puts"]) if row["top5_puts"] else [],
            raw_text="",
        )
        snapshots.append(snap)
    return snapshots


def replay_day(date_str: str):
    print(f"=== REPLAY: {date_str} ===")
    print()

    all_snaps = get_snapshots_for_date(date_str)
    if not all_snaps:
        print(f"No data for {date_str}")
        return

    rth_snaps = [s for s in all_snaps if s.session_tag == "RTH"]
    premarket = [s for s in all_snaps if s.session_tag in ("PREMARKET", "NEXT_DAY_PREVIEW")]

    print(f"Total snapshots: {len(all_snaps)}")
    print(f"  RTH: {len(rth_snaps)}")
    print(f"  Pre-market/overnight: {len(premarket)}")

    if rth_snaps:
        prices = [s.curr_price for s in rth_snaps]
        gammas = [s.net_gamma for s in rth_snaps]
        print(f"  Price range: {min(prices):,.0f} - {max(prices):,.0f}")
        print(f"  Gamma range: {min(gammas)/1e9:+.1f} to {max(gammas)/1e9:+.1f} Bn")
        print(f"  Open: {rth_snaps[0].curr_price:,.2f} | Close: {rth_snaps[-1].curr_price:,.2f}")
    print()

    # Get previous day EOD for overnight drift
    prev_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    prev_eod = get_eod_snapshot(prev_date)
    # If no data for exact previous day, try going back further
    if not prev_eod:
        for d in range(2, 5):
            check_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=d)).strftime("%Y-%m-%d")
            prev_eod = get_eod_snapshot(check_date)
            if prev_eod:
                break

    interpreter = SignalInterpreter()
    all_signals = []
    suppressed_signals = []  # signals blocked by cooldown
    regime_history = []
    long_warnings = []
    short_warnings = []

    # Cooldown simulation: signal_name -> timestamp when cooldown expires
    cooldown_until: Dict[str, datetime] = {}

    # Process snapshots in chronological order, simulating real-time
    for i, snap in enumerate(all_snaps):
        # Build lookback windows from snapshots seen so far
        snap_time = snap.timestamp_pt
        lookback_15m = [s for s in all_snaps[:i+1]
                        if (snap_time - s.timestamp_pt).total_seconds() <= 900]
        lookback_1h = [s for s in all_snaps[:i+1]
                       if (snap_time - s.timestamp_pt).total_seconds() <= 3600]

        fifteen_min = compute_fifteen_min_metrics(lookback_15m) if len(lookback_15m) >= 2 else None
        one_hour = None
        advanced = None

        if fifteen_min and len(lookback_1h) >= 3:
            one_hour = compute_one_hour_metrics(lookback_1h, fifteen_min)

            close_time = snap_time.replace(hour=13, minute=0, second=0)
            mtc = max(0, (close_time - snap_time).total_seconds() / 60)
            advanced = compute_advanced_metrics(lookback_1h, minutes_to_close=mtc)

        # Compute overnight drift for pre-market
        overnight = None
        if snap.session_tag in ("PREMARKET", "NEXT_DAY_PREVIEW") and prev_eod:
            overnight = compute_overnight_drift(prev_eod, snap)

        trend = compute_trend_dashboard(lookback_1h, overnight)

        # Evaluate signals
        signals = interpreter.evaluate_all(
            snapshot=snap,
            fifteen_min=fifteen_min,
            one_hour=one_hour,
            overnight=overnight,
            advanced=advanced,
            trend=trend,
            snapshots_1h=lookback_1h if len(lookback_1h) >= 2 else None,
        )

        for sig in signals:
            sig_name = sig.signal_type.name.lower()
            cooldown_secs = _COOLDOWNS.get(sig_name, 0)

            # Final hour: halve cooldowns (12:00-1:00 PM PT)
            snap_t = snap_time.time()
            if time(12, 0) <= snap_t <= time(13, 0):
                cooldown_secs = cooldown_secs // 2

            # Check if signal is on cooldown
            if sig_name in cooldown_until and snap_time < cooldown_until[sig_name]:
                suppressed_signals.append((snap_time, sig, snap))
                continue

            # Set cooldown
            if cooldown_secs > 0:
                cooldown_until[sig_name] = snap_time + timedelta(seconds=cooldown_secs)

            all_signals.append((snap_time, sig, snap))

        # Track regime
        regime = one_hour.regime if one_hour else None
        if regime:
            regime_confidence = one_hour.regime_confidence
            if not regime_history or regime_history[-1][1] != regime:
                regime_history.append((snap_time, regime, regime_confidence, snap))

        # Track trade level validity
        if snap.session_tag == "RTH" and snap.call_wall > 0 and snap.put_floor > 0:
            long_lvl = compute_long(snap, snap_time)
            short_lvl = compute_short(snap, snap_time)

            if not long_lvl.valid and (not long_warnings or
                (snap_time - long_warnings[-1][0]).total_seconds() > 300):
                long_warnings.append((snap_time, long_lvl, snap))

            if not short_lvl.valid and (not short_warnings or
                (snap_time - short_warnings[-1][0]).total_seconds() > 300):
                short_warnings.append((snap_time, short_lvl, snap))

    # --- Report ---

    print("=" * 60)
    print("REGIME TRANSITIONS:")
    print("=" * 60)
    for ts, regime, conf, snap in regime_history:
        print(f"  {ts.strftime('%I:%M %p PT')} | {regime.value} ({conf:.0f}%) | "
              f"Price {snap.curr_price:,.0f} | Gamma {snap.net_gamma/1e9:+.1f} Bn")
    print()

    print("=" * 60)
    print("ALL SIGNALS FIRED (chronological):")
    print("=" * 60)
    for ts, sig, snap in all_signals:
        print(f"  {ts.strftime('%I:%M %p PT')} | [{sig.signal_type.name}] {sig.title}")
        print(f"    Price: {snap.curr_price:,.0f} | Gamma: {snap.net_gamma/1e9:+.1f} Bn | "
              f"Wall: {snap.call_wall} | Floor: {snap.put_floor}")
        # Print first 2 lines of message for context
        msg_lines = sig.message.strip().split("\n")
        for line in msg_lines[:3]:
            if line.strip():
                print(f"    > {line.strip()}")
        print()

    # Long/Short invalidity tracking
    print("=" * 60)
    print("WHEN LONG BECAME INVALID (R:R < 1.5):")
    print("=" * 60)
    if long_warnings:
        print(f"  FIRST: {long_warnings[0][0].strftime('%I:%M %p PT')} | "
              f"Price {long_warnings[0][2].curr_price:,.0f} | "
              f"R:R {long_warnings[0][1].rr_ratio:.1f}:1 | "
              f"Reason: {long_warnings[0][1].reason}")
        for ts, lvl, snap in long_warnings[:5]:
            print(f"  {ts.strftime('%I:%M %p PT')} | Price {snap.curr_price:,.0f} | "
                  f"R:R {lvl.rr_ratio:.1f}:1 | Wall {snap.call_wall} Floor {snap.put_floor}")
    else:
        print("  Long was always valid (R:R >= 1.5)")
    print()

    print("=" * 60)
    print("WHEN SHORT BECAME INVALID (R:R < 1.5):")
    print("=" * 60)
    if short_warnings:
        print(f"  FIRST: {short_warnings[0][0].strftime('%I:%M %p PT')} | "
              f"Price {short_warnings[0][2].curr_price:,.0f} | "
              f"R:R {short_warnings[0][1].rr_ratio:.1f}:1")
        for ts, lvl, snap in short_warnings[:5]:
            print(f"  {ts.strftime('%I:%M %p PT')} | Price {snap.curr_price:,.0f} | "
                  f"R:R {lvl.rr_ratio:.1f}:1 | Wall {snap.call_wall} Floor {snap.put_floor}")
    else:
        print("  Short was always valid (R:R >= 1.5)")
    print()

    # Cooldown suppression summary
    print("=" * 60)
    print("COOLDOWN SUPPRESSION SUMMARY:")
    print("=" * 60)
    from collections import Counter
    suppressed_counts = Counter(sig.signal_type.name for _, sig, _ in suppressed_signals)
    if suppressed_counts:
        for name, count in suppressed_counts.most_common():
            print(f"  {name}: {count} suppressed by cooldown")
    else:
        print("  No signals suppressed (all passed cooldown)")
    print(f"  Total fired: {len(all_signals)} | Total suppressed: {len(suppressed_signals)}")
    print()

    # Key moments summary
    print("=" * 60)
    print("KEY MOMENTS:")
    print("=" * 60)

    # First signal of the day
    if all_signals:
        ts, sig, snap = all_signals[0]
        print(f"  First signal: {ts.strftime('%I:%M %p PT')} — {sig.signal_type.name}: {sig.title}")

    # First "don't go long" moment
    if long_warnings:
        ts, lvl, snap = long_warnings[0]
        print(f"  First 'don't go long': {ts.strftime('%I:%M %p PT')} — "
              f"R:R collapsed to {lvl.rr_ratio:.1f}:1 (price {snap.curr_price:,.0f}, "
              f"spread {snap.spread} pts)")

    # Biggest regime shift
    regime_shifts = [s for _, s, _ in all_signals if s.signal_type == SignalType.REGIME_SHIFT]
    if regime_shifts:
        print(f"  Regime shifts: {len(regime_shifts)} total")

    # Wall breaches
    breaches = [s for _, s, _ in all_signals if s.signal_type == SignalType.WALL_BREACH]
    if breaches:
        print(f"  Wall breaches: {len(breaches)} total")

    # Strike flips
    flips = [s for _, s, _ in all_signals if s.signal_type == SignalType.STRIKE_FLIP]
    if flips:
        print(f"  Strike flips: {len(flips)} detected")


if __name__ == "__main__":
    # Default to yesterday, or pass a date
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        # Find the most recent trading day with RTH data
        conn = get_conn()
        row = conn.execute(
            "SELECT DISTINCT date_pt FROM gex_snapshots WHERE session_tag = 'RTH' "
            "ORDER BY date_pt DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            date_str = row[0]
        else:
            date_str = (datetime.now(PT) - timedelta(days=1)).strftime("%Y-%m-%d")

    replay_day(date_str)
