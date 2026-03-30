"""Tests for gex_parser, gex_db, trend_engine, trade_levels, and signal_interpreter."""

import os
import sys
import json
from datetime import datetime, time, timedelta

import pytz

# Ensure we can import from the project directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gex_parser import parse_message, GEXSnapshot
from gex_db import (
    init_db,
    save_snapshot,
    get_recent_snapshots,
    get_latest_snapshot,
    get_today_snapshots,
    get_snapshot_count,
    DB_PATH,
)
from trend_engine import (
    compute_fifteen_min_metrics,
    compute_one_hour_metrics,
    compute_overnight_drift,
    compute_advanced_metrics,
    find_pivot_strikes,
    Regime,
    PivotType,
)
from trade_levels import compute_long, compute_short, compute_sell_premium, SetupType
from signal_interpreter import SignalInterpreter, SignalType

PT_TZ = pytz.timezone("US/Pacific")

# --- Sample Messages ---

SAMPLE_1 = """
SpotGamma GEX Update - 01/15/2026

Updated: 10:30 AM ET

Current Price: $6,985.50

Net Gamma: 12,500,000,000
Total Call Gamma: 18,000,000,000
Total Put Gamma: -5,500,000,000

Top 5 Call Positions:
Strike: $7,000 | Gamma: 4,500,000,000
Strike: $6,950 | Gamma: 3,200,000,000
Strike: $7,050 | Gamma: 2,800,000,000
Strike: $6,900 | Gamma: 2,100,000,000
Strike: $7,100 | Gamma: 1,400,000,000

Top 5 Put Positions:
Strike: $6,900 | Gamma: -2,100,000,000
Strike: $6,850 | Gamma: -1,800,000,000
Strike: $6,950 | Gamma: -800,000,000
Strike: $6,800 | Gamma: -500,000,000
Strike: $6,750 | Gamma: -300,000,000
"""

SAMPLE_2 = """
SpotGamma GEX Update

Updated: 11:45 AM ET

Current Price: $6,970.25

Net Gamma: 8,200,000,000
Total Call Gamma: 14,000,000,000
Total Put Gamma: -5,800,000,000

Call Positions:
Strike: $7,000 | Gamma: 3,800,000,000
Strike: $6,970 | Gamma: 3,100,000,000
Strike: $7,050 | Gamma: 2,500,000,000
Strike: $6,950 | Gamma: 2,000,000,000
Strike: $7,100 | Gamma: 1,200,000,000

Put Positions:
Strike: $6,900 | Gamma: -2,300,000,000
Strike: $6,850 | Gamma: -1,600,000,000
Strike: $6,970 | Gamma: -1,200,000,000
Strike: $6,800 | Gamma: -500,000,000
Strike: $6,750 | Gamma: -200,000,000
"""

SAMPLE_3_NEGATIVE = """
SpotGamma GEX Update

Updated: 12:30 PM ET

Current Price: $6,920.00

Net Gamma: -3,500,000,000
Total Call Gamma: 8,000,000,000
Total Put Gamma: -11,500,000,000

Top 5 Call Positions:
Strike: $6,950 | Gamma: 2,800,000,000
Strike: $7,000 | Gamma: 2,200,000,000
Strike: $6,900 | Gamma: 1,500,000,000
Strike: $7,050 | Gamma: 800,000,000
Strike: $7,100 | Gamma: 400,000,000

Top 5 Put Positions:
Strike: $6,900 | Gamma: -4,200,000,000
Strike: $6,850 | Gamma: -3,100,000,000
Strike: $6,800 | Gamma: -2,000,000,000
Strike: $6,750 | Gamma: -1,200,000,000
Strike: $6,700 | Gamma: -800,000,000
"""

SAMPLE_4_PREMARKET = """
SpotGamma Pre-Market Update

Updated: 5:15 AM ET

Current Price: $6,995.00

Net Gamma: 10,000,000,000
Total Call Gamma: 16,000,000,000
Total Put Gamma: -6,000,000,000

Top 5 Call Positions:
Strike: $7,000 | Gamma: 4,000,000,000
Strike: $7,050 | Gamma: 3,500,000,000
Strike: $6,950 | Gamma: 2,500,000,000
Strike: $7,100 | Gamma: 2,000,000,000
Strike: $6,900 | Gamma: 1,500,000,000

Top 5 Put Positions:
Strike: $6,950 | Gamma: -2,500,000,000
Strike: $6,900 | Gamma: -1,800,000,000
Strike: $6,850 | Gamma: -1,000,000,000
Strike: $6,800 | Gamma: -500,000,000
Strike: $6,750 | Gamma: -200,000,000
"""

SAMPLE_5_EMOJI = """
⚖️ SpotGamma GEX Update ⚖️

Updated: 9:30 AM ET

Current Price: $7,010.75

Net Gamma: ⚖️ 15,000,000,000
Total Call Gamma: 20,000,000,000
Total Put Gamma: -5,000,000,000

Top 5 Call Positions:
Strike: $7,050 | Gamma: 5,000,000,000
Strike: $7,000 | Gamma: 4,200,000,000
Strike: $7,100 | Gamma: 3,000,000,000
Strike: $6,950 | Gamma: 2,500,000,000
Strike: $7,150 | Gamma: 1,800,000,000

Top 5 Put Positions:
Strike: $6,950 | Gamma: -2,000,000,000
Strike: $6,900 | Gamma: -1,500,000,000
Strike: $6,850 | Gamma: -800,000,000
Strike: $6,800 | Gamma: -400,000,000
Strike: $6,750 | Gamma: -300,000,000
"""

SAMPLE_INVALID = "Hello everyone, markets looking good today!"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


def test_parser():
    print("\n=== TEST: Parser ===")

    # Sample 1: Full message with date
    snap = parse_message(SAMPLE_1)
    check("Sample 1 parses", snap is not None)
    if snap:
        check("Price extracted", snap.curr_price == 6985.50, f"got {snap.curr_price}")
        check("Net gamma", snap.net_gamma == 12_500_000_000, f"got {snap.net_gamma}")
        check("Total call gamma", snap.total_call_gamma == 18_000_000_000, f"got {snap.total_call_gamma}")
        check("Total put gamma", snap.total_put_gamma == -5_500_000_000, f"got {snap.total_put_gamma}")
        check("Call wall is 7000", snap.call_wall == 7000, f"got {snap.call_wall}")
        check("Put floor is 6900", snap.put_floor == 6900, f"got {snap.put_floor}")
        check("5 call positions", len(snap.top5_calls) == 5, f"got {len(snap.top5_calls)}")
        check("5 put positions", len(snap.top5_puts) == 5, f"got {len(snap.top5_puts)}")
        check("Gamma ratio > 0", snap.gamma_ratio > 0, f"got {snap.gamma_ratio:.2f}")
        check("Spread = 100", snap.spread == 100, f"got {snap.spread}")
        check("Session is RTH", snap.session_tag == "RTH", f"got {snap.session_tag}")

    # Sample 2: Has shared strike 6970 (pivot canary)
    snap2 = parse_message(SAMPLE_2)
    check("Sample 2 parses", snap2 is not None)
    if snap2:
        check("Price 6970.25", snap2.curr_price == 6970.25, f"got {snap2.curr_price}")
        check("Net gamma 8.2Bn", snap2.net_gamma == 8_200_000_000, f"got {snap2.net_gamma}")
        check("Call wall 7000", snap2.call_wall == 7000, f"got {snap2.call_wall}")
        # Check 6970 appears in both calls and puts
        call_strikes = {s for s, g in snap2.top5_calls}
        put_strikes = {s for s, g in snap2.top5_puts}
        shared = call_strikes & put_strikes
        check("6970 is shared pivot", 6970 in shared, f"shared: {shared}")

    # Sample 3: Negative gamma
    snap3 = parse_message(SAMPLE_3_NEGATIVE)
    check("Sample 3 parses", snap3 is not None)
    if snap3:
        check("Negative net gamma", snap3.net_gamma == -3_500_000_000, f"got {snap3.net_gamma}")
        check("Net gamma Bn", abs(snap3.net_gamma_bn - (-3.5)) < 0.01, f"got {snap3.net_gamma_bn}")

    # Sample 4: Premarket timestamp (5:15 AM ET = 2:15 AM PT)
    snap4 = parse_message(SAMPLE_4_PREMARKET)
    check("Sample 4 parses", snap4 is not None)
    if snap4:
        # 5:15 AM ET = 2:15 AM PT → NEXT_DAY_PREVIEW (premarket is 4-6:30 AM PT)
        check("Next day preview session", snap4.session_tag == "NEXT_DAY_PREVIEW", f"got {snap4.session_tag}")

    # Sample 5: Emoji in message
    snap5 = parse_message(SAMPLE_5_EMOJI)
    check("Sample 5 parses with emoji", snap5 is not None)
    if snap5:
        check("Price with emoji", snap5.curr_price == 7010.75, f"got {snap5.curr_price}")
        check("Net gamma with emoji", snap5.net_gamma == 15_000_000_000, f"got {snap5.net_gamma}")

    # Invalid message
    snap_invalid = parse_message(SAMPLE_INVALID)
    check("Invalid message returns None", snap_invalid is None)

    # Empty / short messages
    check("Empty returns None", parse_message("") is None)
    check("Short returns None", parse_message("Hi") is None)


def test_database():
    print("\n=== TEST: Database ===")

    # Use a temporary test DB, not the production one
    import gex_db
    _orig_path = gex_db.DB_PATH
    test_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_gex_data.db")
    gex_db.DB_PATH = test_db_path
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    init_db()

    snap1 = parse_message(SAMPLE_1)
    snap2 = parse_message(SAMPLE_2)

    check("Snap1 valid for DB", snap1 is not None)
    check("Snap2 valid for DB", snap2 is not None)

    if snap1 and snap2:
        # Keep timestamps near now so get_latest_snapshot() includes both rows.
        now = datetime.now(PT_TZ)
        snap1.timestamp_pt = now - timedelta(minutes=2)
        snap2.timestamp_pt = now - timedelta(minutes=1)

        id1 = save_snapshot(snap1)
        check("Saved snap1", id1 > 0, f"id={id1}")

        id2 = save_snapshot(snap2)
        check("Saved snap2", id2 > 0, f"id={id2}")

        count = get_snapshot_count()
        check("Count is 2", count == 2, f"got {count}")

        latest = get_latest_snapshot()
        check("Latest not None", latest is not None)
        if latest:
            check("Latest price matches snap2", latest.curr_price == snap2.curr_price,
                  f"got {latest.curr_price}")

        # Test range query using the stored timestamps
        all_snaps = get_today_snapshots()
        # May or may not match today's date depending on when test runs
        # Instead, test with a wide range query
        from gex_db import get_snapshots_range
        range_snaps = get_snapshots_range(0, datetime.now(PT_TZ).timestamp() + 86400)
        check("Range query finds 2", len(range_snaps) == 2, f"got {len(range_snaps)}")

    # Restore original DB path and clean up test DB
    gex_db.DB_PATH = _orig_path
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


def test_trend_engine():
    print("\n=== TEST: Trend Engine ===")

    # Create a sequence of snapshots for testing
    base_time = datetime.now(PT_TZ).replace(hour=9, minute=0, second=0)

    snapshots = []
    for i in range(6):
        snap = GEXSnapshot(
            timestamp_pt=base_time + timedelta(minutes=i * 3),
            session_tag="RTH",
            curr_price=6980 + i * 2,
            net_gamma=int(12e9 - i * 0.5e9),
            total_call_gamma=int(18e9),
            total_put_gamma=int(-6e9),
            call_wall=7000,
            put_floor=6900,
            top5_calls=[(7000, int(4.5e9)), (6950, int(3.2e9)), (7050, int(2.8e9)),
                        (6900, int(2.1e9)), (7100, int(1.4e9))],
            top5_puts=[(6900, int(-2.1e9)), (6850, int(-1.8e9)), (6950, int(-0.8e9)),
                       (6800, int(-0.5e9)), (6750, int(-0.3e9))],
        )
        snapshots.append(snap)

    # 15-min metrics
    m15 = compute_fifteen_min_metrics(snapshots)
    check("15min metrics computed", m15 is not None)
    if m15:
        check("Price delta positive", m15.price_delta > 0, f"got {m15.price_delta}")
        check("Net gamma delta negative (decaying)", m15.net_gamma_delta < 0,
              f"got {m15.net_gamma_delta}")
        check("Has current price", m15.curr_price > 0)

    # 1-hour metrics
    m1h = compute_one_hour_metrics(snapshots, m15) if m15 else None
    check("1hr metrics computed", m1h is not None)
    if m1h:
        check("Has regime", m1h.regime is not None)
        check("Has slope", isinstance(m1h.net_gamma_slope, float))
        check("Price range > 0", m1h.price_range > 0, f"got {m1h.price_range}")
        print(f"    Regime: {m1h.regime.value} (confidence: {m1h.regime_confidence:.0f}%)")
        print(f"    Slope: {m1h.net_gamma_slope:+.3f} Bn/hr")

    # Overnight drift
    eod = GEXSnapshot(
        timestamp_pt=base_time - timedelta(hours=20),
        session_tag="RTH",
        curr_price=6950,
        net_gamma=int(10e9),
        total_call_gamma=int(15e9),
        total_put_gamma=int(-5e9),
        call_wall=6975,
        put_floor=6875,
        top5_calls=[(6975, int(4e9))],
        top5_puts=[(6875, int(-2e9))],
    )
    drift = compute_overnight_drift(eod, snapshots[-1])
    check("Overnight drift computed", drift is not None)
    if drift:
        check("Gamma shift calculated", drift.gamma_shift != 0, f"got {drift.gamma_shift}")
        check("Call wall moved", drift.call_wall_moved != 0, f"got {drift.call_wall_moved}")
        check("Range forecast set", drift.range_forecast in ("TIGHT", "NORMAL", "WIDE"),
              f"got {drift.range_forecast}")
        print(f"    Gamma shift: {drift.gamma_shift_pct:+.1f}%")
        print(f"    Range forecast: {drift.range_forecast}")

    # Advanced metrics
    advanced = compute_advanced_metrics(snapshots, minutes_to_close=60)
    check("Advanced metrics computed", advanced is not None)
    if advanced:
        check("GCI > 0", advanced.gci > 0, f"got {advanced.gci:.1f}%")
        check("Has GCI label", advanced.gci_label in ("HARD_WALL", "MODERATE", "DIFFUSE"))
        check("Has asymmetry", advanced.gamma_asymmetry > 0)
        check("Squeeze prob 0-100", 0 <= advanced.squeeze_probability <= 100)
        check("Pin prob 0-100", 0 <= advanced.pin_probability <= 100)
        print(f"    GCI: {advanced.gci:.1f}% ({advanced.gci_label})")
        print(f"    Squeeze: {advanced.squeeze_probability:.0f}% | Pin: {advanced.pin_probability:.0f}%")

    # Pivot strikes with shared 6900 strike
    snap_pivot = GEXSnapshot(
        timestamp_pt=base_time,
        session_tag="RTH",
        curr_price=6950,
        net_gamma=int(8e9),
        total_call_gamma=int(14e9),
        total_put_gamma=int(-6e9),
        call_wall=7000,
        put_floor=6900,
        top5_calls=[(7000, int(4e9)), (6970, int(3e9)), (6900, int(2e9))],
        top5_puts=[(6900, int(-2.5e9)), (6850, int(-1.5e9)), (6970, int(-1e9))],
    )
    pivots = find_pivot_strikes(snap_pivot)
    check("Found pivot strikes", len(pivots) > 0, f"got {len(pivots)}")
    if pivots:
        strike_set = {p.strike for p in pivots}
        check("6900 is pivot", 6900 in strike_set, f"pivots at: {strike_set}")
        check("6970 is pivot", 6970 in strike_set, f"pivots at: {strike_set}")
        for p in pivots:
            print(f"    Pivot {p.strike}: {p.pivot_type.value} (ratio {p.put_call_ratio:.2f})")


def test_trade_levels():
    print("\n=== TEST: Trade Levels ===")

    snap = GEXSnapshot(
        timestamp_pt=datetime.now(PT_TZ).replace(hour=10, minute=0),
        session_tag="RTH",
        curr_price=6980,
        net_gamma=int(10e9),
        total_call_gamma=int(16e9),
        total_put_gamma=int(-6e9),
        call_wall=7000,
        put_floor=6900,
        top5_calls=[(7000, int(4e9))],
        top5_puts=[(6900, int(-2e9))],
    )

    # Buffer calculation
    spread = snap.call_wall - snap.put_floor  # 100
    expected_buffer = max(5, spread * 0.10)  # 10
    long = compute_long(snap)
    check("Long setup type", long.setup_type == SetupType.LONG)
    check("Buffer = 10", long.buffer == expected_buffer, f"got {long.buffer}")
    check("Long entry low = floor+buffer", long.entry_low == 6910, f"got {long.entry_low}")
    check("Long entry high = floor+3*buffer", long.entry_high == 6930, f"got {long.entry_high}")
    check("Long stop = floor-buffer", long.stop == 6890, f"got {long.stop}")
    check("Long target = wall-buffer", long.target == 6990, f"got {long.target}")
    check("Long R:R > 1", long.rr_ratio > 1, f"got {long.rr_ratio:.1f}")
    print(f"    LONG: Entry {long.entry_low}-{long.entry_high} | Stop {long.stop} | Target {long.target} | R:R {long.rr_ratio:.1f}")

    short = compute_short(snap)
    check("Short setup type", short.setup_type == SetupType.SHORT)
    check("Short entry low = wall-buffer", short.entry_low == 6990, f"got {short.entry_low}")
    check("Short entry high = wall+buffer", short.entry_high == 7010, f"got {short.entry_high}")
    check("Short stop = wall+2*buffer", short.stop == 7020, f"got {short.stop}")
    check("Short target = floor+2*buffer", short.target == 6920, f"got {short.target}")
    print(f"    SHORT: Entry {short.entry_low}-{short.entry_high} | Stop {short.stop} | Target {short.target} | R:R {short.rr_ratio:.1f}")

    premium = compute_sell_premium(snap)
    check("Premium setup type", premium.setup_type == SetupType.SELL_PREMIUM)
    check("Short put = floor-2*buffer", premium.entry_low == 6880, f"got {premium.entry_low}")
    check("Short call = wall+buffer", premium.entry_high == 7010, f"got {premium.entry_high}")
    print(f"    SELL_PREMIUM: Put {premium.entry_low} / Call {premium.entry_high}")


def test_signal_interpreter():
    print("\n=== TEST: Signal Interpreter ===")

    base_time = datetime.now(PT_TZ).replace(hour=10, minute=0, second=0)

    interpreter = SignalInterpreter()

    # First call to set baseline regime
    snap1 = GEXSnapshot(
        timestamp_pt=base_time,
        session_tag="RTH",
        curr_price=6980,
        net_gamma=int(12e9),
        total_call_gamma=int(18e9),
        total_put_gamma=int(-6e9),
        call_wall=7000,
        put_floor=6900,
        top5_calls=[(7000, int(4.5e9)), (6950, int(3.2e9))],
        top5_puts=[(6900, int(-2.1e9)), (6850, int(-1.8e9))],
    )

    # Create snapshots with distinct timestamps for polyfit
    snap1a = GEXSnapshot(
        timestamp_pt=base_time - timedelta(minutes=10),
        session_tag="RTH", curr_price=6978, net_gamma=int(12.5e9),
        total_call_gamma=int(18e9), total_put_gamma=int(-5.5e9),
        call_wall=7000, put_floor=6900,
        top5_calls=[(7000, int(4.5e9))], top5_puts=[(6900, int(-2.1e9))],
    )
    snap1b = GEXSnapshot(
        timestamp_pt=base_time - timedelta(minutes=5),
        session_tag="RTH", curr_price=6979, net_gamma=int(12.2e9),
        total_call_gamma=int(18e9), total_put_gamma=int(-5.8e9),
        call_wall=7000, put_floor=6900,
        top5_calls=[(7000, int(4.5e9))], top5_puts=[(6900, int(-2.1e9))],
    )
    snaps = [snap1a, snap1b, snap1]
    m15 = compute_fifteen_min_metrics(snaps)
    m1h = compute_one_hour_metrics(snaps, m15)

    signals1 = interpreter.evaluate_all(snap1, m15, m1h, None, None, None, snaps)
    check("Initial eval has no regime_shift", all(s.signal_type != SignalType.REGIME_SHIFT for s in signals1))

    # Second call with negative gamma -> should trigger regime shift
    # Must be 30+ min after initial regime confirmation (minimum time-in-regime for adjacent regimes)
    snap2 = GEXSnapshot(
        timestamp_pt=base_time + timedelta(minutes=45),
        session_tag="RTH",
        curr_price=6920,
        net_gamma=int(-3e9),
        total_call_gamma=int(8e9),
        total_put_gamma=int(-11e9),
        call_wall=6950,
        put_floor=6850,
        top5_calls=[(6950, int(2.8e9))],
        top5_puts=[(6850, int(-4e9))],
    )

    m15_2 = compute_fifteen_min_metrics([snap1, snap2])
    m1h_2 = compute_one_hour_metrics([snap1a, snap1, snap2], m15_2)

    # Debounce requires 3 consecutive snapshots agreeing on the new regime
    interpreter.evaluate_all(snap2, m15_2, m1h_2, None, None, None, [snap1, snap2])
    interpreter.evaluate_all(snap2, m15_2, m1h_2, None, None, None, [snap1, snap2])
    signals2 = interpreter.evaluate_all(snap2, m15_2, m1h_2, None, None, None, [snap1, snap2])
    regime_signals = [s for s in signals2 if s.signal_type == SignalType.REGIME_SHIFT]
    check("Regime shift detected (after debounce)", len(regime_signals) > 0)
    if regime_signals:
        check("Shift to UNCONTROLLED", regime_signals[0].metadata.get("to") == "UNCONTROLLED")
        print(f"    {regime_signals[0].title}")

    # Wall breach test
    snap_breach = GEXSnapshot(
        timestamp_pt=base_time,
        session_tag="RTH",
        curr_price=7010,  # Above call wall 7000
        net_gamma=int(10e9),
        total_call_gamma=int(16e9),
        total_put_gamma=int(-6e9),
        call_wall=7000,
        put_floor=6900,
        top5_calls=[(7000, int(4e9))],
        top5_puts=[(6900, int(-2e9))],
    )
    # Reset interpreter state
    interpreter3 = SignalInterpreter()
    interpreter3._prev_regime = Regime.CONTROLLED_TREND
    signals3 = interpreter3.evaluate_all(snap_breach, m15, m1h, None, None, None, [snap_breach])
    wall_breach = [s for s in signals3 if s.signal_type == SignalType.WALL_BREACH]
    check("Wall breach detected", len(wall_breach) > 0)
    if wall_breach:
        check("Call wall breach", "CALL WALL" in wall_breach[0].message)

    # Bounce zone test
    snap_bounce = GEXSnapshot(
        timestamp_pt=base_time,
        session_tag="RTH",
        curr_price=6910,  # Within 15pts of put floor 6900
        net_gamma=int(10e9),
        total_call_gamma=int(16e9),
        total_put_gamma=int(-6e9),
        call_wall=7000,
        put_floor=6900,
        top5_calls=[(7000, int(4e9))],
        top5_puts=[(6900, int(-2e9))],
    )
    interpreter4 = SignalInterpreter()
    interpreter4._prev_regime = Regime.CONTROLLED_TREND
    signals4 = interpreter4.evaluate_all(snap_bounce, m15, m1h, None, None, None, [snap_bounce])
    bounce = [s for s in signals4 if s.signal_type == SignalType.BOUNCE_ZONE]
    check("Bounce zone detected", len(bounce) > 0)

    # Pivot/Strike flip test
    snap_flip = GEXSnapshot(
        timestamp_pt=base_time,
        session_tag="RTH",
        curr_price=6970,
        net_gamma=int(5e9),
        total_call_gamma=int(12e9),
        total_put_gamma=int(-7e9),
        call_wall=7000,
        put_floor=6900,
        top5_calls=[(7000, int(4e9)), (6970, int(1e9)), (6950, int(2e9))],
        top5_puts=[(6900, int(-3e9)), (6970, int(-2e9)), (6850, int(-1e9))],
    )
    adv = compute_advanced_metrics([snap_flip], minutes_to_close=60)
    interpreter5 = SignalInterpreter()
    interpreter5._prev_regime = Regime.CONTROLLED_TREND
    signals5 = interpreter5.evaluate_all(snap_flip, m15, m1h, None, adv, None, [snap_flip])
    flip = [s for s in signals5 if s.signal_type == SignalType.STRIKE_FLIP]
    check("Strike flip detected (6970 canary)", len(flip) > 0)
    if flip:
        check("6970 in flip message", "6970" in flip[0].message)

    # Scheduled signal generation
    pulse = interpreter.generate_rth_pulse(snap1, m1h, None)
    check("RTH pulse generated", pulse is not None)
    check("Pulse channel is gex_engine", pulse.channel == "gex_engine")

    final = interpreter.generate_final_15(snap1, None)
    check("Final 15 generated", final is not None)
    check("Final 15 channel is gex_context", final.channel == "gex_context")

    brief = interpreter.generate_morning_brief(snap1, None, None)
    check("Morning brief generated", brief is not None)


def main():
    print("GEX 0DTE Trading Engine — Test Suite")
    print("=" * 50)

    test_parser()
    test_database()
    test_trend_engine()
    test_trade_levels()
    test_signal_interpreter()

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failed} TEST(S) FAILED")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
