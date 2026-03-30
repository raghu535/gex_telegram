"""End-to-end simulation: sweep watch → reclaim → Type 1 fire.

The key subtlety: T1-T4 tier checks run BEFORE sweep detection.
If dP_20m is large (big drop over 20min), T1 fires instead of L1.
To isolate the sweep, we need 20m-ago price close to current
so dP_20m is small, but price still breaks below the 1hr range low.

Setup:
  - 1hr history: prices drift 6840→6835 (range_low ≈ 6833)
  - Snap 1 (t=0):   price 6850, gamma -5.5Bn → no signal (range intact)
  - Snap 2 (t+3m):  price 6835 → BUT 20m-ago price is ~6838 from history,
                     so dP_20m = -3 (not enough for T1). Price < range_low
                     triggers L1 sweep watch only.
  - Snap 3 (t+6m):  price 6839 → reclaims level+4 → signal FIRES

Usage: python test_unified_alert.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
from datetime import datetime, timedelta
from typing import List

import pytz

PT_TZ = pytz.timezone("US/Pacific")

from gex_parser import GEXSnapshot
from signal_interpreter import SignalInterpreter


def make_snap(price, gamma_bn, cw, pf, ts):
    return GEXSnapshot(
        timestamp_pt=ts,
        session_tag="RTH",
        curr_price=price,
        net_gamma=int(gamma_bn * 1e9),
        total_call_gamma=0,
        total_put_gamma=int(gamma_bn * 1e9),
        call_wall=cw,
        put_floor=pf,
    )


# ── Timeline ────────────────────────────────────────────────────────
# Base: 10:00 AM PT (well into RTH)
base = datetime.now(PT_TZ).replace(hour=10, minute=0, second=0, microsecond=0)

# Build 30 snapshots over 1hr, prices cycling 6834-6842
# Phase-shifted so the 15m-ago entry is ABOVE range_low (critical for L1 check)
history: List[GEXSnapshot] = []
for i in range(30):
    t = base - timedelta(minutes=(30 - i) * 2)
    # Cycle offset so range_low entries don't land on the 15m lookback
    px = 6834 + ((i + 2) % 5) * 2  # 6834, 6836, 6838, 6840, 6842
    history.append(make_snap(px, -5.5, 6865, 6830, t))

range_prices = [s.curr_price for s in history[:-3]]
range_low = min(range_prices)
range_high = max(range_prices)
print(f"1hr History: {len(history)} snaps, range low={range_low:.0f}, high={range_high:.0f}")
print()

# ── Interpreter setup ───────────────────────────────────────────────
si = SignalInterpreter()
si._rth_snap_count = 50  # past the 15-snap gate
si._last_ma_values = {
    "vwap": 6852.0,
    "day_open": 6855.0,
    "day_high": 6860.0,
    "day_low": 6828.0,
    "5m_20": 6848.0,
    "30m_20": 6855.0,
}


# ═══════════════════════════════════════════════════════════════════
# SNAP 1: price=6850, steady state
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNAP 1 — Steady state: price 6850, gamma -5.5Bn")
print("=" * 70)

t1 = base + timedelta(minutes=2)
snap1 = make_snap(6850, -5.5, 6865, 6830, t1)
snaps_1h = history + [snap1]

sig1 = si._check_gamma_snap(snap1, snaps_1h)
print(f"  Signal fired  : {sig1 is not None}")
print(f"  Sweep watch   : {si._sweep_watch is not None}")

# Snap1 at 6850 is ABOVE the range (6833-6841), so no sweep, no T1
# (dP_20m from ~6837 to 6850 is +13, which is a RISE not a fall)
print()


# ═══════════════════════════════════════════════════════════════════
# SNAP 2: price drops to 6830 (below range_low of 6833)
#   → should SET sweep watch, NOT fire signal
#   dP_20m ≈ 6830 - 6837 = -7 (not enough for T1's -15 threshold)
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNAP 2 — Price drops to 6830 (below 1hr low of {})".format(range_low))
print("  Expect: sweep watch SET, signal NOT fired")
print("=" * 70)

t2 = t1 + timedelta(minutes=3)
snap2 = make_snap(6830, -5.5, 6865, 6830, t2)
snaps_1h_2 = history + [snap1, snap2]

# Verify the 20m lookback price (should be ~6837, not 6850)
now_ts = snap2.timestamp_pt.timestamp()
for s in reversed(snaps_1h_2):
    s_ts = s.timestamp_pt.timestamp()
    age = now_ts - s_ts
    if age >= 1100:  # ~20min
        print(f"  20m-ago price : {s.curr_price:.0f}  (dP_20m ≈ {6830 - s.curr_price:+.0f})")
        break
for s in reversed(snaps_1h_2):
    s_ts = s.timestamp_pt.timestamp()
    age = now_ts - s_ts
    if age >= 800:  # ~15min
        print(f"  15m-ago price : {s.curr_price:.0f}  (was above range_low? {s.curr_price > range_low})")
        break

sig2 = si._check_gamma_snap(snap2, snaps_1h_2)
print(f"  Signal fired  : {sig2 is not None}")
print(f"  Sweep watch   : {si._sweep_watch is not None}")

if sig2 is not None:
    print(f"  UNEXPECTED signal: {sig2.title}")
    print(f"  Metadata: {sig2.metadata}")
    # If T1/T2 fired, show why
    if sig2.metadata.get("tier_label") in ("T1", "T2", "T3"):
        print(f"  → T-tier fired because dP_20m was too large. Adjust history.")
        sys.exit(1)

assert sig2 is None, "FAIL: Signal should NOT fire on sweep"
assert si._sweep_watch is not None, "FAIL: Sweep watch should be set"

w = si._sweep_watch
print(f"  Watch side    : {w['side']}")
print(f"  Watch level   : {w['level']:.0f}")
print(f"  Watch tier    : {w['tier_label']}")
print(f"  Reclaim at    : > {w['level'] + 4:.0f}  (level + 4pts)")
print()
print("  ✅ PASS — Sweep detected, watching for reclaim")
print()


# ═══════════════════════════════════════════════════════════════════
# SNAP 3: price reclaims to level+5 → FIRE the signal
# ═══════════════════════════════════════════════════════════════════
sweep_level = w["level"]
reclaim_price = sweep_level + 5  # comfortably above level+4

print("=" * 70)
print(f"SNAP 3 — Price reclaims to {reclaim_price:.0f} (above {sweep_level:.0f} + 4)")
print("  Expect: signal FIRES with Type 1 format")
print("=" * 70)

t3 = t2 + timedelta(minutes=3)
snap3 = make_snap(reclaim_price, -5.5, 6865, 6830, t3)
snaps_1h_3 = history + [snap1, snap2, snap3]

sig3 = si._check_gamma_snap(snap3, snaps_1h_3)
print(f"  Signal fired  : {sig3 is not None}")
print(f"  Sweep watch   : {si._sweep_watch}")

assert sig3 is not None, "FAIL: Signal should fire on reclaim!"
assert si._sweep_watch is None, "FAIL: Sweep watch should be cleared"
print()
print("  ✅ PASS — Signal fired on reclaim!")
print()


# ═══════════════════════════════════════════════════════════════════
# CHECK 4: Print the full Type 1 output
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("FULL TYPE 1 OUTPUT")
print("=" * 70)
print()
msg = sig3.message
print(msg)
print()


# ═══════════════════════════════════════════════════════════════════
# CHECK 5: Verify confluence has all 5 checks
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHECK 5 — Confluence: 5 checks")
print("=" * 70)

confluence_line = None
for line in msg.split("\n"):
    if "Confluence:" in line:
        confluence_line = line.strip()
        break

print(f"  Line: {confluence_line}")
if confluence_line:
    parts = confluence_line.split("Confluence:")[-1].strip()
    # Format: "X/Y"
    slash_part = parts.split("/")
    total = int(slash_part[-1])
    active = int(slash_part[0].split()[-1])
    print(f"  Active: {active}/{total}")
    assert total == 5, f"FAIL: Expected 5 total, got {total}"
    print(f"  ✅ PASS — All 5 confluence checks present")
else:
    print(f"  ❌ FAIL — No confluence line found")
print()


# ═══════════════════════════════════════════════════════════════════
# CHECK 6: No internal codes
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHECK 6 — No internal codes")
print("=" * 70)

forbidden = ["[T1]", "[T2]", "[T3]", "[T4]", "[L1]", "[S1]", "[S2]", "[S3]",
             "[B1]", "[BEAR]", "[EM]", "CONTROLLED_PIN", "CONTROLLED_TREND",
             "FRAGILE_CONTROL", "UNCONTROLLED"]
found_codes = [code for code in forbidden if code in msg]

if found_codes:
    print(f"  ❌ FAIL: Found internal codes: {found_codes}")
else:
    print(f"  ✅ PASS — No internal codes in output")
    print(f"     Checked: {len(forbidden)} patterns")
print()


# ═══════════════════════════════════════════════════════════════════
# CHECK 7: Gamma as plain English, not raw numbers
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHECK 7 — Plain English gamma (no raw Bn)")
print("=" * 70)

bn_pattern = re.compile(r"[+-]?\d+\.?\d*\s*Bn", re.IGNORECASE)
bn_matches = bn_pattern.findall(msg)

if bn_matches:
    print(f"  ❌ FAIL: Found raw gamma: {bn_matches}")
else:
    print(f"  ✅ PASS — No raw gamma numbers (Bn suffix)")

# Check for plain English
gamma_terms = ["heavy negative gamma", "negative gamma", "extreme negative gamma",
               "positive gamma", "neutral gamma", "gamma ceiling"]
found_terms = [t for t in gamma_terms if t.lower() in msg.lower()]
print(f"  Plain English found: {found_terms}")

# Show structure line
for line in msg.split("\n"):
    if line.startswith("Structure:"):
        print(f"  {line}")
        break
print()


# ═══════════════════════════════════════════════════════════════════
# CHECK 8: 0DTE play present
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHECK 8 — 0DTE play")
print("=" * 70)

has_play = "Play:" in msg and "SPXW" in msg
has_stop_note = "cut below" in msg or "cut above" in msg

print(f"  Play line  : {'✅' if has_play else '❌'}")
print(f"  Stop note  : {'✅' if has_stop_note else '❌'}")
for line in msg.split("\n"):
    if "Play:" in line or "Target:" in line or "cut " in line:
        print(f"    {line}")
print()


# ═══════════════════════════════════════════════════════════════════
# Signal metadata
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("SIGNAL METADATA")
print("=" * 70)
m = sig3.metadata
print(f"  type       : {sig3.signal_type.name}")
print(f"  title      : {sig3.title}")
print(f"  channel    : {sig3.channel}")
print(f"  side       : {m.get('side')}")
print(f"  tier_label : {m.get('tier_label')}")
print(f"  win_pct    : {m.get('win_pct')}")
print(f"  gamma_bn   : {m.get('gamma_bn')}")
print()


# ═══════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
checks = {
    "Sweep does NOT fire signal": sig2 is None,
    "Reclaim DOES fire signal": sig3 is not None,
    "Sweep watch cleared": si._sweep_watch is None,
    "No internal codes": len(found_codes) == 0,
    "No raw gamma (Bn)": len(bn_matches) == 0,
    "0DTE play present": has_play,
    "5 confluence checks": confluence_line is not None and "/5" in confluence_line,
}

all_pass = all(checks.values())
for name, passed in checks.items():
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}")

print()
if all_pass:
    print("  ALL 7 CHECKS PASSED ✅")
else:
    print("  SOME CHECKS FAILED ❌")
print("=" * 70)
