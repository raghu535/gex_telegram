"""Dynamic entry/stop/target computation for 0DTE trades."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Optional

import pytz

from gex_parser import GEXSnapshot
from trend_engine import Regime

PT_TZ = pytz.timezone("US/Pacific")


class SetupType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    SELL_PREMIUM = "SELL_PREMIUM"


@dataclass
class TradeLevels:
    setup_type: SetupType
    entry_low: float
    entry_high: float
    stop: float
    target: float
    rr_ratio: float
    instrument_hint: str
    buffer: float
    valid: bool = True
    reason: str = ""

    @property
    def risk(self) -> float:
        if self.setup_type == SetupType.LONG:
            return abs(self.entry_low - self.stop)
        elif self.setup_type == SetupType.SHORT:
            return abs(self.stop - self.entry_high)
        return abs(self.entry_low - self.stop)

    @property
    def reward(self) -> float:
        if self.setup_type == SetupType.LONG:
            return abs(self.target - self.entry_high)
        elif self.setup_type == SetupType.SHORT:
            return abs(self.entry_low - self.target)
        return abs(self.target - self.entry_high)


@dataclass
class SellPremiumLevels:
    short_put_strike: float
    short_call_strike: float
    buffer: float


def _compute_buffer(call_wall: int, put_floor: int) -> float:
    spread = call_wall - put_floor
    return max(5.0, spread * 0.10)


def _instrument_hint(minutes_to_close: float) -> str:
    if minutes_to_close > 180:
        return "Debit spreads (>3h to close)"
    elif minutes_to_close > 60:
        return "Vertical spreads (1-3h)"
    else:
        return "Sell premium only (<1h)"


def _minutes_to_close(now_pt: Optional[datetime] = None) -> float:
    if now_pt is None:
        now_pt = datetime.now(PT_TZ)
    close = now_pt.replace(hour=13, minute=0, second=0, microsecond=0)
    delta = (close - now_pt).total_seconds() / 60
    return max(0, delta)


def compute_long(snapshot: GEXSnapshot, now_pt: Optional[datetime] = None) -> TradeLevels:
    """Compute LONG setup levels: buy near put floor."""
    buffer = _compute_buffer(snapshot.call_wall, snapshot.put_floor)
    mtc = _minutes_to_close(now_pt)

    entry_low = snapshot.put_floor + buffer
    entry_high = snapshot.put_floor + 3 * buffer
    stop = snapshot.put_floor - buffer
    target = snapshot.call_wall - buffer

    risk = abs(entry_low - stop)
    reward = abs(target - entry_high)
    rr = reward / risk if risk > 0 else 0

    valid = rr >= 1.5 and target > entry_high
    if target <= entry_high:
        reason = f"Target {target:.0f} not above entry {entry_high:.0f}"
    elif not valid:
        reason = f"R:R {rr:.1f}:1 below 1.5:1 minimum"
    else:
        reason = ""

    return TradeLevels(
        setup_type=SetupType.LONG,
        entry_low=entry_low,
        entry_high=entry_high,
        stop=stop,
        target=target,
        rr_ratio=rr,
        instrument_hint=_instrument_hint(mtc),
        buffer=buffer,
        valid=valid,
        reason=reason,
    )


def compute_short(snapshot: GEXSnapshot, now_pt: Optional[datetime] = None) -> TradeLevels:
    """Compute SHORT setup levels: sell near call wall."""
    buffer = _compute_buffer(snapshot.call_wall, snapshot.put_floor)
    mtc = _minutes_to_close(now_pt)

    entry_low = snapshot.call_wall - buffer
    entry_high = snapshot.call_wall + buffer
    stop = snapshot.call_wall + 2 * buffer
    target = snapshot.put_floor + 2 * buffer

    risk = abs(stop - entry_high)
    reward = abs(entry_low - target)
    rr = reward / risk if risk > 0 else 0

    valid = rr >= 1.5 and target < entry_low
    if target >= entry_low:
        reason = f"Target {target:.0f} not below entry {entry_low:.0f}"
    elif not valid:
        reason = f"R:R {rr:.1f}:1 below 1.5:1 minimum"
    else:
        reason = ""

    return TradeLevels(
        setup_type=SetupType.SHORT,
        entry_low=entry_low,
        entry_high=entry_high,
        stop=stop,
        target=target,
        rr_ratio=rr,
        instrument_hint=_instrument_hint(mtc),
        buffer=buffer,
        valid=valid,
        reason=reason,
    )


def compute_sell_premium(
    snapshot: GEXSnapshot, now_pt: Optional[datetime] = None
) -> TradeLevels:
    """Compute SELL_PREMIUM setup (PIN regime only): iron condor / straddle levels."""
    buffer = _compute_buffer(snapshot.call_wall, snapshot.put_floor)
    mtc = _minutes_to_close(now_pt)

    short_put = snapshot.put_floor - 2 * buffer
    short_call = snapshot.call_wall + buffer
    entry_low = short_put
    entry_high = short_call

    # For sell premium, risk is the distance beyond the wings
    # Target is credit received (approximated by buffer distance)
    stop = max(snapshot.call_wall + 2 * buffer, short_call + buffer)
    target = (entry_low + entry_high) / 2  # Midpoint as "target" (stay between)

    # R:R for premium selling is different - we use spread width vs buffer
    spread_width = short_call - short_put
    risk = buffer * 2  # Approximate max loss per side
    reward = spread_width * 0.15  # Approximate premium capture
    rr = reward / risk if risk > 0 else 0

    return TradeLevels(
        setup_type=SetupType.SELL_PREMIUM,
        entry_low=entry_low,
        entry_high=entry_high,
        stop=stop,
        target=target,
        rr_ratio=rr,
        instrument_hint=_instrument_hint(mtc),
        buffer=buffer,
        valid=True,
        reason="PIN regime premium sell",
    )


def compute_levels_for_regime(
    snapshot: GEXSnapshot,
    regime: Regime,
    now_pt: Optional[datetime] = None,
) -> dict:
    """Compute all applicable trade levels based on regime.

    Returns dict with 'long', 'short', and optionally 'sell_premium' TradeLevels.
    """
    result = {}

    if snapshot.call_wall == 0 or snapshot.put_floor == 0:
        return result

    spread = snapshot.call_wall - snapshot.put_floor
    if spread < 20:
        return result  # Zero or tiny spread = bad data, no valid levels

    result["long"] = compute_long(snapshot, now_pt)
    result["short"] = compute_short(snapshot, now_pt)

    if regime == Regime.CONTROLLED_PIN:
        result["sell_premium"] = compute_sell_premium(snapshot, now_pt)

    return result
