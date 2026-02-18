"""Signal detection and alert formatting for 11 signal types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import IntEnum
from typing import Dict, List, Optional

import pytz

from gex_parser import GEXSnapshot
from trend_engine import (
    AdvancedMetrics,
    FifteenMinMetrics,
    OneHourMetrics,
    OvernightDrift,
    TrendDashboard,
    Regime,
    PivotType,
)
from trade_levels import (
    TradeLevels,
    compute_levels_for_regime,
)

PT_TZ = pytz.timezone("US/Pacific")


class SignalType(IntEnum):
    """Signal types in priority order."""
    REGIME_SHIFT = 1
    WALL_BREACH = 2
    GAMMA_SQUEEZE = 3
    BOUNCE_ZONE = 4
    GAMMA_COMPRESSION = 5
    STRIKE_FLIP = 6
    TREND_DASHBOARD = 7
    OVERNIGHT_DRIFT = 8
    RTH_PULSE = 9
    FINAL_15 = 10
    MORNING_BRIEF = 11
    HOURLY_RECAP = 12
    PIN_FORECAST = 13
    DATA_LAG = 14
    DATA_RESTORED = 15
    QUIET_SUMMARY = 16
    SPX_MILLION = 17
    GAP_ALERT = 18
    MORNING_CHECKLIST = 19
    MICRO_PULSE = 20
    GAMMA_SNAP = 21


SIGNAL_CHANNEL_MAP = {
    SignalType.REGIME_SHIFT: "gex_ops",
    SignalType.WALL_BREACH: "gex_ops",
    SignalType.GAMMA_SQUEEZE: "gex_ops",
    SignalType.BOUNCE_ZONE: "gex_ops",
    SignalType.GAMMA_COMPRESSION: "gex_ops",
    SignalType.STRIKE_FLIP: "gex_ops",
    SignalType.TREND_DASHBOARD: "market_pulse",
    SignalType.OVERNIGHT_DRIFT: "daily_intel",
    SignalType.RTH_PULSE: "market_pulse",
    SignalType.FINAL_15: "daily_intel",
    SignalType.MORNING_BRIEF: "daily_intel",
    SignalType.HOURLY_RECAP: "daily_intel",
    SignalType.PIN_FORECAST: "daily_intel",
    SignalType.DATA_LAG: "gex_ops",
    SignalType.DATA_RESTORED: "gex_ops",
    SignalType.QUIET_SUMMARY: "market_pulse",
    SignalType.SPX_MILLION: "market_pulse",
    SignalType.GAP_ALERT: "daily_intel",
    SignalType.MORNING_CHECKLIST: "market_pulse",
    SignalType.MICRO_PULSE: "market_pulse",
    SignalType.GAMMA_SNAP: "gex_ops",
}


@dataclass
class Signal:
    signal_type: SignalType
    title: str
    message: str
    channel: str
    priority: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(PT_TZ))
    metadata: Dict = field(default_factory=dict)


import yaml
import os

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as _f:
    _CFG = yaml.safe_load(_f)

_ANALYSIS = _CFG["analysis"]
_REGIME_CFG = _CFG["regime"]
_TREND_CFG = _CFG.get("trend", {})
_DAY_PROFILE_CFG = _CFG.get("day_profile", {})
_COUNTER_TREND_CFG = _CFG.get("counter_trend_gates", {})

# Noise filter thresholds
MIN_SPREAD = _ANALYSIS.get("min_spread_pts", 20)
WALL_BREACH_BUFFER = _ANALYSIS.get("wall_breach_buffer_pts", 5)
MIN_PIVOT_GAMMA_BN = _ANALYSIS.get("min_pivot_gamma_bn", 0.5)
REGIME_HOLD_COUNT = _REGIME_CFG.get("hold_count", 3)
TREND_HEARTBEAT_SECS = int(_TREND_CFG.get("heartbeat_secs", 0) or 0)
TREND_SCORE_THRESHOLD = int(_TREND_CFG.get("score_threshold", 4))
TREND_MIN_EMIT_SCORE = int(_TREND_CFG.get("min_emit_score", 0) or 0)
_COUNTER_TREND_SUPPRESS_ALL_PTS = float(_COUNTER_TREND_CFG.get("suppress_all_pts", 0) or 0)
_COUNTER_TREND_GATES = []
for _gate in _COUNTER_TREND_CFG.get("gates", []) or []:
    try:
        _COUNTER_TREND_GATES.append(
            (float(_gate.get("move_pts", 0) or 0), int(_gate.get("min_score", 0) or 0))
        )
    except Exception:
        continue
_COUNTER_TREND_GATES.sort(key=lambda x: x[0], reverse=True)

_GAMMA_SNAP_CFG = _CFG.get("gamma_snap", {})


def _is_rth(ts: datetime) -> bool:
    t = ts.astimezone(PT_TZ).time()
    return time(6, 30) <= t < time(13, 0)


class SignalInterpreter:
    """Evaluate all signal conditions and produce formatted alerts."""

    def __init__(self):
        self._prev_regime: Optional[Regime] = None
        self._confirmed_regime: Optional[Regime] = None
        self._regime_candidate: Optional[Regime] = None
        self._regime_candidate_count: int = 0
        self._regime_confirmed_at: Optional[datetime] = None  # time of last regime confirmation
        self._prev_call_wall: int = 0
        self._prev_put_floor: int = 0
        self._prev_breach_side: Optional[str] = None
        self._last_breach_wall_value: int = 0  # track actual wall strike that was breached
        # State-change dedup for noisy signals
        self._last_drift_pct: Optional[float] = None  # last reported overnight drift %
        self._reported_flipped_strikes: set = set()    # strikes already reported as flipped
        self._last_bounce_side: Optional[str] = None   # "call" or "put" — last bounce zone fired
        self._bounce_cleared: bool = True               # price moved away from wall
        self._compression_active: bool = False           # currently in a compression event
        self._last_compression_pct: float = 0.0          # last compression % that fired
        self._squeeze_active: bool = False               # already alerted gamma squeeze
        self._last_trend_bias: Optional[str] = None
        self._last_trend_smell: Optional[bool] = None
        self._last_trend_score: Optional[int] = None
        self._last_trend_flags: set = set()
        self._last_trend_emit_ts: float = 0.0
        self._last_trend_change_ts: float = 0.0
        self._trend_heartbeat_sent: bool = False
        self._trend_state_date: Optional[str] = None
        self._trend_open_gamma_bn: Optional[float] = None
        self._trend_early_min_gamma_bn: Optional[float] = None
        # Gamma snap state
        self._last_gamma_snap_side: Optional[str] = None  # "long" or "short"
        self._gamma_snap_cleared: bool = True

    def evaluate_all(
        self,
        snapshot: GEXSnapshot,
        fifteen_min: Optional[FifteenMinMetrics],
        one_hour: Optional[OneHourMetrics],
        overnight: Optional[OvernightDrift],
        advanced: Optional[AdvancedMetrics],
        trend: Optional[TrendDashboard] = None,
        snapshots_1h: Optional[List[GEXSnapshot]] = None,
    ) -> List[Signal]:
        """Check all signal conditions and return triggered signals, sorted by priority."""
        signals = []
        snap_date = snapshot.timestamp_pt.astimezone(PT_TZ).strftime("%Y-%m-%d")
        if self._trend_state_date != snap_date:
            self._trend_state_date = snap_date
            self._last_trend_bias = None
            self._last_trend_smell = None
            self._last_trend_score = None
            self._last_trend_flags = set()
            self._last_trend_emit_ts = 0.0
            self._last_trend_change_ts = 0.0
            self._trend_heartbeat_sent = False
            self._trend_open_gamma_bn = None
            self._trend_early_min_gamma_bn = None

        regime = one_hour.regime if one_hour else Regime.CONTROLLED_TREND

        # 1. REGIME_SHIFT
        sig = self._check_regime_shift(regime, one_hour, snapshot.timestamp_pt)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 2. WALL_BREACH
        sig = self._check_wall_breach(snapshot)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 3. GAMMA_SQUEEZE
        sig = self._check_gamma_squeeze(snapshot, one_hour, advanced)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 4. BOUNCE_ZONE
        sig = self._check_bounce_zone(snapshot, regime, fifteen_min)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 5. GAMMA_COMPRESSION
        sig = self._check_gamma_compression(snapshot, snapshots_1h)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 6. STRIKE_FLIP
        sig = self._check_strike_flip(advanced)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 7. TREND_DASHBOARD
        sig = self._check_trend_dashboard(trend, snapshot)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 8. OVERNIGHT_DRIFT
        sig = self._check_overnight_drift(overnight)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 9. GAMMA_SNAP (gamma-velocity trade signals — long and short)
        sig = self._check_gamma_snap(snapshot, snapshots_1h)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # Update state for next cycle
        self._prev_regime = self._confirmed_regime or regime
        self._prev_call_wall = snapshot.call_wall
        self._prev_put_floor = snapshot.put_floor

        return sorted(signals, key=lambda s: s.priority)

    def generate_rth_pulse(
        self,
        snapshot: GEXSnapshot,
        one_hour: Optional[OneHourMetrics],
        advanced: Optional[AdvancedMetrics],
    ) -> Signal:
        """Generate RTH_PULSE heartbeat signal."""
        regime = one_hour.regime if one_hour else Regime.CONTROLLED_TREND
        levels = compute_levels_for_regime(snapshot, regime)

        msg = _format_rth_pulse(snapshot, one_hour, advanced, levels)
        return Signal(
            signal_type=SignalType.RTH_PULSE,
            title="RTH Pulse",
            message=msg,
            channel="market_pulse",
            priority=SignalType.RTH_PULSE,
            timestamp=snapshot.timestamp_pt,
        )

    def generate_micro_pulse(
        self,
        snapshot: GEXSnapshot,
        move_5m: float,
        gamma_delta_5m_bn: float,
        fifteen_min: Optional[FifteenMinMetrics],
        move_1m: float,
        recent_high: float,
        recent_low: float,
        micro_cfg: Optional[dict] = None,
    ) -> Signal:
        """Generate a fast intraday micro pulse for 5-10pt move capture."""
        cfg = micro_cfg or {}
        direction = "LONG" if move_5m > 0 else "SHORT" if move_5m < 0 else "NEUTRAL"
        abs_move = abs(move_5m)

        mild_extreme_pts = float(cfg.get("mild_extreme_pts", 5))
        possible_extreme_pts = float(cfg.get("possible_extreme_pts", 8))
        crazy_extreme_pts = float(cfg.get("crazy_extreme_pts", 12))
        breakout_buffer_pts = float(cfg.get("breakout_buffer_pts", 1))
        invalidation_buffer_pts = float(cfg.get("invalidation_buffer_pts", 2))
        target1_pts = float(cfg.get("target1_pts", 5))
        target2_pts = float(cfg.get("target2_pts", 10))

        if abs_move >= crazy_extreme_pts:
            tier = "CRAZY EXTREME"
        elif abs_move >= possible_extreme_pts:
            tier = "POSSIBLE EXTREME"
        elif abs_move >= mild_extreme_pts:
            tier = "MILD EXTREME"
        else:
            tier = "MICRO MOVE"

        if direction == "LONG":
            gas_state = "ACCELERATING" if (move_1m > 0 and gamma_delta_5m_bn > 0) else "LOSING STEAM"
            trigger = recent_high + breakout_buffer_pts
            invalidation = recent_low - invalidation_buffer_pts
            t1 = trigger + target1_pts
            t2 = trigger + target2_pts
            room = (snapshot.call_wall - snapshot.curr_price) if snapshot.call_wall else 0.0
            room_line = f"Room to wall: {room:+.1f} pts"
        elif direction == "SHORT":
            gas_state = "ACCELERATING" if (move_1m < 0 and gamma_delta_5m_bn < 0) else "LOSING STEAM"
            trigger = recent_low - breakout_buffer_pts
            invalidation = recent_high + invalidation_buffer_pts
            t1 = trigger - target1_pts
            t2 = trigger - target2_pts
            room = (snapshot.curr_price - snapshot.put_floor) if snapshot.put_floor else 0.0
            room_line = f"Room to floor: {room:+.1f} pts"
        else:
            gas_state = "STABLE"
            trigger = snapshot.curr_price
            invalidation = snapshot.curr_price
            t1 = snapshot.curr_price
            t2 = snapshot.curr_price
            room_line = "Room: n/a"

        lines = [
            f"**MICRO PULSE** | {snapshot.timestamp_pt.strftime('%I:%M %p PT')}",
            f"Bias: {direction} | Tier: {tier}",
            f"Gas: {gas_state} | Price: {snapshot.curr_price:.2f}",
            f"5m dPx: {move_5m:+.2f} | 1m dPx: {move_1m:+.2f}",
            f"5m dGamma: {gamma_delta_5m_bn:+.2f} Bn | Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn",
            f"Call Wall: {snapshot.call_wall} | Put Floor: {snapshot.put_floor}",
            (
                f"Scalp map: Trigger {trigger:.2f} | Invalidate {invalidation:.2f} | "
                f"T1 {t1:.2f} | T2 {t2:.2f}"
            ),
            room_line,
        ]

        if fifteen_min:
            lines.append(
                f"15m confirm: dPx {fifteen_min.price_delta:+.1f} | "
                f"dG {fifteen_min.net_gamma_delta / 1e9:+.2f} Bn"
            )

        return Signal(
            signal_type=SignalType.MICRO_PULSE,
            title=f"Micro Pulse - {direction}",
            message="\n".join(lines),
            channel="market_pulse",
            priority=SignalType.MICRO_PULSE,
            timestamp=snapshot.timestamp_pt,
            metadata={
                "bias": direction,
                "tier": tier,
                "gas_state": gas_state,
                "move_5m": move_5m,
                "move_1m": move_1m,
                "gamma_delta_5m_bn": gamma_delta_5m_bn,
                "trigger": trigger,
                "invalidation": invalidation,
                "target1": t1,
                "target2": t2,
            },
        )

    def generate_final_15(
        self,
        snapshot: GEXSnapshot,
        advanced: Optional[AdvancedMetrics],
    ) -> Signal:
        """Generate FINAL_15 signal at 12:45 PM PT."""
        msg = _format_final_15(snapshot, advanced)
        return Signal(
            signal_type=SignalType.FINAL_15,
            title="Final 15 Minutes",
            message=msg,
            channel="daily_intel",
            priority=SignalType.FINAL_15,
            timestamp=datetime.now(PT_TZ),
        )

    def generate_quiet_summary(
        self,
        snapshot: GEXSnapshot,
        fifteen_min: Optional[FifteenMinMetrics],
        one_hour: Optional[OneHourMetrics],
        advanced: Optional[AdvancedMetrics],
    ) -> Signal:
        """Generate a periodic trading summary when internal alerts are quiet."""
        msg = _format_quiet_summary(snapshot, fifteen_min, one_hour, advanced)
        return Signal(
            signal_type=SignalType.QUIET_SUMMARY,
            title="Trading Summary",
            message=msg,
            channel="market_pulse",
            priority=SignalType.QUIET_SUMMARY,
            timestamp=snapshot.timestamp_pt,
        )

    def generate_morning_brief(
        self,
        snapshot: GEXSnapshot,
        overnight: Optional[OvernightDrift],
        advanced: Optional[AdvancedMetrics],
    ) -> Signal:
        """Generate MORNING_BRIEF signal at 6:00 AM PT."""
        msg = _format_morning_brief(snapshot, overnight, advanced)
        return Signal(
            signal_type=SignalType.MORNING_BRIEF,
            title="Morning Brief",
            message=msg,
            channel="daily_intel",
            priority=SignalType.MORNING_BRIEF,
            timestamp=datetime.now(PT_TZ),
        )

    # --- Signal checks ---

    # Adjacent regime pairs that flip frequently — require longer hold time
    _ADJACENT_REGIMES = {
        frozenset({Regime.CONTROLLED_TREND, Regime.FRAGILE_CONTROL}),
        frozenset({Regime.FRAGILE_CONTROL, Regime.UNCONTROLLED}),
        frozenset({Regime.CONTROLLED_PIN, Regime.FRAGILE_CONTROL}),
        frozenset({Regime.CONTROLLED_PIN, Regime.CONTROLLED_TREND}),
    }

    def _check_regime_shift(
        self, current: Regime, one_hour: Optional[OneHourMetrics],
        snap_time: Optional[datetime] = None,
    ) -> Optional[Signal]:
        MIN_REGIME_SECS = 900   # 15 min minimum for major shifts (2+ levels apart)
        ADJ_REGIME_SECS = 1800  # 30 min minimum for adjacent regime flips

        if self._confirmed_regime is None:
            self._confirmed_regime = current
            self._regime_candidate = current
            self._regime_candidate_count = REGIME_HOLD_COUNT
            self._regime_confirmed_at = snap_time
            return None

        # Debounce: require N consecutive snapshots agreeing on the new regime
        if current != self._confirmed_regime:
            # Determine if this is an adjacent (noisy) regime flip
            is_adjacent = frozenset({current, self._confirmed_regime}) in self._ADJACENT_REGIMES
            min_secs = ADJ_REGIME_SECS if is_adjacent else MIN_REGIME_SECS

            # Minimum time-in-regime: don't even start counting until elapsed
            if (snap_time and self._regime_confirmed_at and
                    (snap_time - self._regime_confirmed_at).total_seconds() < min_secs):
                return None

            if current == self._regime_candidate:
                self._regime_candidate_count += 1
            else:
                self._regime_candidate = current
                self._regime_candidate_count = 1

            if self._regime_candidate_count < REGIME_HOLD_COUNT:
                return None

            # Confirmed shift
            old_regime = self._confirmed_regime
            self._confirmed_regime = current
            self._regime_candidate_count = 0
            self._regime_confirmed_at = snap_time

            confidence = one_hour.regime_confidence if one_hour else 0
            slope = one_hour.net_gamma_slope if one_hour else 0

            msg = (
                f"**REGIME SHIFT**\n"
                f"{old_regime.value} -> **{current.value}**\n\n"
                f"Confidence: {confidence:.0f}%\n"
                f"Gamma Slope: {slope:+.2f} Bn/hr\n"
            )

            if current == Regime.UNCONTROLLED:
                msg += "\nExpect expanded ranges and directional moves. Fade setups invalid."
            elif current == Regime.CONTROLLED_PIN:
                msg += "\nMean reversion dominant. Sell premium viable."
            elif current == Regime.FRAGILE_CONTROL:
                msg += "\nDual-scenario mode. Walls may hold or fail — prepare for both."

            return Signal(
                signal_type=SignalType.REGIME_SHIFT,
                title=f"Regime: {current.value}",
                message=msg,
                channel="gex_ops",
                priority=SignalType.REGIME_SHIFT,
                metadata={"from": old_regime.value, "to": current.value},
            )
        else:
            # Same as confirmed — reset candidate
            self._regime_candidate = current
            self._regime_candidate_count = 0

        return None

    def _check_wall_breach(self, snapshot: GEXSnapshot) -> Optional[Signal]:
        if snapshot.call_wall == 0 or snapshot.put_floor == 0:
            return None

        # Walls are meaningless when spread is too tight
        if snapshot.spread < MIN_SPREAD:
            return None

        price = snapshot.curr_price
        # Require price to exceed wall by buffer, not just a tick through
        breached_call = price > (snapshot.call_wall + WALL_BREACH_BUFFER)
        breached_put = price < (snapshot.put_floor - WALL_BREACH_BUFFER)

        if not breached_call and not breached_put:
            # Don't clear breach state — wall value may just be flickering
            return None

        # Don't re-fire the same side with the same wall value
        side = "call" if breached_call else "put"
        wall_value = snapshot.call_wall if breached_call else snapshot.put_floor
        if side == self._prev_breach_side and wall_value == self._last_breach_wall_value:
            return None
        self._prev_breach_side = side
        self._last_breach_wall_value = wall_value

        if breached_call:
            dist = price - snapshot.call_wall
            msg = (
                f"**WALL BREACH — CALL WALL**\n"
                f"Price {price:.0f} broke above Call Wall {snapshot.call_wall}\n\n"
                f"Distance: +{dist:.0f} pts above\n"
                f"Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn\n"
                f"Spread: {snapshot.spread} pts\n\n"
                f"Watch for gamma squeeze acceleration or rejection back below."
            )
        else:
            dist = snapshot.put_floor - price
            msg = (
                f"**WALL BREACH — PUT FLOOR**\n"
                f"Price {price:.0f} broke below Put Floor {snapshot.put_floor}\n\n"
                f"Distance: -{dist:.0f} pts below\n"
                f"Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn\n"
                f"Spread: {snapshot.spread} pts\n\n"
                f"Watch for accelerated selling or bounce back above."
            )

        return Signal(
            signal_type=SignalType.WALL_BREACH,
            title="Wall Breach",
            message=msg,
            channel="gex_ops",
            priority=SignalType.WALL_BREACH,
        )

    def _check_gamma_squeeze(
        self,
        snapshot: GEXSnapshot,
        one_hour: Optional[OneHourMetrics],
        advanced: Optional[AdvancedMetrics],
    ) -> Optional[Signal]:
        if snapshot.call_wall == 0 or snapshot.spread < MIN_SPREAD:
            return None

        price = snapshot.curr_price
        above_wall = price > snapshot.call_wall
        accel = one_hour.gamma_accel if one_hour else 0
        squeeze_prob = advanced.squeeze_probability if advanced else 0

        if not above_wall or accel <= 0:
            # Price dropped back below wall — reset so next squeeze can fire
            self._squeeze_active = False
            return None

        # State-change dedup: only fire once per squeeze event
        if self._squeeze_active:
            return None
        self._squeeze_active = True

        msg = (
            f"**GAMMA SQUEEZE ALERT**\n"
            f"Price {price:.0f} above Call Wall {snapshot.call_wall}\n"
            f"Gamma accelerating: {accel:+.2f} Bn/hr^2\n"
            f"Squeeze Probability: {squeeze_prob:.0f}%\n\n"
            f"Dealer hedging may amplify upside move."
        )

        return Signal(
            signal_type=SignalType.GAMMA_SQUEEZE,
            title="Gamma Squeeze",
            message=msg,
            channel="gex_ops",
            priority=SignalType.GAMMA_SQUEEZE,
            metadata={"squeeze_prob": squeeze_prob},
        )

    def _check_bounce_zone(
        self,
        snapshot: GEXSnapshot,
        regime: Regime,
        fifteen_min: Optional[FifteenMinMetrics],
    ) -> Optional[Signal]:
        BOUNCE_THRESHOLD = 15.0
        CLEAR_THRESHOLD = 30.0  # price must move this far from wall to "clear" the zone
        price = snapshot.curr_price

        # Skip when spread is too tight — walls are noise
        if snapshot.spread < MIN_SPREAD:
            return None

        near_call = abs(price - snapshot.call_wall) <= BOUNCE_THRESHOLD if snapshot.call_wall else False
        near_put = abs(price - snapshot.put_floor) <= BOUNCE_THRESHOLD if snapshot.put_floor else False

        if not near_call and not near_put:
            # Price moved away — mark as cleared so next approach can fire
            self._bounce_cleared = True
            self._last_bounce_side = None
            return None

        # State-change dedup: don't re-fire same side until price cleared the zone
        side = "call" if near_call else "put"
        if side == self._last_bounce_side and not self._bounce_cleared:
            return None

        # Mark this side as active (not cleared)
        self._last_bounce_side = side
        self._bounce_cleared = False

        wall_name = "Call Wall" if near_call else "Put Floor"
        wall_val = snapshot.call_wall if near_call else snapshot.put_floor
        dist = abs(price - wall_val)

        # Urgency tiers for Telegram filtering
        if dist < 5:
            urgency = "CONTACT"
        elif dist < 10:
            urgency = "CLOSE"
        else:
            urgency = "APPROACH"

        levels = compute_levels_for_regime(snapshot, regime)

        # Urgency label for message
        _urgency_labels = {
            "CONTACT": f"AT {wall_name} — bounce or break imminent",
            "CLOSE": f"Very close to {wall_name}",
            "APPROACH": f"Approaching {wall_name}",
        }
        urgency_note = _urgency_labels.get(urgency, "")

        if regime == Regime.FRAGILE_CONTROL:
            # Dual-scenario format
            msg = (
                f"**BOUNCE ZONE — {wall_name} (FRAGILE)** [{urgency}]\n"
                f"Price {price:.0f} within {dist:.0f} pts of {wall_name} {wall_val}\n"
                f"{urgency_note}\n\n"
                f"**Scenario A — Wall Holds:**\n"
            )
            if near_call and "short" in levels:
                sl = levels["short"]
                msg += f"  SHORT: Entry {sl.entry_low:.0f}-{sl.entry_high:.0f} | Stop {sl.stop:.0f} | Target {sl.target:.0f}\n"
            elif near_put and "long" in levels:
                ll = levels["long"]
                msg += f"  LONG: Entry {ll.entry_low:.0f}-{ll.entry_high:.0f} | Stop {ll.stop:.0f} | Target {ll.target:.0f}\n"

            msg += f"\n**Scenario B — Wall Fails:**\n"
            if near_call:
                msg += f"  If price sustains above {wall_val}, gamma squeeze in play.\n"
            else:
                msg += f"  If price breaks below {wall_val}, accelerated downside likely.\n"
        else:
            msg = (
                f"**BOUNCE ZONE — {wall_name}** [{urgency}]\n"
                f"Price {price:.0f} within {dist:.0f} pts of {wall_name} {wall_val}\n"
                f"{urgency_note}\n"
                f"Regime: {regime.value}\n\n"
            )
            if near_call and "short" in levels and levels["short"].valid:
                sl = levels["short"]
                msg += f"SHORT: Entry {sl.entry_low:.0f}-{sl.entry_high:.0f} | Stop {sl.stop:.0f} | Target {sl.target:.0f} | R:R {sl.rr_ratio:.1f}:1\n"
                msg += f"Instrument: {sl.instrument_hint}\n"
            elif near_put and "long" in levels and levels["long"].valid:
                ll = levels["long"]
                msg += f"LONG: Entry {ll.entry_low:.0f}-{ll.entry_high:.0f} | Stop {ll.stop:.0f} | Target {ll.target:.0f} | R:R {ll.rr_ratio:.1f}:1\n"
                msg += f"Instrument: {ll.instrument_hint}\n"

        return Signal(
            signal_type=SignalType.BOUNCE_ZONE,
            title=f"Bounce Zone — {wall_name}",
            message=msg,
            channel="gex_ops",
            priority=SignalType.BOUNCE_ZONE,
            metadata={"urgency": urgency, "distance_pts": dist, "gamma_bn": snapshot.net_gamma / 1e9},
        )

    def _check_gamma_snap(
        self,
        snapshot: GEXSnapshot,
        snapshots_1h: Optional[List[GEXSnapshot]],
    ) -> Optional[Signal]:
        """Gamma-velocity trade signal — fires on rubber-band and fake-rally setups.

        Combined backtest (GEX + SPY context, 13 days / 3878 RTH snapshots):
          LONG tiers (neg gamma + price falling = rubber band snap):
            T1: G<-5Bn + px dn >15pts/20m  => 59% win, +13.0 net  [Telegram]
            T2: G<-5Bn + px dn >10pts/20m  => 54% win, +9.9 net   [Discord only]
            T3: G<-10Bn + px dn >5pts/20m  => 57% win, +11.7 net  [Telegram]
            T4: G<-15Bn (any price move)   => 37% win, +2.5 net   [Discord only]
          SHORT (fake rally — price up but gamma falling):
            S1: px up >10pts/15m + dG<-1Bn => 77% win, +9.4 net   [Telegram]

          SPY context findings:
            T1 + RSI<40 = 30% win (-10.5 net) — SUPPRESS on Telegram
            T3 + VIX>20 = 60% win — mild boost
        """
        cfg = _GAMMA_SNAP_CFG
        if not cfg.get("enabled", True):
            return None

        if not snapshots_1h or len(snapshots_1h) < 8:
            return None

        gamma_bn = snapshot.net_gamma / 1e9
        price = snapshot.curr_price

        # Find snapshot ~20 min ago (10 snaps at 2-min) and ~15 min ago (8 snaps)
        # snapshots_1h is sorted by time ascending, snapshot is latest
        lookback_20m = None
        lookback_15m = None
        now_ts = snapshot.timestamp_pt.timestamp() if hasattr(snapshot.timestamp_pt, 'timestamp') else snapshot.timestamp_pt

        for s in reversed(snapshots_1h):
            s_ts = s.timestamp_pt.timestamp() if hasattr(s.timestamp_pt, 'timestamp') else s.timestamp_pt
            age_sec = now_ts - s_ts
            if lookback_15m is None and age_sec >= 800:  # ~13-15 min
                lookback_15m = s
            if lookback_20m is None and age_sec >= 1100:  # ~18-20 min
                lookback_20m = s
            if lookback_20m and lookback_15m:
                break

        if not lookback_20m:
            return None

        g_20m_ago = lookback_20m.net_gamma / 1e9
        p_20m_ago = lookback_20m.curr_price
        dG_20m = gamma_bn - g_20m_ago
        dP_20m = price - p_20m_ago

        # Also compute 15m lookback for short signal
        dG_15m = 0.0
        dP_15m = 0.0
        if lookback_15m:
            dG_15m = gamma_bn - lookback_15m.net_gamma / 1e9
            dP_15m = price - lookback_15m.curr_price

        # Thresholds from config (with backtested defaults)
        long_t1_gamma = float(cfg.get("long_t1_gamma_bn", -5))
        long_t1_px_fall = float(cfg.get("long_t1_px_fall", 15))
        long_t2_gamma = float(cfg.get("long_t2_gamma_bn", -5))
        long_t2_px_fall = float(cfg.get("long_t2_px_fall", 10))
        long_t3_gamma = float(cfg.get("long_t3_gamma_bn", -10))
        long_t3_px_fall = float(cfg.get("long_t3_px_fall", 5))
        long_t4_gamma = float(cfg.get("long_t4_gamma_bn", -15))
        short_px_rise = float(cfg.get("short_px_rise_15m", 10))
        short_dg_fall = float(cfg.get("short_dg_fall_15m", -1))

        side = None
        tier = None
        tier_label = ""
        win_pct = 0
        rr = 0.0

        # --- LONG tiers (most selective first) ---
        if gamma_bn <= long_t1_gamma and dP_20m <= -long_t1_px_fall:
            side, tier, tier_label = "LONG", 1, "T1"
            win_pct, rr = 65, 9.0
        elif gamma_bn <= long_t2_gamma and dP_20m <= -long_t2_px_fall:
            side, tier, tier_label = "LONG", 2, "T2"
            win_pct, rr = 56, 5.5
        elif gamma_bn <= long_t3_gamma and dP_20m <= -long_t3_px_fall:
            side, tier, tier_label = "LONG", 3, "T3"
            win_pct, rr = 47, 4.4
        elif gamma_bn <= long_t4_gamma:
            side, tier, tier_label = "LONG", 4, "T4"
            win_pct, rr = 36, 2.4

        # --- SHORT: fake rally (price up but gamma falling) ---
        if side is None and lookback_15m:
            if dP_15m >= short_px_rise and dG_15m <= short_dg_fall:
                side, tier, tier_label = "SHORT", 1, "S1"
                win_pct, rr = 76, 0.0  # R:R not computed for short

        if side is None:
            # No signal — check if we should clear state
            if self._last_gamma_snap_side is not None:
                # Clear after conditions relax
                if abs(dP_20m) < 3 and abs(gamma_bn) < 3:
                    self._gamma_snap_cleared = True
                    self._last_gamma_snap_side = None
            return None

        # Dedup: don't re-fire same side until conditions cleared
        # Exception: T1 LONG can re-fire (cooldown gates frequency) for 3-4 alerts
        if side == self._last_gamma_snap_side and not self._gamma_snap_cleared:
            if not (side == "LONG" and tier == 1):
                return None

        self._last_gamma_snap_side = side
        self._gamma_snap_cleared = False

        # Time-of-day context: late session 0DTE moves are amplified
        now_pt = datetime.now(PT_TZ)
        hour_pt = now_pt.hour
        late_session = hour_pt >= 11 or (hour_pt == 10 and now_pt.minute >= 30)
        time_tag = ""
        if late_session and side == "LONG" and tier == 1:
            time_tag = " [LATE SESSION]"

        # Build message
        if side == "LONG":
            msg = (
                f"**GAMMA SNAP — LONG [{tier_label}]{time_tag}**\n"
                f"Price: {price:.0f} | Gamma: {gamma_bn:+.1f} Bn\n"
                f"20m move: {dP_20m:+.0f} pts | dG: {dG_20m:+.1f} Bn\n"
                f"Backtest: {win_pct}% win | {rr:.1f}:1 R:R\n"
            )
            if snapshot.put_floor:
                msg += f"Put floor: {snapshot.put_floor} ({price - snapshot.put_floor:.0f} pts away)\n"
            if tier <= 2:
                msg += "Rubber band snap — dealers forced to buy.\n"
            elif tier == 3:
                msg += "Deep gamma (<-10Bn) snap — structural bounce setup.\n"
            elif tier == 4:
                msg += "Extreme negative gamma — elevated bounce probability.\n"
            if late_session and tier == 1:
                msg += "0DTE gamma extreme — late session amplifies this move.\n"
        else:
            msg = (
                f"**GAMMA SNAP — SHORT [FAKE RALLY]**\n"
                f"Price: {price:.0f} | Gamma: {gamma_bn:+.1f} Bn\n"
                f"15m move: {dP_15m:+.0f} pts | dG: {dG_15m:+.1f} Bn\n"
                f"Backtest: {win_pct}% win\n"
                f"Rally not supported by gamma — reversal likely.\n"
            )
            if snapshot.call_wall:
                msg += f"Call wall: {snapshot.call_wall} ({snapshot.call_wall - price:.0f} pts away)\n"

        return Signal(
            signal_type=SignalType.GAMMA_SNAP,
            title=f"Gamma Snap — {side} {tier_label}",
            message=msg,
            channel="gex_ops",
            priority=SignalType.GAMMA_SNAP,
            metadata={
                "side": side,
                "tier": tier,
                "tier_label": tier_label,
                "gamma_bn": gamma_bn,
                "dP_20m": dP_20m,
                "dG_20m": dG_20m,
                "win_pct": win_pct,
                "rr": rr,
                "late_session": late_session,
            },
        )

    def _check_gamma_compression(
        self,
        snapshot: GEXSnapshot,
        snapshots_1h: Optional[List[GEXSnapshot]],
    ) -> Optional[Signal]:
        if not snapshots_1h or len(snapshots_1h) < 2:
            return None

        earliest = snapshots_1h[0]
        if earliest.call_wall == 0 or earliest.put_floor == 0:
            return None

        old_spread = earliest.call_wall - earliest.put_floor
        new_spread = snapshot.call_wall - snapshot.put_floor

        if old_spread <= 0:
            return None

        compression_pct = (old_spread - new_spread) / old_spread * 100

        if compression_pct < 20.0:
            # Compression ended — reset so next compression event can fire
            self._compression_active = False
            self._last_compression_pct = 0.0
            return None

        # Also require meaningful absolute spread on the old side (not just tiny walls)
        if old_spread < MIN_SPREAD:
            return None

        # State-change dedup: only fire once per compression event,
        # or again if compression deepened by another 20%
        if self._compression_active:
            if compression_pct < self._last_compression_pct + 20:
                return None

        self._compression_active = True
        self._last_compression_pct = compression_pct

        msg = (
            f"**GAMMA COMPRESSION**\n"
            f"Walls converging: spread narrowed {compression_pct:.0f}%\n"
            f"Call Wall: {earliest.call_wall} -> {snapshot.call_wall}\n"
            f"Put Floor: {earliest.put_floor} -> {snapshot.put_floor}\n"
            f"Spread: {old_spread} -> {new_spread} pts\n\n"
            f"Compressed range often precedes expansion. Breakout imminent."
        )

        return Signal(
            signal_type=SignalType.GAMMA_COMPRESSION,
            title="Gamma Compression",
            message=msg,
            channel="gex_ops",
            priority=SignalType.GAMMA_COMPRESSION,
        )

    def _check_strike_flip(self, advanced: Optional[AdvancedMetrics]) -> Optional[Signal]:
        if not advanced or not advanced.pivot_strikes:
            return None

        # Only report flips where gamma is significant (not noise from tiny positions)
        flipped = [
            p for p in advanced.pivot_strikes
            if p.pivot_type == PivotType.FLIPPED
            and abs(p.call_gamma) / 1e9 >= MIN_PIVOT_GAMMA_BN
        ]
        if not flipped:
            # If nothing is flipped now, clear history so we can re-detect if they flip again later
            self._reported_flipped_strikes.clear()
            return None

        # State-change dedup: only report strikes we haven't already reported
        new_flips = [p for p in flipped if p.strike not in self._reported_flipped_strikes]
        if not new_flips:
            return None

        # Record these as reported
        for p in new_flips:
            self._reported_flipped_strikes.add(p.strike)

        lines = ["**STRIKE FLIP DETECTED**\n"]
        for p in new_flips:
            lines.append(
                f"Strike {p.strike}: Put gamma ({p.put_gamma:,}) overtaking Call gamma ({p.call_gamma:,})\n"
                f"  Put/Call ratio: {p.put_call_ratio:.2f}\n"
            )
        lines.append("\nThis strike has flipped from support to resistance. Watch for breakdown.")

        return Signal(
            signal_type=SignalType.STRIKE_FLIP,
            title=f"Strike Flip: {new_flips[0].strike}",
            message="".join(lines),
            channel="gex_ops",
            priority=SignalType.STRIKE_FLIP,
            metadata={"strikes": [p.strike for p in new_flips]},
        )

    def _augment_day_profile_flags(self, flags: set, snapshot: GEXSnapshot) -> set:
        if not _is_rth(snapshot.timestamp_pt):
            return flags

        g_bn = snapshot.net_gamma_bn
        if self._trend_open_gamma_bn is None:
            self._trend_open_gamma_bn = g_bn
        if self._trend_early_min_gamma_bn is None:
            self._trend_early_min_gamma_bn = g_bn

        cutoff = time(7, 30)
        snap_t = snapshot.timestamp_pt.astimezone(PT_TZ).time()
        if snap_t <= cutoff:
            self._trend_early_min_gamma_bn = min(self._trend_early_min_gamma_bn, g_bn)

        open_gamma_bn = self._trend_open_gamma_bn
        min_early_gamma_bn = self._trend_early_min_gamma_bn
        open_neg_bn = float(_DAY_PROFILE_CFG.get("open_negative_bn", -4.0))
        early_crash_bn = float(_DAY_PROFILE_CFG.get("early_crash_bn", -9.0))
        open_pos_bn = float(_DAY_PROFILE_CFG.get("open_positive_bn", 5.0))
        early_not_too_negative_bn = float(
            _DAY_PROFILE_CFG.get("early_not_too_negative_bn", -5.0)
        )

        if open_gamma_bn <= open_neg_bn or min_early_gamma_bn <= early_crash_bn:
            flags.add("day_profile_risk_off_fast_down")
        elif open_gamma_bn >= open_pos_bn and min_early_gamma_bn > early_not_too_negative_bn:
            flags.add("day_profile_controlled_up_or_range")
        else:
            flags.add("day_profile_mixed_transition")

        return flags

    def _check_trend_dashboard(
        self,
        trend: Optional[TrendDashboard],
        snapshot: GEXSnapshot,
    ) -> Optional[Signal]:
        if not trend or trend.bias == "NEUTRAL":
            return None

        flags = self._augment_day_profile_flags(set(trend.flags or []), snapshot)
        # Day-profile gating to cut counter-trend noise.
        if "day_profile_risk_off_fast_down" in flags and trend.bias == "BULLISH":
            if snapshot.net_gamma_bn < 2.0:
                return None
            if trend.smell or trend.score < (TREND_SCORE_THRESHOLD + 1):
                return None
        if "day_profile_controlled_up_or_range" in flags and trend.bias == "BEARISH":
            if trend.smell and "price_pressing_floor" not in flags and "gamma_decay" not in flags:
                return None

        # Move-from-open counter-trend gating: require stronger score (or suppress entirely)
        # when bias fights the realized move from the RTH open.
        if trend.open_price is not None:
            move_from_open = snapshot.curr_price - trend.open_price
            is_counter_trend = (
                (trend.bias == "BULLISH" and move_from_open < 0)
                or (trend.bias == "BEARISH" and move_from_open > 0)
            )
            if is_counter_trend:
                abs_move = abs(move_from_open)
                if _COUNTER_TREND_SUPPRESS_ALL_PTS > 0 and abs_move >= _COUNTER_TREND_SUPPRESS_ALL_PTS:
                    return None
                for move_pts, min_score in _COUNTER_TREND_GATES:
                    if abs_move >= move_pts and trend.score < min_score:
                        return None

        now_ts = snapshot.timestamp_pt.timestamp()
        first_of_day = self._last_trend_bias is None
        bias_changed = (not first_of_day) and (trend.bias != self._last_trend_bias)
        heartbeat_due = (
            TREND_HEARTBEAT_SECS > 0
            and (not first_of_day)
            and (not bias_changed)
            and (not self._trend_heartbeat_sent)
            and (now_ts - self._last_trend_change_ts) >= TREND_HEARTBEAT_SECS
            and _is_rth(snapshot.timestamp_pt)
        )
        if not first_of_day and not bias_changed and not heartbeat_due:
            return None

        # Require conviction for emitted trend alerts.
        # Exception: allow the first trend alert of the day even with low score.
        # Heartbeats also obey this threshold.
        if (not first_of_day) and TREND_MIN_EMIT_SCORE > 0 and trend.score < TREND_MIN_EMIT_SCORE:
            return None

        self._last_trend_bias = trend.bias
        self._last_trend_smell = trend.smell
        self._last_trend_score = trend.score
        self._last_trend_flags = flags
        self._last_trend_emit_ts = now_ts

        heartbeat_only = (
            heartbeat_due
            and not first_of_day
            and not bias_changed
        )
        if first_of_day or bias_changed:
            self._last_trend_change_ts = now_ts
            self._trend_heartbeat_sent = False
        elif heartbeat_only:
            self._trend_heartbeat_sent = True
        label = "HEARTBEAT" if heartbeat_only else ("SMELL" if trend.smell else "TREND")
        lines = [
            f"**TREND DASHBOARD - {trend.bias} {label}**",
            f"Score: {trend.score} | Price: {snapshot.curr_price:.0f} | Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn",
            f"Call Wall: {snapshot.call_wall} | Put Floor: {snapshot.put_floor} | Spread: {snapshot.spread} pts",
            "",
        ]

        for w in sorted(trend.windows, key=lambda x: x.minutes):
            tag = "BEAR" if w.bearish else "BULL" if w.bullish else "NEUTRAL"
            lines.append(
                f"{w.minutes}m {tag} | dPx {w.price_delta:+.0f} | dG {w.gamma_delta_bn:+.2f} Bn"
            )

        lines.append("")
        lines.append(f"Support targets: {_format_targets(trend.support_targets)}")
        lines.append(f"Resistance targets: {_format_targets(trend.resistance_targets)}")

        if trend.notes:
            lines.append("")
            lines.append("Notes:")
            for note in trend.notes[:6]:
                lines.append(f"- {note}")

        return Signal(
            signal_type=SignalType.TREND_DASHBOARD,
            title=f"Trend Dashboard - {trend.bias}",
            message="\n".join(lines),
            channel="market_pulse",
            priority=SignalType.TREND_DASHBOARD,
            metadata={
                "bias": trend.bias,
                "score": trend.score,
                "smell": trend.smell,
                "flags": list(flags),
                "heartbeat": heartbeat_only,
            },
        )

    def _check_overnight_drift(self, overnight: Optional[OvernightDrift]) -> Optional[Signal]:
        if not overnight:
            return None

        # Only signal if shift is significant (>15%)
        if abs(overnight.gamma_shift_pct) < 15:
            return None

        # State-change dedup: only fire if drift changed by >10% from last report
        if self._last_drift_pct is not None:
            delta = abs(overnight.gamma_shift_pct - self._last_drift_pct)
            if delta < 10:
                return None

        self._last_drift_pct = overnight.gamma_shift_pct

        direction = "higher" if overnight.gamma_shift > 0 else "lower"
        msg = (
            f"**OVERNIGHT DRIFT**\n"
            f"Net Gamma shifted {direction}: {overnight.gamma_shift_pct:+.1f}%\n"
            f"  EOD: {overnight.eod_net_gamma / 1e9:+.2f} Bn -> Now: {overnight.current_net_gamma / 1e9:+.2f} Bn\n\n"
            f"Call Wall: {overnight.eod_call_wall} -> {overnight.current_call_wall} ({overnight.call_wall_moved:+d})\n"
            f"Put Floor: {overnight.eod_put_floor} -> {overnight.current_put_floor} ({overnight.put_floor_moved:+d})\n\n"
            f"Range Forecast: **{overnight.range_forecast}**"
        )

        return Signal(
            signal_type=SignalType.OVERNIGHT_DRIFT,
            title="Overnight Drift",
            message=msg,
            channel="daily_intel",
            priority=SignalType.OVERNIGHT_DRIFT,
        )


# --- Formatting helpers ---


def _format_rth_pulse(
    snapshot: GEXSnapshot,
    one_hour: Optional[OneHourMetrics],
    advanced: Optional[AdvancedMetrics],
    levels: dict,
) -> str:
    regime = one_hour.regime.value if one_hour else "N/A"
    slope = one_hour.net_gamma_slope if one_hour else 0
    confidence = one_hour.regime_confidence if one_hour else 0

    lines = [
        f"**RTH PULSE** | {datetime.now(PT_TZ).strftime('%I:%M %p PT')}",
        f"",
        f"Price: {snapshot.curr_price:.0f}",
        f"Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn",
        f"Regime: {regime} ({confidence:.0f}%)",
        f"Slope: {slope:+.2f} Bn/hr",
        f"Call Wall: {snapshot.call_wall} | Put Floor: {snapshot.put_floor}",
        f"Spread: {snapshot.spread} pts",
    ]

    if advanced:
        lines.append(f"GCI: {advanced.gci:.0f}% ({advanced.gci_label})")
        lines.append(f"Squeeze Prob: {advanced.squeeze_probability:.0f}% | Pin Prob: {advanced.pin_probability:.0f}%")
        if advanced.pivot_strikes:
            pivots = ", ".join(f"{p.strike}({p.pivot_type.value})" for p in advanced.pivot_strikes)
            lines.append(f"Pivots: {pivots}")

    if "long" in levels and levels["long"].valid:
        ll = levels["long"]
        lines.append(f"\nLONG: {ll.entry_low:.0f}-{ll.entry_high:.0f} | Stop {ll.stop:.0f} | Target {ll.target:.0f} (R:R {ll.rr_ratio:.1f})")
    if "short" in levels and levels["short"].valid:
        sl = levels["short"]
        lines.append(f"SHORT: {sl.entry_low:.0f}-{sl.entry_high:.0f} | Stop {sl.stop:.0f} | Target {sl.target:.0f} (R:R {sl.rr_ratio:.1f})")

    return "\n".join(lines)


def _format_targets(targets: list[tuple[int, float]]) -> str:
    if not targets:
        return "n/a"
    parts = []
    for strike, gamma_bn in targets:
        parts.append(f"{strike} ({gamma_bn:+.2f}Bn)")
    return ", ".join(parts)


def _format_final_15(
    snapshot: GEXSnapshot,
    advanced: Optional[AdvancedMetrics],
) -> str:
    pin_prob = advanced.pin_probability if advanced else 0
    gci = advanced.gci if advanced else 0

    lines = [
        f"**FINAL 15 MINUTES** | 12:45 PM PT",
        f"",
        f"Price: {snapshot.curr_price:.0f}",
        f"Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn",
        f"Call Wall: {snapshot.call_wall} | Put Floor: {snapshot.put_floor}",
        f"",
        f"Pin Probability: **{pin_prob:.0f}%**",
        f"Gamma Concentration: {gci:.0f}%",
    ]

    if pin_prob > 70:
        lines.append(f"\nStrong pin expected near max-gamma strike. Sell premium favored.")
    elif pin_prob > 40:
        lines.append(f"\nModerate pin potential. Watch for late-day drift.")
    else:
        lines.append(f"\nWeak pin signal. Expect volatility into close.")

    if advanced and advanced.pivot_strikes:
        for p in advanced.pivot_strikes:
            lines.append(f"Pivot Strike {p.strike}: {p.pivot_type.value} (ratio {p.put_call_ratio:.2f})")

    return "\n".join(lines)


def _format_morning_brief(
    snapshot: GEXSnapshot,
    overnight: Optional[OvernightDrift],
    advanced: Optional[AdvancedMetrics],
) -> str:
    lines = [
        f"**MORNING BRIEF** | {datetime.now(PT_TZ).strftime('%A %b %d')} | 6:00 AM PT",
        f"",
        f"Pre-Market Price: {snapshot.curr_price:.0f}",
        f"Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn",
        f"Call Wall: {snapshot.call_wall} | Put Floor: {snapshot.put_floor}",
        f"Spread: {snapshot.spread} pts",
    ]

    if overnight:
        lines.append(f"")
        lines.append(f"**Overnight Changes:**")
        lines.append(f"  Gamma: {overnight.gamma_shift_pct:+.1f}%")
        lines.append(f"  Call Wall: {overnight.call_wall_moved:+d} pts")
        lines.append(f"  Put Floor: {overnight.put_floor_moved:+d} pts")
        lines.append(f"  Range Forecast: {overnight.range_forecast}")

    if advanced:
        lines.append(f"")
        lines.append(f"GCI: {advanced.gci:.0f}% ({advanced.gci_label})")
        lines.append(f"Asymmetry: {advanced.gamma_asymmetry:.2f} ({advanced.asymmetry_label})")
        if advanced.pivot_strikes:
            for p in advanced.pivot_strikes:
                lines.append(f"Pivot: {p.strike} — {p.pivot_type.value}")

    lines.append(f"")
    lines.append(f"Key levels to watch: {snapshot.put_floor} (floor) / {snapshot.call_wall} (ceiling)")

    return "\n".join(lines)


def _format_quiet_summary(
    snapshot: GEXSnapshot,
    fifteen_min: Optional[FifteenMinMetrics],
    one_hour: Optional[OneHourMetrics],
    advanced: Optional[AdvancedMetrics],
) -> str:
    regime = one_hour.regime.value if one_hour else "N/A"
    confidence = one_hour.regime_confidence if one_hour else 0

    lines = [
        f"**TRADING SUMMARY** | {snapshot.timestamp_pt.strftime('%I:%M %p PT')}",
        f"Price: {snapshot.curr_price:.0f}",
        f"Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn",
        f"Regime: {regime} ({confidence:.0f}%)",
    ]

    if fifteen_min:
        lines.append(
            f"15m: dPx {fifteen_min.price_delta:+.1f} | dG {fifteen_min.net_gamma_delta/1e9:+.2f} Bn"
        )

    lines.append(f"Call Wall: {snapshot.call_wall} | Put Floor: {snapshot.put_floor}")
    lines.append(f"Spread: {snapshot.spread} pts")

    if advanced:
        lines.append(f"GCI: {advanced.gci:.0f}% ({advanced.gci_label}) | Pin {advanced.pin_probability:.0f}%")

    return "\n".join(lines)
