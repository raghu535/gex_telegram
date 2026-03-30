"""Trend engine: 15-min metrics, 1-hour metrics, regime classification, advanced analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pytz
import yaml
import os

from gex_parser import GEXSnapshot

PT_TZ = pytz.timezone("US/Pacific")

# Load config
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as f:
    CFG = yaml.safe_load(f)

REGIME_CFG = CFG["regime"]
ADVANCED_CFG = CFG["advanced"]
TREND_CFG = CFG.get("trend", {})


class Regime(Enum):
    CONTROLLED_PIN = "CONTROLLED_PIN"
    CONTROLLED_TREND = "CONTROLLED_TREND"
    FRAGILE_CONTROL = "FRAGILE_CONTROL"
    UNCONTROLLED = "UNCONTROLLED"


class PivotType(Enum):
    SUPPORT = "SUPPORT"
    PIVOT = "PIVOT"
    FLIPPED = "FLIPPED"


@dataclass
class FifteenMinMetrics:
    timestamp: datetime
    price_delta: float = 0.0
    net_gamma_delta: int = 0
    net_gamma_delta_pct: float = 0.0
    gamma_ratio: float = 0.0
    gamma_ratio_trend: str = "STABLE"  # RISING, FALLING, STABLE
    call_wall: int = 0
    call_wall_stable: bool = True
    put_floor: int = 0
    put_floor_stable: bool = True
    distance_to_call_wall: float = 0.0
    distance_to_put_floor: float = 0.0
    gamma_flip_proximity_pct: float = 0.0
    curr_price: float = 0.0
    net_gamma: int = 0
    num_snapshots: int = 0


@dataclass
class OneHourMetrics:
    timestamp: datetime
    fifteen_min: FifteenMinMetrics
    price_range: float = 0.0
    net_gamma_slope: float = 0.0  # Bn/hr via linear regression
    gamma_accel: float = 0.0  # 2nd derivative
    regime: Regime = Regime.CONTROLLED_PIN
    regime_confidence: float = 0.0
    num_snapshots: int = 0


@dataclass
class OvernightDrift:
    eod_net_gamma: int = 0
    current_net_gamma: int = 0
    gamma_shift: int = 0
    gamma_shift_pct: float = 0.0
    eod_call_wall: int = 0
    current_call_wall: int = 0
    call_wall_moved: int = 0
    eod_put_floor: int = 0
    current_put_floor: int = 0
    put_floor_moved: int = 0
    range_forecast: str = "NORMAL"  # TIGHT, NORMAL, WIDE


@dataclass
class PivotStrike:
    strike: int
    call_gamma: int
    put_gamma: int
    put_call_ratio: float
    pivot_type: PivotType


@dataclass
class AdvancedMetrics:
    gci: float = 0.0  # Gamma Concentration Index (%)
    gci_label: str = "MODERATE"  # HARD_WALL, MODERATE, DIFFUSE
    gamma_velocity: float = 0.0  # Bn/min
    gamma_velocity_label: str = "STABLE"  # ACCELERATING, DECELERATING, STABLE
    gamma_asymmetry: float = 0.0  # call/put ratio
    asymmetry_label: str = "BALANCED"  # CALL_DOMINANT, PUT_DOMINANT, BALANCED
    squeeze_probability: float = 0.0  # 0-100
    pin_probability: float = 0.0  # 0-100
    pivot_strikes: List[PivotStrike] = field(default_factory=list)


@dataclass
class TrendWindow:
    minutes: int
    price_delta: float
    gamma_delta_bn: float
    weight: int
    num_snapshots: int
    bearish: bool = False
    bullish: bool = False


@dataclass
class TrendDashboard:
    timestamp: datetime
    bias: str  # BEARISH, BULLISH, NEUTRAL
    score: int
    smell: bool
    bearish_score: int
    bullish_score: int
    windows: List[TrendWindow] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    support_targets: List[Tuple[int, float]] = field(default_factory=list)
    resistance_targets: List[Tuple[int, float]] = field(default_factory=list)
    open_price: Optional[float] = None


def compute_fifteen_min_metrics(snapshots: List[GEXSnapshot]) -> Optional[FifteenMinMetrics]:
    """Compute 15-minute metrics from a list of recent snapshots."""
    if not snapshots or len(snapshots) < 2:
        return None

    latest = snapshots[-1]
    earliest = snapshots[0]

    price_delta = latest.curr_price - earliest.curr_price
    net_gamma_delta = latest.net_gamma - earliest.net_gamma
    net_gamma_delta_pct = (
        (net_gamma_delta / abs(earliest.net_gamma) * 100)
        if earliest.net_gamma != 0
        else 0.0
    )

    # Gamma ratio trend
    ratios = [s.gamma_ratio for s in snapshots if s.gamma_ratio > 0]
    if len(ratios) >= 2:
        ratio_change = ratios[-1] - ratios[0]
        if ratio_change > 0.1:
            ratio_trend = "RISING"
        elif ratio_change < -0.1:
            ratio_trend = "FALLING"
        else:
            ratio_trend = "STABLE"
    else:
        ratio_trend = "STABLE"

    # Wall stability
    call_walls = [s.call_wall for s in snapshots if s.call_wall > 0]
    put_floors = [s.put_floor for s in snapshots if s.put_floor > 0]
    call_wall_stable = len(set(call_walls)) <= 1 if call_walls else True
    put_floor_stable = len(set(put_floors)) <= 1 if put_floors else True

    # Distances
    price = latest.curr_price
    call_wall = latest.call_wall
    put_floor = latest.put_floor

    dist_call = abs(price - call_wall) if call_wall and price else 0.0
    dist_put = abs(price - put_floor) if put_floor and price else 0.0

    # Gamma flip proximity: how close net gamma is to zero as % of recent range
    gamma_values = [abs(s.net_gamma) for s in snapshots]
    gamma_range = max(gamma_values) if gamma_values else 1
    flip_proximity = (
        (1 - abs(latest.net_gamma) / gamma_range) * 100
        if gamma_range > 0
        else 0.0
    )

    return FifteenMinMetrics(
        timestamp=latest.timestamp_pt,
        price_delta=price_delta,
        net_gamma_delta=net_gamma_delta,
        net_gamma_delta_pct=net_gamma_delta_pct,
        gamma_ratio=latest.gamma_ratio,
        gamma_ratio_trend=ratio_trend,
        call_wall=call_wall,
        call_wall_stable=call_wall_stable,
        put_floor=put_floor,
        put_floor_stable=put_floor_stable,
        distance_to_call_wall=dist_call,
        distance_to_put_floor=dist_put,
        gamma_flip_proximity_pct=flip_proximity,
        curr_price=price,
        net_gamma=latest.net_gamma,
        num_snapshots=len(snapshots),
    )


def compute_one_hour_metrics(
    snapshots: List[GEXSnapshot],
    fifteen_min: FifteenMinMetrics,
) -> Optional[OneHourMetrics]:
    """Compute 1-hour metrics with linear regression on gamma slope."""
    if not snapshots or len(snapshots) < 3:
        return None

    latest = snapshots[-1]
    prices = [s.curr_price for s in snapshots]
    price_range = max(prices) - min(prices) if prices else 0.0

    # Linear regression on net gamma over time
    times = np.array([(s.timestamp_pt.timestamp() - snapshots[0].timestamp_pt.timestamp()) / 3600
                       for s in snapshots])
    gammas = np.array([s.net_gamma / 1e9 for s in snapshots])

    net_gamma_slope = 0.0
    gamma_accel = 0.0

    # Only fit if we have distinct time points (avoid SVD failure)
    time_range = times[-1] - times[0] if len(times) > 1 else 0
    if time_range > 0 and len(times) >= 3:
        try:
            coeffs = np.polyfit(times, gammas, min(2, len(times) - 1))
            if len(coeffs) >= 2:
                net_gamma_slope = coeffs[-2]  # Bn/hr (1st derivative)
            if len(coeffs) >= 3:
                gamma_accel = coeffs[-3] * 2  # 2nd derivative
        except np.linalg.LinAlgError:
            pass

    elif time_range > 0 and len(times) >= 2:
        try:
            coeffs = np.polyfit(times, gammas, 1)
            net_gamma_slope = coeffs[0]
        except np.linalg.LinAlgError:
            pass

    # Classify regime
    regime, confidence = _classify_regime(
        net_gamma_bn=latest.net_gamma / 1e9,
        slope=net_gamma_slope,
        accel=gamma_accel,
        price_range=price_range,
        fifteen_min=fifteen_min,
    )

    return OneHourMetrics(
        timestamp=latest.timestamp_pt,
        fifteen_min=fifteen_min,
        price_range=price_range,
        net_gamma_slope=net_gamma_slope,
        gamma_accel=gamma_accel,
        regime=regime,
        regime_confidence=confidence,
        num_snapshots=len(snapshots),
    )


def _classify_regime(
    net_gamma_bn: float,
    slope: float,
    accel: float,
    price_range: float,
    fifteen_min: FifteenMinMetrics,
) -> Tuple[Regime, float]:
    """Classify into 4-state regime with confidence score."""

    strong = REGIME_CFG["strong_gamma_bn"]
    moderate = REGIME_CFG["moderate_gamma_bn"]
    flat_slope = REGIME_CFG["flat_slope_range"]
    neg_slope = REGIME_CFG["negative_slope_threshold"]
    severe_slope = REGIME_CFG["severe_slope_threshold"]
    pin_range = REGIME_CFG["pin_range_pts"]

    # UNCONTROLLED: negative gamma or severe decay
    if net_gamma_bn < 0:
        confidence = min(100, abs(net_gamma_bn) / moderate * 100)
        return Regime.UNCONTROLLED, confidence

    if slope < severe_slope:
        confidence = min(100, abs(slope) / abs(severe_slope) * 80)
        return Regime.UNCONTROLLED, confidence

    # CONTROLLED_PIN: strong gamma, flat slope, tight range
    if (
        net_gamma_bn >= strong
        and abs(slope) <= flat_slope
        and price_range <= pin_range
    ):
        confidence = min(100, net_gamma_bn / strong * 60 + (1 - abs(slope) / flat_slope) * 40)
        return Regime.CONTROLLED_PIN, confidence

    # CONTROLLED_TREND: moderate+ gamma, slope aligned with price direction
    if net_gamma_bn >= moderate and slope > -flat_slope:
        price_up = fifteen_min.price_delta > 0
        slope_positive = slope > 0
        aligned = price_up == slope_positive or abs(slope) < flat_slope
        if aligned:
            confidence = min(100, net_gamma_bn / strong * 50 + 50)
            return Regime.CONTROLLED_TREND, confidence

    # FRAGILE_CONTROL: positive but decaying
    if net_gamma_bn > 0:
        decay_severity = 0
        if slope < 0:
            decay_severity += min(50, abs(slope) / abs(neg_slope) * 50)
        if accel < 0:
            decay_severity += 20
        flip_proximity = fifteen_min.gamma_flip_proximity_pct
        if flip_proximity > REGIME_CFG["flip_warning_pct"]:
            decay_severity += 30

        confidence = min(100, decay_severity)
        return Regime.FRAGILE_CONTROL, confidence

    # Fallback
    return Regime.FRAGILE_CONTROL, 50.0


def compute_overnight_drift(
    eod_snapshot: Optional[GEXSnapshot],
    current_snapshot: GEXSnapshot,
) -> Optional[OvernightDrift]:
    """Compute overnight gamma drift from previous EOD to current."""
    if not eod_snapshot:
        return None

    gamma_shift = current_snapshot.net_gamma - eod_snapshot.net_gamma
    gamma_shift_pct = (
        (gamma_shift / abs(eod_snapshot.net_gamma) * 100)
        if eod_snapshot.net_gamma != 0
        else 0.0
    )

    call_wall_moved = current_snapshot.call_wall - eod_snapshot.call_wall
    put_floor_moved = current_snapshot.put_floor - eod_snapshot.put_floor

    # Range forecast based on gamma shift magnitude
    abs_pct = abs(gamma_shift_pct)
    if abs_pct < 10:
        forecast = "TIGHT"
    elif abs_pct < 30:
        forecast = "NORMAL"
    else:
        forecast = "WIDE"

    return OvernightDrift(
        eod_net_gamma=eod_snapshot.net_gamma,
        current_net_gamma=current_snapshot.net_gamma,
        gamma_shift=gamma_shift,
        gamma_shift_pct=gamma_shift_pct,
        eod_call_wall=eod_snapshot.call_wall,
        current_call_wall=current_snapshot.call_wall,
        call_wall_moved=call_wall_moved,
        eod_put_floor=eod_snapshot.put_floor,
        current_put_floor=current_snapshot.put_floor,
        put_floor_moved=put_floor_moved,
        range_forecast=forecast,
    )


def _slice_window(
    snapshots: List[GEXSnapshot],
    minutes: int,
    latest_ts: datetime,
) -> List[GEXSnapshot]:
    cutoff = latest_ts - timedelta(minutes=minutes)
    return [s for s in snapshots if s.timestamp_pt >= cutoff]


def _wall_gamma(snapshot: GEXSnapshot, side: str) -> int:
    if side == "call":
        return snapshot.top5_calls[0][1] if snapshot.top5_calls else 0
    return snapshot.top5_puts[0][1] if snapshot.top5_puts else 0


def _select_targets(
    positions: List[Tuple[int, int]],
    price: float,
    side: str,
    count: int = 3,
) -> List[Tuple[int, float]]:
    """Select up to N targets from top gamma positions relative to price."""
    if not positions:
        return []

    if side == "support":
        below = [p for p in positions if p[0] <= price]
        above = [p for p in positions if p[0] > price]
        below.sort(key=lambda x: x[0], reverse=True)  # nearest below first
        above.sort(key=lambda x: x[0])               # nearest above next
        ordered = below + above
    else:
        above = [p for p in positions if p[0] >= price]
        below = [p for p in positions if p[0] < price]
        above.sort(key=lambda x: x[0])               # nearest above first
        below.sort(key=lambda x: x[0], reverse=True) # nearest below next
        ordered = above + below

    seen = set()
    targets: List[Tuple[int, float]] = []
    for strike, gamma in ordered:
        if strike in seen:
            continue
        seen.add(strike)
        targets.append((strike, gamma / 1e9))
        if len(targets) >= count:
            break
    return targets


def compute_trend_dashboard(
    snapshots: List[GEXSnapshot],
    overnight: Optional[OvernightDrift] = None,
) -> Optional[TrendDashboard]:
    """Compute multi-window trend dashboard for early bias detection."""
    if not snapshots:
        return None

    snapshots = sorted(snapshots, key=lambda s: s.timestamp_pt)
    latest = snapshots[-1]

    windows_cfg = TREND_CFG.get("windows", [])
    if not windows_cfg:
        return None

    bearish_score = 0
    bullish_score = 0
    windows: List[TrendWindow] = []
    notes: List[str] = []
    flags: List[str] = []

    for cfg in windows_cfg:
        minutes = int(cfg.get("minutes", 0))
        min_snaps = int(cfg.get("min_snaps", 2))
        price_pts = float(cfg.get("price_pts", 0))
        gamma_bn = float(cfg.get("gamma_bn", 0))
        weight = int(cfg.get("weight", 1))
        gamma_only_weight = int(cfg.get("gamma_only_weight", 0))

        if minutes <= 0:
            continue

        window_snaps = _slice_window(snapshots, minutes, latest.timestamp_pt)
        if len(window_snaps) < min_snaps:
            continue

        earliest = window_snaps[0]
        price_delta = latest.curr_price - earliest.curr_price
        gamma_delta_bn = (latest.net_gamma - earliest.net_gamma) / 1e9

        bearish = price_delta <= -price_pts and gamma_delta_bn <= -gamma_bn
        bullish = price_delta >= price_pts and gamma_delta_bn >= gamma_bn

        if bearish:
            bearish_score += weight
        if bullish:
            bullish_score += weight

        # Gamma-only decay/build for early "smell" before price moves
        if gamma_only_weight:
            if not bearish and gamma_delta_bn <= -gamma_bn:
                bearish_score += gamma_only_weight
                notes.append(f"{minutes}m gamma decay {gamma_delta_bn:+.2f} Bn")
            if not bullish and gamma_delta_bn >= gamma_bn:
                bullish_score += gamma_only_weight
                notes.append(f"{minutes}m gamma build {gamma_delta_bn:+.2f} Bn")

        windows.append(TrendWindow(
            minutes=minutes,
            price_delta=price_delta,
            gamma_delta_bn=gamma_delta_bn,
            weight=weight,
            num_snapshots=len(window_snaps),
            bearish=bearish,
            bullish=bullish,
        ))

    # Overnight influence
    overnight_pct = TREND_CFG.get("overnight_pct", 15)
    overnight_weight = int(TREND_CFG.get("overnight_weight", 1))
    if overnight and abs(overnight.gamma_shift_pct) >= overnight_pct:
        if overnight.gamma_shift_pct < 0:
            bearish_score += overnight_weight
            notes.append(f"Overnight gamma shift {overnight.gamma_shift_pct:+.1f}% (bearish)")
            flags.append("overnight_bearish")
        else:
            bullish_score += overnight_weight
            notes.append(f"Overnight gamma shift {overnight.gamma_shift_pct:+.1f}% (bullish)")
            flags.append("overnight_bullish")

    decay_pct = float(TREND_CFG.get("decay_pct", 25))
    decay_weight = int(TREND_CFG.get("decay_weight", 2))
    structural_weight = int(TREND_CFG.get("structural_weight", 1))

    # Wall strength changes over the last 60 minutes (if stable strike)
    window_60 = _slice_window(snapshots, 60, latest.timestamp_pt)
    if len(window_60) >= 2:
        earliest_60 = window_60[0]
        # Gamma decay from peak within the last hour
        gamma_peak = max(s.net_gamma for s in window_60)
        if gamma_peak > 0:
            decay = (gamma_peak - latest.net_gamma) / gamma_peak * 100
            if decay >= decay_pct:
                bearish_score += decay_weight
                notes.append(f"Gamma decay from peak: -{decay:.0f}%")
                flags.append("gamma_decay")
        if earliest_60.call_wall and latest.call_wall:
            if earliest_60.call_wall == latest.call_wall:
                start = abs(_wall_gamma(earliest_60, "call"))
                end = abs(_wall_gamma(latest, "call"))
                if start > 0:
                    pct = (end - start) / start * 100
                    if pct <= -40:
                        bearish_score += structural_weight
                        notes.append(f"Call wall weakening: {pct:.0f}% at {latest.call_wall}")
                        flags.append("call_wall_weakening")
                    elif pct >= 40:
                        bullish_score += structural_weight
                        notes.append(f"Call wall strengthening: +{pct:.0f}% at {latest.call_wall}")
                        flags.append("call_wall_strengthening")
            else:
                notes.append(f"Call wall shifted: {earliest_60.call_wall} -> {latest.call_wall}")

        if earliest_60.put_floor and latest.put_floor:
            if earliest_60.put_floor == latest.put_floor:
                start = abs(_wall_gamma(earliest_60, "put"))
                end = abs(_wall_gamma(latest, "put"))
                if start > 0:
                    pct = (end - start) / start * 100
                    if pct >= 40:
                        bearish_score += structural_weight
                        notes.append(f"Put floor building: +{pct:.0f}% at {latest.put_floor}")
                        flags.append("put_floor_building")
                    elif pct <= -40:
                        bullish_score += structural_weight
                        notes.append(f"Put floor weakening: {pct:.0f}% at {latest.put_floor}")
                        flags.append("put_floor_weakening")
            else:
                notes.append(f"Put floor shifted: {earliest_60.put_floor} -> {latest.put_floor}")

    # Pivot pressure (pre-flip)
    pivots = find_pivot_strikes(latest)
    for p in pivots:
        if 0.9 <= p.put_call_ratio < 1.5:
            notes.append(f"Pivot pressure: {p.strike} ratio {p.put_call_ratio:.2f}")
            flags.append("pivot_pressure")

    # Floor/ceiling pressure
    floor_pressure_pts = float(TREND_CFG.get("floor_pressure_pts", 12))
    ceiling_pressure_pts = float(TREND_CFG.get("ceiling_pressure_pts", 12))
    if latest.put_floor:
        dist_floor = latest.curr_price - latest.put_floor
        if dist_floor <= floor_pressure_pts:
            notes.append(f"Price pressing put floor ({dist_floor:.0f} pts)")
            flags.append("price_pressing_floor")
    if latest.call_wall:
        dist_ceiling = latest.call_wall - latest.curr_price
        if dist_ceiling <= ceiling_pressure_pts:
            notes.append(f"Price pressing call wall ({dist_ceiling:.0f} pts)")
            flags.append("price_pressing_ceiling")

    support_targets = _select_targets(latest.top5_puts, latest.curr_price, side="support")
    resistance_targets = _select_targets(latest.top5_calls, latest.curr_price, side="resistance")

    # ---------------------------------------------------------------
    # Gamma regime override: absolute gamma level is directional
    # +15 Bn = dealers net long gamma = buying dips = bullish structure
    # -15 Bn = dealers net short gamma = selling rallies = bearish
    # This was MISSING — the engine only tracked gamma deltas, not
    # absolute regime, causing all-day BEARISH calls on +40Bn days.
    # ---------------------------------------------------------------
    regime_gamma_bn = latest.net_gamma / 1e9
    regime_override_bn = float(TREND_CFG.get("regime_override_bn", 15.0))
    regime_override_weight = int(TREND_CFG.get("regime_override_weight", 3))

    if regime_gamma_bn >= regime_override_bn:
        bullish_score += regime_override_weight
        notes.append(f"Positive gamma regime: {regime_gamma_bn:+.1f}Bn (structural bullish)")
        flags.append("gamma_regime_bullish")
    elif regime_gamma_bn <= -regime_override_bn:
        bearish_score += regime_override_weight
        notes.append(f"Negative gamma regime: {regime_gamma_bn:+.1f}Bn (structural bearish)")
        flags.append("gamma_regime_bearish")

    # Moderate regime bonus (weaker signal but still informative)
    moderate_regime_bn = float(TREND_CFG.get("moderate_regime_bn", 8.0))
    moderate_regime_weight = int(TREND_CFG.get("moderate_regime_weight", 1))
    if moderate_regime_bn < regime_override_bn:
        if moderate_regime_bn <= regime_gamma_bn < regime_override_bn:
            bullish_score += moderate_regime_weight
            flags.append("gamma_regime_mod_bullish")
        elif -regime_override_bn < regime_gamma_bn <= -moderate_regime_bn:
            bearish_score += moderate_regime_weight
            flags.append("gamma_regime_mod_bearish")

    # ---------------------------------------------------------------
    # Price-from-open reality check: a realized move IS directional
    # evidence regardless of window scoring.  Fixes the "slow grind"
    # bug where 5/15/30/60 min thresholds miss gradual rallies.
    # ---------------------------------------------------------------
    rth_snaps = [s for s in snapshots if s.session_tag == "RTH"]
    if len(rth_snaps) >= 5:
        open_price = rth_snaps[0].curr_price
        move_from_open = latest.curr_price - open_price
        move_threshold = float(TREND_CFG.get("move_from_open_pts", 20.0))
        move_weight = int(TREND_CFG.get("move_from_open_weight", 2))
        if move_from_open >= move_threshold:
            bullish_score += move_weight
            notes.append(f"Up {move_from_open:.0f}pts from open (realized bullish)")
            flags.append("move_from_open_bullish")
        elif move_from_open <= -move_threshold:
            bearish_score += move_weight
            notes.append(f"Down {abs(move_from_open):.0f}pts from open (realized bearish)")
            flags.append("move_from_open_bearish")

    # Determine bias and smell (after structural + regime adjustments)
    score_threshold = int(TREND_CFG.get("score_threshold", 4))
    smell_threshold = int(TREND_CFG.get("smell_threshold", 2))
    bias = "NEUTRAL"
    score = 0

    if bearish_score > bullish_score and bearish_score >= smell_threshold:
        bias = "BEARISH"
        score = bearish_score
    elif bullish_score > bearish_score and bullish_score >= smell_threshold:
        bias = "BULLISH"
        score = bullish_score
    else:
        score = max(bearish_score, bullish_score)

    smell = bias != "NEUTRAL" and score < score_threshold

    return TrendDashboard(
        timestamp=latest.timestamp_pt,
        bias=bias,
        score=score,
        smell=smell,
        bearish_score=bearish_score,
        bullish_score=bullish_score,
        windows=windows,
        notes=notes,
        flags=flags,
        support_targets=support_targets,
        resistance_targets=resistance_targets,
    )


# --- Session 5: Advanced Metrics ---


def compute_gamma_concentration_index(snapshot: GEXSnapshot) -> Tuple[float, str]:
    """GCI: top1 gamma / total gamma. >40% = HARD_WALL, <25% = DIFFUSE."""
    if not snapshot.top5_calls or snapshot.total_call_gamma == 0:
        return 0.0, "DIFFUSE"

    top1_gamma = abs(snapshot.top5_calls[0][1])
    total = abs(snapshot.total_call_gamma)
    if total == 0:
        return 0.0, "DIFFUSE"

    gci = (top1_gamma / total) * 100

    if gci >= ADVANCED_CFG["gci_hard_wall_pct"]:
        label = "HARD_WALL"
    elif gci >= ADVANCED_CFG["gci_diffuse_pct"]:
        label = "MODERATE"
    else:
        label = "DIFFUSE"

    return gci, label


def compute_gamma_velocity(snapshots: List[GEXSnapshot]) -> Tuple[float, str]:
    """Gamma velocity: Bn/min rate of change."""
    if len(snapshots) < 2:
        return 0.0, "STABLE"

    dt_minutes = (
        snapshots[-1].timestamp_pt.timestamp() - snapshots[-2].timestamp_pt.timestamp()
    ) / 60
    if dt_minutes <= 0:
        return 0.0, "STABLE"

    delta_bn = (snapshots[-1].net_gamma - snapshots[-2].net_gamma) / 1e9
    velocity = delta_bn / dt_minutes

    # Compare to prior velocity if we have enough data
    if len(snapshots) >= 3:
        dt_prev = (
            snapshots[-2].timestamp_pt.timestamp() - snapshots[-3].timestamp_pt.timestamp()
        ) / 60
        if dt_prev > 0:
            prev_velocity = (snapshots[-2].net_gamma - snapshots[-3].net_gamma) / 1e9 / dt_prev
            if abs(velocity) > abs(prev_velocity) * 1.2:
                label = "ACCELERATING"
            elif abs(velocity) < abs(prev_velocity) * 0.8:
                label = "DECELERATING"
            else:
                label = "STABLE"
        else:
            label = "STABLE"
    else:
        label = "ACCELERATING" if abs(velocity) > 0.01 else "STABLE"

    return velocity, label


def compute_gamma_asymmetry(snapshots: List[GEXSnapshot]) -> Tuple[float, str]:
    """Gamma asymmetry: call/put ratio trend. >2.0 calls dominate, <1.0 puts dominate."""
    if not snapshots:
        return 1.0, "BALANCED"

    latest = snapshots[-1]
    if latest.total_put_gamma == 0:
        return 0.0, "BALANCED"

    ratio = abs(latest.total_call_gamma) / abs(latest.total_put_gamma)

    if ratio >= ADVANCED_CFG["asymmetry_call_dominant"]:
        label = "CALL_DOMINANT"
    elif ratio <= ADVANCED_CFG["asymmetry_put_dominant"]:
        label = "PUT_DOMINANT"
    else:
        label = "BALANCED"

    return ratio, label


def compute_squeeze_probability(
    snapshot: GEXSnapshot,
    velocity: float,
    gci: float,
    asymmetry: float,
) -> float:
    """Squeeze probability 0-100 based on price proximity, velocity, concentration, asymmetry."""
    if not snapshot.call_wall or not snapshot.curr_price:
        return 0.0

    w = ADVANCED_CFG

    # Price proximity to call wall (higher = closer = more squeeze risk)
    dist = abs(snapshot.curr_price - snapshot.call_wall)
    spread = snapshot.spread or 100
    proximity_score = max(0, (1 - dist / spread)) * 100

    # Velocity score (positive velocity toward wall = squeeze)
    velocity_score = min(100, abs(velocity) * 1000) if velocity > 0 else 0

    # Concentration score
    concentration_score = min(100, gci * 2)

    # Asymmetry score (high call dominance = squeeze fuel)
    asymmetry_score = min(100, max(0, (asymmetry - 1) * 50))

    probability = (
        proximity_score * w["squeeze_proximity_weight"]
        + velocity_score * w["squeeze_velocity_weight"]
        + concentration_score * w["squeeze_concentration_weight"]
        + asymmetry_score * w["squeeze_asymmetry_weight"]
    )

    return min(100, max(0, probability))


def compute_pin_probability(
    snapshot: GEXSnapshot,
    gci: float,
    minutes_to_close: float,
) -> float:
    """Pin probability 0-100 based on distance to max-gamma, net gamma strength, time, concentration."""
    if not snapshot.call_wall or not snapshot.curr_price:
        return 0.0

    w = ADVANCED_CFG

    # Find max-gamma strike (the magnet)
    all_positions = snapshot.top5_calls + [(s, abs(g)) for s, g in snapshot.top5_puts]
    if not all_positions:
        return 0.0
    max_gamma_strike = max(all_positions, key=lambda x: abs(x[1]))[0]

    # Distance score (closer to max-gamma = higher pin probability)
    dist = abs(snapshot.curr_price - max_gamma_strike)
    dist_score = max(0, (1 - dist / 50)) * 100  # 50pt scale

    # Net gamma strength (higher positive = stronger pin)
    gamma_bn = snapshot.net_gamma / 1e9
    gamma_score = min(100, max(0, gamma_bn / 8 * 100)) if gamma_bn > 0 else 0

    # Time score (closer to close = stronger pin)
    time_score = max(0, (1 - minutes_to_close / 390)) * 100  # 390 min = full RTH

    # Concentration score
    concentration_score = min(100, gci * 2)

    probability = (
        dist_score * w["pin_distance_weight"]
        + gamma_score * w["pin_gamma_weight"]
        + time_score * w["pin_time_weight"]
        + concentration_score * w["pin_concentration_weight"]
    )

    return min(100, max(0, probability))


def find_pivot_strikes(snapshot: GEXSnapshot) -> List[PivotStrike]:
    """Find strikes present in BOTH top5_calls and top5_puts (the 6970 canary).

    Classify as SUPPORT/PIVOT/FLIPPED based on put/call ratio at that strike.
    """
    if not snapshot.top5_calls or not snapshot.top5_puts:
        return []

    call_map = {strike: gamma for strike, gamma in snapshot.top5_calls}
    put_map = {strike: gamma for strike, gamma in snapshot.top5_puts}

    shared_strikes = set(call_map.keys()) & set(put_map.keys())
    pivots = []

    for strike in sorted(shared_strikes):
        call_gamma = call_map[strike]
        put_gamma = abs(put_map[strike])

        if call_gamma == 0:
            ratio = float("inf")
        else:
            ratio = put_gamma / call_gamma

        if ratio < 0.5:
            pivot_type = PivotType.SUPPORT
        elif ratio > 1.5:
            pivot_type = PivotType.FLIPPED
        else:
            pivot_type = PivotType.PIVOT

        pivots.append(PivotStrike(
            strike=strike,
            call_gamma=call_gamma,
            put_gamma=put_map[strike],
            put_call_ratio=ratio,
            pivot_type=pivot_type,
        ))

    return pivots


def compute_advanced_metrics(
    snapshots: List[GEXSnapshot],
    minutes_to_close: float = 390.0,
) -> Optional[AdvancedMetrics]:
    """Compute all Session 5 advanced metrics."""
    if not snapshots:
        return None

    latest = snapshots[-1]

    gci, gci_label = compute_gamma_concentration_index(latest)
    velocity, velocity_label = compute_gamma_velocity(snapshots)
    asymmetry, asymmetry_label = compute_gamma_asymmetry(snapshots)

    squeeze_prob = compute_squeeze_probability(latest, velocity, gci, asymmetry)
    pin_prob = compute_pin_probability(latest, gci, minutes_to_close)
    pivot_strikes = find_pivot_strikes(latest)

    return AdvancedMetrics(
        gci=gci,
        gci_label=gci_label,
        gamma_velocity=velocity,
        gamma_velocity_label=velocity_label,
        gamma_asymmetry=asymmetry,
        asymmetry_label=asymmetry_label,
        squeeze_probability=squeeze_prob,
        pin_probability=pin_prob,
        pivot_strikes=pivot_strikes,
    )
