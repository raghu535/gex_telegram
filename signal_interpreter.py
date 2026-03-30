"""Signal detection and alert formatting for 11 signal types."""

from __future__ import annotations

import json
import os
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
from output_formatter import (
    format_type1_signal,
    format_type3_market_read,
    compute_conviction_tier,
    compute_urgency,
    generate_narrative,
)
from delta_tracker import DeltaTracker
import historical_stats

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
    LOTTO = 22
    MA_SNAP = 23
    MARKET_CONSENSUS = 24
    LEVEL_APPROACH = 25
    CONVICTION_SIGNAL = 26
    MORNING_READ = 27
    MORNING_BET = 28
    EM_FADE = 29
    PIN_TRADE = 30
    GEX_TREND = 31
    QQQ_CONTEXT = 33


SIGNAL_CHANNEL_MAP = {
    SignalType.REGIME_SHIFT: "gex_context",
    SignalType.WALL_BREACH: "gex_trades",
    SignalType.GAMMA_SQUEEZE: "gex_trades",
    SignalType.BOUNCE_ZONE: "gex_engine",
    SignalType.GAMMA_COMPRESSION: "gex_engine",
    SignalType.STRIKE_FLIP: "gex_engine",
    SignalType.TREND_DASHBOARD: "gex_engine",
    SignalType.OVERNIGHT_DRIFT: "gex_context",
    SignalType.RTH_PULSE: "gex_engine",
    SignalType.FINAL_15: "gex_context",
    SignalType.MORNING_BRIEF: "gex_context",
    SignalType.HOURLY_RECAP: "gex_engine",
    SignalType.PIN_FORECAST: "gex_context",
    SignalType.DATA_LAG: "gex_engine",
    SignalType.DATA_RESTORED: "gex_engine",
    SignalType.QUIET_SUMMARY: "gex_engine",
    SignalType.SPX_MILLION: "gex_relay",
    SignalType.GAP_ALERT: "gex_context",
    SignalType.MORNING_CHECKLIST: "gex_context",
    SignalType.MICRO_PULSE: "gex_engine",
    SignalType.GAMMA_SNAP: "gex_trades",
    SignalType.LOTTO: "gex_trades",
    SignalType.MA_SNAP: "gex_trades",
    SignalType.MARKET_CONSENSUS: "gex_relay",
    SignalType.LEVEL_APPROACH: "gex_trades",
    SignalType.CONVICTION_SIGNAL: "conviction_signal",
    SignalType.MORNING_READ: "gex_context",
    SignalType.MORNING_BET: "gex_trades",
    SignalType.EM_FADE: "gex_trades",
    SignalType.PIN_TRADE: "gex_trades",
    SignalType.GEX_TREND: "gex_engine",
    SignalType.QQQ_CONTEXT: "gex_relay",
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
import math
import time as _time_mod

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as _f:
    _CFG = yaml.safe_load(_f)


def _gamma_english(gamma_bn: float) -> str:
    """Translate gamma to plain English — no raw numbers."""
    if gamma_bn < -10:
        return "extreme negative gamma"
    elif gamma_bn < -5:
        return "heavy negative gamma"
    elif gamma_bn < -2:
        return "negative gamma"
    elif gamma_bn > 8:
        return "strong positive gamma (ceiling)"
    elif gamma_bn > 3:
        return "positive gamma"
    return "neutral gamma"


def _vwap_context(direction: str, price: float, vwap: Optional[float]) -> str:
    """VWAP context line for trade alerts."""
    if not vwap:
        return ""
    if direction == "LONG":
        if price < vwap:
            return f"VWAP: {vwap:.0f} (reclaim target)"
        return f"VWAP: {vwap:.0f} (above, bullish)"
    else:
        if price > vwap:
            return f"VWAP: {vwap:.0f} (rejection zone)"
        return f"VWAP: {vwap:.0f} (below, bearish)"


# Structure lines per signal type — plain English
_STRUCTURE_TEMPLATES = {
    "L1": "Swept 1hr low into {gamma_desc}",
    "T1": "Fell {dP:.0f}pts in 20min — dealers forced to buy on dips",
    "T2": "Fell {dP:.0f}pts into {gamma_desc} — bounce likely",
    "T3": "Deep {gamma_desc} stretch — rubber band loaded",
    "T4": "Extreme gamma zone — speculative bounce only",
    "S1": "Price up but gamma dropping — fake rally",
    "S2": "Rally into gamma ceiling — 100% historical reversal",
    "S3": "Swept 1hr high into {gamma_desc}",
    "S4": "Rally in negative gamma — dealers sell into it",
    "S5": "Wall spread collapsed — market lost guardrails",
    "S6": "Gamma accelerating negative — dealer feedback loop",
    "B1": "Strong opening momentum — tends to persist all day",
    "BEAR": "Early selloff in mild negative gamma — sellers in control",
    "EM": "Price at expected move boundary — reversal zone",
}


def _confluence_bar(active: int, total: int) -> str:
    """Build a visual confluence bar: e.g. 3/4 -> '\u2588\u2588\u2588\u2591'."""
    filled = "\u2588" * active
    empty = "\u2591" * (total - active)
    return f"{filled}{empty}"


def _format_trade_alert(
    direction: str,
    signal_name: str,
    price: float,
    entry: float,
    stop: float,
    target: float,
    win_pct: int,
    gamma_bn: float,
    structure_line: str,
    confluence: dict,
    ma_values: Optional[Dict] = None,
    pf: float = 0,
    cw: float = 0,
    em_levels: Optional[dict] = None,
    late_session: bool = False,
) -> str:
    """Build a Type 1 trade alert — box-drawing format."""
    emoji = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = reward / risk if risk > 0 else 0

    now_pt = datetime.now(PT_TZ)
    time_str = now_pt.strftime("%I:%M %p").lstrip("0")

    lines = [
        "\u2501\u2501\u2501 GEX SIGNAL \u2501\u2501\u2501",
        f"{emoji} {direction} \u2014 {signal_name}",
        f"SPX {price:.0f} | {time_str}",
        "",
        f"Structure: {structure_line}",
    ]

    # VWAP line
    vwap = ma_values.get("vwap") if ma_values else None
    vwap_line = _vwap_context(direction, price, vwap)
    if vwap_line:
        lines.append(vwap_line)

    # SMA line
    sma_5m20 = ma_values.get("5m_20") if ma_values else None
    sma_30m20 = ma_values.get("30m_20") if ma_values else None
    if sma_5m20 or sma_30m20:
        parts = []
        if sma_5m20:
            parts.append(f"5m-20 {sma_5m20:.0f}")
        if sma_30m20:
            parts.append(f"30m-20 {sma_30m20:.0f}")
        lines.append(f"SMA: {' | '.join(parts)}")

    # Levels line — combine GEX walls + EM
    support = pf if pf else 0
    resistance = cw if cw else 0
    if em_levels:
        em_lower = em_levels.get("lower", 0)
        em_upper = em_levels.get("upper", 0)
        if em_lower and support:
            support = max(support, em_lower)
        elif em_lower:
            support = em_lower
        if em_upper and resistance:
            resistance = min(resistance, em_upper)
        elif em_upper:
            resistance = em_upper
    if support or resistance:
        parts = []
        if support:
            parts.append(f"Support {support:.0f}")
        if resistance:
            parts.append(f"Resistance {resistance:.0f}")
        lines.append(f"Levels: {' | '.join(parts)}")

    # Entry/Stop/Target
    lines.append("")
    lines.append(f"Entry: {entry:.0f} | Stop: {stop:.0f} | Target: {target:.0f}")
    total = len(confluence)
    active = sum(1 for v in confluence.values() if v)
    bar = _confluence_bar(active, total)
    lines.append(f"R:R: {rr:.1f}:1 | Confluence: {bar} {active}/{total}")

    # 0DTE play via strike_picker
    try:
        from strike_picker import get_0dte_play
        opt_dir = "CALL" if direction == "LONG" else "PUT"
        play = get_0dte_play(opt_dir, price, target, stop)
        lines.append("")
        bid_str = f"${play['bid']:.2f}" if play['bid'] else "?"
        ask_str = f"${play['ask']:.2f}" if play['ask'] else "?"
        lines.append(f"Play: 0DTE {play['symbol']} @ {bid_str}-{ask_str}")
        if play.get("target_premium"):
            target_desc = "VWAP reclaim" if direction == "LONG" and vwap and target <= vwap + 5 else f"target {target:.0f}"
            lines.append(f"      Target: {play['target_premium']} at {target_desc}")
        lines.append(f"      Stop: {play['stop_note']}")
    except Exception:
        # Fallback: simple estimate
        if direction == "LONG":
            strike = math.ceil(price / 5) * 5
            option = f"SPXW {strike}C"
        else:
            strike = math.floor(price / 5) * 5
            option = f"SPXW {strike}P"
        lines.append("")
        lines.append(f"Play: 0DTE {option}")

    if late_session:
        lines.append("\u26a0 LATE SESSION \u2014 0DTE amplified")

    lines.append("\u2501" * 18)
    return "\n".join(lines)

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
        # B1 Early Rally / BEAR Early Selloff state
        self._b1_fired_today: bool = False
        self._b1_state_date: Optional[str] = None
        self._rth_snap_count: int = 0  # count RTH snapshots to gate first 30min
        # Butterfly pin alert state
        self._butterfly_fired_today: bool = False
        # Lotto short (late-session pin fade) state
        self._lotto_fired_today: bool = False
        # MA Snap state — tracks which MAs have fired today
        self._ma_snap_fired_today: set = set()  # e.g. {"60m_20", "30m_200"}
        # Wall collapse state — fire once per collapse event
        self._wall_collapse_fired: bool = False
        # Level approach — track which levels already alerted
        self._level_alerted: dict = {}  # level_name -> timestamp
        self._ma_snap_state_date: Optional[str] = None
        # Sweep watch state machine (L1/S3 reclaim logic)
        self._sweep_watch: Optional[dict] = None
        # Recent signals for stacking check
        self._recent_signals: list = []  # [(side, signal_name, datetime)]
        # Delta tracker for "Δ Since last" lines
        self._delta_tracker = DeltaTracker()
        # Restore persisted state from disk (survives restarts)
        self._load_interpreter_state()

    _INTERPRETER_STATE_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "interpreter_state.json"
    )

    def _save_interpreter_state(self):
        """Persist key state so engine restarts don't re-fire first-of-day alerts."""
        today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
        state = {
            "date": today,
            "confirmed_regime": self._confirmed_regime.value if self._confirmed_regime else None,
            "last_trend_bias": self._last_trend_bias,
            "last_trend_change_ts": self._last_trend_change_ts,
            "last_trend_emit_ts": self._last_trend_emit_ts,
            "trend_heartbeat_sent": self._trend_heartbeat_sent,
            "rth_snap_count": self._rth_snap_count,
        }
        try:
            with open(self._INTERPRETER_STATE_FILE, "w") as f:
                json.dump(state, f)
        except Exception:
            pass

    def _load_interpreter_state(self):
        """Load persisted state if it's from today."""
        try:
            with open(self._INTERPRETER_STATE_FILE, "r") as f:
                state = json.load(f)
            today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
            if state.get("date") != today:
                return
            # Restore regime
            regime_val = state.get("confirmed_regime")
            if regime_val:
                try:
                    self._confirmed_regime = Regime(regime_val)
                    self._regime_candidate = self._confirmed_regime
                    self._regime_candidate_count = 0
                except ValueError:
                    pass
            # Restore trend state
            if state.get("last_trend_bias"):
                self._last_trend_bias = state["last_trend_bias"]
                self._last_trend_change_ts = state.get("last_trend_change_ts", 0.0)
                self._last_trend_emit_ts = state.get("last_trend_emit_ts", 0.0)
                self._trend_heartbeat_sent = state.get("trend_heartbeat_sent", False)
            self._rth_snap_count = state.get("rth_snap_count", 0)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def evaluate_all(
        self,
        snapshot: GEXSnapshot,
        fifteen_min: Optional[FifteenMinMetrics],
        one_hour: Optional[OneHourMetrics],
        overnight: Optional[OvernightDrift],
        advanced: Optional[AdvancedMetrics],
        trend: Optional[TrendDashboard] = None,
        snapshots_1h: Optional[List[GEXSnapshot]] = None,
        open_price: Optional[float] = None,
        daily_ma20: Optional[float] = None,
        ma_values: Optional[Dict[str, float]] = None,
    ) -> List[Signal]:
        """Check all signal conditions and return triggered signals, sorted by priority."""
        # Store ma_values on instance so _emit_gamma_snap_signal can access it
        self._last_ma_values = ma_values
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
        # Reset B1, butterfly, lotto, and MA snap state on new day
        if self._b1_state_date != snap_date:
            self._b1_state_date = snap_date
            self._b1_fired_today = False
            self._butterfly_fired_today = False
            self._lotto_fired_today = False
            self._rth_snap_count = 0
        if self._ma_snap_state_date != snap_date:
            self._ma_snap_state_date = snap_date
            self._ma_snap_fired_today = set()
        self._rth_snap_count += 1
        if self._rth_snap_count % 10 == 0:
            self._save_interpreter_state()

        regime = one_hour.regime if one_hour else Regime.CONTROLLED_TREND

        # 1. REGIME_SHIFT
        sig = self._check_regime_shift(regime, one_hour, snapshot.timestamp_pt, snapshot)
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

        # 10. B1 EARLY RALLY / BEAR EARLY SELLOFF (first 30min momentum)
        sig = self._check_early_momentum(snapshot, open_price)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 11. BUTTERFLY PIN (gamma > +10Bn = price pins to call wall, butterfly trade)
        sig = self._check_butterfly_pin(snapshot, snapshots_1h)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 12. LOTTO SHORT (late-session pin fade — combined CW proximity + MA extension)
        sig = self._check_lotto(snapshot, snapshots_1h, daily_ma20)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 13. MA SNAP (MA touch + gamma regime = rubber band or dead cat fade)
        sig = self._check_ma_snap(snapshot, ma_values)
        if sig:
            sig.timestamp = snapshot.timestamp_pt
            signals.append(sig)

        # 14. LEVEL APPROACH (price approaching GEX + EM convergence zones)
        sig = self._check_level_approach(snapshot, snapshots_1h)
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

        msg = _format_rth_pulse(snapshot, one_hour, advanced, levels, self._delta_tracker)
        return Signal(
            signal_type=SignalType.RTH_PULSE,
            title="RTH Pulse",
            message=msg,
            channel="gex_engine",
            priority=SignalType.RTH_PULSE,
            timestamp=snapshot.timestamp_pt,
        )

    def generate_gex_trend(
        self,
        snapshot: GEXSnapshot,
        fifteen_min: Optional[FifteenMinMetrics],
        one_hour: Optional[OneHourMetrics],
        overnight: Optional[OvernightDrift],
    ) -> Optional[Signal]:
        """Generate GEX TREND summary: short/medium/long term with real numbers."""
        if not fifteen_min:
            return None

        em_levels = self.compute_expected_move()
        price = snapshot.curr_price
        gamma_bn = fifteen_min.net_gamma / 1e9 if fifteen_min.net_gamma else 0

        # Build dicts for the formatter
        snap_data = {
            "price": price,
            "gamma_bn": gamma_bn,
            "call_wall": snapshot.call_wall or 0,
            "put_floor": snapshot.put_floor or 0,
            "spread": snapshot.spread or 0,
        }

        fm_15 = {
            "price_delta": fifteen_min.price_delta,
            "gamma_bn": gamma_bn,
            "gamma_delta_bn": fifteen_min.net_gamma_delta / 1e9 if fifteen_min.net_gamma_delta else 0,
            "dist_floor": fifteen_min.distance_to_put_floor,
            "dist_ceil": fifteen_min.distance_to_call_wall,
        }

        fm_1h = None
        if one_hour:
            fm_1h = {
                "price_range": one_hour.price_range,
                "slope": one_hour.net_gamma_slope,
                "regime": one_hour.regime.value,
                "confidence": one_hour.regime_confidence,
            }

        # Delta tracking
        regime_val = self._confirmed_regime.value if self._confirmed_regime else ""
        delta_line = self._delta_tracker.format_delta_line("gex_engine", "gex_trend", {
            "price": price, "gamma_bn": gamma_bn,
            "call_wall": snapshot.call_wall or 0,
            "put_floor": snapshot.put_floor or 0,
            "spread": snapshot.spread or 0,
            "regime": regime_val,
        })

        msg = format_type3_market_read(
            read_type="GEX_TREND",
            snapshot_data=snap_data,
            delta_line=delta_line,
            fifteen_min=fm_15,
            one_hour=fm_1h,
            em_levels=em_levels,
        )

        return Signal(
            signal_type=SignalType.GEX_TREND,
            title="GEX Trend",
            message=msg,
            channel="gex_engine",
            priority=SignalType.GEX_TREND,
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
            room = (snapshot.call_wall - snapshot.curr_price) if snapshot.call_wall else 0.0
            # Enhanced targets: when room >= 30 pts and big move, use wall-level targets
            if room >= 30 and tier in ("POSSIBLE EXTREME", "CRAZY EXTREME") and snapshot.call_wall:
                t1 = snapshot.curr_price + (room / 2)
                t2 = float(snapshot.call_wall)
            else:
                t1 = trigger + target1_pts
                t2 = trigger + target2_pts
            room_line = f"Room to wall: {room:+.1f} pts"
        elif direction == "SHORT":
            gas_state = "ACCELERATING" if (move_1m < 0 and gamma_delta_5m_bn < 0) else "LOSING STEAM"
            trigger = recent_low - breakout_buffer_pts
            invalidation = recent_high + invalidation_buffer_pts
            room = (snapshot.curr_price - snapshot.put_floor) if snapshot.put_floor else 0.0
            # Enhanced targets: when room >= 30 pts and big move, use floor-level targets
            if room >= 30 and tier in ("POSSIBLE EXTREME", "CRAZY EXTREME") and snapshot.put_floor:
                t1 = snapshot.curr_price - (room / 2)
                t2 = float(snapshot.put_floor)
            else:
                t1 = trigger - target1_pts
                t2 = trigger - target2_pts
            room_line = f"Room to floor: {room:+.1f} pts"
        else:
            gas_state = "STABLE"
            trigger = snapshot.curr_price
            invalidation = snapshot.curr_price
            t1 = snapshot.curr_price
            t2 = snapshot.curr_price
            room_line = "Room: n/a"

        # Gas state to plain English
        _gas_english = {
            "ACCELERATING": "Momentum building",
            "LOSING STEAM": "Momentum fading",
            "STABLE": "Holding steady",
        }
        gas_plain = _gas_english.get(gas_state, gas_state)

        # Tier to urgency
        _tier_english = {
            "CRAZY EXTREME": "Extreme move — high conviction",
            "POSSIBLE EXTREME": "Big move in progress",
            "MILD EXTREME": "Notable move developing",
            "MICRO MOVE": "Small move",
        }
        tier_plain = _tier_english.get(tier, tier)

        gamma_bn = snapshot.net_gamma / 1e9
        gd = _gamma_english(gamma_bn)
        emoji = "\U0001f7e2" if direction == "LONG" else "\U0001f534"

        lines = [
            f"{emoji} SCALP {direction} | {snapshot.timestamp_pt.strftime('%I:%M %p PT')}",
            f"SPX {snapshot.curr_price:.0f} | {tier_plain}",
            f"{gas_plain} | {gd}",
            "",
            f"Entry: {trigger:.0f} | Stop: {invalidation:.0f} | T1: {t1:.0f} | T2: {t2:.0f}",
            room_line,
        ]

        if fifteen_min:
            confirm_dir = "confirms" if (
                (direction == "LONG" and fifteen_min.price_delta > 0) or
                (direction == "SHORT" and fifteen_min.price_delta < 0)
            ) else "diverges"
            lines.append(f"15m trend {confirm_dir}: dPx {fifteen_min.price_delta:+.1f}")

        return Signal(
            signal_type=SignalType.MICRO_PULSE,
            title=f"Micro Pulse - {direction}",
            message="\n".join(lines),
            channel="gex_engine",
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
        msg = _format_final_15(snapshot, advanced, self._delta_tracker)
        return Signal(
            signal_type=SignalType.FINAL_15,
            title="Final 15 Minutes",
            message=msg,
            channel="gex_context",
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
        msg = _format_quiet_summary(snapshot, fifteen_min, one_hour, advanced, self._delta_tracker)
        return Signal(
            signal_type=SignalType.QUIET_SUMMARY,
            title="Trading Summary",
            message=msg,
            channel="gex_engine",
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
        msg = _format_morning_brief(snapshot, overnight, advanced, self._delta_tracker)
        return Signal(
            signal_type=SignalType.MORNING_BRIEF,
            title="Morning Brief",
            message=msg,
            channel="gex_context",
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
        snap_time: Optional[datetime] = None, snapshot: Optional[GEXSnapshot] = None,
    ) -> Optional[Signal]:
        MIN_REGIME_SECS = 900   # 15 min minimum for major shifts (2+ levels apart)
        ADJ_REGIME_SECS = 1800  # 30 min minimum for adjacent regime flips

        if self._confirmed_regime is None:
            self._confirmed_regime = current
            self._regime_candidate = current
            self._regime_candidate_count = REGIME_HOLD_COUNT
            self._regime_confirmed_at = snap_time
            self._save_interpreter_state()
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
            self._save_interpreter_state()

            confidence = one_hour.regime_confidence if one_hour else 0
            slope = one_hour.net_gamma_slope if one_hour else 0

            # Type 3: Market Commentary format
            if snapshot:
                msg = _format_market_read(
                    old_regime.value, current.value, snapshot, confidence, slope, self._delta_tracker,
                )
            else:
                # Fallback if no snapshot available
                old_plain = _REGIME_PLAIN.get(old_regime.value, old_regime.value)
                new_plain = _REGIME_PLAIN.get(current.value, current.value)
                msg = f"\U0001f4ca MARKET READ\n\nFlipped from {old_plain} to {new_plain}"

            return Signal(
                signal_type=SignalType.REGIME_SHIFT,
                title=f"Market Read — Regime Shift",
                message=msg,
                channel="gex_context",
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

        gamma_bn = snapshot.net_gamma / 1e9
        if breached_call:
            dist = price - snapshot.call_wall
            gd = _gamma_english(gamma_bn)
            if gamma_bn > 5:
                direction, bias = "SHORT", "FADE SHORT"
                stop = price + 15
                target = snapshot.call_wall - 10
                structure = f"Price {dist:.0f}pts above CW {snapshot.call_wall}. {gd.capitalize()} — dealers sell into this."
            elif gamma_bn < -3:
                direction, bias = "LONG", "BREAKOUT LONG"
                stop = snapshot.call_wall - 5
                target = price + 25
                structure = f"Price broke above CW {snapshot.call_wall} by {dist:.0f}pts. {gd.capitalize()} — real breakout."
            else:
                direction, bias = "LONG", "NEUTRAL"
                stop = snapshot.call_wall - 5
                target = price + 15
                structure = f"Price {dist:.0f}pts above CW {snapshot.call_wall}. {gd.capitalize()} — watch for direction."
        else:
            dist = snapshot.put_floor - price
            gd = _gamma_english(gamma_bn)
            if gamma_bn < -5:
                direction, bias = "LONG", "BOUNCE LONG"
                stop = price - 10
                target = snapshot.put_floor + 15
                structure = f"Price {dist:.0f}pts below PF {snapshot.put_floor}. {gd.capitalize()} — dealers amplify dip, bounce likely."
            elif gamma_bn > 3:
                direction, bias = "SHORT", "BREAKDOWN SHORT"
                stop = snapshot.put_floor + 5
                target = price - 25
                structure = f"Price broke below PF {snapshot.put_floor}. {gd.capitalize()} — real breakdown."
            else:
                direction, bias = "SHORT", "NEUTRAL"
                stop = snapshot.put_floor + 5
                target = price - 15
                structure = f"Price {dist:.0f}pts below PF {snapshot.put_floor}. {gd.capitalize()} — watch for direction."

        # Delta tracking
        delta_line = self._delta_tracker.format_delta_line("gex_trades", "wall_breach", {
            "price": price, "gamma_bn": gamma_bn,
            "call_wall": snapshot.call_wall, "put_floor": snapshot.put_floor,
            "spread": snapshot.spread,
        })

        confluence = {"Wall Breach": True, "Gamma Regime": (gamma_bn < -3 if direction == "LONG" else gamma_bn > 3)}
        msg = format_type1_signal(
            direction=direction,
            signal_name=f"WALL BREACH [{bias}]",
            price=price,
            entry=price,
            stop=stop,
            target=target,
            win_pct=38 if bias == "FADE SHORT" else 55,
            gamma_bn=gamma_bn,
            structure_line=structure,
            confluence=confluence,
            delta_line=delta_line,
            pf=snapshot.put_floor,
            cw=snapshot.call_wall,
        )

        return Signal(
            signal_type=SignalType.WALL_BREACH,
            title="Wall Breach",
            message=msg,
            channel="gex_trades",
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

        gamma_bn = snapshot.net_gamma / 1e9
        gd = _gamma_english(gamma_bn)
        if gamma_bn > 10:
            bias = "CAUTION \u2014 FADE ZONE"
            note = f"{gd.capitalize()} \u2014 100% historical reversal from this level. Squeeze may be the top."
        else:
            bias = "MOMENTUM"
            note = f"{gd.capitalize()} \u2014 dealer hedging amplifies upside."
        msg = (
            f"GAMMA SQUEEZE [{bias}]\n"
            f"Price {price:.0f} above call wall {snapshot.call_wall}\n"
            f"Squeeze probability: {squeeze_prob:.0f}%\n"
            f"{note}\n"
        )

        return Signal(
            signal_type=SignalType.GAMMA_SQUEEZE,
            title="Gamma Squeeze",
            message=msg,
            channel="gex_trades",
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

        gamma_bn = snapshot.net_gamma / 1e9
        gd = _gamma_english(gamma_bn)
        regime_plain = _REGIME_PLAIN.get(regime.value, regime.value.lower().replace("_", " "))

        # Direction considers gamma regime, not just wall proximity.
        # With strongly positive gamma (>15Bn), dealers are net long gamma
        # and buying dips — approaching call wall = breakout likely, not bounce.
        # With strongly negative gamma (<-15Bn), approaching put floor =
        # breakdown likely, not bounce.
        _bz_gamma_flip_bn = 15.0
        if near_call:
            direction = "LONG" if gamma_bn >= _bz_gamma_flip_bn else "SHORT"
        else:
            direction = "SHORT" if gamma_bn <= -_bz_gamma_flip_bn else "LONG"
        now_pt = datetime.now(PT_TZ)
        time_str = now_pt.strftime("%I:%M %p").lstrip("0")

        if near_put:
            structure = f"Approaching put floor support at {wall_val:.0f} in {gd}"
        else:
            structure = f"Approaching call wall resistance at {wall_val:.0f} in {gd}"

        # Delta tracking
        delta_line = self._delta_tracker.format_delta_line("gex_engine", "bounce_zone", {
            "price": price, "gamma_bn": gamma_bn,
            "call_wall": snapshot.call_wall, "put_floor": snapshot.put_floor,
            "spread": snapshot.spread,
        })

        # Build structured message with conviction + urgency
        from output_formatter import compute_conviction_tier, _conviction_dots, compute_urgency, _box, _gamma_english as _ge

        urgency_label = compute_urgency(dist, 0, 390, False)
        conviction = compute_conviction_tier(55, 2 if dist < 10 else 1, 3)
        dots = _conviction_dots(conviction)

        lines = [
            f"{dots} {urgency_label} | {'🟢' if direction == 'LONG' else '🔴'} {direction} — BOUNCE ZONE",
            f"SPX {price:.0f} | {time_str}",
            "",
            f"{structure}",
            f"Regime: {regime_plain} | {urgency_note}",
        ]

        if regime == Regime.FRAGILE_CONTROL:
            scenario_lines = []
            scenario_lines.append("Scenario A — Wall holds:")
            if near_call and "short" in levels:
                sl = levels["short"]
                scenario_lines.append(f"  SHORT: Entry {sl.entry_low:.0f}-{sl.entry_high:.0f} | Stop {sl.stop:.0f} | Target {sl.target:.0f}")
            elif near_put and "long" in levels:
                ll = levels["long"]
                scenario_lines.append(f"  LONG: Entry {ll.entry_low:.0f}-{ll.entry_high:.0f} | Stop {ll.stop:.0f} | Target {ll.target:.0f}")
            scenario_lines.append("Scenario B — Wall fails:")
            if near_call:
                scenario_lines.append(f"  Price sustains above {wall_val:.0f} = squeeze in play")
            else:
                scenario_lines.append(f"  Price breaks below {wall_val:.0f} = accelerated downside")
            lines.append("")
            lines.append(_box("SCENARIOS", scenario_lines))
        else:
            if near_call and "short" in levels and levels["short"].valid:
                sl = levels["short"]
                lines.append("")
                lines.append(f"Entry: {sl.entry_low:.0f}-{sl.entry_high:.0f} | Stop: {sl.stop:.0f} | Target: {sl.target:.0f}")
                lines.append(f"R:R: {sl.rr_ratio:.1f}:1")
            elif near_put and "long" in levels and levels["long"].valid:
                ll = levels["long"]
                lines.append("")
                lines.append(f"Entry: {ll.entry_low:.0f}-{ll.entry_high:.0f} | Stop: {ll.stop:.0f} | Target: {ll.target:.0f}")
                lines.append(f"R:R: {ll.rr_ratio:.1f}:1")

        lines.append("")
        lines.append(delta_line)

        msg = "\n".join(lines)

        return Signal(
            signal_type=SignalType.BOUNCE_ZONE,
            title=f"Bounce Zone — {wall_name}",
            message=msg,
            channel="gex_engine",
            priority=SignalType.BOUNCE_ZONE,
            metadata={"urgency": urgency, "distance_pts": dist, "gamma_bn": snapshot.net_gamma / 1e9},
        )

    def _compute_confluence(self, side: str, price: float, gamma_bn: float,
                            pf: float, cw: float, ma_values: Optional[dict],
                            em_levels: Optional[dict]) -> dict:
        """5 standardized confluence checks — display only, does not gate signals."""
        checks = {}
        # 1. Gamma Regime
        if side == "LONG":
            checks["Gamma Regime"] = gamma_bn < -3
        else:
            checks["Gamma Regime"] = gamma_bn > 3

        # 2. Wall Proximity
        if side == "LONG":
            checks["Wall Support"] = bool(pf and abs(price - pf) < 15)
        else:
            checks["Wall Resistance"] = bool(cw and abs(price - cw) < 15)

        # 3. VWAP Alignment
        vwap = ma_values.get("vwap") if ma_values else None
        if side == "LONG" and vwap:
            checks["VWAP Alignment"] = price < vwap
        elif side == "SHORT" and vwap:
            checks["VWAP Alignment"] = price > vwap
        else:
            checks["VWAP Alignment"] = False

        # 4. Signal Stacking
        checks["Signal Stacking"] = self._has_recent_signal(side, minutes=5)

        # 5. EM Position
        if em_levels:
            em_lower = em_levels.get("lower", 0)
            em_upper = em_levels.get("upper", 0)
            if side == "LONG" and em_lower:
                checks["EM Position"] = price <= em_lower + 10
            elif side == "SHORT" and em_upper:
                checks["EM Position"] = price >= em_upper - 10
            else:
                checks["EM Position"] = False
        else:
            checks["EM Position"] = False

        return checks

    def _has_recent_signal(self, side: str, minutes: int = 5) -> bool:
        """Check if another signal fired for same direction within N minutes."""
        now = datetime.now(PT_TZ)
        cutoff = (now - __import__('datetime').timedelta(minutes=minutes))
        for s_side, s_name, s_time in self._recent_signals:
            if s_side == side and s_time >= cutoff:
                return True
        return False

    def _record_signal(self, side: str, signal_name: str):
        """Record a signal for stacking check. Prune old entries."""
        now = datetime.now(PT_TZ)
        self._recent_signals.append((side, signal_name, now))
        # Keep only last 30 minutes
        cutoff = now - __import__('datetime').timedelta(minutes=30)
        self._recent_signals = [(s, n, t) for s, n, t in self._recent_signals if t >= cutoff]

    def _emit_gamma_snap_signal(
        self, snapshot, snapshots_1h, side, tier, tier_label,
        win_pct, rr, gamma_bn, price, dP_20m, dG_20m, dP_15m, dG_15m,
        reclaim_level=None,
    ) -> Signal:
        """Build and return a GAMMA_SNAP signal with Type 1 formatting."""
        pf = snapshot.put_floor or 0
        cw = snapshot.call_wall or 0
        now_pt = datetime.now(PT_TZ)
        late_session = now_pt.hour >= 11 or (now_pt.hour == 10 and now_pt.minute >= 30)

        # Get EM levels (self-contained first, fallback to API)
        em_levels = self.compute_expected_move()

        # Get ma_values from evaluate_all flow — stored on instance during evaluate
        ma_values = getattr(self, '_last_ma_values', None)

        # Build structure line
        gamma_desc = _gamma_english(gamma_bn)
        template = _STRUCTURE_TEMPLATES.get(tier_label, "Gamma signal detected")
        try:
            structure_line = template.format(
                gamma_desc=gamma_desc, dP=abs(dP_20m), gamma_bn=gamma_bn,
            )
        except (KeyError, IndexError):
            structure_line = template

        if reclaim_level is not None:
            structure_line = f"Swept {'1hr low' if side == 'LONG' else '1hr high'}, reclaimed {reclaim_level:.0f} — confirmed"

        # Define signal_name per tier (no tier codes!)
        SIGNAL_NAMES = {
            "L1": "LIQUIDITY SWEEP", "T1": "RUBBER BAND", "T2": "GAMMA SNAP",
            "T3": "DEEP GAMMA", "T4": "EXTREME GAMMA", "S1": "FAKE RALLY",
            "S2": "GAMMA CEILING", "S3": "LIQUIDITY SWEEP", "S4": "NEG GAMMA FADE",
            "S5": "WALL COLLAPSE", "S6": "GAMMA ACCEL",
            "B1": "EARLY RALLY", "BEAR": "EARLY SELLOFF",
        }
        signal_name = SIGNAL_NAMES.get(tier_label, "GAMMA SNAP")

        # Define entry/stop/target
        TRADE_PARAMS = {
            "L1": (price, price - 8, price + 20),
            "T1": (price, price - 10, price + 25),
            "T2": (price, price - 10, price + 15),
            "T3": (price, price - 10, price + 15),
            "T4": (price, price - 10, price + 15),
            "S1": (price, price + 10, price - 10),
            "S2": (price, price + 15, price - 28),
            "S3": (price, price + 10, price - 10),
            "S4": (price, price + 12, price - 20),
            "S5": (price, price + 15, price - 20),
            "S6": (price, price + 12, price - 15),
        }
        entry, stop, target = TRADE_PARAMS.get(tier_label, (price, price - 10, price + 15))

        # Compute 5-check confluence
        confluence = self._compute_confluence(side, price, gamma_bn, pf, cw, ma_values, em_levels)

        # Record for stacking
        self._record_signal(side, signal_name)

        # Delta tracking
        regime_val = self._confirmed_regime.value if self._confirmed_regime else ""
        delta_line = self._delta_tracker.format_delta_line("gex_trades", "gamma_snap", {
            "price": price, "gamma_bn": gamma_bn,
            "call_wall": cw, "put_floor": pf,
            "spread": cw - pf if cw and pf else 0,
            "regime": regime_val,
        })

        # Historical stats
        r_stats = historical_stats.regime_stats(regime_val) if regime_val else None
        hist_note = ""
        if r_stats and r_stats.get("sample_n", 0) > 10:
            hist_note = (
                f"Regime avg duration: {r_stats['avg_duration_snapshots']:.0f} snapshots, "
                f"avg range: {r_stats['avg_range_pts']:.0f}pts (n={r_stats['sample_n']})"
            )

        msg = format_type1_signal(
            direction=side,
            signal_name=signal_name,
            price=price,
            entry=entry,
            stop=stop,
            target=target,
            win_pct=win_pct,
            gamma_bn=gamma_bn,
            structure_line=structure_line,
            confluence=confluence,
            delta_line=delta_line,
            ma_values=ma_values,
            pf=pf,
            cw=cw,
            em_levels=em_levels,
            late_session=late_session,
            sample_n=0,
            regime=regime_val,
            historical_note=hist_note,
        )

        return Signal(
            signal_type=SignalType.GAMMA_SNAP,
            title=f"Gamma Snap \u2014 {side} {signal_name}",
            message=msg,
            channel="gex_trades",
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

        # --- Sweep watch reclaim check (L1/S3 state machine) ---
        if self._sweep_watch:
            watch = self._sweep_watch
            elapsed = (snapshot.timestamp_pt - watch["ts"]).total_seconds()
            if elapsed > 600:  # 10-min timeout — real breakout, cancel
                self._sweep_watch = None
            elif watch["side"] == "LONG" and price > watch["level"] + 4:
                # Reclaimed! Emit the signal now
                self._sweep_watch = None
                return self._emit_gamma_snap_signal(
                    snapshot, snapshots_1h, watch["side"], 5, watch["tier_label"],
                    watch["win_pct"], watch.get("rr", 3.8), gamma_bn, price,
                    watch.get("dP_20m", 0), watch.get("dG_20m", 0),
                    watch.get("dP_15m", 0), watch.get("dG_15m", 0),
                    reclaim_level=watch["level"],
                )
            elif watch["side"] == "SHORT" and price < watch["level"] - 4:
                self._sweep_watch = None
                return self._emit_gamma_snap_signal(
                    snapshot, snapshots_1h, watch["side"], 3, watch["tier_label"],
                    watch["win_pct"], watch.get("rr", 0), gamma_bn, price,
                    watch.get("dP_20m", 0), watch.get("dG_20m", 0),
                    watch.get("dP_15m", 0), watch.get("dG_15m", 0),
                    reclaim_level=watch["level"],
                )

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

        # --- SHORT S1: fake rally (price up but gamma falling) ---
        if side is None and lookback_15m:
            if dP_15m >= short_px_rise and dG_15m <= short_dg_fall:
                side, tier, tier_label = "SHORT", 1, "S1"
                win_pct, rr = 76, 0.0

        # --- SHORT S2: gamma ceiling (rally into strong positive gamma) ---
        # Backtest: price up >15pts/20m + G>+10Bn = 100% reversal, -27.8 avg
        short_s2_px_rise = float(cfg.get("short_s2_px_rise_20m", 15))
        short_s2_gamma = float(cfg.get("short_s2_gamma_bn", 10))
        if side is None:
            if dP_20m >= short_s2_px_rise and gamma_bn >= short_s2_gamma:
                side, tier, tier_label = "SHORT", 2, "S2"
                win_pct, rr = 100, 0.0

        # --- LIQUIDITY SWEEPS: price breaks 1hr high/low then reverses ---
        # Uses snapshots_1h to find session high/low over last ~30 snaps
        # LOW SWEEP LONG: new low + G<-5Bn = 67% win, +23.3 avg (74%/+27.8 if shallow)
        # HIGH SWEEP SHORT: new high + G>+5Bn = 68% win, +9.0 avg
        if side is None and snapshots_1h and len(snapshots_1h) >= 15:
            sweep_gamma = float(cfg.get("sweep_gamma_bn", 5))
            sweep_lookback = min(30, len(snapshots_1h) - 1)
            recent = snapshots_1h[-sweep_lookback:-3]  # exclude last 3 (~5min)
            if recent:
                range_high = max(s.curr_price for s in recent)
                range_low = min(s.curr_price for s in recent)

                # LOW SWEEP: price just broke below range low → set sweep watch
                if price < range_low and 0 < range_low - price < 8 and gamma_bn <= -sweep_gamma:
                    if lookback_15m and lookback_15m.curr_price > range_low:
                        self._sweep_watch = {
                            "level": range_low, "side": "LONG", "ts": snapshot.timestamp_pt,
                            "tier_label": "L1", "win_pct": 67, "rr": 3.8,
                            "gamma_bn": gamma_bn, "price_at_sweep": price,
                            "dP_20m": dP_20m, "dG_20m": dG_20m,
                            "dP_15m": dP_15m, "dG_15m": dG_15m,
                        }
                        # Don't emit yet — wait for reclaim

                # HIGH SWEEP: price just broke above range high → set sweep watch
                if side is None and price > range_high and 0 < price - range_high < 8 and gamma_bn >= sweep_gamma:
                    if lookback_15m and lookback_15m.curr_price < range_high:
                        self._sweep_watch = {
                            "level": range_high, "side": "SHORT", "ts": snapshot.timestamp_pt,
                            "tier_label": "S3", "win_pct": 68, "rr": 0.0,
                            "gamma_bn": gamma_bn, "price_at_sweep": price,
                            "dP_20m": dP_20m, "dG_20m": dG_20m,
                            "dP_15m": dP_15m, "dG_15m": dG_15m,
                        }

        # --- S4: NEG GAMMA RALLY FADE --- (DISABLED: audit showed unreliable)
        if side is None and cfg.get("s4_enabled", True) and snapshots_1h and len(snapshots_1h) >= 10:
            s4_rally_pts = float(cfg.get("s4_rally_pts", 15))
            s4_gamma_bn = float(cfg.get("s4_gamma_bn", -2))
            # Gate: skip first ~15 snapshots of RTH (~30 min) — opening noise
            if self._rth_snap_count > 15:
                s4_lookback = min(20, len(snapshots_1h))
                recent_prices = [s.curr_price for s in snapshots_1h[-s4_lookback:]]
                recent_gammas = [s.net_gamma / 1e9 for s in snapshots_1h[-s4_lookback:]]
                recent_low = min(recent_prices)
                rally_from_low = price - recent_low
                # Require gamma was deeper negative earlier (confirms it's a neg gamma regime,
                # not just a mild pullback). Min gamma in lookback must be < -3Bn.
                min_gamma_lookback = min(recent_gammas) if recent_gammas else 0
                if (rally_from_low >= s4_rally_pts
                        and gamma_bn <= s4_gamma_bn
                        and min_gamma_lookback <= -3):
                    side, tier, tier_label = "SHORT", 4, "S4"
                    win_pct = 70
                    rr = 2.5

        # --- S5: WALL COLLAPSE --- (DISABLED: audit showed flickering spreads)
        if side is None and cfg.get("s5_enabled", True) and snapshots_1h and len(snapshots_1h) >= 5:
            s5_spread_threshold = float(cfg.get("s5_spread_threshold", 5))
            s5_prior_spread_min = float(cfg.get("s5_prior_spread_min", 10))
            curr_spread = abs((snapshot.call_wall or 0) - (snapshot.put_floor or 0))

            # Reset fired flag when spread widens
            if curr_spread >= s5_prior_spread_min:
                self._wall_collapse_fired = False

            if not getattr(self, '_wall_collapse_fired', False):
                # Check if spread was wider in IMMEDIATE prior snapshots (last 3-5)
                prior_wide = False
                for s in snapshots_1h[-6:-1]:
                    s_spread = abs((s.call_wall or 0) - (s.put_floor or 0))
                    if s_spread >= s5_prior_spread_min:
                        prior_wide = True
                        break
                if curr_spread <= s5_spread_threshold and prior_wide and gamma_bn < -1:
                    side, tier, tier_label = "SHORT", 5, "S5"
                    win_pct = 65
                    rr = 2.0
                    self._wall_collapse_fired = True

        # --- S6: GAMMA ACCELERATION --- (DISABLED: audit showed 50/50)
        if side is None and cfg.get("s6_enabled", True) and lookback_20m:
            s6_dg_threshold = float(cfg.get("s6_dg_threshold", -2))
            s6_gamma_floor = float(cfg.get("s6_gamma_floor", -4))
            # Gate: gamma must have been milder 20min ago (> -3Bn)
            # and current gamma must not be too extreme (> -6Bn = exhaustion)
            if (dG_20m <= s6_dg_threshold
                    and gamma_bn < 0
                    and gamma_bn > s6_gamma_floor  # not exhaustion
                    and g_20m_ago > -3  # was milder before
                    and self._rth_snap_count > 10):  # skip first 20min of RTH
                side, tier, tier_label = "SHORT", 6, "S6"
                win_pct = 60
                rr = 2.0

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

        return self._emit_gamma_snap_signal(
            snapshot, snapshots_1h, side, tier, tier_label,
            win_pct, rr, gamma_bn, price, dP_20m, dG_20m, dP_15m, dG_15m,
        )

    def _check_early_momentum(
        self,
        snapshot: GEXSnapshot,
        open_price: Optional[float],
    ) -> Optional[Signal]:
        """B1 Early Rally / BEAR Early Selloff — first 30min momentum signal.

        Backtest (129 days):
          B1 LONG: +15pts from open in 30min => 82% win, +16.1 EOD (N=11)
          B1 + G<0: 100% win, +56.4 avg (N=2)
          B1 + G>+5: 100% win, +13.2 avg (N=5)
        Fires once per day max.
        """
        if self._b1_fired_today:
            return None
        if open_price is None:
            return None
        # Only check in first ~30min of RTH (roughly 15 snapshots at 2-min intervals)
        if self._rth_snap_count > 15:
            return None

        price = snapshot.curr_price
        gamma_bn = snapshot.net_gamma / 1e9
        move = price - open_price
        pf = snapshot.put_floor or 0
        cw = snapshot.call_wall or 0

        if move > 15:
            self._b1_fired_today = True
            return self._emit_gamma_snap_signal(
                snapshot, None, "LONG", 6, "B1", 82, 0, gamma_bn, price,
                move, 0, 0, 0,
            )

        if move < -15:
            self._b1_fired_today = True
            if -5 <= gamma_bn < 0:
                return self._emit_gamma_snap_signal(
                    snapshot, None, "SHORT", 7, "BEAR", 80, 0, gamma_bn, price,
                    move, 0, 0, 0,
                )

        return None

    def _check_butterfly_pin(
        self,
        snapshot: GEXSnapshot,
        snapshots_1h: Optional[List[GEXSnapshot]],
    ) -> Optional[Signal]:
        """Butterfly pin trade alert — when gamma > +10Bn, price pins to call wall.

        Backtest (116 trading days):
          Avg G > +10Bn (N=18): 89% pin within 5pts, 100% within 10pts
          Butterfly (20-pt wings at call wall): 56% win, +$339 avg P/L on $500 risk
          Avg G > +15Bn (N=14): 93% pin within 5pts, 64% win, +$436 avg
        Fires once per day max, after 10:00 AM PT (need gamma to establish).
        """
        if self._butterfly_fired_today:
            return None

        # Only fire after 10:00 AM PT — gamma pin strengthens as day progresses
        now_pt = datetime.now(PT_TZ)
        if now_pt.hour < 10:
            return None

        gamma_bn = snapshot.net_gamma / 1e9
        if gamma_bn < 10:
            return None

        price = snapshot.curr_price
        cw = snapshot.call_wall
        if not cw or cw == 0:
            return None

        dist = abs(price - cw)
        # Only alert when price is within 20pts of call wall (butterfly wing width)
        if dist > 20:
            return None

        # Need some history to confirm gamma has been elevated
        if snapshots_1h and len(snapshots_1h) >= 5:
            recent_gamma = [s.net_gamma / 1e9 for s in snapshots_1h[-5:]]
            avg_recent = sum(recent_gamma) / len(recent_gamma)
            if avg_recent < 8:
                return None
        else:
            return None

        self._butterfly_fired_today = True

        # Round call wall to nearest 5 for clean strike
        center = round(cw / 5) * 5
        wing_low = center - 20
        wing_high = center + 20

        # Determine confidence tier
        if gamma_bn >= 15:
            tier = "HIGH"
            win_pct = 64
            avg_pl = 436
            pin_pct = 93
        else:
            tier = "STANDARD"
            win_pct = 56
            avg_pl = 339
            pin_pct = 89

        late_session = now_pt.hour >= 11 or (now_pt.hour == 10 and now_pt.minute >= 30)
        late_tag = " [LATE SESSION]" if late_session else ""

        gd = _gamma_english(gamma_bn)
        late_note = "\n\u26a0 LATE SESSION \u2014 pin effect strongest last 2hrs." if late_session else ""
        msg = (
            f"\U0001f7e2 LONG \u2014 GAMMA PIN{late_tag}\n"
            f"SPX {price:.0f} | {now_pt.strftime('%I:%M %p').lstrip('0')}\n"
            f"\n"
            f"Structure: {gd.capitalize()} pinning price to call wall at {cw}\n"
            f"Translation: Dealers are locked in. Price gravitates to {center}.\n"
            f"\n"
            f"Play: 0DTE Butterfly at {center}\n"
            f"  BUY  {wing_low} call\n"
            f"  SELL 2x {center} call\n"
            f"  BUY  {wing_high} call\n"
            f"  Cost: ~$5 ($500 risk) | Max payout: $20 ($2,000)\n"
            f"\n"
            f"Pin rate: {pin_pct}% within 5pts | Win: {win_pct}% | Avg: +${avg_pl}"
            f"{late_note}"
        )

        return Signal(
            signal_type=SignalType.GAMMA_SNAP,
            title=f"Butterfly Pin {tier}",
            message=msg,
            channel="gex_trades",
            priority=SignalType.GAMMA_SNAP,
            metadata={
                "side": "BUTTERFLY",
                "tier": 9,
                "tier_label": f"PIN_{tier}",
                "gamma_bn": gamma_bn,
                "call_wall": cw,
                "center_strike": center,
                "dist_from_cw": dist,
                "win_pct": win_pct,
            },
        )

    def _check_lotto(
        self,
        snapshot: GEXSnapshot,
        snapshots_1h: Optional[List[GEXSnapshot]],
        daily_ma20: Optional[float] = None,
    ) -> Optional[Signal]:
        """CW Fade / Lotto short — price above call wall with strong positive gamma.

        The setup: G > +10Bn pins price to CW. When price pushes above CW,
        dealers sell into it. Cheap puts have 6-7x payoff when the pin snaps back.

        Today's example (2026-02-18): SPX 6905, CW 6900, G=+17Bn at 9:55 AM.
        6885 PUT entered at $3.80 → hit $20+ as price dropped to 6868. 5x return.

        Fires once per day during RTH. No time gate — the setup works all day.
        """
        if self._lotto_fired_today:
            return None

        if not _is_rth(snapshot.timestamp_pt):
            return None

        gamma_bn = snapshot.net_gamma / 1e9

        # G must be 10-20Bn — strong positive gamma = dealers pinning
        if gamma_bn < 10 or gamma_bn > 20:
            return None

        price = snapshot.curr_price
        cw = snapshot.call_wall
        if not cw or cw == 0:
            return None

        # Price must be above call wall
        if price <= cw:
            return None

        dist_above_cw = price - cw

        # Check confidence layers
        # Layer 1: CW proximity (3-15 pts above — widened from 3-8)
        near_cw = 3 <= dist_above_cw <= 15

        # Layer 2: MA extension (price > MA20 + 50pts)
        ma_extended = False
        ma_dist = 0.0
        if daily_ma20 is not None and daily_ma20 > 0:
            ma_dist = price - daily_ma20
            if ma_dist > 50:
                ma_extended = True

        # At least one confidence layer must be true
        if not near_cw and not ma_extended:
            return None

        self._lotto_fired_today = True

        now_pt = datetime.now(PT_TZ)
        late_session = now_pt.hour >= 11 or (now_pt.hour == 10 and now_pt.minute >= 30)
        time_tag = " [LATE SESSION]" if late_session else ""

        # Determine confidence tier
        if near_cw and ma_extended:
            conf_tier = "HIGH CONF"
            tags = "ABOVE CW + EXTENDED"
            win_range = "70-75%"
            mfe_range = "12-36"
        elif near_cw:
            conf_tier = "STANDARD"
            tags = "ABOVE CW"
            win_range = "70%"
            mfe_range = "12-25"
        else:
            conf_tier = "STANDARD"
            tags = "EXTENDED"
            win_range = "75%"
            mfe_range = "20-36"

        # Suggested put strike: at or slightly below call wall
        put_strike = round(cw / 5) * 5
        gd = _gamma_english(gamma_bn)

        structure_parts = []
        if near_cw:
            structure_parts.append(f"{dist_above_cw:.0f}pts above call wall at {cw}")
        if ma_extended:
            structure_parts.append(f"{ma_dist:.0f}pts above daily MA20 ({daily_ma20:.0f})")
        structure_line = " + ".join(structure_parts) if structure_parts else "Price extended above resistance"

        # Delta tracking
        delta_line = self._delta_tracker.format_delta_line("gex_trades", "lotto", {
            "price": price, "gamma_bn": gamma_bn,
            "call_wall": cw, "put_floor": snapshot.put_floor or 0,
        })

        # Build via format_type1_signal
        confluence = {
            "Gamma Regime": gamma_bn > 10,
            "Above CW": near_cw,
            "MA Extended": ma_extended,
        }

        msg = format_type1_signal(
            direction="SHORT",
            signal_name=f"CALL WALL FADE{time_tag}",
            price=price,
            entry=price,
            stop=price + 15,
            target=cw - 5,
            win_pct=72,
            gamma_bn=gamma_bn,
            structure_line=structure_line,
            confluence=confluence,
            delta_line=delta_line,
            pf=snapshot.put_floor or 0,
            cw=cw,
            late_session=late_session,
            sample_n=0,
            regime=self._confirmed_regime.value if self._confirmed_regime else "",
        )

        return Signal(
            signal_type=SignalType.LOTTO,
            title=f"Lotto Short {conf_tier}",
            message=msg,
            channel="gex_trades",
            priority=SignalType.LOTTO,
            metadata={
                "side": "LOTTO",
                "conf_tier": conf_tier,
                "gamma_bn": gamma_bn,
                "dist_above_cw": dist_above_cw,
                "ma_dist": ma_dist,
                "near_cw": near_cw,
                "ma_extended": ma_extended,
            },
        )

    # --- MA Snap qualifying MAs (priority order) ---
    _MA_SNAP_LONG_MAS = [
        # (priority, ma_key, label, win_pct, avg_r15, sample_n)
        (1, "60m_20",  "60-Minute 20-Period Moving Average",  92, 15.1, 13),
        (2, "30m_20",  "30-Minute 20-Period Moving Average",  78, 11.8, 18),
        (3, "15m_50",  "15-Minute 50-Period Moving Average",  88, 15.8, 16),
        (4, "5m_200",  "5-Minute 200-Period Moving Average",  82, 13.5, 11),
        (5, "3m_200",  "3-Minute 200-Period Moving Average",  78, 12.3, 18),
        (6, "5m_20",   "5-Minute 20-Period Moving Average",   64, 12.1, 36),
    ]

    _MA_SNAP_SHORT_MAS = [
        (1, "30m_200", "30-Minute 200-Period Moving Average", 94, -36.0, 16),
        (2, "60m_50",  "60-Minute 50-Period Moving Average",  61, -16.9, 23),
        (3, "15m_200", "15-Minute 200-Period Moving Average", 57, -15.5, 21),
    ]

    def _check_ma_snap(
        self,
        snapshot: GEXSnapshot,
        ma_values: Optional[Dict[str, float]] = None,
    ) -> Optional[Signal]:
        """MA touch + gamma regime signal — fires when price retraces to a key MA.

        Backtested 4,256 MA touch events over 98 trading days (Oct 2025 - Feb 2026).
        LONG (Rubber Band): G < -10Bn + price retraces down to support MA → 78-92% win
        SHORT (Dead Cat Fade): G -10 to -5Bn + price rallies up to resistance MA → 61-94% win
        """
        if not ma_values:
            return None

        if not _is_rth(snapshot.timestamp_pt):
            return None

        gamma_bn = snapshot.net_gamma / 1e9
        price = snapshot.curr_price

        # Determine which side(s) qualify based on gamma regime
        check_long = gamma_bn < -10
        check_short = -10 <= gamma_bn <= -5

        if not check_long and not check_short:
            return None

        best_signal = None

        if check_long:
            for priority, ma_key, label, win_pct, avg_r15, sample_n in self._MA_SNAP_LONG_MAS:
                if ma_key in self._ma_snap_fired_today:
                    continue
                ma_val = ma_values.get(ma_key)
                if ma_val is None:
                    continue
                dist = abs(price - ma_val)
                if dist > 3:
                    continue
                # Require price was 8+ pts away recently — checked via ma_values metadata
                recent_dist_key = f"{ma_key}_recent_max_dist"
                recent_dist = ma_values.get(recent_dist_key, 0)
                if recent_dist < 8:
                    continue
                # Found a qualifying touch — take highest priority (first match)
                self._ma_snap_fired_today.add(ma_key)
                best_signal = self._format_ma_snap_long(
                    snapshot, price, gamma_bn, ma_key, ma_val, label,
                    win_pct, avg_r15, sample_n, dist, recent_dist,
                )
                break

        if check_short and best_signal is None:
            for priority, ma_key, label, win_pct, avg_r15, sample_n in self._MA_SNAP_SHORT_MAS:
                if ma_key in self._ma_snap_fired_today:
                    continue
                ma_val = ma_values.get(ma_key)
                if ma_val is None:
                    continue
                dist = abs(price - ma_val)
                if dist > 3:
                    continue
                recent_dist_key = f"{ma_key}_recent_max_dist"
                recent_dist = ma_values.get(recent_dist_key, 0)
                if recent_dist < 8:
                    continue
                self._ma_snap_fired_today.add(ma_key)
                best_signal = self._format_ma_snap_short(
                    snapshot, price, gamma_bn, ma_key, ma_val, label,
                    win_pct, avg_r15, sample_n, dist, recent_dist,
                )
                break

        return best_signal

    def _format_ma_snap_long(
        self, snapshot, price, gamma_bn, ma_key, ma_val, label,
        win_pct, avg_r15, sample_n, dist, recent_dist,
    ) -> Signal:
        # Parse timeframe info from ma_key for context
        tf, period = ma_key.split("_")
        tf_map = {"3m": "3-minute", "5m": "5-minute", "15m": "15-minute", "30m": "30-minute", "60m": "60-minute"}
        period_map = {"20": "20-period", "50": "50-period", "200": "200-period"}
        tf_name = tf_map.get(tf, tf)
        period_name = period_map.get(period, f"{period}-period")

        # Time context descriptions
        hours_map = {
            "60m_20": "the last 20 hours of trading",
            "30m_20": "the last 10 hours of trading",
            "15m_50": "the last 12.5 hours of trading",
            "5m_200": "the last 16.7 hours of trading",
            "3m_200": "the last 10 hours of trading",
            "5m_20": "the last 100 minutes of trading",
        }
        hours_desc = hours_map.get(ma_key, "a significant period")

        stop = round(price - 10)
        target = round(price + round(avg_r15))
        strike = round(price / 5) * 5 + 5

        gd = _gamma_english(gamma_bn)
        now_pt = datetime.now(PT_TZ)
        time_str = now_pt.strftime("%I:%M %p").lstrip("0")

        structure_line = (
            f"Price pulled back to {tf_name} {period_name} MA at {ma_val:.0f}. "
            f"{gd.capitalize()} + MA support = dealers forced to buy here. "
            f"Price was {recent_dist:.0f}pts above this level recently — fresh retrace."
        )

        delta_line = self._delta_tracker.format_delta_line("gex_trades", "ma_snap", {
            "price": price, "gamma_bn": gamma_bn,
            "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
        })

        confluence = {
            "Gamma Regime": gamma_bn < -3,
            "MA Support": True,
            "Fresh Retrace": recent_dist > 10,
        }

        msg = format_type1_signal(
            direction="LONG",
            signal_name="MA RUBBER BAND",
            price=price,
            entry=price,
            stop=stop,
            target=target,
            win_pct=win_pct,
            gamma_bn=gamma_bn,
            structure_line=structure_line,
            confluence=confluence,
            delta_line=delta_line,
            ma_values={f"{tf}_{period}": ma_val},
            pf=snapshot.put_floor or 0,
            cw=snapshot.call_wall or 0,
            sample_n=sample_n,
        )

        return Signal(
            signal_type=SignalType.MA_SNAP,
            title=f"MA Snap — LONG ({tf} {period}MA)",
            message=msg,
            channel="gex_trades",
            priority=SignalType.MA_SNAP,
            metadata={
                "side": "LONG",
                "ma_key": ma_key,
                "ma_val": ma_val,
                "gamma_bn": gamma_bn,
                "win_pct": win_pct,
                "avg_r15": avg_r15,
                "sample_n": sample_n,
            },
        )

    def _format_ma_snap_short(
        self, snapshot, price, gamma_bn, ma_key, ma_val, label,
        win_pct, avg_r15, sample_n, dist, recent_dist,
    ) -> Signal:
        tf, period = ma_key.split("_")
        tf_map = {"3m": "3-minute", "5m": "5-minute", "15m": "15-minute", "30m": "30-minute", "60m": "60-minute"}
        period_map = {"20": "20-period", "50": "50-period", "200": "200-period"}
        tf_name = tf_map.get(tf, tf)
        period_name = period_map.get(period, f"{period}-period")

        hours_map = {
            "30m_200": "the last 100+ hours of trading",
            "60m_50": "the last 50 hours of trading",
            "15m_200": "the last 50 hours of trading",
        }
        hours_desc = hours_map.get(ma_key, "a significant period")

        stop = round(price + 12)
        target = round(price + avg_r15)  # avg_r15 is negative for shorts
        strike = round(price / 5) * 5 - 5

        gd = _gamma_english(gamma_bn)
        now_pt = datetime.now(PT_TZ)
        time_str = now_pt.strftime("%I:%M %p").lstrip("0")

        structure_line = (
            f"Rally into {tf_name} {period_name} MA at {ma_val:.0f}. "
            f"{gd.capitalize()} + MA resistance = rejection zone. "
            f"Price was {recent_dist:.0f}pts below this level recently — dead cat bounce."
        )

        delta_line = self._delta_tracker.format_delta_line("gex_trades", "ma_snap", {
            "price": price, "gamma_bn": gamma_bn,
            "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
        })

        confluence = {
            "Gamma Regime": gamma_bn > 3,
            "MA Resistance": True,
            "Dead Cat": recent_dist > 10,
        }

        msg = format_type1_signal(
            direction="SHORT",
            signal_name="DEAD CAT FADE",
            price=price,
            entry=price,
            stop=stop,
            target=target,
            win_pct=win_pct,
            gamma_bn=gamma_bn,
            structure_line=structure_line,
            confluence=confluence,
            delta_line=delta_line,
            ma_values={f"{tf}_{period}": ma_val},
            pf=snapshot.put_floor or 0,
            cw=snapshot.call_wall or 0,
            sample_n=sample_n,
        )

        return Signal(
            signal_type=SignalType.MA_SNAP,
            title=f"MA Snap — SHORT ({tf} {period}MA)",
            message=msg,
            channel="gex_trades",
            priority=SignalType.MA_SNAP,
            metadata={
                "side": "SHORT",
                "ma_key": ma_key,
                "ma_val": ma_val,
                "gamma_bn": gamma_bn,
                "win_pct": win_pct,
                "avg_r15": avg_r15,
                "sample_n": sample_n,
            },
        )

    def _check_level_approach(
        self,
        snapshot: GEXSnapshot,
        snapshots_1h: Optional[List[GEXSnapshot]],
    ) -> Optional[Signal]:
        """Alert when price approaches FIXED EM levels (VIX/16).

        DESIGN PHILOSOPHY (from user):
        - EM levels are FIXED (yesterday's close) — no lag, no chasing
        - GEX walls move every 2min and lag ~15min — use for CONFIRMATION only, not entry
        - When CW=PF (spread=0), GEX walls are meaningless — ignore them
        - Focus on EM ±1.0 boundaries (68% close inside) and key fibs (0.618, 0.768)
        - Alert ONCE per level approach (30min cooldown), not on every snapshot
        - Skip first 10 min of RTH (opening noise)
        """
        if not _is_rth(snapshot.timestamp_pt):
            return None

        # Skip first 10 min of RTH
        if self._rth_snap_count < 5:
            return None

        price = snapshot.curr_price
        gamma_bn = snapshot.net_gamma / 1e9
        cw = snapshot.call_wall or 0
        pf = snapshot.put_floor or 0
        gex_spread = abs(cw - pf) if cw and pf else 0

        em = self._get_em_levels()
        if not em:
            return None

        now_ts = snapshot.timestamp_pt.timestamp() if hasattr(snapshot.timestamp_pt, 'timestamp') else float(snapshot.timestamp_pt)

        # PRIMARY levels: EM (fixed, no lag)
        levels = []
        proximity_pts = 8

        if em.get("upper"):
            levels.append(("EM +1.0", em["upper"], "SHORT",
                f"Upper EM boundary {em['upper']:.0f} — 68% of days close inside. Options premiums spike here."))
        if em.get("lower"):
            levels.append(("EM -1.0", em["lower"], "LONG",
                f"Lower EM boundary {em['lower']:.0f} — 68% of days close inside. Options premiums spike here."))
        for fib in em.get("fibs", []):
            if fib["ratio"] == 0.618:
                levels.append((f"EM -{fib['ratio']}", fib["lower"], "LONG",
                    f"Golden ratio support {fib['lower']:.0f} — strong reversal zone"))
                levels.append((f"EM +{fib['ratio']}", fib["upper"], "SHORT",
                    f"Golden ratio resistance {fib['upper']:.0f} — strong reversal zone"))
            elif fib["ratio"] == 0.768:
                levels.append((f"EM -{fib['ratio']}", fib["lower"], "LONG",
                    f"Deep fib support {fib['lower']:.0f} — approaching extreme"))
                levels.append((f"EM +{fib['ratio']}", fib["upper"], "SHORT",
                    f"Deep fib resistance {fib['upper']:.0f} — approaching extreme"))

        # GEX walls are CONFIRMATION only — NOT entry triggers.
        # They lag ~15min and flicker. EM levels are the only triggers.

        best = None
        for name, level, bias, desc in levels:
            dist = abs(price - level)
            if dist > proximity_pts:
                continue
            last = self._level_alerted.get(name, 0)
            if (now_ts - last) < 1800:
                continue
            # Pick closest level
            if best is None or dist < best[5]:
                # Count confluence with other levels
                nearby = [(n, l) for n, l, _, _ in levels if abs(l - level) < 15 and n != name]
                best = (name, level, bias, desc, nearby, dist)

        if not best:
            return None

        name, level, bias, desc, nearby, dist = best
        self._level_alerted[name] = now_ts

        nearby_str = " + ".join(f"{n}({l:.0f})" for n, l in nearby) if nearby else ""
        confluence = len(nearby) + 1

        em_pos = em.get("position", 0)

        # Build structure description — plain English
        gamma_desc = _gamma_english(gamma_bn)
        why_lines = [desc]
        if bias == "LONG" and gamma_bn < -3:
            why_lines.append(f"{gamma_desc.capitalize()} amplifies bounce")
        elif bias == "LONG" and gamma_bn > 3:
            why_lines.append(f"{gamma_desc.capitalize()} supports level")
        elif bias == "SHORT" and gamma_bn > 3:
            why_lines.append(f"{gamma_desc.capitalize()} pins — fade likely")
        elif bias == "SHORT" and gamma_bn < -3:
            why_lines.append(f"{gamma_desc.capitalize()} — breakout risk if fails")
        if gex_spread > 15:
            if bias == "LONG" and pf and abs(price - pf) < 15:
                why_lines.append(f"Put floor at {pf:.0f} confirms support")
            elif bias == "SHORT" and cw and abs(price - cw) < 15:
                why_lines.append(f"Call wall at {cw:.0f} confirms resistance")

        if bias == "LONG":
            stop = level - 8
            target = level + 15
        else:
            stop = level + 8
            target = level - 15

        # Build structure line from why_lines
        structure_line = why_lines[0] if why_lines else _STRUCTURE_TEMPLATES.get("EM", "Price at expected move boundary")
        em_levels_data = self.compute_expected_move()
        ma_vals = getattr(self, '_last_ma_values', None)

        # 5-check confluence
        confluence_map = self._compute_confluence(bias, price, gamma_bn, pf, cw, ma_vals, em_levels_data)

        msg = _format_trade_alert(
            direction=bias,
            signal_name=f"EM LEVEL ({name})",
            price=price,
            entry=price,
            stop=stop,
            target=target,
            win_pct=68,
            gamma_bn=gamma_bn,
            structure_line=structure_line,
            confluence=confluence_map,
            ma_values=ma_vals,
            pf=pf,
            cw=cw,
            em_levels=em_levels_data,
        )

        return Signal(
            signal_type=SignalType.LEVEL_APPROACH,
            title=f"Level Alert — {name}",
            message=msg,
            channel="gex_trades",
            priority=SignalType.LEVEL_APPROACH,
            metadata={
                "level_name": name,
                "level_price": level,
                "bias": bias,
                "distance": dist,
                "confluence": confluence,
                "gamma_bn": gamma_bn,
                "em_position": em_pos,
            },
        )

    def _get_em_levels(self) -> Optional[dict]:
        """Fetch expected move levels from Command Center API. Cached per day."""
        import time as _time
        cache_attr = "_em_cache"
        cache_ts_attr = "_em_cache_ts"
        now = _time.time()
        # Cache for 1 hour
        if hasattr(self, cache_attr) and hasattr(self, cache_ts_attr):
            if now - getattr(self, cache_ts_attr) < 3600:
                return getattr(self, cache_attr)
        try:
            import urllib.request
            import json
            req = urllib.request.urlopen("http://localhost:5500/api/expected-move", timeout=5)
            data = json.loads(req.read())
            if data.get("error"):
                return None
            setattr(self, cache_attr, data)
            setattr(self, cache_ts_attr, now)
            return data
        except Exception:
            return getattr(self, cache_attr, None)

    def compute_expected_move(self) -> Optional[dict]:
        """Self-contained expected move calculation from DB + yfinance VIX.

        Returns dict with prev_close, vix, em_pts, upper, lower, anchor, position.
        Falls back to _get_em_levels() API call if yfinance fails.
        """
        cache_attr = "_em_self_cache"
        cache_ts_attr = "_em_self_cache_ts"
        now = _time_mod.time()
        if hasattr(self, cache_attr) and hasattr(self, cache_ts_attr):
            if now - getattr(self, cache_ts_attr) < 3600:
                return getattr(self, cache_attr)

        try:
            from gex_db import get_conn
            conn = get_conn()
            # Get yesterday's last RTH snapshot for closing price
            row = conn.execute(
                "SELECT curr_price FROM gex_snapshots "
                "WHERE session_tag='RTH' AND date_pt < date('now', 'localtime') "
                "ORDER BY timestamp_pt DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if not row:
                return self._get_em_levels()
            prev_close = float(row["curr_price"])
        except Exception:
            return self._get_em_levels()

        try:
            import yfinance as yf
            vix = float(yf.Ticker("^VIX").fast_info["lastPrice"])
        except Exception:
            return self._get_em_levels()

        # Full-day EM = prev_close * vix / 100 / 16
        em_full = prev_close * (vix / 100) / 16
        # Time-decay: scale by sqrt(fraction of RTH remaining)
        # RTH = 6.5 hours (9:30-16:00 ET = 6:30-13:00 PT)
        import datetime as _dt_mod, pytz as _pytz_mod
        _pt = _pytz_mod.timezone("US/Pacific")
        _now_pt = _dt_mod.datetime.now(_pt)
        _mkt_open_min = 6 * 60 + 30   # 6:30 PT
        _mkt_close_min = 13 * 60       # 13:00 PT
        _total_rth = _mkt_close_min - _mkt_open_min  # 390 min
        _now_min = _now_pt.hour * 60 + _now_pt.minute
        if _mkt_open_min < _now_min < _mkt_close_min:
            _remaining = max(_mkt_close_min - _now_min, 1)
            em_pts = em_full * (_remaining / _total_rth) ** 0.5
        else:
            em_pts = em_full

        result = {
            "prev_close": prev_close,
            "anchor": prev_close,
            "vix": vix,
            "em_pts": em_pts,
            "upper": prev_close + em_pts,
            "lower": prev_close - em_pts,
            "fibs": [
                {"ratio": 0.618, "upper": prev_close + em_pts * 0.618, "lower": prev_close - em_pts * 0.618},
                {"ratio": 0.768, "upper": prev_close + em_pts * 0.768, "lower": prev_close - em_pts * 0.768},
            ],
        }

        setattr(self, cache_attr, result)
        setattr(self, cache_ts_attr, now)
        return result

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
            channel="gex_engine",
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
            channel="gex_engine",
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
        self._save_interpreter_state()

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
        gamma_bn = snapshot.net_gamma / 1e9
        price = snapshot.curr_price
        emoji = "\U0001f7e2" if trend.bias == "BULLISH" else "\U0001f534" if trend.bias == "BEARISH" else "\u26aa"

        # Build plain English story
        # 1. What's the move from open?
        move_str = ""
        if trend.open_price:
            move = price - trend.open_price
            if abs(move) < 3:
                move_str = "Flat from open"
            else:
                move_str = f"{'Up' if move > 0 else 'Down'} {abs(move):.0f}pts from open"

        # 2. What's gamma doing? (the WHY)
        if gamma_bn > 5:
            gamma_str = f"Strong positive gamma ({gamma_bn:+.1f}Bn) — dealers pinning price, fade the edges"
        elif gamma_bn > 1:
            gamma_str = f"Positive gamma ({gamma_bn:+.1f}Bn) — dealers cushion dips, hard to break out"
        elif gamma_bn > -1:
            gamma_str = f"Neutral gamma ({gamma_bn:+.1f}Bn) — dealers not driving, follow the tape"
        elif gamma_bn > -5:
            gamma_str = f"Negative gamma ({gamma_bn:+.1f}Bn) — moves get amplified, be quick"
        else:
            gamma_str = f"Deep negative gamma ({gamma_bn:+.1f}Bn) — chaotic, dealers selling into drops"

        # 3. What to do? (the SO WHAT)
        cw = snapshot.call_wall or 0
        pf = snapshot.put_floor or 0
        dist_cw = cw - price if cw else 0
        dist_pf = price - pf if pf else 0

        if trend.bias == "BULLISH" and trend.score >= 8:
            action = "Buy dips. Dealers are supporting this move."
        elif trend.bias == "BULLISH" and dist_cw < 10 and cw:
            action = f"Careful — only {dist_cw:.0f}pts to call wall {cw}. Could stall here."
        elif trend.bias == "BEARISH" and trend.score >= 8:
            action = "Sell rips. Momentum is to the downside."
        elif trend.bias == "BEARISH" and dist_pf < 10 and pf:
            action = f"Careful — only {dist_pf:.0f}pts to put floor {pf}. Could bounce here."
        elif trend.score < 6:
            action = "No clear edge. Stay patient or reduce size."
        elif gamma_bn > 3 and abs(dist_cw - dist_pf) < 15:
            action = "Pinned between walls. Sell premium or wait for a break."
        else:
            side = "calls on dips" if trend.bias == "BULLISH" else "puts on rips" if trend.bias == "BEARISH" else "nothing"
            action = f"Lean {trend.bias.lower()}. Look for {side}."

        lines = [
            f"{emoji} SPX {price:.0f} | {move_str}",
            gamma_str,
            f"Walls: {pf} floor — {cw} ceiling ({pf and cw and (cw-pf) or 0:.0f}pt spread)",
            f"Play: {action}",
        ]

        # Add specific catalysts if any
        action_notes = [n for n in (trend.notes or []) if any(kw in n.lower() for kw in
            ["shifted", "pressure", "pressing", "overnight", "floor", "wall", "flip"])]
        if action_notes:
            lines.append("Watch: " + "; ".join(n for n in action_notes[:2]))

        return Signal(
            signal_type=SignalType.TREND_DASHBOARD,
            title=f"Trend Dashboard - {trend.bias}",
            message="\n".join(lines),
            channel="gex_engine",
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

        shift = overnight.gamma_shift_pct
        curr_bn = overnight.current_net_gamma / 1e9

        if shift > 50:
            impact = "Major gamma build — expect a controlled, trending day"
        elif shift > 20:
            impact = "Gamma building — dealers gaining control"
        elif shift < -50:
            impact = "Gamma collapsed — expect fast, amplified moves today"
        elif shift < -20:
            impact = "Gamma fading — swings will be bigger than yesterday"
        else:
            impact = "Minor shift — similar environment to yesterday"

        gd = _gamma_english(curr_bn)
        lines = [
            f"OVERNIGHT SHIFT",
            f"",
            f"{impact}",
            f"Gamma: {overnight.eod_net_gamma / 1e9:+.1f} -> {curr_bn:+.1f} Bn ({shift:+.0f}%)",
            f"Now: {gd}",
        ]
        if overnight.call_wall_moved != 0 or overnight.put_floor_moved != 0:
            lines.append(f"Walls moved: CW {overnight.eod_call_wall}->{overnight.current_call_wall} | PF {overnight.eod_put_floor}->{overnight.current_put_floor}")
        lines.append(f"Expected range: {overnight.range_forecast}")
        msg = "\n".join(lines)

        return Signal(
            signal_type=SignalType.OVERNIGHT_DRIFT,
            title="Overnight Drift",
            message=msg,
            channel="gex_context",
            priority=SignalType.OVERNIGHT_DRIFT,
        )


# --- Formatting helpers ---


def _format_rth_pulse(
    snapshot: GEXSnapshot,
    one_hour: Optional[OneHourMetrics],
    advanced: Optional[AdvancedMetrics],
    levels: dict,
    delta_tracker: Optional['DeltaTracker'] = None,
) -> str:
    gamma_bn = snapshot.net_gamma / 1e9
    regime = one_hour.regime.value if one_hour else "N/A"

    snap_data = {
        "price": snapshot.curr_price, "gamma_bn": gamma_bn,
        "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
        "spread": snapshot.spread or 0,
    }

    oh_data = {"regime": regime, "confidence": one_hour.regime_confidence if one_hour else 0} if one_hour else None
    adv_data = {
        "squeeze_prob": advanced.squeeze_probability if advanced else 0,
        "pin_prob": advanced.pin_probability if advanced else 0,
    } if advanced else None

    # Convert trade_levels to dict format for formatter
    lvl_data = None
    if levels:
        lvl_data = {}
        if "long" in levels and levels["long"].valid:
            ll = levels["long"]
            lvl_data.update({"long_valid": True, "long_entry_low": ll.entry_low, "long_entry_high": ll.entry_high,
                           "long_stop": ll.stop, "long_target": ll.target, "long_rr": ll.rr_ratio})
        if "short" in levels and levels["short"].valid:
            sl = levels["short"]
            lvl_data.update({"short_valid": True, "short_entry_low": sl.entry_low, "short_entry_high": sl.entry_high,
                           "short_stop": sl.stop, "short_target": sl.target, "short_rr": sl.rr_ratio})

    delta_line = ""
    if delta_tracker:
        delta_line = delta_tracker.format_delta_line("gex_engine", "rth_pulse", {
            "price": snapshot.curr_price, "gamma_bn": gamma_bn,
            "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
            "spread": snapshot.spread or 0, "regime": regime,
        })

    return format_type3_market_read(
        read_type="RTH_PULSE", snapshot_data=snap_data, delta_line=delta_line,
        one_hour=oh_data, advanced=adv_data, levels=lvl_data,
    )


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
    delta_tracker: Optional['DeltaTracker'] = None,
) -> str:
    gamma_bn = snapshot.net_gamma / 1e9
    snap_data = {
        "price": snapshot.curr_price, "gamma_bn": gamma_bn,
        "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
    }
    adv_data = {"pin_prob": advanced.pin_probability if advanced else 0} if advanced else None

    delta_line = ""
    if delta_tracker:
        delta_line = delta_tracker.format_delta_line("gex_context", "final_15", snap_data)

    return format_type3_market_read(
        read_type="FINAL_15", snapshot_data=snap_data, delta_line=delta_line, advanced=adv_data,
    )


def _format_morning_brief(
    snapshot: GEXSnapshot,
    overnight: Optional[OvernightDrift],
    advanced: Optional[AdvancedMetrics],
    delta_tracker: Optional['DeltaTracker'] = None,
) -> str:
    gamma_bn = snapshot.net_gamma / 1e9
    snap_data = {
        "price": snapshot.curr_price, "gamma_bn": gamma_bn,
        "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
        "spread": snapshot.spread or 0,
    }

    overnight_data = None
    if overnight:
        overnight_data = {
            "gamma_shift_pct": overnight.gamma_shift_pct,
            "cw_moved": overnight.call_wall_moved,
            "pf_moved": overnight.put_floor_moved,
            "range_forecast": overnight.range_forecast,
        }

    adv_data = None
    if advanced and advanced.pivot_strikes:
        adv_data = {"pivot_strikes": [p.strike for p in advanced.pivot_strikes]}

    delta_line = ""
    if delta_tracker:
        delta_line = delta_tracker.format_delta_line("gex_context", "morning_brief", snap_data)

    return format_type3_market_read(
        read_type="MORNING_BRIEF", snapshot_data=snap_data, delta_line=delta_line,
        overnight=overnight_data, advanced=adv_data,
    )


def _format_quiet_summary(
    snapshot: GEXSnapshot,
    fifteen_min: Optional[FifteenMinMetrics],
    one_hour: Optional[OneHourMetrics],
    advanced: Optional[AdvancedMetrics],
    delta_tracker: Optional['DeltaTracker'] = None,
) -> str:
    gamma_bn = snapshot.net_gamma / 1e9
    regime = one_hour.regime.value if one_hour else "N/A"
    time_str = snapshot.timestamp_pt.strftime("%I:%M %p").lstrip("0")

    snap_data = {
        "price": snapshot.curr_price, "gamma_bn": gamma_bn,
        "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
        "spread": snapshot.spread or 0, "time_str": time_str,
    }
    fm_15 = {"price_delta": fifteen_min.price_delta} if fifteen_min else None
    oh_data = {"regime": regime} if one_hour else None

    delta_line = ""
    if delta_tracker:
        delta_line = delta_tracker.format_delta_line("gex_engine", "quiet_summary", {
            "price": snapshot.curr_price, "gamma_bn": gamma_bn,
            "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
        })

    return format_type3_market_read(
        read_type="QUIET_SUMMARY", snapshot_data=snap_data, delta_line=delta_line,
        fifteen_min=fm_15, one_hour=oh_data,
    )


# --- Plain English regime translations ---
_REGIME_PLAIN = {
    "CONTROLLED_PIN": "pinning regime (price stuck near key strikes)",
    "CONTROLLED_TREND": "controlled trend (dealers hedge in your favor)",
    "FRAGILE_CONTROL": "fragile balance (could break either way)",
    "UNCONTROLLED": "negative gamma chaos (dealers amplify moves)",
}

# Translation logic for regime shift combos
_REGIME_SHIFT_TRANSLATIONS = {
    ("CONTROLLED_PIN", "UNCONTROLLED"): "Yesterday was quiet. Today dealers amplify everything. Trade the extremes.",
    ("CONTROLLED_TREND", "FRAGILE_CONTROL"): "Trend is losing dealer support. Tighten stops.",
    ("CONTROLLED_TREND", "UNCONTROLLED"): "Controlled move turning chaotic. Dealers now amplify in both directions.",
    ("UNCONTROLLED", "CONTROLLED_PIN"): "Chaos settling down. Expect narrow range. Fade the edges.",
    ("UNCONTROLLED", "CONTROLLED_TREND"): "Chaos resolving into a trend. Pick a side and ride it.",
    ("FRAGILE_CONTROL", "UNCONTROLLED"): "Balance broke. Dealers selling into moves now. Expect fast swings.",
    ("FRAGILE_CONTROL", "CONTROLLED_PIN"): "Settling into a pin. Range will tighten. Sell premium.",
    ("FRAGILE_CONTROL", "CONTROLLED_TREND"): "Clarity emerging. Dealers now supporting the trend direction.",
    ("CONTROLLED_PIN", "CONTROLLED_TREND"): "Pin releasing into a trend. Follow the direction.",
    ("CONTROLLED_PIN", "FRAGILE_CONTROL"): "Pin weakening. Could break either way. Watch for the trigger.",
    ("UNCONTROLLED", "FRAGILE_CONTROL"): "Chaos easing but not resolved. Stay cautious.",
    ("CONTROLLED_TREND", "CONTROLLED_PIN"): "Trend stalling into a pin. Expect a range day.",
}

_REGIME_STANCE = {
    "CONTROLLED_PIN": "Fade the edges. Sell premium.",
    "CONTROLLED_TREND": "Follow the trend. Buy dips or sell rips.",
    "FRAGILE_CONTROL": "Wait for the break. Don't force trades.",
    "UNCONTROLLED": "Wait for the sweep, trade the snap.",
}


def _format_market_read(
    old_regime: str,
    new_regime: str,
    snapshot: GEXSnapshot,
    confidence: float = 0,
    slope: float = 0,
    delta_tracker: Optional['DeltaTracker'] = None,
) -> str:
    """Format a Type 3 Market Commentary message — plain English."""
    gamma_bn = snapshot.net_gamma / 1e9
    snap_data = {
        "price": snapshot.curr_price, "gamma_bn": gamma_bn,
        "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
        "spread": snapshot.spread or 0,
    }

    delta_line = ""
    if delta_tracker:
        delta_line = delta_tracker.format_delta_line("gex_context", "regime_shift", {
            "price": snapshot.curr_price, "gamma_bn": gamma_bn,
            "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
            "regime": new_regime,
        })

    return format_type3_market_read(
        read_type="REGIME_SHIFT", snapshot_data=snap_data, delta_line=delta_line,
        regime=new_regime, old_regime=old_regime,
    )
