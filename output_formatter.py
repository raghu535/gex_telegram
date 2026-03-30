"""Central output formatter — CLAUDE.md-aligned Discord messages.

Three main formatters matching the 3 output types:
  format_type1_signal()  — GEX SIGNAL (trade alerts)
  format_type2_relay()   — CHANNEL RELAY (external validation)
  format_type3_market_read() — MARKET READ (regime commentary)

All outputs follow the formatting bible:
  - Conviction tiers first (3/2/1 green dots), then urgency
  - Plain English narrative, 2-3 sentences, every claim backed by DB data
  - Every message ends with "Δ Since last: {changes}"
  - Option plays as ranges, never single contracts
  - Win rates always show sample size (n=X)
  - SHORT signals: always flip the sign on returns
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional

import pytz

PT_TZ = pytz.timezone("US/Pacific")

# ── Conviction + Urgency Helpers ──────────────────────────────────────────

def compute_conviction_tier(
    win_pct: float,
    confluence_active: int,
    confluence_total: int,
    sample_n: int = 0,
    has_conflicts: bool = False,
) -> int:
    """Map confluence + win rate to 1-3 conviction tier.

    Returns:
        3 = high conviction (🟢🟢🟢)
        2 = medium (🟢🟢)
        1 = low (🟢)
    """
    score = 0.0

    # Win rate component (0-2)
    if win_pct >= 75:
        score += 2.0
    elif win_pct >= 60:
        score += 1.5
    elif win_pct >= 50:
        score += 1.0
    else:
        score += 0.5

    # Confluence component (0-1.5)
    ratio = confluence_active / confluence_total if confluence_total > 0 else 0
    score += ratio * 1.5

    # Sample size bonus (0-0.5)
    if sample_n >= 30:
        score += 0.5
    elif sample_n >= 15:
        score += 0.25

    # Conflict penalty
    if has_conflicts:
        score -= 0.5

    if score >= 3.0:
        return 3
    elif score >= 2.0:
        return 2
    return 1


def _conviction_dots(tier: int) -> str:
    """Return green circle dots for conviction tier."""
    return "\U0001f7e2" * tier


def compute_urgency(
    distance_to_level: float = 999,
    price_velocity: float = 0,
    minutes_remaining: float = 390,
    late_session: bool = False,
) -> str:
    """Map distance/velocity/time to urgency label.

    Returns: "ACT NOW" | "LIMIT ORDER" | "WATCH"
    """
    if late_session and distance_to_level < 5:
        return "ACT NOW"

    if distance_to_level < 3 and abs(price_velocity) > 5:
        return "ACT NOW"
    elif distance_to_level < 10 or abs(price_velocity) > 3:
        return "LIMIT ORDER"
    return "WATCH"


# ── Narrative Generator ───────────────────────────────────────────────────

def generate_narrative(
    signal_name: str,
    direction: str,
    gamma_bn: float,
    regime: str = "",
    win_pct: float = 0,
    sample_n: int = 0,
    extra_context: str = "",
) -> str:
    """Template-based 2-3 sentence plain English hook.

    Every claim references DB data (win rate, sample size, regime stats).
    """
    gamma_desc = _gamma_english(gamma_bn)

    # Direction-specific opener
    if direction == "LONG":
        if gamma_bn < -3:
            opener = f"Negative gamma means dealers are forced to buy here."
        else:
            opener = f"Momentum building with {gamma_desc}."
    else:
        if gamma_bn > 3:
            opener = f"Positive gamma means dealers sell into every rally."
        else:
            opener = f"Selling pressure with {gamma_desc}."

    # Signal-specific middle sentence
    _signal_hooks = {
        "RUBBER BAND": "Price stretched too far, too fast — snap-back is the highest-probability setup we track.",
        "GAMMA SNAP": "Gamma dislocation detected — dealers forced to chase.",
        "LIQUIDITY SWEEP": "Swept the extreme, now reclaiming — this is the reversal pattern.",
        "GAMMA CEILING": "100% historical reversal from this gamma level. Every rally sold.",
        "FAKE RALLY": "Gamma fading while price rises — the move has no support underneath.",
        "DEEP GAMMA": "Extreme stretch into deep gamma — rubber band fully loaded.",
        "EARLY RALLY": "Strong opening momentum tends to persist all day.",
        "EARLY SELLOFF": "Morning selling in negative gamma — historically continues.",
        "NEG GAMMA FADE": "Rally into negative gamma — dealers amplify the reversal.",
        "WALL COLLAPSE": "Spread collapsed — market lost its guardrails. Expect a breakout.",
        "GAMMA ACCEL": "Gamma accelerating negative — dealer feedback loop engaged.",
        "MA RUBBER BAND": "Price pulled back to key moving average support — bounce zone.",
        "DEAD CAT FADE": "Rally into moving average resistance — rejection likely.",
        "CALL WALL FADE": "Price above call wall with strong positive gamma — dealers pin it back down.",
        "BOUNCE ZONE": "Price approaching key gamma wall — watch for the bounce or break.",
    }
    hook = _signal_hooks.get(signal_name, f"{signal_name} signal detected.")

    # Win rate closer
    if win_pct > 0 and sample_n > 0:
        closer = f"Win rate: {win_pct:.0f}% (n={sample_n})."
    elif win_pct > 0:
        closer = f"Win rate: {win_pct:.0f}%."
    else:
        closer = ""

    parts = [opener, hook]
    if extra_context:
        parts.append(extra_context)
    if closer:
        parts.append(closer)

    return " ".join(parts)


# ── Box Drawing ───────────────────────────────────────────────────────────

def _box(title: str, lines: List[str]) -> str:
    """Build a compact box-drawing section."""
    content = "\n".join(f"  {l}" for l in lines if l)
    return f"┌─ {title}\n{content}\n└─"


def _confluence_bar(active: int, total: int) -> str:
    filled = "\u2588" * active
    empty = "\u2591" * (total - active)
    return f"{filled}{empty}"


def _gamma_english(gamma_bn: float) -> str:
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


_REGIME_PLAIN = {
    "CONTROLLED_PIN": "pinning regime (price stuck near key strikes)",
    "CONTROLLED_TREND": "controlled trend (dealers hedge in your favor)",
    "FRAGILE_CONTROL": "fragile balance (could break either way)",
    "UNCONTROLLED": "negative gamma chaos (dealers amplify moves)",
}

_REGIME_STANCE = {
    "CONTROLLED_PIN": "Fade the edges. Sell premium.",
    "CONTROLLED_TREND": "Follow the trend. Buy dips or sell rips.",
    "FRAGILE_CONTROL": "Wait for the break. Don't force trades.",
    "UNCONTROLLED": "Wait for the sweep, trade the snap.",
}

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


# ── TYPE 1: GEX SIGNAL (Trade Alerts) ────────────────────────────────────

def format_type1_signal(
    direction: str,
    signal_name: str,
    price: float,
    entry: float,
    stop: float,
    target: float,
    win_pct: float,
    gamma_bn: float,
    structure_line: str,
    confluence: dict,
    delta_line: str = "",
    ma_values: Optional[Dict] = None,
    pf: float = 0,
    cw: float = 0,
    em_levels: Optional[dict] = None,
    late_session: bool = False,
    sample_n: int = 0,
    regime: str = "",
    historical_note: str = "",
) -> str:
    """Build a Type 1 trade alert — CLAUDE.md aligned.

    Structure:
        Conviction dots + Urgency → Header → Narrative → SIGNAL DATA box →
        TRADE box → Play → Delta
    """
    now_pt = datetime.now(PT_TZ)
    time_str = now_pt.strftime("%I:%M %p").lstrip("0")

    # Conviction + Urgency
    total = len(confluence)
    active = sum(1 for v in confluence.values() if v)
    tier = compute_conviction_tier(win_pct, active, total, sample_n)
    dots = _conviction_dots(tier)

    # Distance to nearest level for urgency
    dist = min(abs(price - pf) if pf else 999, abs(price - cw) if cw else 999)
    urgency = compute_urgency(dist, 0, 390, late_session)

    # Narrative
    narrative = generate_narrative(
        signal_name, direction, gamma_bn, regime, win_pct, sample_n
    )

    # Build message
    emoji = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = reward / risk if risk > 0 else 0

    lines = [
        f"{dots} {urgency} | {emoji} {direction} — {signal_name}",
        f"SPX {price:.0f} | {time_str}",
        "",
        narrative,
    ]

    # SIGNAL DATA box
    signal_data = []
    signal_data.append(f"Structure: {structure_line}")

    # VWAP
    vwap = ma_values.get("vwap") if ma_values else None
    if vwap:
        if direction == "LONG":
            vwap_ctx = "reclaim target" if price < vwap else "above, bullish"
        else:
            vwap_ctx = "rejection zone" if price > vwap else "below, bearish"
        signal_data.append(f"VWAP: {vwap:.0f} ({vwap_ctx})")

    # SMA
    sma_parts = []
    if ma_values:
        for k in ["5m_20", "30m_20", "60m_20"]:
            v = ma_values.get(k)
            if v:
                sma_parts.append(f"{k.replace('_', '-')} {v:.0f}")
    if sma_parts:
        signal_data.append(f"SMA: {' | '.join(sma_parts)}")

    # Levels
    support = pf or 0
    resistance = cw or 0
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
    lvl_parts = []
    if support:
        lvl_parts.append(f"Support {support:.0f}")
    if resistance:
        lvl_parts.append(f"Resistance {resistance:.0f}")
    if lvl_parts:
        signal_data.append(f"Levels: {' | '.join(lvl_parts)}")

    # Confluence bar + win rate
    bar = _confluence_bar(active, total)
    wr_str = f"WR {win_pct:.0f}%"
    if sample_n > 0:
        wr_str += f" (n={sample_n})"
    signal_data.append(f"Confluence: {bar} {active}/{total} | {wr_str}")

    lines.append("")
    lines.append(_box("SIGNAL DATA", signal_data))

    # TRADE box
    trade_lines = [
        f"Entry: {entry:.0f} | Stop: {stop:.0f} | Target: {target:.0f}",
        f"R:R: {rr:.1f}:1",
    ]

    # 0DTE play as range
    try:
        from strike_picker import get_0dte_play
        opt_dir = "CALL" if direction == "LONG" else "PUT"
        play = get_0dte_play(opt_dir, price, target, stop)
        bid_str = f"${play['bid']:.2f}" if play.get('bid') else "?"
        ask_str = f"${play['ask']:.2f}" if play.get('ask') else "?"
        trade_lines.append(f"Play: 0DTE {play['symbol']} @ {bid_str}-{ask_str}")
        if play.get("target_premium"):
            trade_lines.append(f"Target: {play['target_premium']} at target {target:.0f}")
        trade_lines.append(f"Stop: {play.get('stop_note', f'cut at {stop:.0f}')}")
    except Exception:
        if direction == "LONG":
            strike = math.ceil(price / 5) * 5
            option = f"SPXW {strike}C"
        else:
            strike = math.floor(price / 5) * 5
            option = f"SPXW {strike}P"
        trade_lines.append(f"Play: 0DTE {option}")

    lines.append(_box("TRADE", trade_lines))

    if late_session:
        lines.append("\u26a0 LATE SESSION — 0DTE amplified, tighten stops")

    if historical_note:
        lines.append(f"\U0001f4ca {historical_note}")

    # Delta line (always last)
    if delta_line:
        lines.append("")
        lines.append(delta_line)

    msg = "\n".join(lines)

    # Discord 2000-char limit — progressive trim
    if len(msg) > 1950:
        # Remove historical note first
        if historical_note:
            lines = [l for l in lines if not l.startswith("\U0001f4ca")]
            msg = "\n".join(lines)
    if len(msg) > 1950:
        # Remove SMA lines from signal data
        lines = [l for l in lines if not l.strip().startswith("SMA:")]
        msg = "\n".join(lines)
    if len(msg) > 1950:
        msg = msg[:1945] + "\n..."

    return msg


# ── TYPE 2: CHANNEL RELAY (External Validation) ──────────────────────────

def format_type2_relay(
    source: str,
    direction: str,
    confidence: float,
    checks: Dict[str, bool],
    price: float,
    gamma_bn: float,
    entry: float = 0,
    stop: float = 0,
    target: float = 0,
    scenario: str = "",
    delta_line: str = "",
) -> str:
    """Build a Type 2 channel relay — external signal validated against GEX.

    Structure:
        Header → Source attribution → 4-check validation grid →
        GEX verdict → Trade if aligned → Delta
    """
    now_pt = datetime.now(PT_TZ)
    time_str = now_pt.strftime("%I:%M %p").lstrip("0")

    emoji = "\U0001f7e2" if direction == "LONG" else "\U0001f534"

    # Build check grid
    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    check_lines = []
    for name, ok in checks.items():
        icon = "\u2705" if ok else "\u274c"
        check_lines.append(f"{icon} {name}")

    # Verdict
    if passed >= 3:
        verdict = "CONFIRMED — GEX aligns with external call"
    elif passed == 2:
        verdict = "PARTIAL — some GEX support, trade with caution"
    else:
        verdict = "REJECTED — GEX contradicts external call"

    lines = [
        f"\U0001f4e1 RELAY | {emoji} {direction} | {time_str}",
        f"Source: {source} (confidence: {confidence:.0f}%)",
        f"SPX {price:.0f} | {_gamma_english(gamma_bn)}",
        "",
    ]

    lines.append(_box("VALIDATION", check_lines))
    lines.append("")
    lines.append(f"Verdict: {verdict}")

    if scenario:
        lines.append(f"Scenario: {scenario}")

    if entry and stop and target:
        rr = abs(target - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0
        lines.append("")
        lines.append(_box("TRADE", [
            f"Entry: {entry:.0f} | Stop: {stop:.0f} | Target: {target:.0f}",
            f"R:R: {rr:.1f}:1",
        ]))

    if delta_line:
        lines.append("")
        lines.append(delta_line)

    msg = "\n".join(lines)
    if len(msg) > 1950:
        msg = msg[:1945] + "\n..."
    return msg


# ── TYPE 3: MARKET READ (Regime Commentary) ──────────────────────────────

def format_type3_market_read(
    read_type: str,
    snapshot_data: dict,
    delta_line: str = "",
    regime: str = "",
    old_regime: str = "",
    overnight: Optional[dict] = None,
    em_levels: Optional[dict] = None,
    advanced: Optional[dict] = None,
    fifteen_min: Optional[dict] = None,
    one_hour: Optional[dict] = None,
    levels: Optional[dict] = None,
    historical_note: str = "",
) -> str:
    """Build a Type 3 market commentary — CLAUDE.md aligned.

    read_type: "REGIME_SHIFT" | "GEX_TREND" | "RTH_PULSE" | "MORNING_BRIEF" |
               "FINAL_15" | "QUIET_SUMMARY"

    Structure:
        Header → Regime box → Narrative → Key levels → EM map → Delta
    """
    now_pt = datetime.now(PT_TZ)
    time_str = now_pt.strftime("%I:%M %p").lstrip("0")
    price = snapshot_data.get("price", 0)
    gamma_bn = snapshot_data.get("gamma_bn", 0)
    cw = snapshot_data.get("call_wall", 0)
    pf = snapshot_data.get("put_floor", 0)
    spread = snapshot_data.get("spread", 0)
    gamma_desc = _gamma_english(gamma_bn)
    regime_plain = _REGIME_PLAIN.get(regime, regime.lower().replace("_", " ") if regime else "unknown")

    if read_type == "REGIME_SHIFT":
        return _format_regime_shift(
            old_regime, regime, snapshot_data, delta_line,
        )
    elif read_type == "GEX_TREND":
        return _format_gex_trend(
            snapshot_data, fifteen_min, one_hour, em_levels, delta_line,
        )
    elif read_type == "RTH_PULSE":
        return _format_rth_pulse_v2(
            snapshot_data, one_hour, advanced, levels, delta_line,
        )
    elif read_type == "MORNING_BRIEF":
        return _format_morning_brief_v2(
            snapshot_data, overnight, advanced, delta_line,
        )
    elif read_type == "FINAL_15":
        return _format_final_15_v2(
            snapshot_data, advanced, delta_line,
        )
    elif read_type == "QUIET_SUMMARY":
        return _format_quiet_summary_v2(
            snapshot_data, fifteen_min, one_hour, delta_line,
        )

    # Fallback
    lines = [
        f"\U0001f4ca MARKET READ | {time_str}",
        f"SPX {price:.0f} | {gamma_desc}",
        f"Regime: {regime_plain}",
    ]
    if delta_line:
        lines.append("")
        lines.append(delta_line)
    return "\n".join(lines)


# ── Type 3 Subtypes ──────────────────────────────────────────────────────

def _format_regime_shift(old_regime, new_regime, snap, delta_line):
    now_pt = datetime.now(PT_TZ)
    time_str = now_pt.strftime("%I:%M %p").lstrip("0")
    price = snap.get("price", 0)
    cw = snap.get("call_wall", 0)
    pf = snap.get("put_floor", 0)

    old_plain = _REGIME_PLAIN.get(old_regime, old_regime.lower().replace("_", " "))
    new_plain = _REGIME_PLAIN.get(new_regime, new_regime.lower().replace("_", " "))
    translation = _REGIME_SHIFT_TRANSLATIONS.get(
        (old_regime, new_regime),
        f"Shifted from {old_plain} to {new_plain}. Reassess your trades.",
    )

    pivot = round((cw + pf) / 2) if cw and pf else round(price)
    pivot_explain = {
        "UNCONTROLLED": "Below = dealers sell into you. Above = dealers buy into you.",
        "CONTROLLED_PIN": "Price gravitates here. Fade moves away from it.",
        "CONTROLLED_TREND": "Support if trending up, resistance if trending down.",
    }.get(new_regime, "Watch for a break above or below.")

    stance = _REGIME_STANCE.get(new_regime, "Stay nimble.")

    regime_box = _box("REGIME", [
        f"Shift: {old_plain} → {new_plain}",
        f"Gamma: {snap.get('gamma_bn', 0):+.1f}Bn",
        f"Walls: PF {pf} — CW {cw} ({cw - pf}pts)",
    ])

    lines = [
        f"\U0001f4ca MARKET READ | {time_str}",
        f"SPX {price:.0f}",
        "",
        regime_box,
        "",
        f"Translation: {translation}",
        f"Key: {pivot} is the pivot. {pivot_explain}",
        f"Stance: {stance}",
    ]
    if delta_line:
        lines.append("")
        lines.append(delta_line)
    return "\n".join(lines)


def _format_gex_trend(snap, fifteen_min, one_hour, em_levels, delta_line):
    price = snap.get("price", 0)
    gamma_bn = snap.get("gamma_bn", 0)

    # --- 15m line ---
    if fifteen_min:
        delta_15 = fifteen_min.get("price_delta", 0)
        g_bn = fifteen_min.get("gamma_bn", gamma_bn)
        g_delta = fifteen_min.get("gamma_delta_bn", 0)
        dist_floor = fifteen_min.get("dist_floor", 0)
        dist_ceil = fifteen_min.get("dist_ceil", 0)

        if abs(delta_15) < 3:
            read_15 = "Flat — no directional pressure"
        elif delta_15 > 0 and g_delta > 0:
            read_15 = "Bid with gamma support — momentum intact"
        elif delta_15 > 0 and g_delta < 0:
            read_15 = "Bid but gamma fading — rally may stall"
        elif delta_15 < 0 and g_delta < 0:
            read_15 = "Offered with gamma dropping — selling has legs"
        else:
            read_15 = "Offered but gamma rising — dip may find support"

        line_15 = (
            f"15m: {delta_15:+.0f}pts | G: {g_bn:+.1f}Bn ({g_delta:+.1f}) "
            f"| floor {dist_floor:.0f} away, ceiling {dist_ceil:.0f}\n"
            f"  → {read_15}"
        )
    else:
        line_15 = "15m: Insufficient data"

    # --- 1hr line ---
    if one_hour:
        price_range = one_hour.get("price_range", 0)
        slope = one_hour.get("slope", 0)
        oh_regime = one_hour.get("regime", "N/A")
        conf = one_hour.get("confidence", 0)

        if slope > 1:
            read_1h = "Gamma building — dealers adding support"
        elif slope < -1:
            read_1h = "Gamma decaying — dealer hedging weakening"
        elif price_range > 30:
            read_1h = "Wide range — volatile session, pick entries carefully"
        else:
            read_1h = "Tight range — controlled, fade the edges"

        line_1h = (
            f"1hr: {price_range:.0f}pt range | slope {slope:+.1f}Bn/hr "
            f"| {oh_regime} {conf:.0f}%\n"
            f"  → {read_1h}"
        )
    else:
        line_1h = "1hr: Insufficient data"

    # --- D1 line ---
    if em_levels:
        vix = em_levels.get("vix", 0)
        em_pts = em_levels.get("em_pts", 0)
        anchor = em_levels.get("anchor", 0)
        upper = em_levels.get("upper", 0)
        lower = em_levels.get("lower", 0)
        em_pos = (price - anchor) / em_pts if em_pts > 0 else 0

        if abs(em_pos) < 0.25:
            read_d1 = "Near anchor — no EM edge, wait for stretch"
        elif em_pos > 0.5:
            read_d1 = "Extended above anchor — fade zone approaching"
        elif em_pos < -0.5:
            read_d1 = "Extended below anchor — fade zone approaching"
        elif em_pos > 0:
            read_d1 = "Modest bid — room to run but fading odds"
        else:
            read_d1 = "Modest offer — room to drop but odds thinning"

        line_d1 = (
            f"D1: VIX {vix:.1f} | EM ±{em_pts:.0f}pts | "
            f"anchor {anchor:.0f} | pos {em_pos:+.2f}\n"
            f"  → {read_d1}"
        )
        pf_val = snap.get("put_floor", 0)
        cw_val = snap.get("call_wall", 0)
        line_levels = (
            f"Floor {pf_val} | Ceiling {cw_val} | "
            f"Spread {cw_val - pf_val}\n"
            f"EM: {lower:.0f} lower — {upper:.0f} upper"
        )
    else:
        line_d1 = "D1: EM data unavailable"
        line_levels = ""

    lines = [
        f"GEX TREND | SPX {price:.0f}",
        "",
        line_15,
        "",
        line_1h,
        "",
        line_d1,
    ]
    if line_levels:
        lines.append("")
        lines.append(line_levels)
    if delta_line:
        lines.append("")
        lines.append(delta_line)

    return "\n".join(lines)


def _format_rth_pulse_v2(snap, one_hour, advanced, levels, delta_line):
    now_pt = datetime.now(PT_TZ)
    time_str = now_pt.strftime("%I:%M %p").lstrip("0")
    price = snap.get("price", 0)
    gamma_bn = snap.get("gamma_bn", 0)
    cw = snap.get("call_wall", 0)
    pf = snap.get("put_floor", 0)
    spread = snap.get("spread", 0)
    gamma_desc = _gamma_english(gamma_bn)

    regime = one_hour.get("regime", "N/A") if one_hour else "N/A"
    regime_plain = _REGIME_PLAIN.get(regime, regime.lower().replace("_", " ") if isinstance(regime, str) else str(regime))

    # Gamma consistency override
    if gamma_bn > 3 and regime == "UNCONTROLLED":
        regime_plain = "positive gamma but regime unstable (trust the gamma)"
    elif gamma_bn < -2 and regime == "CONTROLLED_PIN":
        regime_plain = "negative gamma but pinning structure (trust the gamma)"

    # Outlook
    squeeze_prob = advanced.get("squeeze_prob", 0) if advanced else 0
    pin_prob = advanced.get("pin_prob", 0) if advanced else 0
    if squeeze_prob > 60:
        outlook = "Squeeze building — momentum trades favored"
    elif pin_prob > 60:
        outlook = "Pinning likely — fade the edges, sell premium"
    elif gamma_bn < -5:
        outlook = "Heavy negative gamma — expect fast moves both ways"
    elif gamma_bn > 5:
        outlook = "Strong positive gamma — moves will be slow, reversals quick"
    elif spread and spread < 20:
        outlook = "Tight range — wait for the breakout"
    else:
        outlook = "Mixed signals — trade the levels, respect the stops"

    status_box = _box("STATUS", [
        f"SPX {price:.0f} | {gamma_desc}",
        f"Regime: {regime_plain}",
        f"Range: {pf} support / {cw} resistance ({spread}pts)",
        f"Outlook: {outlook}",
    ])

    lines = [
        f"MARKET STATUS | {time_str}",
        "",
        status_box,
    ]

    if spread and spread < 20:
        lines.append("\u26a0 Spread < 20pts — levels unreliable, no trade setups")

    # Trade levels
    if levels:
        trade_lines = []
        if levels.get("long_valid"):
            ll = levels
            trade_lines.append(
                f"LONG: {ll.get('long_entry_low', 0):.0f}-{ll.get('long_entry_high', 0):.0f} "
                f"| Stop {ll.get('long_stop', 0):.0f} | Target {ll.get('long_target', 0):.0f} "
                f"| R:R {ll.get('long_rr', 0):.1f}"
            )
        if levels.get("short_valid"):
            sl = levels
            trade_lines.append(
                f"SHORT: {sl.get('short_entry_low', 0):.0f}-{sl.get('short_entry_high', 0):.0f} "
                f"| Stop {sl.get('short_stop', 0):.0f} | Target {sl.get('short_target', 0):.0f} "
                f"| R:R {sl.get('short_rr', 0):.1f}"
            )
        if trade_lines:
            lines.append("")
            lines.append(_box("LEVELS", trade_lines))

    if delta_line:
        lines.append("")
        lines.append(delta_line)

    return "\n".join(lines)


def _format_morning_brief_v2(snap, overnight, advanced, delta_line):
    now_pt = datetime.now(PT_TZ)
    gamma_bn = snap.get("gamma_bn", 0)
    gamma_desc = _gamma_english(gamma_bn)
    price = snap.get("price", 0)
    cw = snap.get("call_wall", 0)
    pf = snap.get("put_floor", 0)
    spread = snap.get("spread", 0)

    # Regime narrative
    _regime_sentences = {
        "CONTROLLED_PIN": "Dealers fully in control — expect tight range, fade the edges.",
        "CONTROLLED_TREND": "Dealers supporting the trend — follow momentum, buy dips.",
        "FRAGILE_CONTROL": "Balance is fragile — one catalyst breaks it. Wait for the trigger.",
        "UNCONTROLLED": "Negative gamma chaos — dealers amplify everything. Trade the extremes.",
        "MILD_POSITIVE": "Mild positive gamma — cushioned moves, expect chop near levels.",
        "NEUTRAL": "Gamma near zero — dealers not driving price. Trade the technicals.",
    }
    regime = snap.get("regime", "")
    narrative = _regime_sentences.get(regime, f"Current regime: {gamma_desc}.")

    lines = [
        f"MORNING BRIEF | {now_pt.strftime('%A %b %d')}",
        "",
    ]

    gex_box = _box("GEX REGIME", [
        f"SPX {price:.0f} | {gamma_desc} ({gamma_bn:+.1f}Bn)",
        f"Battlefield: {pf} support / {cw} resistance ({spread}pts)",
        narrative,
    ])
    lines.append(gex_box)

    if overnight:
        shift = overnight.get("gamma_shift_pct", 0)
        if shift > 30:
            gamma_read = "Gamma surged overnight — dealers shifting to support upside"
        elif shift < -30:
            gamma_read = "Gamma collapsed overnight — expect wilder swings today"
        elif shift > 10:
            gamma_read = "Gamma building — environment getting more controlled"
        elif shift < -10:
            gamma_read = "Gamma fading — dealers losing grip"
        else:
            gamma_read = "Gamma relatively unchanged overnight"

        overnight_lines = [gamma_read]
        cw_moved = overnight.get("cw_moved", 0)
        pf_moved = overnight.get("pf_moved", 0)
        if cw_moved != 0 or pf_moved != 0:
            overnight_lines.append(f"Walls shifted: CW {cw_moved:+d} / PF {pf_moved:+d}")
        range_fc = overnight.get("range_forecast", "")
        if range_fc:
            overnight_lines.append(f"Expected range: {range_fc}")

        lines.append("")
        lines.append(_box("OVERNIGHT", overnight_lines))

    if advanced:
        pivots = advanced.get("pivot_strikes", [])
        if pivots:
            pivots_str = ", ".join(str(p) for p in pivots)
            lines.append(f"Pivots to watch: {pivots_str}")

    lines.append("")
    lines.append(f"Key levels: {pf} (floor) / {cw} (ceiling)")

    if delta_line:
        lines.append("")
        lines.append(delta_line)

    return "\n".join(lines)


def _format_final_15_v2(snap, advanced, delta_line):
    price = snap.get("price", 0)
    gamma_bn = snap.get("gamma_bn", 0)
    gamma_desc = _gamma_english(gamma_bn)
    cw = snap.get("call_wall", 0)
    pf = snap.get("put_floor", 0)
    pin_prob = advanced.get("pin_prob", 0) if advanced else 0

    lines = [f"\U0001f4ca MARKET READ | 12:45 PM", ""]

    if pin_prob > 70:
        lines.extend([
            f"Into the close: Strong pin expected near {cw}",
            f"Translation: Dealers are locking price down. Don't fight the pin.",
            f"Key: {cw} is the magnet. Fade moves away from it.",
            f"Stance: Sell premium or take profits. Don't hold 0DTE into close.",
        ])
    elif pin_prob > 40:
        lines.extend([
            f"Into the close: Moderate pin potential around {cw}",
            f"Translation: Price may drift but unlikely to break out.",
            f"Key: Watch {pf}-{cw} range.",
            f"Stance: Tighten stops. Late moves are traps.",
        ])
    else:
        lines.extend([
            f"Into the close: Weak pin — expect volatility",
            f"Translation: {gamma_desc.capitalize()} means dealers aren't controlling the close.",
            f"Key: {pf} support / {cw} resistance",
            f"Stance: Fast in, fast out. Don't hold through the bell.",
        ])

    if delta_line:
        lines.append("")
        lines.append(delta_line)

    return "\n".join(lines)


def _format_quiet_summary_v2(snap, fifteen_min, one_hour, delta_line):
    time_str = snap.get("time_str", datetime.now(PT_TZ).strftime("%I:%M %p").lstrip("0"))
    price = snap.get("price", 0)
    gamma_bn = snap.get("gamma_bn", 0)
    gamma_desc = _gamma_english(gamma_bn)
    cw = snap.get("call_wall", 0)
    pf = snap.get("put_floor", 0)
    spread = snap.get("spread", 0)

    regime = one_hour.get("regime", "N/A") if one_hour else "N/A"
    regime_plain = _REGIME_PLAIN.get(regime, regime.lower().replace("_", " ") if isinstance(regime, str) else str(regime))

    # Momentum
    momentum = "Quiet — no significant moves"
    if fifteen_min:
        dp = fifteen_min.get("price_delta", 0)
        if abs(dp) > 10:
            momentum = f"Moving {'up' if dp > 0 else 'down'} {abs(dp):.0f}pts in last 15m"
        elif abs(dp) > 3:
            momentum = f"Drifting {'higher' if dp > 0 else 'lower'} ({dp:+.0f}pts / 15m)"
        else:
            momentum = "Chopping sideways"

    lines = [
        f"QUICK UPDATE | {time_str}",
        f"SPX {price:.0f} | {gamma_desc}",
        f"Regime: {regime_plain}",
        momentum,
        f"Range: {pf}-{cw} ({spread}pts)",
    ]

    if delta_line:
        lines.append("")
        lines.append(delta_line)

    return "\n".join(lines)
