"""Shared recap helpers for hourly summaries."""

from __future__ import annotations

from datetime import datetime, timedelta, time

import pytz

from gex_parser import GEXSnapshot
from trend_engine import (
    compute_fifteen_min_metrics,
    compute_one_hour_metrics,
    compute_advanced_metrics,
    compute_trend_dashboard,
)

PT_TZ = pytz.timezone("US/Pacific")


def is_rth_time(dt: datetime) -> bool:
    t = dt.astimezone(PT_TZ).time()
    return time(6, 30) <= t < time(13, 0)


def build_hourly_recap_message(snapshots_1h: list[GEXSnapshot]) -> str:
    earliest = snapshots_1h[0]
    latest = snapshots_1h[-1]

    prices = [s.curr_price for s in snapshots_1h]
    gammas = [s.net_gamma for s in snapshots_1h]

    def window_delta(minutes: int):
        cutoff = latest.timestamp_pt - timedelta(minutes=minutes)
        window = [s for s in snapshots_1h if s.timestamp_pt >= cutoff]
        if len(window) < 2:
            return None
        start = window[0]
        return {
            "count": len(window),
            "price_delta": latest.curr_price - start.curr_price,
            "gamma_delta_bn": (latest.net_gamma - start.net_gamma) / 1e9,
        }

    w5 = window_delta(5)
    w15 = window_delta(15)
    w30 = window_delta(30)
    w60 = window_delta(60)

    snaps_15m = [
        s for s in snapshots_1h
        if s.timestamp_pt >= latest.timestamp_pt - timedelta(minutes=15)
    ]
    fifteen_min = compute_fifteen_min_metrics(snaps_15m) if len(snaps_15m) >= 2 else None
    one_hour = compute_one_hour_metrics(snapshots_1h, fifteen_min) if fifteen_min and len(snapshots_1h) >= 3 else None
    advanced = compute_advanced_metrics(snapshots_1h) if len(snapshots_1h) >= 2 else None
    trend = compute_trend_dashboard(snapshots_1h, None)

    def fmt_targets(targets: list[tuple[int, float]]) -> str:
        if not targets:
            return "n/a"
        return ", ".join([f"{s} ({g:+.2f}Bn)" for s, g in targets])

    # Plain English regime translations
    _REGIME_PLAIN = {
        "CONTROLLED_PIN": "pinning regime (price stuck near key strikes)",
        "CONTROLLED_TREND": "controlled trend (dealers hedge in your favor)",
        "FRAGILE_CONTROL": "fragile balance (could break either way)",
        "UNCONTROLLED": "negative gamma chaos (dealers amplify moves)",
    }

    def _gamma_english(gbn: float) -> str:
        if gbn < -10: return "extreme negative gamma"
        elif gbn < -5: return "heavy negative gamma"
        elif gbn < -2: return "negative gamma"
        elif gbn > 8: return "strong positive gamma (ceiling)"
        elif gbn > 3: return "positive gamma"
        return "neutral gamma"

    price_change = latest.curr_price - earliest.curr_price
    gamma_change_bn = (latest.net_gamma - earliest.net_gamma) / 1e9
    price_range = max(prices) - min(prices)
    gamma_bn = latest.net_gamma / 1e9
    time_range = f"{earliest.timestamp_pt.strftime('%I:%M %p').lstrip('0')} - {latest.timestamp_pt.strftime('%I:%M %p').lstrip('0')} PT"

    # What happened summary
    if abs(price_change) > 20:
        what = f"SPX moved {'+' if price_change > 0 else ''}{price_change:.0f} pts ({'up' if price_change > 0 else 'down'})"
    elif abs(price_change) > 5:
        what = f"SPX drifted {'+' if price_change > 0 else ''}{price_change:.0f} pts"
    else:
        what = f"SPX chopped in a {price_range:.0f} pt range"

    # Gamma context
    if abs(gamma_change_bn) > 3:
        gamma_what = f"Gamma shifted {'up' if gamma_change_bn > 0 else 'down'} {abs(gamma_change_bn):.1f} Bn"
    elif abs(gamma_change_bn) > 1:
        gamma_what = f"Gamma {'built' if gamma_change_bn > 0 else 'decayed'} {abs(gamma_change_bn):.1f} Bn"
    else:
        gamma_what = "Gamma stable"

    regime_str = one_hour.regime.value if one_hour else "N/A"
    regime_plain = _REGIME_PLAIN.get(regime_str, regime_str.lower().replace("_", " "))

    lines = [
        f"HOUR IN REVIEW | {time_range}",
        f"{what} | {gamma_what}",
        f"Now: SPX {latest.curr_price:.0f} | {_gamma_english(gamma_bn)}",
        f"Environment: {regime_plain}",
        f"Range: {latest.put_floor}-{latest.call_wall} ({latest.spread} pts)",
    ]

    # Squeeze/pin outlook
    if advanced:
        if advanced.squeeze_probability > 60:
            lines.append(f"Alert: {advanced.squeeze_probability:.0f}% squeeze probability — breakout watch")
        elif advanced.pin_probability > 60:
            lines.append(f"Alert: {advanced.pin_probability:.0f}% pin probability — expect range contraction")

    # Trend
    if trend:
        emoji = "\U0001f7e2" if trend.bias == "BULLISH" else "\U0001f534" if trend.bias == "BEARISH" else ""
        conviction = "confirmed" if not trend.smell else "early signal"
        lines.append(f"Trend: {emoji} {trend.bias} ({conviction}, score {trend.score})")

    # Key support/resistance
    if trend and trend.support_targets:
        sup = ", ".join(str(s) for s, _ in trend.support_targets[:3])
        lines.append(f"Support: {sup}")
    if trend and trend.resistance_targets:
        res = ", ".join(str(s) for s, _ in trend.resistance_targets[:3])
        lines.append(f"Resistance: {res}")

    return "\n".join(lines)
