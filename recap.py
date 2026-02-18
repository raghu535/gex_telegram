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

    lines = [
        f"**HOURLY RECAP** | {earliest.timestamp_pt.strftime('%I:%M %p')} - {latest.timestamp_pt.strftime('%I:%M %p')} PT",
        f"Snapshots: {len(snapshots_1h)}",
        f"Price: {earliest.curr_price:.2f} -> {latest.curr_price:.2f} | Range {min(prices):.2f}-{max(prices):.2f}",
        f"Net Gamma: {earliest.net_gamma/1e9:+.2f} -> {latest.net_gamma/1e9:+.2f} Bn | Range {min(gammas)/1e9:+.2f} to {max(gammas)/1e9:+.2f} Bn",
        "",
    ]

    for label, w in [("5m", w5), ("15m", w15), ("30m", w30), ("60m", w60)]:
        if w:
            lines.append(
                f"{label}: dPx {w['price_delta']:+.2f} | dG {w['gamma_delta_bn']:+.2f} Bn | snaps {w['count']}"
            )

    lines += [
        "",
        f"Call Wall: {latest.call_wall} | Put Floor: {latest.put_floor} | Spread: {latest.spread} pts",
        f"Top Calls: {', '.join([f'{s}({g/1e9:+.2f}Bn)' for s, g in latest.top5_calls[:3]])}",
        f"Top Puts:  {', '.join([f'{s}({g/1e9:+.2f}Bn)' for s, g in latest.top5_puts[:3]])}",
        "",
    ]

    if fifteen_min:
        lines.append(
            f"15m: dPx {fifteen_min.price_delta:+.2f} | dG {fifteen_min.net_gamma_delta/1e9:+.2f} Bn | "
            f"Ratio {fifteen_min.gamma_ratio:.2f} ({fifteen_min.gamma_ratio_trend})"
        )
    if one_hour:
        lines.append(
            f"1h: Regime {one_hour.regime.value} ({one_hour.regime_confidence:.0f}%) | "
            f"Slope {one_hour.net_gamma_slope:+.2f} Bn/hr | Accel {one_hour.gamma_accel:+.2f}"
        )

    if advanced:
        lines.append(
            f"Advanced: GCI {advanced.gci:.0f}% ({advanced.gci_label}) | "
            f"Asym {advanced.gamma_asymmetry:.2f} ({advanced.asymmetry_label})"
        )
        lines.append(
            f"          Squeeze {advanced.squeeze_probability:.0f}% | Pin {advanced.pin_probability:.0f}%"
        )

    if trend:
        label = "SMELL" if trend.smell else "TREND"
        lines.append(f"Trend: {trend.bias} ({label}) | Score {trend.score}")
        lines.append(f"Support targets: {fmt_targets(trend.support_targets)}")
        lines.append(f"Resistance targets: {fmt_targets(trend.resistance_targets)}")
        if trend.notes:
            lines.append("Notes:")
            for note in trend.notes[:6]:
                lines.append(f"- {note}")

    return "\n".join(lines)
