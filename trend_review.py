"""Nightly trend dashboard review and drop correlation summary."""

from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
from collections import defaultdict

import pytz
import yaml

from gex_db import get_conn, get_eod_snapshot
from gex_parser import GEXSnapshot
from trend_engine import compute_trend_dashboard, compute_overnight_drift
from signal_interpreter import SignalInterpreter, SignalType

PT = pytz.timezone("US/Pacific")

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as _f:
    _CFG = yaml.safe_load(_f)

TREND_REVIEW_CFG = _CFG.get("trend_review", {})


def _load_snapshots(days: int) -> dict[str, list[GEXSnapshot]]:
    cutoff = (datetime.now(PT) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM gex_snapshots WHERE session_tag='RTH' AND date_pt >= ? "
        "ORDER BY timestamp_pt ASC",
        (cutoff,),
    ).fetchall()
    conn.close()

    by_date: dict[str, list[GEXSnapshot]] = defaultdict(list)
    for r in rows:
        snap = GEXSnapshot(
            timestamp_pt=datetime.fromtimestamp(r["timestamp_pt"], tz=PT),
            session_tag=r["session_tag"],
            curr_price=r["curr_price"] or 0.0,
            net_gamma=r["net_gamma"] or 0,
            total_call_gamma=r["total_call_gamma"] or 0,
            total_put_gamma=r["total_put_gamma"] or 0,
            call_wall=r["call_wall"] or 0,
            put_floor=r["put_floor"] or 0,
            top5_calls=json.loads(r["top5_calls"]) if r["top5_calls"] else [],
            top5_puts=json.loads(r["top5_puts"]) if r["top5_puts"] else [],
            raw_text="",
        )
        by_date[r["date_pt"]].append(snap)
    return by_date


def _find_drop_events(
    snaps: list[GEXSnapshot],
    drop_pts: float,
    drop_window_min: int,
    separation_min: int,
) -> list[datetime]:
    events = []
    i = 0
    while i < len(snaps):
        start = snaps[i]
        end_ts = start.timestamp_pt + timedelta(minutes=drop_window_min)
        min_price = start.curr_price
        for j in range(i + 1, len(snaps)):
            if snaps[j].timestamp_pt > end_ts:
                break
            min_price = min(min_price, snaps[j].curr_price)
        if start.curr_price - min_price >= drop_pts:
            events.append(start.timestamp_pt)
            skip_ts = start.timestamp_pt + timedelta(minutes=separation_min)
            while i < len(snaps) and snaps[i].timestamp_pt < skip_ts:
                i += 1
            continue
        i += 1
    return events


def _trend_alerts(snaps: list[GEXSnapshot]) -> list[tuple[datetime, dict]]:
    interpreter = SignalInterpreter()
    alerts: list[tuple[datetime, dict]] = []

    # Get previous day EOD for overnight context
    date = snaps[0].timestamp_pt.strftime("%Y-%m-%d")
    prev_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    prev_eod = get_eod_snapshot(prev_date)

    for i, snap in enumerate(snaps):
        lookback_1h = [
            s for s in snaps[: i + 1]
            if (snap.timestamp_pt - s.timestamp_pt).total_seconds() <= 3600
        ]
        overnight = compute_overnight_drift(prev_eod, snap) if prev_eod else None
        trend = compute_trend_dashboard(lookback_1h, overnight)

        signals = interpreter.evaluate_all(
            snapshot=snap,
            fifteen_min=None,
            one_hour=None,
            overnight=None,
            advanced=None,
            trend=trend,
            snapshots_1h=lookback_1h if len(lookback_1h) >= 2 else None,
        )

        for sig in signals:
            if sig.signal_type == SignalType.TREND_DASHBOARD:
                alerts.append((snap.timestamp_pt, sig.metadata))

    return alerts


def run_trend_review(
    days: int | None = None,
    drop_pts: float | None = None,
    drop_window_min: int | None = None,
    separation_min: int | None = None,
    lead_min: int | None = None,
    output_dir: str | None = None,
) -> str:
    days = days or int(TREND_REVIEW_CFG.get("days", 10))
    drop_pts = drop_pts or float(TREND_REVIEW_CFG.get("drop_pts", 30))
    drop_window_min = drop_window_min or int(TREND_REVIEW_CFG.get("drop_window_min", 60))
    separation_min = separation_min or int(TREND_REVIEW_CFG.get("separation_min", 60))
    lead_min = lead_min or int(TREND_REVIEW_CFG.get("lead_min", 60))
    output_dir = output_dir or TREND_REVIEW_CFG.get("output_dir", "reports")

    by_date = _load_snapshots(days)
    if not by_date:
        return "No data for trend review."

    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append(f"Trend Review ({datetime.now(PT).strftime('%Y-%m-%d %I:%M %p PT')})")
    lines.append(f"Window: last {days} days | Drop: {drop_pts} pts in {drop_window_min}m | Lead: {lead_min}m")
    lines.append("")
    lines.append("Date | Alerts | Bearish Alerts | Heartbeats | Drop Events | Events With Lead Alert")

    total_alerts = 0
    total_bear = 0
    total_hb = 0
    total_events = 0
    total_hits = 0

    for date in sorted(by_date.keys()):
        snaps = sorted(by_date[date], key=lambda s: s.timestamp_pt)
        if not snaps:
            continue

        alerts = _trend_alerts(snaps)
        bearish_alerts = [a for a in alerts if a[1].get("bias") == "BEARISH"]
        hb_count = sum(1 for _, meta in alerts if meta.get("heartbeat"))

        events = _find_drop_events(snaps, drop_pts, drop_window_min, separation_min)
        hits = 0
        for evt in events:
            lead_start = evt - timedelta(minutes=lead_min)
            if any(lead_start <= ts <= evt for ts, meta in bearish_alerts):
                hits += 1

        lines.append(
            f"{date} | {len(alerts)} | {len(bearish_alerts)} | {hb_count} | "
            f"{len(events)} | {hits}"
        )

        total_alerts += len(alerts)
        total_bear += len(bearish_alerts)
        total_hb += hb_count
        total_events += len(events)
        total_hits += hits

    lines.append("")
    if total_events:
        hit_rate = total_hits / total_events * 100
        lines.append(f"Event hit rate: {total_hits}/{total_events} ({hit_rate:.1f}%)")
    lines.append(f"Total alerts: {total_alerts} (bearish {total_bear}, heartbeat {total_hb})")

    output_path = os.path.join(
        output_dir,
        f"trend_review_{datetime.now(PT).strftime('%Y-%m-%d')}.txt",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


if __name__ == "__main__":
    path = run_trend_review()
    print(f"Wrote: {path}")
