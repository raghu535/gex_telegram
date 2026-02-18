"""Run hourly recap on demand (for Task Scheduler)."""

from __future__ import annotations

from datetime import datetime
import os
import yaml

import pytz

from gex_db import get_recent_snapshots
from recap import build_hourly_recap_message, is_rth_time
from signal_interpreter import Signal, SignalType
from discord_alerts import DiscordAlertSender

PT_TZ = pytz.timezone("US/Pacific")


def main():
    now = datetime.now(PT_TZ)
    if not is_rth_time(now):
        return

    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    window_min = cfg.get("scheduled", {}).get("hourly_recap_window_min", 5)

    # Only run near the top of the hour (first 5 minutes)
    if now.minute >= window_min:
        return

    snapshots_1h = get_recent_snapshots(60, session_tag="RTH")
    if len(snapshots_1h) < 2:
        return

    msg = build_hourly_recap_message(snapshots_1h)
    signal = Signal(
        signal_type=SignalType.HOURLY_RECAP,
        title="Hourly Recap",
        message=msg,
        channel="daily_intel",
        priority=SignalType.HOURLY_RECAP,
    )

    DiscordAlertSender().send_signal(signal)


if __name__ == "__main__":
    main()
