"""Delivery watchdog: logs every alert sent, detects missed scheduled events,
and fires mid-day / EOD health reports to gex_engine."""

from __future__ import annotations

import logging
import time as time_mod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional

import pytz
import yaml
import os

from signal_interpreter import Signal, SignalType

PT_TZ = pytz.timezone("US/Pacific")
log = logging.getLogger(__name__)

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as f:
    CFG = yaml.safe_load(f)

_wd_cfg = CFG.get("delivery_watchdog", {})


@dataclass
class DeliveryRecord:
    signal_type: str
    channel: str
    destination: str  # "discord" / "telegram"
    success: bool
    timestamp: datetime
    cooldown_skipped: bool = False


class DeliveryLog:
    """In-memory log of all delivery attempts, auto-cleared daily."""

    def __init__(self):
        self._records: list[DeliveryRecord] = []
        self._date: Optional[str] = None

    def _check_date(self):
        today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
        if self._date != today:
            self._records.clear()
            self._date = today

    def record(
        self,
        signal_type: str,
        channel: str,
        destination: str,
        success: bool,
        cooldown_skipped: bool = False,
    ):
        self._check_date()
        self._records.append(
            DeliveryRecord(
                signal_type=signal_type,
                channel=channel,
                destination=destination,
                success=success,
                timestamp=datetime.now(PT_TZ),
                cooldown_skipped=cooldown_skipped,
            )
        )

    def summary(self) -> dict:
        """Return counts: total, sent_ok, failed, cooldown, by_channel, by_type."""
        self._check_date()
        out = {
            "total": len(self._records),
            "discord": {"sent": 0, "failed": 0, "cooldown": 0},
            "telegram": {"sent": 0, "failed": 0, "cooldown": 0},
            "by_channel": defaultdict(list),
            "by_type": defaultdict(int),
        }
        for r in self._records:
            dest = out.get(r.destination)
            if dest is None:
                continue
            if r.cooldown_skipped:
                dest["cooldown"] += 1
            elif r.success:
                dest["sent"] += 1
            else:
                dest["failed"] += 1
            if r.success:
                out["by_channel"][r.channel].append(r.signal_type)
                out["by_type"][r.signal_type] += 1
        return out

    def failed_signals(self) -> list[DeliveryRecord]:
        """Return real failures (not cooldown skips)."""
        self._check_date()
        return [r for r in self._records if not r.success and not r.cooldown_skipped]

    def reset(self):
        self._records.clear()
        self._date = None


class DeliveryWatchdog:
    """Monitors alert delivery, detects missed events, fires health reports."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.log = DeliveryLog()
        self._orig_discord_send = None
        self._orig_telegram_send = None
        self._missed_today: set[str] = set()
        self._midday_fired: Optional[str] = None
        self._eod_fired: Optional[str] = None
        self._enabled = _wd_cfg.get("enabled", True)
        self._threshold_min = int(_wd_cfg.get("missed_event_threshold_min", 5))
        self._midday_time = _parse_time(_wd_cfg.get("midday_report_time", "11:00"))
        self._eod_time = _parse_time(_wd_cfg.get("eod_report_time", "13:15"))
        self._alert_channel = _wd_cfg.get("alert_channel", "gex_engine")
        # Idle detection
        self._last_successful_delivery: float = time_mod.time()
        self._idle_threshold_min = int(_wd_cfg.get("idle_threshold_min", 20))
        self._idle_cooldown_sec = int(_wd_cfg.get("idle_cooldown_sec", 1800))
        self._last_idle_alert: float = 0
        self._idle_rth_start = time(7, 0)   # skip first 30min of RTH (6:30-7:00)
        self._idle_rth_end = time(12, 45)

    def wrap_senders(self):
        """Monkey-patch discord/telegram send_signal to log every call."""
        pipeline = self.pipeline
        self._orig_discord_send = pipeline.discord.send_signal
        self._orig_telegram_send = pipeline.telegram.send_signal

        wd = self  # capture for closures

        def discord_wrapper(signal: Signal) -> bool:
            signal_name = signal.signal_type.name.lower()
            could_fire = pipeline.discord.cooldown.can_fire(signal_name)
            result = wd._orig_discord_send(signal)
            # If send returned False and cooldown said no BEFORE the call, it's a cooldown skip.
            # If send returned False but cooldown was clear, it's a real failure.
            cooldown_skipped = not result and not could_fire
            wd.log.record(
                signal.signal_type.name,
                signal.channel,
                "discord",
                result,
                cooldown_skipped=cooldown_skipped,
            )
            if result:
                wd._last_successful_delivery = time_mod.time()
            return result

        def telegram_wrapper(signal: Signal) -> bool:
            if not pipeline.telegram.is_eligible(signal):
                return False
            signal_name = signal.signal_type.name.lower()
            could_fire = pipeline.telegram.cooldown.can_fire(signal_name)
            result = wd._orig_telegram_send(signal)
            cooldown_skipped = not result and not could_fire
            wd.log.record(
                signal.signal_type.name,
                signal.channel,
                "telegram",
                result,
                cooldown_skipped=cooldown_skipped,
            )
            if result:
                wd._last_successful_delivery = time_mod.time()
            return result

        pipeline.discord.send_signal = discord_wrapper
        pipeline.telegram.send_signal = telegram_wrapper
        log.info("DeliveryWatchdog: senders wrapped")

    # ------------------------------------------------------------------
    # Scheduled-event miss detection
    # ------------------------------------------------------------------

    def check_scheduled_events(self):
        """Check if any scheduled event is overdue by >threshold minutes."""
        if not self._enabled:
            return
        now_pt = datetime.now(PT_TZ)
        today = now_pt.strftime("%Y-%m-%d")
        current_min = now_pt.hour * 60 + now_pt.minute

        for event in self.pipeline.scheduler._events:
            trigger_min = event.trigger_time.hour * 60 + event.trigger_time.minute
            overdue = current_min - trigger_min
            if overdue > self._threshold_min and event._fired_today != today:
                if event.name not in self._missed_today:
                    self._missed_today.add(event.name)
                    self._fire_missed_alert(event, now_pt)

    def check_signal_flow(self):
        """Detect if the engine has gone idle (no successful deliveries) during RTH."""
        if not self._enabled:
            return
        now_pt = datetime.now(PT_TZ)
        current_time = now_pt.time()

        # Only check during active RTH window (7:00 AM - 12:45 PM PT)
        if not (self._idle_rth_start <= current_time <= self._idle_rth_end):
            return
        # Weekdays only
        if now_pt.weekday() >= 5:
            return

        now_ts = time_mod.time()
        idle_min = (now_ts - self._last_successful_delivery) / 60

        if idle_min >= self._idle_threshold_min:
            # Cooldown: don't spam idle alerts
            if (now_ts - self._last_idle_alert) < self._idle_cooldown_sec:
                return
            self._last_idle_alert = now_ts
            msg = (
                f"\u26a0\ufe0f ENGINE IDLE: No signals delivered in {idle_min:.0f}+ minutes\n"
                f"Check data feed, Telegram listener, or GEX engine health.\n"
                f"Time: {now_pt.strftime('%I:%M %p PT')}"
            )
            log.warning(f"DeliveryWatchdog: {msg}")
            self._send_engine_alert(msg)

    def _fire_missed_alert(self, event, now_pt: datetime):
        trigger_str = event.trigger_time.strftime("%I:%M %p PT")
        now_str = now_pt.strftime("%I:%M %p PT")
        msg = (
            f"\u26a0\ufe0f MISSED: {event.name}\n"
            f"Expected: {trigger_str} | Now: {now_str}\n"
            f"Check data feed or callback errors in logs."
        )
        log.warning(f"DeliveryWatchdog: {msg}")
        self._send_engine_alert(msg)

    # ------------------------------------------------------------------
    # Health reports
    # ------------------------------------------------------------------

    def check_health_reports(self):
        if not self._enabled:
            return
        now_pt = datetime.now(PT_TZ)
        today = now_pt.strftime("%Y-%m-%d")
        current_min = now_pt.hour * 60 + now_pt.minute

        midday_min = self._midday_time.hour * 60 + self._midday_time.minute
        if self._midday_fired != today and 0 <= (current_min - midday_min) < 2:
            self._midday_fired = today
            self._fire_health_report("Mid-Day", now_pt)

        eod_min = self._eod_time.hour * 60 + self._eod_time.minute
        if self._eod_fired != today and 0 <= (current_min - eod_min) < 2:
            self._eod_fired = today
            self._fire_health_report("EOD", now_pt)

    def _fire_health_report(self, report_type: str, now_pt: datetime):
        s = self.log.summary()
        today = now_pt.strftime("%Y-%m-%d")

        lines = [
            "\u2501\u2501\u2501 ENGINE HEALTH \u2501\u2501\u2501",
            f"\U0001f4cb {report_type} Report | {today}",
            "",
            "Delivery:",
        ]

        for dest in ("discord", "telegram"):
            d = s[dest]
            parts = [f"{d['sent']} sent"]
            if d["failed"]:
                parts.append(f"{d['failed']} failed")
            if d["cooldown"]:
                parts.append(f"{d['cooldown']} cooldown")
            lines.append(f"  {dest.capitalize()}: {' / '.join(parts)}")

        # Scheduled events
        lines.append("")
        lines.append("Scheduled:")
        for event in self.pipeline.scheduler._events:
            fired = event._fired_today == today
            icon = "\u2705" if fired else "\u23f3"
            t_str = event.trigger_time.strftime("%H:%M")
            lines.append(f"  {icon} {event.name:<20s} {t_str}")

        # Routing breakdown
        by_ch = s["by_channel"]
        if by_ch:
            lines.append("")
            lines.append("Routing:")
            for ch, types in sorted(by_ch.items()):
                type_counts = defaultdict(int)
                for t in types:
                    type_counts[t] += 1
                detail = ", ".join(f"{t} x{c}" for t, c in type_counts.items())
                lines.append(f"  {ch}: {len(types)} ({detail})")

        # Failures
        failures = self.log.failed_signals()
        lines.append("")
        if failures:
            lines.append(f"Failures ({len(failures)}):")
            for f in failures[:5]:
                lines.append(f"  {f.signal_type} -> {f.destination}/{f.channel}")
        else:
            lines.append("Failures: None")

        lines.append("\u2501" * 18)

        msg = "\n".join(lines)
        log.info(f"DeliveryWatchdog: firing {report_type} health report")
        self._send_engine_alert(msg)

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self):
        self.log.reset()
        self._missed_today.clear()
        self._midday_fired = None
        self._eod_fired = None
        log.info("DeliveryWatchdog: daily reset")

    # ------------------------------------------------------------------
    # Internal: send to gex_engine via original (unwrapped) discord
    # ------------------------------------------------------------------

    def _send_engine_alert(self, message: str):
        """Send a watchdog alert to gex_engine using the original discord sender."""
        signal = Signal(
            signal_type=SignalType.DATA_RESTORED,  # reuse low-priority type
            title="Delivery Watchdog",
            message=message,
            priority=99,
            channel=self._alert_channel,
            timestamp=datetime.now(PT_TZ),
        )
        try:
            if self._orig_discord_send:
                self._orig_discord_send(signal)
            else:
                self.pipeline.discord.send_signal(signal)
        except Exception as e:
            log.error(f"DeliveryWatchdog: failed to send alert: {e}")


def _parse_time(t_str: str) -> time:
    h, m = t_str.split(":")
    return time(int(h), int(m))
