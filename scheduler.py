"""Scheduled events: morning brief, EOD summary, final 15."""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Callable, Optional

import pytz
import yaml
import os

PT_TZ = pytz.timezone("US/Pacific")
log = logging.getLogger(__name__)

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as f:
    CFG = yaml.safe_load(f)


class ScheduledEvent:
    """A time-triggered event that fires once per day."""

    def __init__(self, name: str, trigger_time: str, callback: Callable):
        self.name = name
        h, m = trigger_time.split(":")
        self.trigger_time = time(int(h), int(m))
        self.callback = callback
        self._fired_today: Optional[str] = None

    def check(self, now_pt: Optional[datetime] = None) -> bool:
        """Check if this event should fire. Returns True if it fired."""
        if now_pt is None:
            now_pt = datetime.now(PT_TZ)

        today = now_pt.strftime("%Y-%m-%d")
        if self._fired_today == today:
            return False

        current_time = now_pt.time()
        # Fire if we're within 2 minutes of the trigger time
        trigger_minutes = self.trigger_time.hour * 60 + self.trigger_time.minute
        current_minutes = current_time.hour * 60 + current_time.minute

        if 0 <= (current_minutes - trigger_minutes) < 2:
            log.info(f"Firing scheduled event: {self.name}")
            try:
                self.callback()
                self._fired_today = today
                return True
            except Exception as e:
                log.error(f"Scheduled event {self.name} failed: {e}")
                return False

        return False

    def reset(self):
        """Reset for a new day."""
        self._fired_today = None


class Scheduler:
    """Manage scheduled events."""

    def __init__(self):
        self._events: list[ScheduledEvent] = []

    def add_event(self, name: str, trigger_time: str, callback: Callable):
        self._events.append(ScheduledEvent(name, trigger_time, callback))

    def check_all(self, now_pt: Optional[datetime] = None) -> list[str]:
        """Check all events and return names of fired events."""
        fired = []
        for event in self._events:
            if event.check(now_pt):
                fired.append(event.name)
        return fired

    def reset_daily(self):
        """Reset all events for a new day."""
        for event in self._events:
            event.reset()
