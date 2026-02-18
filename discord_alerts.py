"""Discord webhook sender with cooldown management."""

from __future__ import annotations

import json
import logging
import os
import time as time_mod
from datetime import datetime, time
from typing import Dict, Optional

import requests
import yaml
import pytz

from signal_interpreter import Signal, SignalType

PT_TZ = pytz.timezone("US/Pacific")
log = logging.getLogger(__name__)

# Load config
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as f:
    CFG = yaml.safe_load(f)


class CooldownManager:
    """Track per-signal cooldowns with final-hour halving."""

    def __init__(self):
        self._cooldowns: Dict[str, float] = CFG["cooldowns"]
        self._last_fired: Dict[str, float] = {}
        self._state_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cooldown_state.json"
        )
        self._load_state()

    def _load_state(self):
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    self._last_fired = json.load(f)
            except Exception:
                self._last_fired = {}

    def _save_state(self):
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(self._last_fired, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save cooldown state: {e}")

    def _is_final_hour(self) -> bool:
        """Check if we're in the final hour (12:00-1:00 PM PT)."""
        now = datetime.now(PT_TZ).time()
        return time(12, 0) <= now < time(13, 0)

    def can_fire(self, signal_name: str) -> bool:
        """Check if a signal can fire based on cooldown."""
        base_cooldown = self._cooldowns.get(signal_name, 300)
        cooldown = base_cooldown / 2 if self._is_final_hour() else base_cooldown

        last = self._last_fired.get(signal_name, 0)
        return (time_mod.time() - last) >= cooldown

    def mark_fired(self, signal_name: str):
        """Record that a signal has fired."""
        self._last_fired[signal_name] = time_mod.time()
        self._save_state()

    def reset_daily(self):
        """Reset all cooldowns for a new trading day."""
        self._last_fired = {}
        self._save_state()


class DiscordAlertSender:
    """Send formatted alerts to Discord webhooks."""

    def __init__(self):
        self._webhooks: Dict[str, str] = {}
        self._cooldown = CooldownManager()
        self._load_webhooks()

    def _load_webhooks(self):
        """Load webhook URLs from environment variables."""
        default_webhook = os.getenv(
            "DISCORD_WEBHOOK",
            "https://discord.com/api/webhooks/1470940051977408564/WpsNBFm5buSH2H-5M5hL32DVWrVnVmWSR0ANzm4Jzik1w3nxjeW_OZ8DKLKWHje8CDB8",
        )
        self._webhooks = {
            "gex_ops": os.getenv("DISCORD_GEX_OPS_WEBHOOK", default_webhook),
            "market_pulse": os.getenv("DISCORD_MARKET_PULSE_WEBHOOK", default_webhook),
            "daily_intel": os.getenv("DISCORD_DAILY_INTEL_WEBHOOK", default_webhook),
            "trading_summary": os.getenv(
                "DISCORD_TRADING_SUMMARY_WEBHOOK",
                os.getenv("DISCORD_MARKET_PULSE_WEBHOOK", default_webhook),
            ),
            "intraday_fast": os.getenv(
                "DISCORD_INTRADAY_FAST_WEBHOOK",
                os.getenv("DISCORD_MARKET_PULSE_WEBHOOK", default_webhook),
            ),
        }

        if not any(self._webhooks.values()):
            log.warning("No Discord webhook URLs configured.")

    def send_signal(self, signal: Signal) -> bool:
        """Send a signal to the appropriate Discord channel.

        Returns True if sent, False if cooled down or failed.
        """
        signal_name = signal.signal_type.name.lower()

        if not self._cooldown.can_fire(signal_name):
            log.debug(f"Signal {signal_name} on cooldown, skipping.")
            return False

        webhook_url = self._webhooks.get(signal.channel)
        if not webhook_url:
            log.warning(f"No webhook configured for channel '{signal.channel}'")
            return False

        success = self._send_webhook(webhook_url, signal)
        if success:
            self._cooldown.mark_fired(signal_name)

        return success

    def _send_webhook(self, url: str, signal: Signal) -> bool:
        """Send a Discord webhook message."""
        ts = signal.timestamp
        if ts.tzinfo is None:
            ts = PT_TZ.localize(ts)
        else:
            ts = ts.astimezone(PT_TZ)
        ts_line = f"Timestamp: {ts.strftime('%Y-%m-%d %I:%M %p PT')}"

        content = signal.message
        if not content.startswith("Timestamp:"):
            content = f"{ts_line}\n{content}"

        payload = {
            "content": content,
            "username": "GEX Engine",
        }

        # Discord has a 2000 char limit for content
        if len(payload["content"]) > 1990:
            payload["content"] = payload["content"][:1987] + "..."

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )

            if resp.status_code == 204:
                log.info(f"Sent {signal.signal_type.name} to {signal.channel}")
                return True
            elif resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 5)
                log.warning(f"Discord rate limited. Retry after {retry_after}s")
                return False
            else:
                log.error(f"Discord webhook failed: {resp.status_code} {resp.text}")
                return False

        except requests.RequestException as e:
            log.error(f"Discord webhook error: {e}")
            return False

    def send_batch(self, signals: list[Signal]) -> int:
        """Send multiple signals, respecting cooldowns and priority order.

        Returns count of signals actually sent.
        """
        sent = 0
        for signal in sorted(signals, key=lambda s: s.priority):
            if self.send_signal(signal):
                sent += 1
        return sent

    @property
    def cooldown(self) -> CooldownManager:
        return self._cooldown
