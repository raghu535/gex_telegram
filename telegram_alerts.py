"""Telegram alert sender for high-priority GEX signals."""

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

# Daily SPY RSI cache — refreshed once per trading day
_rsi_cache: Dict[str, Optional[float]] = {"date": "", "rsi": None}

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as f:
    CFG = yaml.safe_load(f)

# Signal types eligible for Telegram delivery
TELEGRAM_ELIGIBLE = {
    # High-priority intraday signals
    SignalType.BOUNCE_ZONE,
    SignalType.REGIME_SHIFT,
    SignalType.WALL_BREACH,
    SignalType.GAMMA_SQUEEZE,
    SignalType.GAMMA_SNAP,
    # Once-per-day scheduled alerts
    SignalType.MORNING_CHECKLIST,
    SignalType.FINAL_15,
    SignalType.GAP_ALERT,
    # Operational
    SignalType.DATA_LAG,
    SignalType.DATA_RESTORED,
}


class TelegramCooldownManager:
    """Separate cooldown tracker for Telegram (longer than Discord)."""

    def __init__(self):
        tg_cfg = CFG.get("telegram_alerts", {})
        self._cooldowns: Dict[str, int] = tg_cfg.get("cooldowns", {})
        self._last_fired: Dict[str, float] = {}
        self._state_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "telegram_cooldown_state.json"
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
            log.error(f"Failed to save Telegram cooldown state: {e}")

    def _is_final_hour(self) -> bool:
        now = datetime.now(PT_TZ).time()
        return time(12, 0) <= now < time(13, 0)

    def can_fire(self, signal_name: str, sub_key: str = "") -> bool:
        key = f"{signal_name}:{sub_key}" if sub_key else signal_name
        base_cooldown = self._cooldowns.get(key, self._cooldowns.get(signal_name, 900))
        cooldown = base_cooldown / 2 if self._is_final_hour() else base_cooldown
        last = self._last_fired.get(key, 0)
        return (time_mod.time() - last) >= cooldown

    def mark_fired(self, signal_name: str, sub_key: str = ""):
        key = f"{signal_name}:{sub_key}" if sub_key else signal_name
        self._last_fired[key] = time_mod.time()
        self._save_state()

    def reset_daily(self):
        self._last_fired = {}
        self._save_state()


def _get_daily_rsi() -> Optional[float]:
    """Fetch SPY RSI(14) once per day, cached. Returns None on failure."""
    today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
    if _rsi_cache["date"] == today:
        return _rsi_cache["rsi"]
    try:
        import yfinance as yf
        import numpy as np
        data = yf.download("SPY", period="25d", progress=False)
        if data.empty or len(data) < 15:
            log.warning("RSI: insufficient SPY data from yfinance")
            return None
        close = data["Close"].values.flatten()
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        _rsi_cache["date"] = today
        _rsi_cache["rsi"] = float(rsi)
        log.info(f"Daily SPY RSI(14) = {rsi:.1f}")
        return float(rsi)
    except Exception as e:
        log.warning(f"RSI fetch failed: {e}")
        return None


class TelegramAlertSender:
    """Send high-priority GEX signals to Telegram channels."""

    def __init__(self):
        tg_cfg = CFG.get("telegram_alerts", {})
        self._enabled = tg_cfg.get("enabled", False)
        self._bot_token = os.getenv(tg_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"), "")
        self._channels: Dict[str, str] = {
            k: str(v) for k, v in tg_cfg.get("channels", {}).items()
        }
        self._cooldown = TelegramCooldownManager()

        if self._enabled and not self._bot_token:
            log.warning("Telegram alerts enabled but TELEGRAM_BOT_TOKEN not set.")
            self._enabled = False

        if self._enabled:
            log.info(
                "Telegram alerts initialized. Channels: %s",
                list(self._channels.keys()),
            )
        else:
            log.info("Telegram alerts disabled.")

    @property
    def cooldown(self) -> TelegramCooldownManager:
        return self._cooldown

    def is_eligible(self, signal: Signal) -> bool:
        """Check if a signal should be sent to Telegram.

        Combined backtest (GEX + SPY, 13 days / 3878 RTH snapshots):
        - BOUNCE_ZONE: Only when gamma negative + not CONTACT
        - GAMMA_SNAP LONG T1: 63%/+25.3 net (Telegram) — SUPPRESS when RSI<40 (drops to 30%)
        - GAMMA_SNAP LONG T2: 47%/+4.8 net (Discord only — coin flip, low net)
        - GAMMA_SNAP LONG T3: 39%/+3.2 net (Discord only — fails both criteria)
        - GAMMA_SNAP LONG T4: 22%/-2.1 net (Discord only — negative EV)
        - GAMMA_SNAP SHORT S1: 67%/+1.1 net (Telegram — high win%, small sample)
        """
        if not self._enabled:
            return False
        if signal.signal_type not in TELEGRAM_ELIGIBLE:
            return False
        if signal.signal_type == SignalType.BOUNCE_ZONE:
            urgency = signal.metadata.get("urgency")
            gamma_bn = signal.metadata.get("gamma_bn")
            # CONTACT (<5 pts) = wall already breaking, not actionable
            if urgency == "CONTACT":
                return False
            # Positive gamma = chop zone, walls are noise
            if gamma_bn is not None and gamma_bn > 0:
                return False
        if signal.signal_type == SignalType.GAMMA_SNAP:
            tier = signal.metadata.get("tier", 99)
            side = signal.metadata.get("side")
            if side == "LONG":
                # T1 only to phone (63% win, +25.3 avg net)
                # T2 (47%/+4.8), T3 (39%/+3.2), T4 (22%/-2.1) all Discord only
                if tier != 1:
                    return False
                # T1 when RSI<40 = 30% win, -10.5 net — actively harmful
                if tier == 1:
                    rsi = _get_daily_rsi()
                    if rsi is not None and rsi < 40:
                        log.info(f"Telegram: suppressing T1 LONG — RSI={rsi:.0f} < 40")
                        return False
        return True

    def send_signal(self, signal: Signal) -> bool:
        """Send a signal to Telegram if eligible and not on cooldown."""
        if not self.is_eligible(signal):
            return False

        signal_name = signal.signal_type.name.lower()
        # GAMMA_SNAP uses tier-level cooldowns so T1 can alert more often
        # Late session (after 11:30 PT) gets even shorter cooldown — 0DTE gamma is extreme
        sub_key = ""
        if signal.signal_type == SignalType.GAMMA_SNAP:
            tier_label = signal.metadata.get("tier_label", "")
            side = signal.metadata.get("side", "")
            now_pt = datetime.now(PT_TZ).time()
            is_late = now_pt >= time(11, 30)
            if side == "LONG" and tier_label == "T1" and is_late:
                sub_key = f"{side}_{tier_label}_LATE"
            else:
                sub_key = f"{side}_{tier_label}"  # e.g. "LONG_T1", "SHORT_S1"
        if not self._cooldown.can_fire(signal_name, sub_key):
            log.debug(f"Telegram: {signal_name}:{sub_key} on cooldown, skipping.")
            return False

        # Map GEX channel name to Telegram chat ID
        chat_id = self._channels.get(signal.channel)
        if not chat_id:
            # Fallback: intraday_fast -> gex_ops channel
            if signal.channel == "intraday_fast":
                chat_id = self._channels.get("gex_ops")
            if not chat_id:
                log.warning(f"No Telegram channel mapped for '{signal.channel}'")
                return False

        success = self._send_message(chat_id, signal)
        if success:
            self._cooldown.mark_fired(signal_name, sub_key)
        return success

    def send_batch(self, signals: list[Signal]) -> int:
        """Send eligible signals from a batch. Returns count sent."""
        sent = 0
        for signal in sorted(signals, key=lambda s: s.priority):
            if self.send_signal(signal):
                sent += 1
        return sent

    def _send_message(self, chat_id: str, signal: Signal) -> bool:
        """Send a message via Telegram Bot API."""
        ts = signal.timestamp
        if ts.tzinfo is None:
            ts = PT_TZ.localize(ts)
        else:
            ts = ts.astimezone(PT_TZ)

        ts_line = ts.strftime("%I:%M %p PT")

        # Format for Telegram (plain text, no markdown parse issues)
        text = f"{ts_line}\n{signal.message}"

        # Telegram message limit is 4096 chars
        if len(text) > 4090:
            text = text[:4087] + "..."

        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }

        backoff = [1, 2, 4]
        for attempt in range(len(backoff) + 1):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    log.info(f"Sent {signal.signal_type.name} to Telegram ({signal.channel})")
                    return True
                elif resp.status_code == 429:
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                    log.warning(f"Telegram rate limited. Retry after {retry_after}s")
                    if attempt < len(backoff):
                        time_mod.sleep(retry_after)
                        continue
                    return False
                else:
                    log.error(f"Telegram API error: {resp.status_code} {resp.text}")
                    return False
            except requests.RequestException as e:
                log.error(f"Telegram request error: {e}")
                if attempt < len(backoff):
                    time_mod.sleep(backoff[attempt])
                    continue
                return False
        return False
