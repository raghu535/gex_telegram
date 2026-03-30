"""Consensus tracker for Arabic SPX channel directional calls.

Tracks rolling directional consensus across configured channels.
No intelligence logic here — that lives in telegram_listener._check_market_intel().
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytz

PT_TZ = pytz.timezone("US/Pacific")
log = logging.getLogger(__name__)


@dataclass
class DirectionalCall:
    """A single directional call from a channel."""
    channel: str
    direction: str  # "LONG" or "SHORT"
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    targets: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(PT_TZ))


@dataclass
class ConsensusState:
    """Current consensus across channels."""
    direction: Optional[str] = None  # "LONG", "SHORT", or None
    strength: int = 0  # number of agreeing channels
    calls: List[DirectionalCall] = field(default_factory=list)
    price_levels: List[float] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        if self.strength >= 3:
            return "HIGH"
        elif self.strength >= 2:
            return "MEDIUM"
        elif self.strength >= 1:
            return "LOW"
        return "NONE"


# Arabic keyword patterns for direction detection
_LONG_KEYWORDS = [
    "كول",      # "call"
    "شراء",     # "buy"
    "صعود",     # "rise"
    "ارتداد",   # "bounce"
    "اخضر",     # "green"
    "طالع",     # "going up"
    "لونق",     # "long" transliterated
    "buy",
    "long",
    "calls",
    "call",
    "bounce",
    "bullish",
]

_SHORT_KEYWORDS = [
    "بوت",      # "put"
    "بيع",      # "sell"
    "هبوط",     # "drop"
    "احمر",     # "red"
    "نازل",     # "going down"
    "شورت",     # "short" transliterated
    "sell",
    "short",
    "puts",
    "put",
    "bearish",
]

# SPX price pattern (4-digit number in typical SPX range)
_PRICE_RE = re.compile(r'\b(5[5-9]\d{2}|6[0-9]\d{2}|7[0-4]\d{2})\b')


def _extract_direction(text: str) -> Optional[str]:
    """Extract directional bias from text using keyword matching."""
    text_lower = text.lower()
    long_score = sum(1 for kw in _LONG_KEYWORDS if kw in text_lower)
    short_score = sum(1 for kw in _SHORT_KEYWORDS if kw in text_lower)

    if long_score > short_score:
        return "LONG"
    elif short_score > long_score:
        return "SHORT"
    return None


def _extract_prices(text: str) -> List[float]:
    """Extract SPX-range prices from text."""
    return [float(m) for m in _PRICE_RE.findall(text)]


class ConsensusTracker:
    """Rolling consensus tracker across Arabic SPX channels.

    - 1 vote per channel (latest wins)
    - 15-min rolling window
    - Daily reset
    """

    def __init__(self, window_minutes: int = 15, min_channels: int = 1):
        self._window_minutes = window_minutes
        self._min_channels = min_channels
        self._calls: Dict[str, DirectionalCall] = {}  # channel -> latest call
        self._current_date: Optional[str] = None

    def _maybe_reset(self):
        """Reset on new trading day."""
        today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
        if self._current_date != today:
            self._calls.clear()
            self._current_date = today
            log.info("ConsensusTracker: daily reset")

    def _prune_stale(self):
        """Remove calls older than the rolling window."""
        cutoff = datetime.now(PT_TZ) - timedelta(minutes=self._window_minutes)
        stale = [ch for ch, call in self._calls.items() if call.timestamp < cutoff]
        for ch in stale:
            del self._calls[ch]

    def ingest(self, channel: str, spx_signal) -> Optional[DirectionalCall]:
        """Ingest a parsed SPXMillionSignal. Returns DirectionalCall if direction detected."""
        self._maybe_reset()

        if not spx_signal or not spx_signal.direction:
            return None

        direction = "LONG" if spx_signal.direction == "CALL" else "SHORT"
        call = DirectionalCall(
            channel=channel,
            direction=direction,
            entry_price=spx_signal.entry,
            stop_price=spx_signal.stop,
            targets=spx_signal.targets or [],
        )
        self._calls[channel] = call
        log.info(
            f"ConsensusTracker: {channel} → {direction}"
            f" (entry={spx_signal.entry}, {len(self._calls)} channels active)"
        )
        return call

    def ingest_raw_text(self, channel: str, text: str) -> Optional[DirectionalCall]:
        """Ingest raw text when SPX parser doesn't match. Keyword fallback."""
        self._maybe_reset()

        direction = _extract_direction(text)
        if not direction:
            return None

        prices = _extract_prices(text)
        entry = prices[0] if prices else None

        call = DirectionalCall(
            channel=channel,
            direction=direction,
            entry_price=entry,
        )
        self._calls[channel] = call
        log.info(
            f"ConsensusTracker (keyword): {channel} → {direction}"
            f" (entry={entry}, {len(self._calls)} channels active)"
        )
        return call

    def get_consensus(self) -> ConsensusState:
        """Get current consensus state."""
        self._maybe_reset()
        self._prune_stale()

        if not self._calls:
            return ConsensusState()

        long_calls = [c for c in self._calls.values() if c.direction == "LONG"]
        short_calls = [c for c in self._calls.values() if c.direction == "SHORT"]

        if len(long_calls) > len(short_calls) and len(long_calls) >= self._min_channels:
            prices = [c.entry_price for c in long_calls if c.entry_price]
            return ConsensusState(
                direction="LONG",
                strength=len(long_calls),
                calls=long_calls,
                price_levels=prices,
            )
        elif len(short_calls) > len(long_calls) and len(short_calls) >= self._min_channels:
            prices = [c.entry_price for c in short_calls if c.entry_price]
            return ConsensusState(
                direction="SHORT",
                strength=len(short_calls),
                calls=short_calls,
                price_levels=prices,
            )
        elif len(long_calls) == len(short_calls) and len(long_calls) >= self._min_channels:
            # Tie — no consensus
            return ConsensusState()

        return ConsensusState()
