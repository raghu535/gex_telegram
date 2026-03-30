"""Parse SpotGamma Telegram messages into GEXSnapshot dataclass."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pytz

ET_TZ = pytz.timezone("US/Eastern")
PT_TZ = pytz.timezone("US/Pacific")

# --- Real SpotGamma format ---
# 📊 SPX Gamma Exposure Update (20260210)
# 🕒 2026-02-11 03:30:08
# 💵 Current Price: $6,941.81
#
# Top 5 Call Positions (Positive Gamma):
# - Strike: $6,945 | Gamma: 2,925,349,624
# ...
# Top 5 Put Positions (Negative Gamma):
# - Strike: $6,940 | Gamma: -5,373,756,151
# ...
#  Total Call Gamma: 5,578,057,074
#  Total Put Gamma: -15,162,168,237
# ⚖️ Net Gamma: -9,584,111,163

RE_PRICE = re.compile(r"Current Price:\s*\$?([\d,]+\.?\d*)")
RE_NET_GAMMA = re.compile(r"Net Gamma:\s*([-\d,]+)")
RE_TOTAL_CALL = re.compile(r"Total Call Gamma:\s*([-\d,]+)")
RE_TOTAL_PUT = re.compile(r"Total Put Gamma:\s*([-\d,]+)")
RE_SECTION_SPLIT = re.compile(
    r"(Top 5 Call Positions|Top 5 Put Positions|Call Positions|Put Positions)"
)
RE_STRIKE_GAMMA = re.compile(r"Strike:\s*\$?([\d,]+)\s*\|\s*Gamma:\s*([-\d,]+)")

# ISO timestamp from SpotGamma: 🕒 2026-02-11 03:30:08
RE_ISO_TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
# Legacy AM/PM format
RE_AMPM_TIMESTAMP = re.compile(
    r"(?:Updated?|As of|Time)[:.]?\s*(\d{1,2}[:/]\d{2}\s*(?:AM|PM|am|pm)?(?:\s*(?:ET|EST|EDT))?)",
    re.IGNORECASE,
)
RE_DATE = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})")


# Unicode minus signs that SpotGamma messages may use
UNICODE_MINUSES = str.maketrans({
    '\u2212': '-',  # MINUS SIGN
    '\u2013': '-',  # EN DASH
    '\u2014': '-',  # EM DASH
    '\u00AD': '-',  # SOFT HYPHEN
})


@dataclass
class GEXSnapshot:
    timestamp_pt: datetime
    session_tag: str
    curr_price: float = 0.0
    net_gamma: int = 0
    total_call_gamma: int = 0
    total_put_gamma: int = 0
    call_wall: int = 0
    put_floor: int = 0
    top5_calls: List[Tuple[int, int]] = field(default_factory=list)
    top5_puts: List[Tuple[int, int]] = field(default_factory=list)
    raw_text: str = ""

    @property
    def gamma_ratio(self) -> float:
        if self.total_put_gamma == 0:
            return 0.0
        return abs(self.total_call_gamma) / abs(self.total_put_gamma)

    @property
    def net_gamma_bn(self) -> float:
        return self.net_gamma / 1e9

    @property
    def spread(self) -> int:
        if self.call_wall and self.put_floor:
            return self.call_wall - self.put_floor
        return 0


def _clean_text(text: str) -> str:
    """Aggressively strip all non-essential Unicode, keeping only ASCII + basic punctuation."""
    # Normalize unicode minus signs to regular hyphen
    text = text.translate(UNICODE_MINUSES)
    # Strip everything that isn't printable ASCII (keep letters, digits, punctuation, whitespace)
    # But preserve the text structure (newlines, etc.)
    cleaned = []
    for ch in text:
        if ch == '\n' or ch == '\r' or ch == '\t':
            cleaned.append(ch)
        elif 32 <= ord(ch) <= 126:
            cleaned.append(ch)
        # Skip all emoji, Arabic text, variation selectors, etc.
    result = "".join(cleaned)
    # Strip Markdown formatting: **bold** and `backticks`
    result = result.replace("**", "").replace("`", "")
    return result


def _to_int(value: str) -> int:
    return int(value.replace(",", "").replace("$", "").strip())


def _parse_positions(block: str) -> List[Tuple[int, int]]:
    matches = RE_STRIKE_GAMMA.findall(block)
    positions = []
    for strike_str, gamma_str in matches:
        try:
            positions.append((_to_int(strike_str), int(gamma_str.replace(",", ""))))
        except (ValueError, AttributeError):
            continue
    return positions


def _classify_session(dt_pt: datetime) -> str:
    t = dt_pt.time()
    from datetime import time

    rth_start = time(6, 30)
    rth_end = time(13, 0)
    transition_end = time(13, 15)
    premarket_start = time(4, 0)

    if rth_start <= t < rth_end:
        return "RTH"
    if rth_end <= t < transition_end:
        return "TRANSITION"
    if transition_end <= t or t < premarket_start:
        return "NEXT_DAY_PREVIEW"
    if premarket_start <= t < rth_start:
        return "PREMARKET"
    return "RTH"


def classify_session(dt_pt: datetime) -> str:
    """Public helper to classify session tag from PT timestamp."""
    return _classify_session(dt_pt)


def _is_expired_echo(text: str, dt_pt: datetime) -> bool:
    """Detect same-day expiry messages arriving post-close."""
    from datetime import time

    if dt_pt.time() < time(13, 0):
        return False
    if re.search(r"0\s*DTE|same.day|expir", text, re.IGNORECASE):
        return True
    return False


def _extract_timestamp_pt(text: str, envelope_dt: Optional[datetime] = None) -> datetime:
    """Extract timestamp from message body and convert to PT.

    SpotGamma uses ISO format: 🕒 2026-02-11 03:30:08 (this is UTC/Saudi time).
    We convert to PT. Falls back to envelope_dt or current time.
    """
    now_pt = datetime.now(PT_TZ)

    # Try ISO format first (real SpotGamma format)
    iso_match = RE_ISO_TIMESTAMP.search(text)
    if iso_match:
        try:
            dt_raw = datetime.strptime(iso_match.group(1).strip(), "%Y-%m-%d %H:%M:%S")
            dt_riyadh = pytz.timezone("Asia/Riyadh").localize(dt_raw)  # SpotGamma timestamps are Riyadh (UTC+3)
            return dt_riyadh.astimezone(PT_TZ)
        except ValueError:
            pass

    # Try legacy AM/PM format
    match = RE_AMPM_TIMESTAMP.search(text)
    if match:
        time_str = match.group(1).strip()
        time_str = re.sub(r"\s*(ET|EST|EDT)$", "", time_str, flags=re.IGNORECASE).strip()
        for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M"):
            try:
                parsed_time = datetime.strptime(time_str, fmt).time()
                date_match = RE_DATE.search(text)
                if date_match:
                    date_str = date_match.group(1)
                    for dfmt in ("%m/%d/%Y", "%m/%d/%y"):
                        try:
                            parsed_date = datetime.strptime(date_str, dfmt).date()
                            break
                        except ValueError:
                            continue
                    else:
                        parsed_date = now_pt.date()
                else:
                    parsed_date = now_pt.date()
                dt_et = ET_TZ.localize(datetime.combine(parsed_date, parsed_time))
                return dt_et.astimezone(PT_TZ)
            except ValueError:
                continue

    # Fallback: use envelope timestamp if available, otherwise now
    if envelope_dt:
        if envelope_dt.tzinfo is None:
            envelope_dt = pytz.utc.localize(envelope_dt)
        return envelope_dt.astimezone(PT_TZ)

    return now_pt


def parse_message(text: str, envelope_dt: Optional[datetime] = None) -> Optional[GEXSnapshot]:
    """Parse a SpotGamma Telegram message into a GEXSnapshot.

    Args:
        text: Raw message text from Telegram
        envelope_dt: Message datetime from Telegram envelope (UTC). Ignored for
                     timestamp extraction (we use body timestamp) but used as fallback.

    Returns:
        GEXSnapshot if message contains valid GEX data, None otherwise.
    """
    if not text or len(text) < 50:
        return None

    cleaned = _clean_text(text)

    # Must have at least Current Price to be a valid GEX message
    price_match = RE_PRICE.search(cleaned)
    if not price_match:
        return None

    timestamp_pt = _extract_timestamp_pt(cleaned, envelope_dt)
    session_tag = _classify_session(timestamp_pt)

    # Check for expired echo
    if _is_expired_echo(cleaned, timestamp_pt):
        session_tag = "EXPIRED_ECHO"

    # Parse core fields
    curr_price = float(price_match.group(1).replace(",", ""))

    net_gamma = 0
    net_match = RE_NET_GAMMA.search(cleaned)
    if net_match:
        net_gamma = int(net_match.group(1).replace(",", ""))

    total_call = 0
    call_match = RE_TOTAL_CALL.search(cleaned)
    if call_match:
        total_call = int(call_match.group(1).replace(",", ""))

    total_put = 0
    put_match = RE_TOTAL_PUT.search(cleaned)
    if put_match:
        total_put = int(put_match.group(1).replace(",", ""))

    # Split into call/put sections and parse positions
    call_section = ""
    put_section = ""
    parts = RE_SECTION_SPLIT.split(cleaned)
    for i, part in enumerate(parts):
        if "Call Positions" in part and i + 1 < len(parts):
            call_section = parts[i + 1]
        if "Put Positions" in part and i + 1 < len(parts):
            put_section = parts[i + 1]

    top5_calls = _parse_positions(call_section) if call_section else []
    top5_puts = _parse_positions(put_section) if put_section else []

    # Sort: calls by gamma descending, puts by gamma ascending (most negative first)
    top5_calls.sort(key=lambda x: x[1], reverse=True)
    top5_puts.sort(key=lambda x: x[1])

    # Derive walls: highest-gamma strike ABOVE price (call wall), BELOW price (put floor)
    # These are the levels where dealer hedging creates resistance/support
    calls_above = [(s, g) for s, g in top5_calls if s > curr_price]
    puts_below = [(s, g) for s, g in top5_puts if s < curr_price]
    if calls_above:
        call_wall = max(calls_above, key=lambda x: x[1])[0]  # highest gamma above price
    else:
        call_wall = top5_calls[0][0] if top5_calls else 0  # fallback: top gamma overall
    if puts_below:
        put_floor = min(puts_below, key=lambda x: x[1])[0]  # most negative gamma below price
    else:
        put_floor = top5_puts[0][0] if top5_puts else 0  # fallback: top gamma overall

    # Trim to top 5
    top5_calls = top5_calls[:5]
    top5_puts = top5_puts[:5]

    return GEXSnapshot(
        timestamp_pt=timestamp_pt,
        session_tag=session_tag,
        curr_price=curr_price,
        net_gamma=net_gamma,
        total_call_gamma=total_call,
        total_put_gamma=total_put,
        call_wall=call_wall,
        put_floor=put_floor,
        top5_calls=top5_calls,
        top5_puts=top5_puts,
        raw_text=text,
    )
