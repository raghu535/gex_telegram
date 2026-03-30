"""Parse SPX Million (Arabic) channel trade messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import re


_NUM = r"(\d+(?:\.\d+)?)"

RE_ENTRY = re.compile(r"بسعر\s*[:：]?\s*" + _NUM)
RE_ENTRY_ALT = re.compile(r"سعر\s*(?:الدخول|الشراء|البيع)\s*[:：]?\s*" + _NUM)
RE_ENTRY_EN = re.compile(r"Entry\s*[:：]?\s*" + _NUM, re.IGNORECASE)
RE_STOP = re.compile(r"وقف\s*كسر\s*[:：]?\s*" + _NUM)
RE_TARGETS = re.compile(r"نقاط\s*البيع\s*[:：]?\s*([0-9\.\s\-–—=٪%،]+)")
RE_TARGET_SINGLE = re.compile(r"الهدف\s*[:：]?\s*" + _NUM)
RE_INDEX_NOW = re.compile(r"المؤشر\s*الان\s*[:：]?\s*" + _NUM)
RE_FILLED = re.compile(r"تم\s*عند\s*[:：]?\s*" + _NUM)
RE_TARGET_HIT = re.compile(r"تم\s*تحقيق\s*الهدف(?:\s*عند)?\s*[:：]?\s*" + _NUM)
RE_TARGET1 = re.compile(r"حقق\s*نقطة\s*البيع\s*الاولى\s*[:：]?\s*" + _NUM)
RE_PROFIT = re.compile(r"(?:الربح|Profit)\s*[:：]?\s*" + _NUM, re.IGNORECASE)
# SPXW option contract: SPXW $6805 20 FEB 26 PUT/CALL
RE_SPXW = re.compile(r"SPXW\s*\$?" + _NUM + r".*?\b(PUT|CALL)\b", re.IGNORECASE)
RE_NOW_PRICE = re.compile(r"(?:السعر\s*الآن|السعر\s*الان|Now)\s*[:：]?\s*" + _NUM, re.IGNORECASE)


@dataclass
class SPXMillionSignal:
    kind: str
    entry: Optional[float] = None
    stop: Optional[float] = None
    targets: List[float] = field(default_factory=list)
    filled: Optional[float] = None
    target_hit: Optional[float] = None
    target_hit_first: Optional[float] = None
    index_now: Optional[float] = None
    index_target: Optional[float] = None
    no_stop: bool = False
    direction: Optional[str] = None
    raw: str = ""


def _parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_targets(raw: str) -> List[float]:
    if not raw:
        return []
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", raw)]
    if not nums:
        return []
    small = [n for n in nums if n <= 20]
    return small if small else nums


def parse_spx_million_message(text: str) -> Optional[SPXMillionSignal]:
    """Return a structured signal if text matches SPX Million patterns."""
    if not text:
        return None

    key_phrases = (
        "بسعر",
        "نقاط البيع",
        "وقف كسر",
        "بدون وقف",
        "تم عند",
        "تم تحقيق الهدف",
        "المؤشر الان",
        "الهدف",
        "سعر الدخول",
        "سعر الشراء",
        "سعر البيع",
        "الربح",
        "SPXW",
    )
    if not any(k in text for k in key_phrases):
        return None

    # Entry price: try multiple patterns
    entry = _parse_float(RE_ENTRY.search(text).group(1)) if RE_ENTRY.search(text) else None
    if entry is None:
        m = RE_ENTRY_ALT.search(text)
        if m:
            entry = _parse_float(m.group(1))
    if entry is None:
        m = RE_ENTRY_EN.search(text)
        if m:
            entry = _parse_float(m.group(1))

    stop = _parse_float(RE_STOP.search(text).group(1)) if RE_STOP.search(text) else None

    targets = []
    m_targets = RE_TARGETS.search(text)
    if m_targets:
        targets = _parse_targets(m_targets.group(1))

    index_now = _parse_float(RE_INDEX_NOW.search(text).group(1)) if RE_INDEX_NOW.search(text) else None
    index_target = _parse_float(RE_TARGET_SINGLE.search(text).group(1)) if RE_TARGET_SINGLE.search(text) else None

    filled = _parse_float(RE_FILLED.search(text).group(1)) if RE_FILLED.search(text) else None
    target_hit = _parse_float(RE_TARGET_HIT.search(text).group(1)) if RE_TARGET_HIT.search(text) else None
    target_hit_first = _parse_float(RE_TARGET1.search(text).group(1)) if RE_TARGET1.search(text) else None

    profit = _parse_float(RE_PROFIT.search(text).group(1)) if RE_PROFIT.search(text) else None
    now_price = _parse_float(RE_NOW_PRICE.search(text).group(1)) if RE_NOW_PRICE.search(text) else None

    # SPXW option contract detection
    spxw_match = RE_SPXW.search(text)
    spxw_strike = None
    spxw_side = None
    if spxw_match:
        spxw_strike = _parse_float(spxw_match.group(1))
        spxw_side = spxw_match.group(2).upper()  # PUT or CALL

    no_stop = "بدون وقف" in text or "هيرو زيرو" in text
    direction = None
    if "كول" in text or (spxw_side == "CALL"):
        direction = "CALL"
    elif "بوت" in text or (spxw_side == "PUT"):
        direction = "PUT"

    kind = None
    if target_hit or target_hit_first:
        kind = "TARGET_HIT"
    elif profit is not None:
        kind = "PROFIT_UPDATE"
    elif filled:
        kind = "FILL"
    elif entry or targets or stop:
        kind = "ENTRY"
    elif spxw_strike:
        kind = "OPTION_CALL"
    elif index_now or index_target:
        kind = "INDEX_CALL"
    elif now_price is not None:
        kind = "PRICE_UPDATE"
    else:
        return None

    return SPXMillionSignal(
        kind=kind,
        entry=entry,
        stop=stop,
        targets=targets,
        filled=filled,
        target_hit=target_hit,
        target_hit_first=target_hit_first,
        index_now=index_now,
        index_target=index_target,
        no_stop=no_stop,
        direction=direction,
        raw=text,
    )


def format_spx_million_signal(sig: SPXMillionSignal) -> str:
    lines = [f"**SPX MILLION** | {sig.kind}"]
    if sig.direction:
        lines.append(f"Direction: {sig.direction}")
    if sig.entry is not None:
        lines.append(f"Entry: {sig.entry:g}")
    if sig.stop is not None:
        lines.append(f"Stop: {sig.stop:g}")
    if sig.no_stop:
        lines.append("Stop: none (hero zero)")
    if sig.targets:
        lines.append("Targets: " + ", ".join(f"{t:g}" for t in sig.targets))
    if sig.filled is not None:
        lines.append(f"Filled: {sig.filled:g}")
    if sig.target_hit is not None:
        lines.append(f"Target hit: {sig.target_hit:g}")
    if sig.target_hit_first is not None:
        lines.append(f"Target hit (first): {sig.target_hit_first:g}")
    if sig.index_now is not None:
        lines.append(f"Index now: {sig.index_now:g}")
    if sig.index_target is not None:
        lines.append(f"Index target: {sig.index_target:g}")

    # Extract SPXW contract info from raw text for display
    spxw_m = RE_SPXW.search(sig.raw)
    if spxw_m:
        lines.append(f"Contract: SPXW ${spxw_m.group(1)} {spxw_m.group(2).upper()}")

    return "\n".join(lines)
