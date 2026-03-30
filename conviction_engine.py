"""Conviction Engine — cross-system intelligence for high-conviction trade signals.

Polls Antigravity + Options Flow APIs, cross-references with live GEX data,
and fires CONVICTION SIGNAL when 2+ systems agree on direction.
"""

from __future__ import annotations

import logging
import math
import time as _time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pytz
import requests

from gex_parser import GEXSnapshot
from signal_interpreter import Signal, SignalType

PT_TZ = pytz.timezone("US/Pacific")
log = logging.getLogger(__name__)


class _CachedResult:
    """Simple TTL cache for an API response."""

    def __init__(self, ttl: float):
        self.ttl = ttl
        self.data: Optional[Dict] = None
        self.fetched_at: float = 0

    def is_fresh(self) -> bool:
        return self.data is not None and (_time.time() - self.fetched_at) < self.ttl

    def set(self, data: Dict):
        self.data = data
        self.fetched_at = _time.time()


class ConvictionEngine:
    """Cross-references GEX, Antigravity, and Options Flow to score conviction."""

    def __init__(self, cfg: dict):
        conv = cfg.get("conviction", {})
        self.enabled = conv.get("enabled", False)
        self.min_score = float(conv.get("min_score", 6.0))
        self.runner_score = float(conv.get("runner_score", 7.0))
        self.runner_room_pts = float(conv.get("runner_room_pts", 30))
        self.cooldown_secs = float(conv.get("cooldown_secs", 600))

        ag_ttl = float(conv.get("antigravity_cache_ttl", 120))
        of_ttl = float(conv.get("optionsflow_cache_ttl", 300))

        self._ag_url_rec = conv.get(
            "antigravity_recommendation_url",
            "http://localhost:5000/api/trading-recommendation",
        )
        self._ag_url_trap = conv.get(
            "antigravity_trap_url",
            "http://localhost:5000/api/market-data/trap-signal",
        )
        self._of_url = conv.get(
            "optionsflow_url",
            "http://localhost:9080/api/intelligence",
        )

        self._ag_cache = _CachedResult(ag_ttl)
        self._of_cache = _CachedResult(of_ttl)
        self._last_fire_ts: float = 0
        self._request_timeout = float(conv.get("request_timeout", 5))

    # ------------------------------------------------------------------
    # API polling (cached)
    # ------------------------------------------------------------------

    def poll_antigravity(self) -> Optional[Dict]:
        """Fetch recommendation + trap signal from Antigravity. Returns merged dict."""
        if self._ag_cache.is_fresh():
            return self._ag_cache.data

        merged: Dict[str, Any] = {}
        try:
            r = requests.get(self._ag_url_rec, timeout=self._request_timeout)
            r.raise_for_status()
            merged["recommendation"] = r.json()
        except Exception as e:
            log.debug(f"conviction: antigravity recommendation fetch failed: {e}")

        try:
            r = requests.get(self._ag_url_trap, timeout=self._request_timeout)
            r.raise_for_status()
            merged["trap"] = r.json()
        except Exception as e:
            log.debug(f"conviction: antigravity trap fetch failed: {e}")

        if merged:
            self._ag_cache.set(merged)
            return merged
        return self._ag_cache.data  # return stale if available

    def poll_optionsflow(self) -> Optional[Dict]:
        """Fetch intelligence summary from Options Flow."""
        if self._of_cache.is_fresh():
            return self._of_cache.data

        try:
            r = requests.get(self._of_url, timeout=self._request_timeout)
            r.raise_for_status()
            data = r.json()
            self._of_cache.set(data)
            return data
        except Exception as e:
            log.debug(f"conviction: options flow fetch failed: {e}")
            return self._of_cache.data  # stale fallback

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_gex(self, snapshot: GEXSnapshot, trend: Optional[Any]) -> Tuple[float, str]:
        """Score GEX data 0-3.3. Returns (score, summary_line)."""
        score = 0.0
        gamma_bn = snapshot.net_gamma / 1e9

        # Bias detection from trend
        bias = "NEUTRAL"
        trend_score = 0
        if trend is not None:
            bias = getattr(trend, "bias", "NEUTRAL")
            trend_score = getattr(trend, "score", 0)

        # 1) Trend bias strength (0-1.2)
        if abs(trend_score) >= 7:
            score += 1.2
        elif abs(trend_score) >= 5:
            score += 0.8
        elif abs(trend_score) >= 3:
            score += 0.4

        # 2) Room to wall/floor (0-1.0)
        if bias in ("BULLISH",):
            room = (snapshot.call_wall - snapshot.curr_price) if snapshot.call_wall else 0
        elif bias in ("BEARISH",):
            room = (snapshot.curr_price - snapshot.put_floor) if snapshot.put_floor else 0
        else:
            room_call = (snapshot.call_wall - snapshot.curr_price) if snapshot.call_wall else 0
            room_put = (snapshot.curr_price - snapshot.put_floor) if snapshot.put_floor else 0
            room = max(room_call, room_put)

        if room >= 40:
            score += 1.0
        elif room >= 25:
            score += 0.7
        elif room >= 15:
            score += 0.4

        # 3) Gamma regime (0-1.1): negative gamma = more momentum
        if gamma_bn <= -10:
            score += 1.1
        elif gamma_bn <= -5:
            score += 0.8
        elif gamma_bn <= -2:
            score += 0.5
        elif gamma_bn >= 10:
            score += 0.3  # strong positive = pinning, less movement

        summary = f"Trend {bias} ({trend_score}/10), {room:.0f} pts to {'wall' if bias != 'BEARISH' else 'floor'}"
        return min(score, 3.3), summary

    def _score_antigravity(self, ag_data: Dict) -> Tuple[float, str, bool]:
        """Score Antigravity 0-3.3. Returns (score, summary, is_stale)."""
        if not ag_data:
            return 0.0, "Antigravity: unavailable", False

        score = 0.0
        rec = ag_data.get("recommendation", {})
        trap = ag_data.get("trap", {})

        # Check staleness of trap signal
        today_str = datetime.now(PT_TZ).strftime("%Y-%m-%d")
        trap_date = str(trap.get("data_date", trap.get("date", "")))
        is_stale = bool(trap_date and trap_date != today_str and trap_date != "")

        # 1) Active signal (0-1.5)
        action = str(rec.get("action", rec.get("signal", ""))).upper()
        if action in ("BUY", "STRONG BUY"):
            score += 1.5
        elif action in ("SELL", "STRONG SELL"):
            score += 1.5
        elif action in ("HOLD",):
            score += 0.3

        # 2) RSI extremes (0-1.0)
        rsi5 = rec.get("rsi5", rec.get("rsi_5", None))
        if rsi5 is not None:
            rsi5 = float(rsi5)
            if rsi5 < 25 or rsi5 > 75:
                score += 1.0
            elif rsi5 < 35 or rsi5 > 65:
                score += 0.5

        # 3) Win rate + p-value (0-0.8)
        win_rate = rec.get("win_rate", rec.get("winRate", None))
        p_val = rec.get("p_value", rec.get("pValue", None))
        if win_rate is not None:
            wr_str = str(win_rate)
            has_pct = '%' in wr_str
            win_rate = float(wr_str.replace('%', ''))
            if has_pct or win_rate > 1:
                win_rate = win_rate / 100
            if win_rate >= 0.70:
                score += 0.5
            elif win_rate >= 0.55:
                score += 0.3
        if p_val is not None:
            p_val = float(p_val)
            if p_val <= 0.05:
                score += 0.3

        rsi_str = f"RSI5={rsi5:.1f}" if rsi5 is not None else "RSI n/a"
        wr_str = f"{win_rate*100:.1f}% WR" if win_rate is not None else "WR n/a"
        p_str = f"p={p_val:.2f}" if p_val is not None else ""
        stale_flag = " [STALE TRAP]" if is_stale else ""
        summary = f"{action} ({rsi_str}, {wr_str}{', ' + p_str if p_str else ''}){stale_flag}"

        return min(score, 3.3), summary, is_stale

    def _score_optionsflow(self, of_data: Dict) -> Tuple[float, str]:
        """Score Options Flow 0-3.3. Returns (score, summary)."""
        if not of_data:
            return 0.0, "Options Flow: unavailable"

        score = 0.0

        # 1) Regime (0-1.5)
        regime = str(of_data.get("regime", of_data.get("market_regime", ""))).upper()
        if regime in ("OVERSOLD", "OVERBOUGHT"):
            score += 1.5
        elif regime in ("NEUTRAL_BEARISH", "NEUTRAL_BULLISH"):
            score += 0.5

        # 2) Breadth (0-1.0)
        breadth = of_data.get("breadth", of_data.get("market_breadth", None))
        if breadth is not None:
            breadth = float(breadth)
            if breadth < 30 or breadth > 70:
                score += 1.0
            elif breadth < 40 or breadth > 60:
                score += 0.5

        # 3) Weights / sentiment (0-0.8)
        long_wt = of_data.get("long_weight", of_data.get("longWeight", None))
        if long_wt is not None:
            long_wt = float(long_wt)
            if long_wt >= 0.8 or long_wt <= 0.2:
                score += 0.8
            elif long_wt >= 0.65 or long_wt <= 0.35:
                score += 0.4

        breadth_str = f"breadth {breadth:.1f}%" if breadth is not None else "breadth n/a"
        wt_str = f"long wt {long_wt:.1f}" if long_wt is not None else ""
        summary = f"{regime} ({breadth_str}{', ' + wt_str if wt_str else ''})"

        return min(score, 3.3), summary

    def compute_conviction(
        self,
        snapshot: GEXSnapshot,
        trend: Optional[Any],
        ag_data: Optional[Dict],
        of_data: Optional[Dict],
    ) -> Dict:
        """Compute unified conviction score 0-10 from all 3 systems.

        Returns dict with: score, direction, mode, gex/ag/of summaries,
        room, targets, stale flags, etc.
        """
        gex_score, gex_summary = self._score_gex(snapshot, trend)
        ag_score, ag_summary, ag_stale = self._score_antigravity(ag_data) if ag_data else (0.0, "Antigravity: unavailable", False)
        of_score, of_summary = self._score_optionsflow(of_data) if of_data else (0.0, "Options Flow: unavailable")

        total = gex_score + ag_score + of_score

        # Determine direction from trend bias
        bias = "NEUTRAL"
        if trend is not None:
            bias = getattr(trend, "bias", "NEUTRAL")

        # If antigravity has a strong signal and GEX is neutral, defer to AG
        if bias == "NEUTRAL" and ag_data:
            rec = ag_data.get("recommendation", {})
            action = str(rec.get("action", rec.get("signal", ""))).upper()
            if action in ("BUY", "STRONG BUY"):
                bias = "BULLISH"
            elif action in ("SELL", "STRONG SELL"):
                bias = "BEARISH"

        # Determine direction string
        if bias == "BULLISH":
            direction = "LONG"
        elif bias == "BEARISH":
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        # Room to wall/floor
        if direction == "LONG":
            room = (snapshot.call_wall - snapshot.curr_price) if snapshot.call_wall else 0
            wall_label = "wall"
            wall_level = snapshot.call_wall
        elif direction == "SHORT":
            room = (snapshot.curr_price - snapshot.put_floor) if snapshot.put_floor else 0
            wall_label = "floor"
            wall_level = snapshot.put_floor
        else:
            room = 0
            wall_label = "n/a"
            wall_level = 0

        # Mode: RUNNER vs SCALP
        if total >= self.runner_score and room >= self.runner_room_pts:
            mode = "RUNNER"
        else:
            mode = "SCALP"

        # Stale trap suppression: if trap data is stale, reduce AG score contribution
        if ag_stale and ag_score > 0:
            penalty = min(0.5, ag_score * 0.3)
            total -= penalty
            ag_summary += " (score reduced)"

        # Targets
        if direction == "LONG":
            entry = snapshot.curr_price
            stop = entry - 12
            t1 = entry + (room / 2) if mode == "RUNNER" and room > 0 else entry + 10
            t2 = wall_level if mode == "RUNNER" and wall_level else entry + 20
        elif direction == "SHORT":
            entry = snapshot.curr_price
            stop = entry + 12
            t1 = entry - (room / 2) if mode == "RUNNER" and room > 0 else entry - 10
            t2 = wall_level if mode == "RUNNER" and wall_level else entry - 20
        else:
            entry = snapshot.curr_price
            stop = entry
            t1 = entry
            t2 = entry

        # Strike selection
        atm_strike = _round_strike(snapshot.curr_price, direction)
        if direction == "LONG" and snapshot.call_wall:
            lotto_strike = snapshot.call_wall + 5
        elif direction == "SHORT" and snapshot.put_floor:
            lotto_strike = snapshot.put_floor - 5
        else:
            lotto_strike = atm_strike

        suffix = "C" if direction == "LONG" else "P"
        atm_symbol = f"SPXW {atm_strike}{suffix}"
        lotto_symbol = f"SPXW {lotto_strike}{suffix}"

        # Count how many systems are contributing
        sources_active = sum([
            gex_score > 0.5,
            ag_score > 0.5,
            of_score > 0.5,
        ])

        return {
            "score": round(total, 1),
            "direction": direction,
            "mode": mode,
            "room": room,
            "wall_label": wall_label,
            "wall_level": wall_level,
            "gex_score": round(gex_score, 1),
            "ag_score": round(ag_score, 1),
            "of_score": round(of_score, 1),
            "gex_summary": gex_summary,
            "ag_summary": ag_summary,
            "of_summary": of_summary,
            "ag_stale": ag_stale,
            "entry": round(entry),
            "stop": round(stop),
            "t1": round(t1),
            "t2": round(t2),
            "atm_symbol": atm_symbol,
            "lotto_symbol": lotto_symbol,
            "sources_active": sources_active,
            "timestamp": snapshot.timestamp_pt,
        }

    # ------------------------------------------------------------------
    # Signal firing
    # ------------------------------------------------------------------

    def maybe_fire(
        self,
        snapshot: GEXSnapshot,
        trend: Optional[Any],
    ) -> Optional[Signal]:
        """Poll external systems, compute conviction, fire if threshold met.

        Returns Signal if fired, None otherwise.
        """
        if not self.enabled:
            return None

        # Cooldown check
        now = _time.time()
        if (now - self._last_fire_ts) < self.cooldown_secs:
            return None

        # Poll external systems
        ag_data = self.poll_antigravity()
        of_data = self.poll_optionsflow()

        # Compute conviction
        result = self.compute_conviction(snapshot, trend, ag_data, of_data)

        score = result["score"]
        direction = result["direction"]

        # Must have a directional bias
        if direction == "NEUTRAL":
            log.debug(f"conviction: score={score} but direction NEUTRAL, skipping")
            return None

        # Must meet minimum score
        if score < self.min_score:
            log.debug(f"conviction: score={score} < min {self.min_score}, skipping")
            return None

        # Must have at least 2 systems contributing
        if result["sources_active"] < 2:
            log.debug(f"conviction: only {result['sources_active']} source(s), need 2+")
            return None

        # Build signal
        signal = self._format_signal(result, snapshot)
        self._last_fire_ts = now

        log.info(
            f"conviction: FIRED {direction} score={score} mode={result['mode']} "
            f"sources={result['sources_active']}"
        )
        return signal

    def _format_signal(self, result: Dict, snapshot: GEXSnapshot) -> Signal:
        """Format conviction result into a Signal for Discord/Telegram."""
        ts = result["timestamp"]
        time_str = ts.strftime("%I:%M %p PT") if ts else ""

        emoji = "\U0001f7e2" if result["direction"] == "LONG" else "\U0001f534"
        conviction_bar = _conviction_bar(result["score"])

        lines = [
            f"{emoji} CONVICTION {result['direction']} ({result['score']}/10) | {result['mode']} | {time_str}",
            f"SPX {snapshot.curr_price:.0f} | Room to {result['wall_label']}: {result['room']:.0f} pts",
            "",
            f"GEX: {result['gex_summary']}",
            f"Antigravity: {result['ag_summary']}",
            f"Options Flow: {result['of_summary']}",
            "",
            f"Entry: {result['entry']} | Stop: {result['stop']} | T1: {result['t1']} | T2: {result['t2']}",
            f"Play: {result['atm_symbol']} | Lotto: {result['lotto_symbol']}",
            conviction_bar,
        ]

        return Signal(
            signal_type=SignalType.CONVICTION_SIGNAL,
            title=f"Conviction {result['direction']} ({result['score']}/10)",
            message="\n".join(lines),
            channel="conviction_signal",
            priority=1,  # highest priority
            timestamp=ts,
            metadata={
                "score": result["score"],
                "direction": result["direction"],
                "mode": result["mode"],
                "room": result["room"],
                "gex_score": result["gex_score"],
                "ag_score": result["ag_score"],
                "of_score": result["of_score"],
                "sources_active": result["sources_active"],
                "ag_stale": result["ag_stale"],
            },
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _round_strike(price: float, direction: str) -> int:
    """Round price to nearest $5 strike."""
    if direction == "LONG":
        return int(math.ceil(price / 5) * 5)
    elif direction == "SHORT":
        return int(math.floor(price / 5) * 5)
    return int(round(price / 5) * 5)


def _conviction_bar(score: float) -> str:
    """Visual conviction bar."""
    filled = int(score)
    empty = 10 - filled
    return f"[{'█' * filled}{'░' * empty}] {score}/10"
