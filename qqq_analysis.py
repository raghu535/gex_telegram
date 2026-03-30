"""QQQ/SPX session context analyzer — time-adjusted expected move and beta-mapped levels."""

from __future__ import annotations

import logging
import math
import time as _time_mod
from datetime import datetime, time
from typing import Optional

import pytz

from gex_parser import GEXSnapshot

PT_TZ = pytz.timezone("US/Pacific")
log = logging.getLogger(__name__)

RTH_OPEN_MIN = 6 * 60 + 30   # 6:30 PT = 390 minutes into day
RTH_CLOSE_MIN = 13 * 60       # 13:00 PT
RTH_TOTAL_MIN = RTH_CLOSE_MIN - RTH_OPEN_MIN  # 390


class QQQAnalyzer:
    """Computes time-adjusted expected move levels and QQQ beta-mapped levels."""

    def __init__(self, db_path: str, config: dict):
        self._db_path = db_path
        self._cfg = config.get("qqq_analysis", {})
        self._beta: float = float(self._cfg.get("beta", 1.2))
        self._window_min: int = int(self._cfg.get("intraday_window_min", 30))
        self._cache_secs: int = int(self._cfg.get("cache_seconds", 120))

        # Per-session state (reset daily)
        self._session_date: Optional[str] = None
        self._spx_open: Optional[float] = None
        self._qqq_open: Optional[float] = None

        # yfinance TTL cache
        self._yf_cache: dict = {}
        self._yf_cache_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build_signal(self, snapshot: GEXSnapshot, now_pt: datetime) -> Optional[dict]:
        """Compute QQQ/SPX session context and return a signal dict, or None on failure.

        Returns a dict compatible with Signal construction:
          {"title": ..., "message": ..., "channel": ..., "priority": ...}
        """
        try:
            return self._build(snapshot, now_pt)
        except Exception as exc:
            log.warning("QQQAnalyzer.build_signal failed: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _reset_if_new_day(self, date_str: str) -> None:
        if self._session_date != date_str:
            self._session_date = date_str
            self._spx_open = None
            self._qqq_open = None

    def _fetch_yf(self) -> dict:
        """Fetch VIX, VXN, QQQ price with TTL cache."""
        now = _time_mod.time()
        if now - self._yf_cache_ts < self._cache_secs and self._yf_cache:
            return self._yf_cache

        try:
            import yfinance as yf
            vix = float(yf.Ticker("^VIX").fast_info["lastPrice"])
            vxn = float(yf.Ticker("^VXN").fast_info["lastPrice"])
            qqq = float(yf.Ticker("QQQ").fast_info["lastPrice"])
            self._yf_cache = {"vix": vix, "vxn": vxn, "qqq": qqq}
            self._yf_cache_ts = now
        except Exception as exc:
            log.warning("QQQAnalyzer yfinance fetch failed: %s", exc)
            # Return stale cache if available
            if not self._yf_cache:
                raise

        return self._yf_cache

    def _get_recent_prices(self) -> tuple[float, float]:
        """Return (intraday_high, intraday_low) from DB over last window_min RTH snapshots."""
        from gex_db import get_recent_snapshots
        snaps = get_recent_snapshots(self._window_min, session_tag="RTH")
        if not snaps:
            return 0.0, 0.0
        prices = [s.curr_price for s in snaps if s.curr_price]
        if not prices:
            return 0.0, 0.0
        return max(prices), min(prices)

    def _time_remaining(self, now_pt: datetime) -> tuple[int, float]:
        """Return (remaining_min, remaining_fraction) based on current PT time."""
        now_min = now_pt.hour * 60 + now_pt.minute
        remaining_min = max(0, RTH_CLOSE_MIN - now_min)
        remaining_fraction = math.sqrt(remaining_min / RTH_TOTAL_MIN)
        return remaining_min, remaining_fraction

    def _build(self, snapshot: GEXSnapshot, now_pt: datetime) -> Optional[dict]:
        date_str = now_pt.strftime("%Y-%m-%d")
        self._reset_if_new_day(date_str)

        spx_price = snapshot.curr_price
        if not spx_price:
            return None

        # Cache SPX open (first RTH snapshot price for today)
        if self._spx_open is None:
            self._spx_open = spx_price

        # Fetch yfinance data
        yf_data = self._fetch_yf()
        vix: float = yf_data["vix"]
        vxn: float = yf_data["vxn"]
        qqq_price: float = yf_data["qqq"]

        # Cache QQQ open
        if self._qqq_open is None:
            self._qqq_open = qqq_price

        spx_open = self._spx_open
        qqq_open = self._qqq_open

        # --- Time-remaining expected move ---
        remaining_min, remaining_fraction = self._time_remaining(now_pt)

        spx_full_1sig = (vix / 100) / math.sqrt(252) * spx_price
        spx_remaining_1sig = spx_full_1sig * remaining_fraction

        qqq_full_1sig = (vxn / 100) / math.sqrt(252) * qqq_price
        qqq_remaining_1sig = qqq_full_1sig * remaining_fraction

        # --- Move budget used ---
        move_so_far = abs(spx_price - spx_open)
        budget_used_pct = (move_so_far / spx_full_1sig * 100) if spx_full_1sig > 0 else 0
        budget_left_pct = max(0, 100 - budget_used_pct)

        # --- 30-min tight levels ---
        intraday_high, intraday_low = self._get_recent_prices()
        intraday_range = intraday_high - intraday_low if (intraday_high and intraday_low) else 0

        # --- QQQ beta-mapped levels ---
        if spx_open and qqq_open and intraday_high and intraday_low:
            spx_high_pct = (intraday_high - spx_open) / spx_open
            spx_low_pct = (intraday_low - spx_open) / spx_open
            qqq_mapped_high = qqq_open * (1 + spx_high_pct * self._beta)
            qqq_mapped_low = qqq_open * (1 + spx_low_pct * self._beta)
        else:
            qqq_mapped_high = qqq_price
            qqq_mapped_low = qqq_price

        # --- Remaining range from current ---
        qqq_remaining_up = qqq_price + qqq_remaining_1sig
        qqq_remaining_down = qqq_price - qqq_remaining_1sig
        spx_remaining_up = spx_price + spx_remaining_1sig
        spx_remaining_down = spx_price - spx_remaining_1sig

        # --- Leadership signal ---
        qqq_chg_pct = (qqq_price - qqq_open) / qqq_open * 100 if qqq_open else 0
        spx_chg_pct = (spx_price - spx_open) / spx_open * 100 if spx_open else 0
        lead = qqq_chg_pct - spx_chg_pct
        if lead > 0.15:
            leadership_str = f"QQQ LEADING  (+{lead:.2f}%)"
        elif lead < -0.15:
            leadership_str = f"QQQ LAGGING  ({lead:.2f}%)"
        else:
            leadership_str = f"ALIGNED  ({lead:+.2f}%)"

        # --- GEX context ---
        net_gamma_bn = snapshot.net_gamma / 1e9 if snapshot.net_gamma else 0
        cw = snapshot.call_wall or 0
        pf = snapshot.put_floor or 0
        cw_dist = abs(spx_price - cw) if cw else 0
        pf_dist = abs(spx_price - pf) if pf else 0

        # --- Time display ---
        hrs_remaining = remaining_min // 60
        mins_remaining = remaining_min % 60
        time_str = now_pt.strftime("%I:%M %p PT").lstrip("0")

        # --- Format message ---
        vxn_vix_spread = vxn / vix if vix > 0 else 1.0
        gamma_sign = "+" if net_gamma_bn >= 0 else ""

        msg = (
            f"📊 QQQ/SPX SESSION CONTEXT | {time_str}\n"
            f"\n"
            f"SPX  {spx_price:,.0f} | QQQ ${qqq_price:.2f}  |  "
            f"{hrs_remaining}h {mins_remaining:02d}m left\n"
            f"\n"
            f"TIME-ADJUSTED MOVE BUDGET\n"
            f"  SPX ±{spx_remaining_1sig:.0f}pts remaining "
            f"({budget_used_pct:.0f}% used, {budget_left_pct:.0f}% left)\n"
            f"  QQQ ±${qqq_remaining_1sig:.2f} remaining\n"
            f"\n"
            f"TIGHT LEVELS ({self._window_min}m high/low)\n"
            f"  SPX  {intraday_low:,.0f} — {intraday_high:,.0f}  ({intraday_range:.0f}pt range)\n"
            f"  QQQ  ${qqq_mapped_low:.2f} — ${qqq_mapped_high:.2f}  (beta-mapped)\n"
            f"\n"
            f"REMAINING RANGE (from current)\n"
            f"  SPX up to {spx_remaining_up:,.0f} | down to {spx_remaining_down:,.0f}  (1σ remaining)\n"
            f"  QQQ up to ${qqq_remaining_up:.2f} | down to ${qqq_remaining_down:.2f}\n"
            f"\n"
            f"LEADERSHIP  QQQ {qqq_chg_pct:+.2f}% vs SPX {spx_chg_pct:+.2f}% → {leadership_str}\n"
            f"GEX PIN  {gamma_sign}{net_gamma_bn:.1f}Bn | "
            f"CW {cw} ({cw_dist:.0f}pts) | PF {pf} ({pf_dist:.0f}pts away)\n"
            f"VIX {vix:.2f} | VXN {vxn:.2f} ({vxn_vix_spread:.2f}x spread)"
        )

        return {
            "title": "QQQ/SPX SESSION CONTEXT",
            "message": msg,
            "channel": "gex_relay",
            "priority": 2,
        }
