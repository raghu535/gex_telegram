"""Historical statistics helper — DB queries for enriching alert messages.

Provides win rates, regime stats, EM reach percentages, and day-group stats
from gex_data.db and spy_analyst.db. Results are cached per day to avoid
repeated DB hits during fast signal processing.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from functools import lru_cache
from typing import Dict, Optional

GEX_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gex_data.db")
SPY_ANALYST_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spy_analyst.db")


def _today_str() -> str:
    import pytz
    return datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d")


# ── EM Reach Stats ────────────────────────────────────────────────────────

def em_reach_pct(em_level: str = "upper") -> Optional[float]:
    """How often does price reach the upper/lower EM boundary in a day?

    Queries daily_summaries from gex_data.db + VIX history.
    Returns percentage (0-100) or None if data unavailable.

    Args:
        em_level: "upper" or "lower"
    """
    try:
        conn = sqlite3.connect(GEX_DB, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT open_price, close_price, high_gamma, low_gamma,
                   close_call_wall, close_put_floor, date_pt
            FROM daily_summaries
            WHERE open_price > 0 AND close_price > 0
            ORDER BY date_pt DESC
            LIMIT 90
        """).fetchall()
        conn.close()

        if len(rows) < 20:
            return None

        # Approximate: count days where price range > 0.75 * avg daily range
        # This is a proxy for EM reach since we don't store exact EM in DB
        total = len(rows)
        reached = 0
        for r in rows:
            spread = abs(r["close_call_wall"] - r["close_put_floor"]) if r["close_call_wall"] and r["close_put_floor"] else 0
            daily_range = abs(r["close_price"] - r["open_price"])
            if spread > 0 and daily_range > spread * 0.5:
                reached += 1

        return round(reached / total * 100, 1) if total > 0 else None
    except Exception:
        return None


# ── Regime Stats ──────────────────────────────────────────────────────────

def regime_stats(regime_name: str) -> Optional[Dict]:
    """Get average duration, range, and typical follow-through for a regime.

    Args:
        regime_name: e.g. "CONTROLLED_PIN", "UNCONTROLLED"

    Returns dict with:
        avg_duration_snapshots, avg_range_pts, typical_follow, sample_n
    """
    try:
        conn = sqlite3.connect(GEX_DB, timeout=5)
        rows = conn.execute("""
            SELECT regime_sequence, open_price, close_price
            FROM daily_summaries
            WHERE regime_sequence IS NOT NULL
            ORDER BY date_pt DESC
            LIMIT 120
        """).fetchall()
        conn.close()

        if not rows:
            return None

        import json
        durations = []
        ranges = []

        for row in rows:
            try:
                seq = json.loads(row[0]) if row[0] else []
            except (json.JSONDecodeError, TypeError):
                continue
            for regime_entry in seq:
                if isinstance(regime_entry, dict) and regime_entry.get("regime") == regime_name:
                    dur = regime_entry.get("duration_snapshots", 0)
                    if dur > 0:
                        durations.append(dur)
            # Range
            if row[1] and row[2]:
                ranges.append(abs(row[2] - row[1]))

        if not durations:
            return None

        avg_dur = sum(durations) / len(durations)
        avg_range = sum(ranges) / len(ranges) if ranges else 0

        _follow_map = {
            "CONTROLLED_PIN": "Range-bound, fade edges",
            "CONTROLLED_TREND": "Trend continuation",
            "FRAGILE_CONTROL": "Breakout imminent either direction",
            "UNCONTROLLED": "Fast reversals, trade the snaps",
        }

        return {
            "avg_duration_snapshots": round(avg_dur, 1),
            "avg_range_pts": round(avg_range, 1),
            "typical_follow": _follow_map.get(regime_name, "Mixed"),
            "sample_n": len(durations),
        }
    except Exception:
        return None


# ── Day Group Stats (stub) ────────────────────────────────────────────────

def day_group_stats(day_type: str) -> Optional[Dict]:
    """Stats for day classification groups (needs spy_analyst.db integration).

    Returns None until spy_analyst.db tables are available.
    """
    return None


# ── Gap Stats (stub) ──────────────────────────────────────────────────────

def gap_stats() -> Optional[Dict]:
    """Gap-fill statistics. Returns None until data integration complete."""
    return None
