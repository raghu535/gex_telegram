"""Delta tracker — computes human-readable diffs between consecutive messages.

Tracks previous message state per channel+signal_type. Produces the mandatory
"Δ Since last: {changes}" line for every Discord output.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional

_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delta_state.json")


class DeltaTracker:
    """Track previous message state per channel+type and produce diff strings."""

    def __init__(self):
        self._state: Dict[str, dict] = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────────────

    def _load(self):
        try:
            with open(_STATE_FILE, "r") as f:
                raw = json.load(f)
            # Expire stale entries (> 24h old)
            now = datetime.now().timestamp()
            self._state = {
                k: v for k, v in raw.items()
                if now - v.get("_ts", 0) < 86400
            }
        except (FileNotFoundError, json.JSONDecodeError):
            self._state = {}

    def _save(self):
        try:
            with open(_STATE_FILE, "w") as f:
                json.dump(self._state, f)
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────

    def compute_delta(self, channel: str, msg_type: str, current: dict) -> str:
        """Compare *current* snapshot dict against previously stored state.

        Returns a compact delta string like:
            "Price +12 | Gamma -1.2Bn | Regime: FRAGILE→UNCONTROLLED | PF 6850→6800"

        If no previous state exists, returns "First update".
        """
        key = f"{channel}:{msg_type}"
        prev = self._state.get(key)

        # Store current for next comparison
        self._state[key] = {**current, "_ts": datetime.now().timestamp()}
        self._save()

        if not prev:
            return "First update"

        parts = []

        # Price delta
        if "price" in current and "price" in prev:
            dp = current["price"] - prev["price"]
            if abs(dp) >= 0.5:
                parts.append(f"Price {dp:+.0f}")

        # Gamma delta
        if "gamma_bn" in current and "gamma_bn" in prev:
            dg = current["gamma_bn"] - prev["gamma_bn"]
            if abs(dg) >= 0.1:
                parts.append(f"Gamma {dg:+.1f}Bn")

        # Regime change
        if "regime" in current and "regime" in prev:
            if current["regime"] != prev["regime"]:
                parts.append(f"Regime: {prev['regime']}→{current['regime']}")

        # Call wall shift
        if "call_wall" in current and "call_wall" in prev:
            dcw = current["call_wall"] - prev["call_wall"]
            if dcw != 0:
                parts.append(f"CW {prev['call_wall']}→{current['call_wall']}")

        # Put floor shift
        if "put_floor" in current and "put_floor" in prev:
            dpf = current["put_floor"] - prev["put_floor"]
            if dpf != 0:
                parts.append(f"PF {prev['put_floor']}→{current['put_floor']}")

        # Spread change
        if "spread" in current and "spread" in prev:
            ds = current["spread"] - prev["spread"]
            if abs(ds) >= 5:
                parts.append(f"Spread {ds:+.0f}pts")

        # VIX change
        if "vix" in current and "vix" in prev:
            dv = current["vix"] - prev["vix"]
            if abs(dv) >= 0.2:
                parts.append(f"VIX {dv:+.1f}")

        # EM position change
        if "em_position" in current and "em_position" in prev:
            dep = current["em_position"] - prev["em_position"]
            if abs(dep) >= 0.1:
                parts.append(f"EM pos {dep:+.2f}")

        if not parts:
            return "No change"

        return " | ".join(parts)

    def format_delta_line(self, channel: str, msg_type: str, current: dict) -> str:
        """Return the full delta footer line."""
        delta = self.compute_delta(channel, msg_type, current)
        return f"\u0394 Since last: {delta}"
