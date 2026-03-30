"""Day Classifier — 4 actionable alerts for #gex-trades.

Morning Read (9:30 AM ET), Morning Bet (10:00 AM ET),
Pin Butterfly (2:00 PM ET), EM Fade (anytime during RTH).
"""

from __future__ import annotations

import json
import logging
import os
import time as _time_mod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pytz

from gex_parser import GEXSnapshot
from signal_interpreter import _gamma_english
from output_formatter import format_type1_signal, format_type3_market_read, _box
from delta_tracker import DeltaTracker

_dc_delta_tracker = DeltaTracker()

PT_TZ = pytz.timezone("US/Pacific")
log = logging.getLogger(__name__)

_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier_state.json")


@dataclass
class ActiveTrade:
    trade_type: str          # "MORNING_BET", "EM_FADE_LONG", "EM_FADE_SHORT"
    direction: str           # "CALL" or "PUT"
    entry: float
    stop: float
    target: float
    time_exit_ts: float      # epoch timestamp for time exit
    entry_time_str: str      # display string


class DayClassifier:
    """Daily trade engine: max 1 directional + 1 pin per day."""

    def __init__(self):
        self._morning_bet_fired = False
        self._em_fade_long_fired = False
        self._em_fade_short_fired = False
        self._em_early_long_fired = False
        self._em_early_short_fired = False
        self._pin_fired = False
        self._active_trade: Optional[ActiveTrade] = None
        self._first_30_ratio: Optional[float] = None
        self._trades_today = 0
        self._load_state()

    def _save_state(self):
        """Persist fired flags to disk so restarts don't re-fire."""
        today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
        state = {
            "date": today,
            "morning_bet_fired": self._morning_bet_fired,
            "em_fade_long_fired": self._em_fade_long_fired,
            "em_fade_short_fired": self._em_fade_short_fired,
            "em_early_long_fired": self._em_early_long_fired,
            "em_early_short_fired": self._em_early_short_fired,
            "pin_fired": self._pin_fired,
            "trades_today": self._trades_today,
        }
        try:
            with open(_STATE_FILE, "w") as f:
                json.dump(state, f)
        except Exception as e:
            log.warning(f"DayClassifier: failed to save state: {e}")

    def _load_state(self):
        """Load persisted state if it's from today."""
        try:
            with open(_STATE_FILE, "r") as f:
                state = json.load(f)
            today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
            if state.get("date") != today:
                log.info("DayClassifier: state file is from a different day, ignoring")
                return
            self._morning_bet_fired = state.get("morning_bet_fired", False)
            self._em_fade_long_fired = state.get("em_fade_long_fired", False)
            self._em_fade_short_fired = state.get("em_fade_short_fired", False)
            self._em_early_long_fired = state.get("em_early_long_fired", False)
            self._em_early_short_fired = state.get("em_early_short_fired", False)
            self._pin_fired = state.get("pin_fired", False)
            self._trades_today = state.get("trades_today", 0)
            log.info(f"DayClassifier: restored state from disk (trades_today={self._trades_today})")
        except FileNotFoundError:
            pass
        except Exception as e:
            log.warning(f"DayClassifier: failed to load state: {e}")

    def reset_day(self):
        """Called at RTH open / EOD to reset daily state."""
        self._morning_bet_fired = False
        self._em_fade_long_fired = False
        self._em_fade_short_fired = False
        self._em_early_long_fired = False
        self._em_early_short_fired = False
        self._pin_fired = False
        self._active_trade = None
        self._first_30_ratio = None
        self._trades_today = 0
        self._save_state()
        log.info("DayClassifier: daily state reset")

    # ------------------------------------------------------------------
    # ALERT 1: MORNING READ — 9:30 AM ET (6:30 PT) — always fires
    # ------------------------------------------------------------------
    def morning_read(
        self,
        snapshot: GEXSnapshot,
        eod: Optional[GEXSnapshot],
        em_levels: Optional[dict],
        prior_range: float,
        prior_em: float,
    ) -> str:
        if not em_levels:
            return "MORNING READ\nEM levels unavailable."

        anchor = em_levels["anchor"]
        em_pts = em_levels["em_pts"]
        vix = em_levels.get("vix", 0)
        open_gamma_bn = snapshot.net_gamma / 1e9
        gamma_desc = _gamma_english(open_gamma_bn)

        range_low = anchor - 0.75 * em_pts
        range_high = anchor + 0.75 * em_pts

        compression = prior_range < prior_em if prior_em > 0 else False
        comp_str = "COMPRESSED" if compression else "Normal"

        eod_gamma = eod.net_gamma if eod else 0
        coil = abs(eod_gamma / 1e9 - open_gamma_bn) > 5
        coil_str = "YES" if coil else "No"

        delta_line = _dc_delta_tracker.format_delta_line("gex_context", "morning_read", {
            "price": snapshot.curr_price, "gamma_bn": open_gamma_bn,
            "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
        })

        snap_data = {
            "price": snapshot.curr_price, "gamma_bn": open_gamma_bn,
            "call_wall": snapshot.call_wall or 0, "put_floor": snapshot.put_floor or 0,
            "spread": snapshot.spread or 0,
        }

        gex_box = _box("GEX REGIME", [
            f"Gamma: {open_gamma_bn:+.1f}Bn — {gamma_desc}",
            f"Anchor: {anchor:.0f} | EM: ±{em_pts:.0f}pts (VIX {vix:.1f})",
            f"Range: {range_low:.0f} to {range_high:.0f}",
        ])

        context_box = _box("CONTEXT", [
            f"Prior day: {comp_str} (range {prior_range:.0f} vs EM {prior_em:.0f})",
            f"Coil: {coil_str}",
        ])

        msg = f"MORNING READ\n\n{gex_box}\n\n{context_box}\n\n{delta_line}"
        return msg

    # ------------------------------------------------------------------
    # ALERT 2: MORNING BET — 10:00 AM ET (7:00 PT) — conditional
    # ------------------------------------------------------------------
    def morning_bet(
        self,
        snapshot: GEXSnapshot,
        today_snaps: list[GEXSnapshot],
        em_levels: Optional[dict],
        eod: Optional[GEXSnapshot],
        prior_day_low: float,
        prior_day_high: float,
        prior_range: float,
        prior_em: float,
    ) -> Optional[str]:
        if self._morning_bet_fired:
            return None
        if not em_levels:
            return None
        self._morning_bet_fired = True

        anchor = em_levels["anchor"]
        em_pts = em_levels["em_pts"]

        # First 30 min snaps: RTH with time_pt <= "07:00" PT
        first_30 = [s for s in today_snaps if s.timestamp_pt.strftime("%H:%M") <= "07:00"]
        if not first_30:
            return None

        open_price = today_snaps[0].curr_price
        prices_30 = [s.curr_price for s in first_30]
        high_30 = max(prices_30)
        low_30 = min(prices_30)
        first_30_range = high_30 - low_30
        ratio = first_30_range / em_pts if em_pts > 0 else 0
        self._first_30_ratio = ratio

        price_10am = snapshot.curr_price
        direction_pts = price_10am - open_price

        if ratio <= 0.3:
            delta_line = _dc_delta_tracker.format_delta_line("gex_trades", "morning_bet", {
                "price": snapshot.curr_price, "gamma_bn": snapshot.net_gamma / 1e9,
            })
            return (
                f"\U0001f535 MORNING BET — QUIET\n"
                f"First 30m ratio: {ratio:.2f} (<= 0.3)\n"
                f"Range: {first_30_range:.0f}pts vs EM {em_pts:.0f}pts\n"
                f"No trade today — low energy open.\n\n"
                f"{delta_line}"
            )

        # Direction
        if direction_pts > 0:
            side = "CALL"
            emoji = "LONG"
            stop = price_10am - 15
            target = em_levels["upper"]
            dir_word = "up"
        else:
            side = "PUT"
            emoji = "SHORT"
            stop = price_10am + 15
            target = em_levels["lower"]
            dir_word = "down"

        # Big move score
        open_gamma_bn = today_snaps[0].net_gamma / 1e9 if today_snaps else 0
        eod_gamma_bn = eod.net_gamma / 1e9 if eod else 0
        gamma_10am_bn = snapshot.net_gamma / 1e9

        coil = abs(eod_gamma_bn - open_gamma_bn) > 5
        expansion = True  # ratio > 0.3 guaranteed here
        gamma_accel = snapshot.net_gamma < -3e9 and gamma_10am_bn < open_gamma_bn - 1
        compression = prior_range < prior_em if prior_em > 0 else False
        sweep = low_30 < prior_day_low or high_30 > prior_day_high

        score = sum([coil, expansion, gamma_accel, compression, sweep])
        if score >= 2:
            label = "BIG MOVE — hold for runner, trail 10pts"
        else:
            label = "Normal — take profit at target"

        # Set active trade — time exit at 2:00 PM ET = 11:00 PT
        now_pt = datetime.now(PT_TZ)
        time_exit = now_pt.replace(hour=11, minute=0, second=0, microsecond=0)
        if time_exit <= now_pt:
            time_exit += timedelta(days=1)

        self._active_trade = ActiveTrade(
            trade_type="MORNING_BET",
            direction=side,
            entry=price_10am,
            stop=stop,
            target=target,
            time_exit_ts=time_exit.timestamp(),
            entry_time_str="10:00 AM",
        )
        self._trades_today += 1
        self._save_state()

        gamma_bn = snapshot.net_gamma / 1e9
        structure_line = (
            f"First 30m: {dir_word} {abs(direction_pts):.0f}pts | Ratio {ratio:.2f}. "
            f"Big Move: {score}/5 — {label}."
        )
        confluence = {
            "First 30m Ratio": ratio > 0.3,
            "Direction": abs(direction_pts) > 5,
            "Big Move": score >= 2,
        }
        delta_line = _dc_delta_tracker.format_delta_line("gex_trades", "morning_bet", {
            "price": price_10am, "gamma_bn": gamma_bn,
        })
        msg = format_type1_signal(
            direction=emoji,
            signal_name="MORNING BET",
            price=price_10am,
            entry=price_10am,
            stop=stop,
            target=target,
            win_pct=55,
            gamma_bn=gamma_bn,
            structure_line=structure_line,
            confluence=confluence,
            delta_line=delta_line,
            pf=snapshot.put_floor or 0,
            cw=snapshot.call_wall or 0,
            em_levels=em_levels,
        )
        return msg

    # ------------------------------------------------------------------
    # ALERT 3: PIN BUTTERFLY — 2:00 PM ET (11:00 PT) — conditional
    # ------------------------------------------------------------------
    def pin_butterfly(self, snapshot: GEXSnapshot) -> Optional[str]:
        if self._pin_fired:
            return None
        if self._trades_today >= 2:
            return None

        gamma = snapshot.net_gamma
        price = snapshot.curr_price
        call_wall = snapshot.call_wall
        distance = abs(price - call_wall)

        if gamma <= 3e9 or distance >= 15:
            return None

        self._pin_fired = True
        self._trades_today += 1
        self._save_state()

        gamma_bn = gamma / 1e9
        structure_line = (
            f"Strong positive gamma ({gamma_bn:+.1f}Bn) pinning near CW {call_wall}. "
            f"Distance: {distance:.0f}pts. Butterfly earns if price stays near {call_wall}."
        )
        delta_line = _dc_delta_tracker.format_delta_line("gex_trades", "pin_butterfly", {
            "price": price, "gamma_bn": gamma_bn, "call_wall": call_wall,
        })
        confluence = {"Gamma Pin": gamma_bn > 3, "Near CW": distance < 15}
        msg = format_type1_signal(
            direction="NEUTRAL",
            signal_name="PIN BUTTERFLY",
            price=price,
            entry=price,
            stop=call_wall - 10,
            target=call_wall,
            win_pct=60,
            gamma_bn=gamma_bn,
            structure_line=structure_line,
            confluence=confluence,
            delta_line=delta_line,
            cw=call_wall,
            pf=snapshot.put_floor or 0,
        )
        return msg

    # ------------------------------------------------------------------
    # ALERT 4: EM FADE — every snapshot — conditional
    # ------------------------------------------------------------------
    def check_em_fade(
        self,
        snapshot: GEXSnapshot,
        em_levels: Optional[dict],
    ) -> Optional[str]:
        if not em_levels:
            return None
        if self._trades_today >= 2:
            return None
        # Skip if first_30_ratio > 0.5 (trend day)
        if self._first_30_ratio is not None and self._first_30_ratio > 0.5:
            return None
        # Skip if morning bet is active
        if self._active_trade and self._active_trade.trade_type == "MORNING_BET":
            return None
        # Skip if a directional trade already fired (morning bet OR em fade)
        if self._morning_bet_fired and self._first_30_ratio is not None and self._first_30_ratio > 0.3:
            return None

        anchor = em_levels["anchor"]
        em_pts = em_levels["em_pts"]
        price = snapshot.curr_price

        # T2 (early) at 0.5 EM — tighter stop/target
        early_long_level = anchor - 0.5 * em_pts
        early_short_level = anchor + 0.5 * em_pts
        # T1 (strong) at 0.75 EM — original thresholds
        fade_long_level = anchor - 0.75 * em_pts
        fade_short_level = anchor + 0.75 * em_pts
        em_position = (price - anchor) / em_pts if em_pts > 0 else 0

        now_pt = datetime.now(PT_TZ)
        time_str = now_pt.strftime("%I:%M %p").lstrip("0")
        time_exit = now_pt + timedelta(minutes=60)

        # T2 LONG — early fade at 0.5 EM
        if not self._em_early_long_fired and price < early_long_level:
            self._em_early_long_fired = True
            stop = fade_long_level  # 0.75 EM
            target = anchor

            self._active_trade = ActiveTrade(
                trade_type="EM_FADE_LONG",
                direction="CALL",
                entry=price,
                stop=stop,
                target=target,
                time_exit_ts=time_exit.timestamp(),
                entry_time_str=time_str,
            )
            self._trades_today += 1
            self._save_state()

            delta_line = _dc_delta_tracker.format_delta_line("gex_trades", "em_fade", {
                "price": price, "gamma_bn": snapshot.net_gamma / 1e9,
            })
            return format_type1_signal(
                direction="LONG", signal_name="EM FADE EARLY",
                price=price, entry=price, stop=stop, target=target,
                win_pct=55, gamma_bn=snapshot.net_gamma / 1e9,
                structure_line=f"Hit {early_long_level:.0f} (0.5 EM | position: {em_position:.2f}). Exit in 60 min.",
                confluence={"EM Stretch": True, "Early Fade": True},
                delta_line=delta_line, em_levels=em_levels,
                pf=snapshot.put_floor or 0, cw=snapshot.call_wall or 0,
            )

        # T2 SHORT — early fade at 0.5 EM
        if not self._em_early_short_fired and price > early_short_level:
            self._em_early_short_fired = True
            stop = fade_short_level  # 0.75 EM
            target = anchor

            self._active_trade = ActiveTrade(
                trade_type="EM_FADE_SHORT",
                direction="PUT",
                entry=price,
                stop=stop,
                target=target,
                time_exit_ts=time_exit.timestamp(),
                entry_time_str=time_str,
            )
            self._trades_today += 1
            self._save_state()

            delta_line = _dc_delta_tracker.format_delta_line("gex_trades", "em_fade", {
                "price": price, "gamma_bn": snapshot.net_gamma / 1e9,
            })
            return format_type1_signal(
                direction="SHORT", signal_name="EM FADE EARLY",
                price=price, entry=price, stop=stop, target=target,
                win_pct=55, gamma_bn=snapshot.net_gamma / 1e9,
                structure_line=f"Hit {early_short_level:.0f} (0.5 EM | position: {em_position:.2f}). Exit in 60 min.",
                confluence={"EM Stretch": True, "Early Fade": True},
                delta_line=delta_line, em_levels=em_levels,
                pf=snapshot.put_floor or 0, cw=snapshot.call_wall or 0,
            )

        # T1 LONG — strong fade at 0.75 EM
        if not self._em_fade_long_fired and price < fade_long_level:
            self._em_fade_long_fired = True
            stop = em_levels["lower"]
            target = anchor - 0.25 * em_pts

            self._active_trade = ActiveTrade(
                trade_type="EM_FADE_LONG",
                direction="CALL",
                entry=price,
                stop=stop,
                target=target,
                time_exit_ts=time_exit.timestamp(),
                entry_time_str=time_str,
            )
            self._trades_today += 1
            self._save_state()

            delta_line = _dc_delta_tracker.format_delta_line("gex_trades", "em_fade", {
                "price": price, "gamma_bn": snapshot.net_gamma / 1e9,
            })
            return format_type1_signal(
                direction="LONG", signal_name="EM FADE",
                price=price, entry=price, stop=stop, target=target,
                win_pct=60, gamma_bn=snapshot.net_gamma / 1e9,
                structure_line=f"Hit {fade_long_level:.0f} (0.75 EM | position: {em_position:.2f}). Strong fade zone. Exit in 60 min.",
                confluence={"EM Stretch": True, "Strong Fade": True},
                delta_line=delta_line, em_levels=em_levels,
                pf=snapshot.put_floor or 0, cw=snapshot.call_wall or 0,
            )

        # T1 SHORT — strong fade at 0.75 EM
        if not self._em_fade_short_fired and price > fade_short_level:
            self._em_fade_short_fired = True
            stop = em_levels["upper"]
            target = anchor + 0.25 * em_pts

            self._active_trade = ActiveTrade(
                trade_type="EM_FADE_SHORT",
                direction="PUT",
                entry=price,
                stop=stop,
                target=target,
                time_exit_ts=time_exit.timestamp(),
                entry_time_str=time_str,
            )
            self._trades_today += 1
            self._save_state()

            delta_line = _dc_delta_tracker.format_delta_line("gex_trades", "em_fade", {
                "price": price, "gamma_bn": snapshot.net_gamma / 1e9,
            })
            return format_type1_signal(
                direction="SHORT", signal_name="EM FADE",
                price=price, entry=price, stop=stop, target=target,
                win_pct=60, gamma_bn=snapshot.net_gamma / 1e9,
                structure_line=f"Hit {fade_short_level:.0f} (0.75 EM | position: {em_position:.2f}). Strong fade zone. Exit in 60 min.",
                confluence={"EM Stretch": True, "Strong Fade": True},
                delta_line=delta_line, em_levels=em_levels,
                pf=snapshot.put_floor or 0, cw=snapshot.call_wall or 0,
            )

        return None

    # ------------------------------------------------------------------
    # TRADE EXIT — every snapshot
    # ------------------------------------------------------------------
    def check_trade_exit(self, snapshot: GEXSnapshot) -> Optional[str]:
        trade = self._active_trade
        if not trade:
            return None

        price = snapshot.curr_price
        now_ts = _time_mod.time()
        now_pt = datetime.now(PT_TZ)
        time_str = now_pt.strftime("%I:%M %p").lstrip("0")
        result = None

        if trade.direction == "CALL":
            if price <= trade.stop:
                result = f"\u274c STOPPED \u2014 {trade.trade_type}\nSPX {price:.0f} | {time_str}"
            elif price >= trade.target:
                result = f"\u2705 TARGET \u2014 {trade.trade_type}\nSPX {price:.0f} | {time_str}"
        else:  # PUT
            if price >= trade.stop:
                result = f"\u274c STOPPED \u2014 {trade.trade_type}\nSPX {price:.0f} | {time_str}"
            elif price <= trade.target:
                result = f"\u2705 TARGET \u2014 {trade.trade_type}\nSPX {price:.0f} | {time_str}"

        # Time exit
        if result is None and now_pt.timestamp() >= trade.time_exit_ts:
            result = f"\u23f0 TIME EXIT \u2014 {trade.trade_type}\nSPX {price:.0f} | {time_str}"

        if result:
            self._active_trade = None
            # Check if all trades are done
            done_msg = ""
            if self._trades_today >= 2 or (
                self._morning_bet_fired and (self._em_fade_long_fired or self._em_fade_short_fired)
            ):
                done_msg = "\n\n\U0001f507 DONE FOR TODAY"
            elif not self._active_trade and self._pin_fired:
                done_msg = "\n\n\U0001f507 DONE FOR TODAY"
            return result + done_msg

        return None
