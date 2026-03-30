"""Telethon async listener + main processing pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import json
from datetime import datetime, time, timedelta
from typing import Optional

import pytz
import yaml
from dotenv import load_dotenv
from telethon import TelegramClient, events

from gex_parser import GEXSnapshot, parse_message, classify_session
from gex_db import (
    save_snapshot,
    get_recent_snapshots,
    get_latest_snapshot,
    get_previous_trading_day_eod,
    get_today_snapshots,
    save_daily_summary,
    purge_old_data,
    get_conn,
    get_eod_snapshot,
)
from trend_engine import (
    compute_fifteen_min_metrics,
    compute_one_hour_metrics,
    compute_overnight_drift,
    compute_advanced_metrics,
    compute_trend_dashboard,
    compute_gamma_concentration_index,
    compute_pin_probability,
    Regime,
)
from signal_interpreter import SignalInterpreter, Signal, SignalType
from discord_alerts import DiscordAlertSender
from telegram_alerts import TelegramAlertSender
from recap import build_hourly_recap_message, is_rth_time
from scheduler import Scheduler
from trend_review import run_trend_review
from spx_million_parser import parse_spx_million_message, format_spx_million_signal
from consensus_tracker import ConsensusTracker
from day_classifier import DayClassifier
from delivery_watchdog import DeliveryWatchdog
from conviction_engine import ConvictionEngine
from qqq_analysis import QQQAnalyzer

PT_TZ = pytz.timezone("US/Pacific")
log = logging.getLogger(__name__)

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path, "r") as f:
    CFG = yaml.safe_load(f)


class GEXPipeline:
    """Main processing pipeline for GEX data."""

    def __init__(self):
        self.interpreter = SignalInterpreter()
        self.discord = DiscordAlertSender()
        self.telegram = TelegramAlertSender()
        self.scheduler = Scheduler()
        self._last_regime_compute: float = 0
        self._last_overnight_compute: float = 0
        self._last_one_hour = None
        self._last_advanced = None
        self._cached_eod = None
        self._cached_eod_date = None
        self._last_hourly_recap_hour = None
        self._hourly_recap_window_min = CFG.get("scheduled", {}).get("hourly_recap_window_min", 5)
        self._last_message_received_at: Optional[datetime] = None
        self._last_snapshot_received_at: Optional[datetime] = None
        self._last_snapshot_timestamp_pt: Optional[datetime] = None
        self._last_snapshot_price: Optional[float] = None
        # Rolling price buffer to detect stale-stream oscillation
        from collections import deque
        self._recent_prices: deque = deque(maxlen=10)
        self._data_lag_active: bool = False
        self._data_lag_started_at: Optional[datetime] = None
        self._data_feed_cfg = CFG.get("data_feed", {})
        self._data_feed_stale_secs = int(self._data_feed_cfg.get("stale_seconds", 300))
        self._data_feed_channel = self._data_feed_cfg.get("alert_channel", "gex_engine")
        self._flip_band_cfg = CFG.get("flip_bands", {})
        self._flip_band_cache = []
        self._flip_band_last_compute: Optional[datetime] = None
        self._move_ctx_cfg = CFG.get("move_context", {})
        self._move_ctx_stats = None
        self._move_ctx_last_compute: Optional[datetime] = None
        self._quiet_cfg = CFG.get("quiet_summary", {})
        self._quiet_channel = str(self._quiet_cfg.get("alert_channel", "gex_engine"))
        self._micro_cfg = CFG.get("micro_pulse", {})
        self._micro_channel = str(self._micro_cfg.get("alert_channel", "gex_engine"))
        self._last_micro_pulse_ts: Optional[datetime] = None
        self._last_micro_pulse_price: Optional[float] = None
        self._last_internal_alert_ts: Optional[datetime] = None
        self._last_quiet_summary_ts: Optional[datetime] = None
        self._gap_alert_cfg = CFG.get("gap_alert", {})
        self._gap_alert_date: Optional[str] = None
        self._day_profile_cfg = CFG.get("day_profile", {})
        self._day_profile_cache_date: Optional[str] = None
        self._day_profile_cache = None
        self._day_profile_last_compute: Optional[datetime] = None
        self._rth_open_date: Optional[str] = None
        self._rth_open_price: Optional[float] = None
        self._daily_ma20: Optional[float] = None
        self._daily_ma20_date: Optional[str] = None
        # Live MA cache for MA_SNAP signals
        self._live_ma_cache: Optional[dict] = None
        self._live_ma_cache_ts: float = 0
        alerts_cfg = CFG.get("alerts", {})
        alert_mode = str(alerts_cfg.get("mode", "full")).strip().lower()
        if alert_mode not in {"full", "deep_context"}:
            log.warning(f"Unknown alerts.mode='{alert_mode}'. Falling back to 'full'.")
            alert_mode = "full"
        self._alert_mode = alert_mode
        # Consensus tracker for Arabic channel directional calls
        cons_cfg = CFG.get("consensus", {})
        self.consensus = ConsensusTracker(
            window_minutes=int(cons_cfg.get("window_minutes", 15)),
            min_channels=int(cons_cfg.get("min_channels", 1)),
        )
        self._consensus_channels = set(cons_cfg.get("channels", []))
        self._consensus_cooldown_secs = int(cons_cfg.get("cooldown_seconds", 900))
        self._last_consensus_alert_ts: Optional[float] = None
        self.classifier = DayClassifier()
        self.conviction = ConvictionEngine(CFG)
        _db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CFG["database"]["path"])
        self.qqq = QQQAnalyzer(db_path=_db_path, config=CFG)
        self._setup_scheduler()
        self.watchdog = DeliveryWatchdog(self)
        self.watchdog.wrap_senders()

    def _intraday_alerts_enabled(self) -> bool:
        return self._alert_mode == "full"

    def _context_alerts_enabled(self) -> bool:
        return self._alert_mode in {"full", "deep_context"}

    def _setup_scheduler(self):
        """Scheduler disabled — data-only pipeline, no signal dispatch."""
        pass

    def _get_day_profile(self, now: datetime):
        today = now.strftime("%Y-%m-%d")
        interval_sec = int(self._day_profile_cfg.get("compute_interval_seconds", 60))

        if self._day_profile_cache_date != today:
            self._day_profile_cache_date = today
            self._day_profile_cache = None
            self._day_profile_last_compute = None

        if (
            self._day_profile_last_compute is not None
            and (now - self._day_profile_last_compute).total_seconds() < interval_sec
            and self._day_profile_cache is not None
        ):
            return self._day_profile_cache

        self._day_profile_cache = self._compute_day_profile(now)
        self._day_profile_last_compute = now
        return self._day_profile_cache

    def _compute_day_profile(self, now: datetime):
        rth_today = get_today_snapshots(session_tag="RTH")
        if not rth_today:
            return None

        open_snap = rth_today[0]
        early_cutoff = now.replace(hour=7, minute=30, second=0, microsecond=0)
        early = [s for s in rth_today if s.timestamp_pt <= early_cutoff]
        if not early:
            early = [open_snap]

        open_gamma_bn = open_snap.net_gamma / 1e9
        min_early_gamma_bn = min(s.net_gamma / 1e9 for s in early)
        max_early_gamma_bn = max(s.net_gamma / 1e9 for s in early)

        open_neg_bn = float(self._day_profile_cfg.get("open_negative_bn", -5.0))
        early_crash_bn = float(self._day_profile_cfg.get("early_crash_bn", -10.0))
        open_pos_bn = float(self._day_profile_cfg.get("open_positive_bn", 5.0))
        early_not_too_negative_bn = float(
            self._day_profile_cfg.get("early_not_too_negative_bn", -5.0)
        )

        if open_gamma_bn <= open_neg_bn or min_early_gamma_bn <= early_crash_bn:
            return {
                "id": "risk_off_fast_down",
                "flag": "day_profile_risk_off_fast_down",
                "label": "RISK-OFF FAST-DOWN",
                "note": (
                    "Day profile: RISK-OFF FAST-DOWN "
                    f"(open G {open_gamma_bn:+.2f} Bn, early min {min_early_gamma_bn:+.2f} Bn)"
                ),
                "open_gamma_bn": open_gamma_bn,
                "min_early_gamma_bn": min_early_gamma_bn,
                "max_early_gamma_bn": max_early_gamma_bn,
            }

        if open_gamma_bn >= open_pos_bn and min_early_gamma_bn > early_not_too_negative_bn:
            return {
                "id": "controlled_up_or_range",
                "flag": "day_profile_controlled_up_or_range",
                "label": "CONTROLLED UP/RANGE",
                "note": (
                    "Day profile: CONTROLLED UP/RANGE "
                    f"(open G {open_gamma_bn:+.2f} Bn, early min {min_early_gamma_bn:+.2f} Bn)"
                ),
                "open_gamma_bn": open_gamma_bn,
                "min_early_gamma_bn": min_early_gamma_bn,
                "max_early_gamma_bn": max_early_gamma_bn,
            }

        return {
            "id": "mixed_transition",
            "flag": "day_profile_mixed_transition",
            "label": "MIXED/TRANSITION",
            "note": (
                "Day profile: MIXED/TRANSITION "
                f"(open G {open_gamma_bn:+.2f} Bn, early range "
                f"{min_early_gamma_bn:+.2f}..{max_early_gamma_bn:+.2f} Bn)"
            ),
            "open_gamma_bn": open_gamma_bn,
            "min_early_gamma_bn": min_early_gamma_bn,
            "max_early_gamma_bn": max_early_gamma_bn,
        }

    def _update_rth_open_price(self, snapshot: GEXSnapshot):
        date_pt = snapshot.timestamp_pt.strftime("%Y-%m-%d")
        if self._rth_open_date != date_pt:
            self._rth_open_date = date_pt
            self._rth_open_price = None

        if self._rth_open_price is not None:
            return

        rth_today = get_today_snapshots(session_tag="RTH")
        if rth_today:
            self._rth_open_price = rth_today[0].curr_price
        else:
            self._rth_open_price = snapshot.curr_price

    def _get_daily_ma20(self) -> Optional[float]:
        """Get cached daily MA20, computing once per day."""
        today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
        if self._daily_ma20_date == today and self._daily_ma20 is not None:
            return self._daily_ma20
        self._daily_ma20 = self._compute_daily_ma20()
        self._daily_ma20_date = today
        if self._daily_ma20 is not None:
            log.info(f"Daily MA20 = {self._daily_ma20:.2f}")
        return self._daily_ma20

    def _compute_daily_ma20(self) -> Optional[float]:
        """Get daily MA20 from DB closes. Called once per day."""
        try:
            conn = get_conn()
            rows = conn.execute("""
                SELECT s1.date_pt,
                    (SELECT curr_price FROM gex_snapshots s2
                     WHERE s2.date_pt = s1.date_pt AND s2.session_tag='RTH'
                     ORDER BY s2.timestamp_pt DESC LIMIT 1) as close_px
                FROM (SELECT DISTINCT date_pt FROM gex_snapshots
                      WHERE session_tag='RTH'
                      ORDER BY date_pt DESC LIMIT 25) s1
                ORDER BY s1.date_pt
            """).fetchall()
            conn.close()
            closes = [r["close_px"] for r in rows if r["close_px"] is not None]
            if not closes:
                return None
            n = min(len(closes), 20)
            return sum(closes[-n:]) / n
        except Exception as e:
            log.error(f"Failed to compute daily MA20: {e}")
            return None

    def _get_live_ma_values(self) -> Optional[dict]:
        """Compute live MA values from raw snapshots for MA_SNAP signals.

        Queries last ~6000 RTH snapshots (~20 trading days), resamples into
        3m/5m/15m/30m/60m bars, computes rolling MAs. Cached for 5 minutes.
        Returns dict of {ma_key: float, ma_key_recent_max_dist: float}.
        """
        import pandas as pd

        now_ts = datetime.now(PT_TZ).timestamp()
        if self._live_ma_cache is not None and (now_ts - self._live_ma_cache_ts) < 300:
            return self._live_ma_cache

        try:
            conn = get_conn()
            rows = conn.execute(
                "SELECT timestamp_pt, curr_price FROM gex_snapshots "
                "WHERE session_tag='RTH' "
                "ORDER BY timestamp_pt DESC LIMIT 6000"
            ).fetchall()
            conn.close()
        except Exception as e:
            log.error(f"Failed to query snapshots for live MAs: {e}")
            return self._live_ma_cache

        if not rows or len(rows) < 100:
            return None

        # Build DataFrame — rows are descending, reverse for chronological order
        timestamps = []
        prices = []
        for r in reversed(rows):
            ts_val = r["timestamp_pt"]
            if isinstance(ts_val, (int, float)):
                timestamps.append(datetime.fromtimestamp(ts_val, tz=PT_TZ))
            else:
                timestamps.append(ts_val)
            prices.append(r["curr_price"])

        df = pd.DataFrame({"price": prices}, index=pd.DatetimeIndex(timestamps))
        # Drop duplicate timestamps (keep last)
        df = df[~df.index.duplicated(keep="last")]

        current_price = prices[-1]

        # Define MAs to compute: {ma_key: (resample_freq, period)}
        ma_specs = {
            "3m_200":  ("3min",  200),
            "5m_20":   ("5min",   20),
            "5m_200":  ("5min",  200),
            "15m_50":  ("15min",  50),
            "15m_200": ("15min", 200),
            "30m_20":  ("30min",  20),
            "30m_200": ("30min", 200),
            "60m_20":  ("60min",  20),
            "60m_50":  ("60min",  50),
        }

        result = {}
        for ma_key, (freq, period) in ma_specs.items():
            try:
                bars = df["price"].resample(freq).last().dropna()
                if len(bars) < period:
                    continue
                ma_series = bars.rolling(period).mean()
                ma_val = ma_series.iloc[-1]
                if pd.isna(ma_val):
                    continue
                result[ma_key] = float(ma_val)

                # Compute recent max distance (last ~10 min of bars) for retrace check
                # Look at last 5-10 bars of the resampled timeframe
                lookback_bars = max(3, min(10, int(600 / int(freq.replace("min", "")))))
                recent_slice = bars.iloc[-lookback_bars:]
                recent_prices = recent_slice.values
                max_dist = max(abs(p - ma_val) for p in recent_prices) if len(recent_prices) > 0 else 0
                result[f"{ma_key}_recent_max_dist"] = float(max_dist)
            except Exception as e:
                log.debug(f"MA computation failed for {ma_key}: {e}")
                continue

        # VWAP and intraday stats from today's prices
        try:
            today_date = datetime.now(PT_TZ).date()
            today_mask = [ts.date() == today_date for ts in timestamps]
            today_prices = [p for p, m in zip(prices, today_mask) if m]
            if today_prices:
                result["vwap"] = sum(today_prices) / len(today_prices)
                result["day_open"] = today_prices[0]
                result["day_high"] = max(today_prices)
                result["day_low"] = min(today_prices)
        except Exception as e:
            log.debug(f"VWAP computation failed: {e}")

        if result:
            self._live_ma_cache = result
            self._live_ma_cache_ts = now_ts
            log.info(f"Live MAs computed: {len([k for k in result if '_recent' not in k and k not in ('vwap','day_open','day_high','day_low')])} MAs")

        return result

    def process_message(self, text: str, envelope_dt: Optional[datetime] = None):
        """Main pipeline: parse -> store -> compute -> signal -> alert."""
        # Log raw message for debugging
        log.info("Raw message (%d chars): %s...", len(text), text[:300].encode('ascii', 'replace').decode())
        now = datetime.now(PT_TZ)
        self._last_message_received_at = now

        snapshot = parse_message(text, envelope_dt)
        if not snapshot:
            log.debug("Message did not parse into valid GEX snapshot.")
            return
        self._sanitize_snapshot_timestamp(snapshot, envelope_dt, now)

        # Skip expired echoes entirely
        if snapshot.session_tag == "EXPIRED_ECHO":
            log.debug("Skipping EXPIRED_ECHO message.")
            return

        # Reject stale-stream oscillation: SpotGamma channel sends two streams —
        # a live feed and a stale replay. The stale stream repeats the same price
        # while the real stream moves. Detect: if this price appeared 2+ times in
        # the last 10 snapshots while other significantly different prices also
        # appeared, this is a stale echo.
        if snapshot.session_tag == "RTH" and len(self._recent_prices) >= 4:
            price = snapshot.curr_price
            same_count = sum(1 for p in self._recent_prices if abs(p - price) < 0.02)
            diff_count = sum(1 for p in self._recent_prices if abs(p - price) > 5)
            if same_count >= 2 and diff_count >= 2:
                log.info(
                    f"Dropping stale echo: price={price:.2f} appeared {same_count}x "
                    f"in last {len(self._recent_prices)} snapshots while {diff_count} "
                    f"others were >5pts away"
                )
                return

        # Store to SQLite
        row_id = save_snapshot(snapshot)
        log.info(
            f"Saved snapshot #{row_id}: price={snapshot.curr_price}, "
            f"gamma={snapshot.net_gamma_bn:+.2f}Bn, session={snapshot.session_tag}"
        )
        self._last_snapshot_received_at = now
        self._last_snapshot_timestamp_pt = snapshot.timestamp_pt
        self._last_snapshot_price = snapshot.curr_price
        self._recent_prices.append(snapshot.curr_price)

        if self._data_lag_active:
            self._fire_data_restored(now)

        # Check scheduled events
        self.scheduler.check_all()

        now = datetime.now(PT_TZ)
        now_ts = now.timestamp()

        if snapshot.session_tag == "RTH":
            self._process_rth(snapshot, now_ts)
        elif snapshot.session_tag in ("NEXT_DAY_PREVIEW", "PREMARKET"):
            self._process_off_hours(snapshot, now_ts)

    def _process_rth(self, snapshot: GEXSnapshot, now_ts: float):
        """RTH processing: always compute 15-min, regime every 5 min, check all signals."""
        self._update_rth_open_price(snapshot)

        # Always compute 15-min metrics
        snapshots_15m = get_recent_snapshots(15, session_tag="RTH")
        fifteen_min = compute_fifteen_min_metrics(snapshots_15m)

        one_hour = self._last_one_hour
        advanced = self._last_advanced
        regime_interval = CFG["analysis"]["regime_compute_interval"] * 60

        # Compute 1-hr metrics + regime every 5 minutes
        snapshots_1h = get_recent_snapshots(60, session_tag="RTH")
        if now_ts - self._last_regime_compute >= regime_interval:
            if fifteen_min and len(snapshots_1h) >= 3:
                one_hour = compute_one_hour_metrics(snapshots_1h, fifteen_min)

                # Compute advanced metrics
                close_time = datetime.now(PT_TZ).replace(hour=13, minute=0, second=0)
                minutes_to_close = max(0, (close_time - datetime.now(PT_TZ)).total_seconds() / 60)
                advanced = compute_advanced_metrics(snapshots_1h, minutes_to_close)

                self._last_regime_compute = now_ts
                self._last_one_hour = one_hour
                self._last_advanced = advanced

        # Overnight drift for trend context (do not emit overnight alerts during RTH)
        today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
        if self._cached_eod_date != today:
            self._cached_eod = get_previous_trading_day_eod()
            self._cached_eod_date = today

        trend_overnight = None
        if self._cached_eod:
            trend_overnight = compute_overnight_drift(self._cached_eod, snapshot)

        trend_dashboard = compute_trend_dashboard(snapshots_1h, trend_overnight)
        if trend_dashboard and self._rth_open_price is not None:
            trend_dashboard.open_price = self._rth_open_price
        flip_bands = self._get_flip_bands()
        if trend_dashboard and flip_bands:
            band_pts = float(self._flip_band_cfg.get("band_pts", 5))
            band_pts_int = int(round(band_pts))
            band_str = ", ".join(
                [f"{b['strike']} (n={b['count']}, max {b['max_ratio']:.2f})" for b in flip_bands]
            )
            trend_dashboard.notes.append(f"Hist flip bands: {band_str}")
            for b in flip_bands:
                if abs(snapshot.curr_price - b["strike"]) <= band_pts:
                    trend_dashboard.notes.append(
                        f"Price within {band_pts_int} pts of flip band {b['strike']} (n={b['count']})"
                    )
                    trend_dashboard.flags.append("hist_flip_band")

        move_ctx_note, move_ctx_flag = self._compute_move_context(snapshots_1h, one_hour)
        if trend_dashboard and move_ctx_note:
            trend_dashboard.notes.append(move_ctx_note)
            if move_ctx_flag:
                trend_dashboard.flags.append(move_ctx_flag)

        day_profile = self._get_day_profile(snapshot.timestamp_pt)
        if trend_dashboard and day_profile:
            trend_dashboard.notes.append(day_profile["note"])
            trend_dashboard.flags.append(day_profile["flag"])

        # --- Data pipeline complete ---
        # Metrics (fifteen_min, one_hour, advanced, trend_dashboard) are computed
        # and cached for API consumption. No signal generation or alert dispatch.

    def _check_classifier_snapshot(self, snapshot: GEXSnapshot):
        """Run EM fade + trade exit checks on every RTH snapshot."""
        em_levels = self.interpreter.compute_expected_move()

        fade_msg = self.classifier.check_em_fade(snapshot, em_levels)
        if fade_msg:
            self._send_classifier_alert("EM Fade", fade_msg)

        exit_msg = self.classifier.check_trade_exit(snapshot)
        if exit_msg:
            self._send_classifier_alert("Trade Exit", exit_msg)

    def _send_classifier_alert(self, title: str, message: str):
        """Send a day_classifier alert to Discord AND Telegram."""
        from signal_interpreter import Signal, SignalType, SIGNAL_CHANNEL_MAP
        _classifier_type_map = {
            "Morning Read": SignalType.MORNING_READ,
            "Morning Bet": SignalType.MORNING_BET,
            "Pin Butterfly": SignalType.PIN_TRADE,
            "EM Fade": SignalType.EM_FADE,
            "Trade Exit": SignalType.EM_FADE,  # exits share EM_FADE cooldown
        }
        sig_type = _classifier_type_map.get(title, SignalType.EM_FADE)
        channel = SIGNAL_CHANNEL_MAP.get(sig_type, "gex_trades")
        signal = Signal(
            signal_type=sig_type,
            title=title,
            message=message,
            channel=channel,
            priority=1,
            timestamp=datetime.now(PT_TZ),
        )
        disc_ok = self.discord.send_signal(signal)
        tg_ok = self.telegram.send_signal(signal)
        log.info(f"Classifier [{title}]: Discord={'OK' if disc_ok else 'SKIP'}, Telegram={'OK' if tg_ok else 'SKIP'}")

    def _check_market_intel(self, snapshot: GEXSnapshot, one_hour):
        """Gamma brain: when Arabic channels call a direction, check gamma mechanics.

        Scenarios that make money:
        1. CW Fade: SHORT + pos gamma + price above CW = dealers sell rallies
        2. Neg gamma acceleration: SHORT + neg gamma = selling feeds on itself
        3. Rubber band: LONG + neg gamma + near put floor = dealers buy dips
        4. CW Breakout: LONG + price punches THROUGH call wall with momentum
           = dealers forced to chase, pin thesis broken, ride the breakout
        5. PF Breakdown: SHORT + price crashes THROUGH put floor with momentum
           = put floor failed, acceleration down
        6. Late session amplifier: final hour 0DTE gamma is extreme, moves are fast
        """
        import time as time_mod

        consensus = self.consensus.get_consensus()
        if not consensus.direction:
            return None

        # Cooldown check
        now_ts = time_mod.time()
        if (
            self._last_consensus_alert_ts is not None
            and (now_ts - self._last_consensus_alert_ts) < self._consensus_cooldown_secs
        ):
            return None

        gamma_bn = snapshot.net_gamma / 1e9
        price = snapshot.curr_price
        cw = snapshot.call_wall
        pf = snapshot.put_floor

        # Price velocity: how fast is price moving? (5-min lookback)
        recents = get_recent_snapshots(5, session_tag="RTH")
        price_velocity = 0.0  # pts per 5 min
        gamma_velocity = 0.0  # Bn per 5 min
        if recents and len(recents) >= 2:
            price_velocity = recents[-1].curr_price - recents[0].curr_price
            gamma_velocity = (recents[-1].net_gamma - recents[0].net_gamma) / 1e9

        # Late session detection (final hour = 12:00-13:00 PT)
        now_pt = datetime.now(PT_TZ).time()
        is_late = now_pt >= time(12, 0)
        late_label = " [LATE SESSION]" if is_late else ""

        # Targets are more aggressive in final hour
        target_mult = 1.5 if is_late else 1.0

        verdict = None
        reason = None
        trade_side = None
        entry = None
        stop = None
        target = None
        instrument = None
        setup_name = None

        if consensus.direction == "LONG":
            # === SCENARIO 3: Rubber band (neg gamma bounce) ===
            if gamma_bn < -5:
                reason = (
                    f"Gamma at {gamma_bn:+.1f}Bn — dealers forced to buy on dips. "
                    f"Rubber band effect near put floor {pf:.0f}. "
                    f"Negative gamma = dealer hedging creates mechanical bounce."
                )
                trade_side = "CALL"
                entry = price
                stop = pf - 5
                target = price + int(15 * target_mult)
                instrument = f"SPX 0DTE {round(price / 5) * 5 + 5}C"
                verdict = "CONFIRMED"
                setup_name = "RUBBER BAND"

            # === SCENARIO 4: CW Breakout ===
            # Price above call wall + rising = dealers forced to chase
            # This is what the 12:45 trade was — NOT a pin, a breakout
            elif gamma_bn > 0 and price > cw + 3 and price_velocity > 3:
                reason = (
                    f"Price {price:.0f} broke above Call Wall {cw:.0f} with momentum "
                    f"({price_velocity:+.1f} pts/5min). Gamma at {gamma_bn:+.1f}Bn. "
                    f"Dealers who were short delta are now forced to buy to hedge. "
                    f"Pin thesis BROKEN — this is a breakout chase."
                )
                trade_side = "CALL"
                entry = price
                stop = cw - 2  # back below CW = breakout failed
                target = price + int(15 * target_mult)
                instrument = f"SPX 0DTE {round(price / 5) * 5 + 5}C"
                verdict = "BREAKOUT"
                setup_name = "CW BREAKOUT"

            # === Gamma rising fast toward consensus direction ===
            elif gamma_velocity > 1.0 and price_velocity > 5:
                reason = (
                    f"Gamma surging {gamma_velocity:+.1f}Bn/5min with price "
                    f"velocity {price_velocity:+.1f} pts/5min. "
                    f"Momentum building in consensus direction. "
                    f"Gamma {gamma_bn:+.1f}Bn, CW {cw:.0f}."
                )
                trade_side = "CALL"
                entry = price
                stop = price - 8
                target = price + int(12 * target_mult)
                instrument = f"SPX 0DTE {round(price / 5) * 5 + 5}C"
                verdict = "MOMENTUM"
                setup_name = "GAMMA SURGE"

            else:
                log.info(
                    f"Market Intel: consensus LONG | G={gamma_bn:+.1f}Bn "
                    f"CW={cw:.0f} px={price:.0f} vel={price_velocity:+.1f} — SKIP"
                )
                return None

        elif consensus.direction == "SHORT":
            # === SCENARIO 1: CW Fade (pos gamma ceiling) ===
            if gamma_bn > 5 and price >= cw - 5:
                reason = (
                    f"Gamma at {gamma_bn:+.1f}Bn — dealers short delta above "
                    f"Call Wall {cw:.0f}. Every rally attempt gets sold. "
                    f"CW Fade: extreme positive gamma = mechanical ceiling."
                )
                trade_side = "PUT"
                entry = price
                stop = cw + 12
                target = price - int(20 * target_mult)
                instrument = f"SPX 0DTE {round(price / 5) * 5 - 5}P"
                verdict = "CONFIRMED"
                setup_name = "CW FADE"

            # === SCENARIO 2: Neg gamma acceleration ===
            elif gamma_bn < -5:
                reason = (
                    f"Gamma at {gamma_bn:+.1f}Bn — no floor. Dealer hedging "
                    f"amplifies every downtick. Selling feeds on itself."
                )
                trade_side = "PUT"
                entry = price
                stop = price + 12
                target = price - int(25 * target_mult)
                instrument = f"SPX 0DTE {round(price / 5) * 5 - 5}P"
                verdict = "CONFIRMED"
                setup_name = "NEG GAMMA ACCEL"

            # === SCENARIO 5: PF Breakdown ===
            elif price < pf - 3 and price_velocity < -3:
                reason = (
                    f"Price {price:.0f} crashed below Put Floor {pf:.0f} with "
                    f"momentum ({price_velocity:+.1f} pts/5min). "
                    f"Gamma {gamma_bn:+.1f}Bn. Put floor FAILED — no support. "
                    f"Acceleration down."
                )
                trade_side = "PUT"
                entry = price
                stop = pf + 2
                target = price - int(20 * target_mult)
                instrument = f"SPX 0DTE {round(price / 5) * 5 - 5}P"
                verdict = "BREAKDOWN"
                setup_name = "PF BREAKDOWN"

            # === Gamma falling + price falling = momentum short ===
            elif gamma_velocity < -1.0 and price_velocity < -5:
                reason = (
                    f"Gamma collapsing {gamma_velocity:+.1f}Bn/5min with price "
                    f"falling {price_velocity:+.1f} pts/5min. "
                    f"Momentum building to downside. "
                    f"Gamma {gamma_bn:+.1f}Bn, PF {pf:.0f}."
                )
                trade_side = "PUT"
                entry = price
                stop = price + 8
                target = price - int(12 * target_mult)
                instrument = f"SPX 0DTE {round(price / 5) * 5 - 5}P"
                verdict = "MOMENTUM"
                setup_name = "GAMMA COLLAPSE"

            else:
                log.info(
                    f"Market Intel: consensus SHORT | G={gamma_bn:+.1f}Bn "
                    f"PF={pf:.0f} px={price:.0f} vel={price_velocity:+.1f} — SKIP"
                )
                return None
        else:
            return None

        # Build the alert
        direction_label = "SHORT" if trade_side == "PUT" else "LONG"
        dir_emoji = "\U0001f7e2" if direction_label == "LONG" else "\U0001f534"
        confidence = consensus.confidence
        channels_str = ", ".join(c.channel for c in consensus.calls)
        risk = abs(stop - entry)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0

        # Type 2: Channel Relay with GEX validation
        from output_formatter import format_type2_relay
        from delta_tracker import DeltaTracker

        if not hasattr(self, '_relay_delta_tracker'):
            self._relay_delta_tracker = DeltaTracker()

        checks = {
            "Consensus Direction": True,
            "Gamma Alignment": (gamma_bn < -3 if direction_label == "LONG" else gamma_bn > 3),
            "Price Velocity": abs(price_velocity) > 3,
            "Wall Proximity": (abs(price - pf) < 15 if direction_label == "LONG" else abs(price - cw) < 15),
        }

        delta_line = self._relay_delta_tracker.format_delta_line("gex_relay", "consensus", {
            "price": price, "gamma_bn": gamma_bn,
            "call_wall": cw, "put_floor": pf,
        })

        message = format_type2_relay(
            source=f"Consensus ({consensus.strength} channels)",
            direction=direction_label,
            confidence=confidence,
            checks=checks,
            price=price,
            gamma_bn=gamma_bn,
            entry=entry,
            stop=stop,
            target=target,
            scenario=f"{setup_name}: {reason[:120]}..." if len(reason) > 120 else f"{setup_name}: {reason}",
            delta_line=delta_line,
        )

        signal = Signal(
            signal_type=SignalType.MARKET_CONSENSUS,
            title=f"Market Intel \u2014 {direction_label} {verdict}",
            message=message,
            channel="gex_relay",
            priority=SignalType.MARKET_CONSENSUS,
        )

        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)
        self._last_consensus_alert_ts = now_ts

        log.info(
            f"Market Intel FIRED: {setup_name} {direction_label} {verdict} | "
            f"G={gamma_bn:+.1f}Bn vel={price_velocity:+.1f} | "
            f"{consensus.strength} ch"
        )
        return signal

    def _maybe_fire_micro_pulse(
        self,
        snapshot: GEXSnapshot,
        fifteen_min,
    ):
        cfg = self._micro_cfg or {}
        if not cfg.get("enabled", False):
            return

        now = datetime.now(PT_TZ)
        if not is_rth_time(now):
            return

        interval_sec = int(cfg.get("interval_seconds", 120))
        if self._last_micro_pulse_ts:
            if (now - self._last_micro_pulse_ts).total_seconds() < interval_sec:
                return

        min_sep_pts = float(cfg.get("min_price_separation_pts", 0))
        if (
            min_sep_pts > 0
            and self._last_micro_pulse_price is not None
            and abs(snapshot.curr_price - self._last_micro_pulse_price) < min_sep_pts
        ):
            return

        snaps_5m = get_recent_snapshots(5, session_tag="RTH")
        if len(snaps_5m) < 2:
            return

        move_5m = snaps_5m[-1].curr_price - snaps_5m[0].curr_price
        move_1m = snaps_5m[-1].curr_price - snaps_5m[-2].curr_price
        gamma_delta_5m_bn = (snaps_5m[-1].net_gamma - snaps_5m[0].net_gamma) / 1e9
        recent_high = max(s.curr_price for s in snaps_5m)
        recent_low = min(s.curr_price for s in snaps_5m)

        min_move_pts = float(cfg.get("min_move_pts", 5.0))
        min_gamma_delta_bn = float(cfg.get("min_gamma_delta_bn", 0.5))
        if abs(move_5m) < min_move_pts and abs(gamma_delta_5m_bn) < min_gamma_delta_bn:
            return

        if not self.discord.cooldown.can_fire("micro_pulse"):
            return

        signal = self.interpreter.generate_micro_pulse(
            snapshot=snapshot,
            move_5m=move_5m,
            gamma_delta_5m_bn=gamma_delta_5m_bn,
            fifteen_min=fifteen_min,
            move_1m=move_1m,
            recent_high=recent_high,
            recent_low=recent_low,
            micro_cfg=cfg,
        )
        signal.channel = self._micro_channel
        if self.discord.send_signal(signal):
            self._last_micro_pulse_ts = now
            self._last_micro_pulse_price = snapshot.curr_price
        self.telegram.send_signal(signal)

    def _maybe_fire_conviction_signal(
        self,
        snapshot: GEXSnapshot,
        trend,
    ):
        """Poll external systems and fire conviction signal if 2+ systems agree."""
        if not is_rth_time(datetime.now(PT_TZ)):
            return

        if not self.discord.cooldown.can_fire("conviction_signal"):
            return

        signal = self.conviction.maybe_fire(snapshot, trend)
        if signal is None:
            return

        if self.discord.send_signal(signal):
            log.info(f"Conviction signal sent to Discord: {signal.metadata.get('score')}/10 {signal.metadata.get('direction')}")

    def _process_off_hours(self, snapshot: GEXSnapshot, now_ts: float):
        """Off-hours: store, compute overnight drift every 30 min."""
        drift_interval = CFG["analysis"]["overnight_drift_interval"] * 60

        if now_ts - self._last_overnight_compute >= drift_interval:
            eod = get_previous_trading_day_eod()
            if eod:
                overnight = compute_overnight_drift(eod, snapshot)
                # Signal generation stripped — data-only pipeline

            self._last_overnight_compute = now_ts

    def _maybe_fire_gap_alert(self, snapshot: GEXSnapshot):
        if not self._context_alerts_enabled():
            return
        if snapshot.session_tag not in ("NEXT_DAY_PREVIEW", "PREMARKET"):
            return

        delay_min = int(self._gap_alert_cfg.get("delay_minutes", 10))
        close_time = snapshot.timestamp_pt.replace(hour=13, minute=0, second=0, microsecond=0)
        if snapshot.timestamp_pt < close_time + timedelta(minutes=delay_min):
            return

        date_pt = snapshot.timestamp_pt.strftime("%Y-%m-%d")
        if self._gap_alert_date == date_pt:
            return

        if not self.discord.cooldown.can_fire("gap_alert"):
            return

        if snapshot.session_tag == "NEXT_DAY_PREVIEW":
            eod = get_eod_snapshot(date_pt)
        else:
            eod = get_previous_trading_day_eod()
        if not eod:
            return

        close_px = eod.curr_price
        gap_pts = snapshot.curr_price - close_px
        gap_pct = (gap_pts / close_px * 100) if close_px else 0.0
        direction = "GAP UP" if gap_pts > 0 else "GAP DOWN" if gap_pts < 0 else "FLAT"

        channel = self._gap_alert_cfg.get("alert_channel", "gex_context")
        msg = (
            f"**{direction}**\n"
            f"Reference Close: {close_px:.2f}\n"
            f"Now: {snapshot.curr_price:.2f}\n"
            f"Gap: {gap_pts:+.2f} pts ({gap_pct:+.2f}%)\n"
            f"Session: {snapshot.session_tag}"
        )

        signal = Signal(
            signal_type=SignalType.GAP_ALERT,
            title="Gap Alert",
            message=msg,
            channel=channel,
            priority=SignalType.GAP_ALERT,
            timestamp=snapshot.timestamp_pt,
        )
        if self.discord.send_signal(signal):
            self._gap_alert_date = date_pt
            self.telegram.send_signal(signal)

    def _fire_gap_alert_scheduled(self):
        now = datetime.now(PT_TZ)
        delay_min = int(self._gap_alert_cfg.get("delay_minutes", 10))
        close_time = now.replace(hour=13, minute=0, second=0, microsecond=0)
        if now < close_time + timedelta(minutes=delay_min):
            return

        latest = get_latest_snapshot()
        if not latest or latest.session_tag not in ("NEXT_DAY_PREVIEW", "PREMARKET"):
            return

        self._maybe_fire_gap_alert(latest)

    def check_data_feed(self):
        now = datetime.now(PT_TZ)
        if not is_rth_time(now):
            # Reset state outside RTH so we don't carry lag across days
            self._data_lag_active = False
            self._data_lag_started_at = None
            return

        stale_secs = self._data_feed_stale_secs
        rth_open = now.replace(hour=6, minute=30, second=0, microsecond=0)

        if self._last_snapshot_received_at and self._last_snapshot_received_at.date() != now.date():
            self._last_snapshot_received_at = None
            self._last_snapshot_timestamp_pt = None

        if self._last_snapshot_received_at is None:
            if (now - rth_open).total_seconds() >= stale_secs:
                self._fire_data_lag(now, "no snapshots received since RTH open")
            return

        age_secs = (now - self._last_snapshot_received_at).total_seconds()
        if age_secs >= stale_secs and not self._data_lag_active:
            self._fire_data_lag(now, f"no new snapshots for {age_secs/60:.1f} minutes")

    def _fire_data_lag(self, now: datetime, reason: str):
        if self._data_lag_active:
            return

        self._data_lag_active = True
        self._data_lag_started_at = now

        last_snap_ts = (
            self._last_snapshot_timestamp_pt.strftime("%I:%M %p PT")
            if self._last_snapshot_timestamp_pt else "n/a"
        )
        last_recv_ts = (
            self._last_snapshot_received_at.strftime("%I:%M %p PT")
            if self._last_snapshot_received_at else "n/a"
        )

        msg = (
            f"**DATA FEED LAG**\n"
            f"Reason: {reason}\n"
            f"Last snapshot time: {last_snap_ts}\n"
            f"Last received: {last_recv_ts}\n"
            f"Action: check Telegram feed / credentials / network."
        )

        signal = Signal(
            signal_type=SignalType.DATA_LAG,
            title="Data Feed Lag",
            message=msg,
            channel=self._data_feed_channel,
            priority=SignalType.DATA_LAG,
            timestamp=now,
        )
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)

    def _fire_data_restored(self, now: datetime):
        if not self._data_lag_active:
            return

        downtime_min = None
        if self._data_lag_started_at:
            downtime_min = (now - self._data_lag_started_at).total_seconds() / 60

        last_snap_ts = (
            self._last_snapshot_timestamp_pt.strftime("%I:%M %p PT")
            if self._last_snapshot_timestamp_pt else "n/a"
        )

        msg = (
            f"**DATA FEED RESTORED**\n"
            f"Latest snapshot time: {last_snap_ts}\n"
        )
        if downtime_min is not None:
            msg += f"Downtime: {downtime_min:.1f} minutes\n"

        signal = Signal(
            signal_type=SignalType.DATA_RESTORED,
            title="Data Feed Restored",
            message=msg.rstrip(),
            channel=self._data_feed_channel,
            priority=SignalType.DATA_RESTORED,
            timestamp=now,
        )
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)

        self._data_lag_active = False
        self._data_lag_started_at = None

    def _sanitize_snapshot_timestamp(
        self,
        snapshot: GEXSnapshot,
        envelope_dt: Optional[datetime],
        now: datetime,
    ):
        if snapshot.session_tag == "EXPIRED_ECHO":
            return
        cfg = CFG.get("timestamps", {})
        max_future_min = int(cfg.get("max_future_minutes", 10))
        max_skew_min = int(cfg.get("max_skew_minutes", 15))
        envelope_grace_min = int(cfg.get("envelope_grace_minutes", 5))

        if not envelope_dt:
            return

        if envelope_dt.tzinfo is None:
            envelope_dt = pytz.utc.localize(envelope_dt)
        envelope_pt = envelope_dt.astimezone(PT_TZ)

        # Only adjust when the message envelope is close to "now" (live feed),
        # to avoid corrupting historical/backfill timestamps.
        if abs((now - envelope_pt).total_seconds()) > envelope_grace_min * 60:
            return

        snap_ts = snapshot.timestamp_pt
        if snap_ts.tzinfo is None:
            snap_ts = PT_TZ.localize(snap_ts)
        else:
            snap_ts = snap_ts.astimezone(PT_TZ)

        future_min = (snap_ts - now).total_seconds() / 60
        skew_min = abs((snap_ts - envelope_pt).total_seconds()) / 60

        if future_min > max_future_min or skew_min > max_skew_min:
            log.warning(
                "Snapshot timestamp skew detected. Using envelope timestamp. "
                f"snap={snap_ts}, envelope={envelope_pt}, now={now}"
            )
            # Mark as skewed so process_message can dedup against stale streams
            snapshot._timestamp_was_skewed = True
            snapshot.timestamp_pt = envelope_pt
            snapshot.session_tag = classify_session(envelope_pt)

    def _get_flip_bands(self):
        now = datetime.now(PT_TZ)
        interval_min = int(self._flip_band_cfg.get("compute_interval_min", 30))
        if (
            self._flip_band_last_compute is None
            or (now - self._flip_band_last_compute).total_seconds() >= interval_min * 60
        ):
            self._flip_band_cache = self._compute_flip_bands()
            self._flip_band_last_compute = now
        return self._flip_band_cache

    def _compute_flip_bands(self):
        lookback_days = int(self._flip_band_cfg.get("lookback_days", 3))
        min_occ = int(self._flip_band_cfg.get("min_occurrences", 5))
        top_n = int(self._flip_band_cfg.get("top_n", 3))
        ratio_threshold = float(self._flip_band_cfg.get("ratio_threshold", 1.5))

        cutoff = (datetime.now(PT_TZ) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        conn = get_conn()
        rows = conn.execute(
            "SELECT date_pt, top5_calls, top5_puts FROM gex_snapshots "
            "WHERE session_tag='RTH' AND date_pt >= ?",
            (cutoff,),
        ).fetchall()
        conn.close()

        counts = {}
        for r in rows:
            try:
                calls = json.loads(r["top5_calls"]) if r["top5_calls"] else []
                puts = json.loads(r["top5_puts"]) if r["top5_puts"] else []
            except Exception:
                continue

            call_map = {int(s): int(g) for s, g in calls}
            put_map = {int(s): int(g) for s, g in puts}
            shared = set(call_map) & set(put_map)
            for strike in shared:
                call_g = call_map[strike]
                put_g = abs(put_map[strike])
                ratio = float("inf") if call_g == 0 else (put_g / call_g)
                if ratio < ratio_threshold:
                    continue
                if strike not in counts:
                    counts[strike] = {"count": 0, "max_ratio": 0.0, "last_date": r["date_pt"]}
                counts[strike]["count"] += 1
                if ratio > counts[strike]["max_ratio"]:
                    counts[strike]["max_ratio"] = ratio
                counts[strike]["last_date"] = r["date_pt"]

        filtered = [
            {"strike": s, **info}
            for s, info in counts.items()
            if info["count"] >= min_occ
        ]
        filtered.sort(key=lambda x: (x["count"], x["max_ratio"]), reverse=True)
        return filtered[:top_n]

    def _get_move_context_stats(self):
        now = datetime.now(PT_TZ)
        interval_min = int(self._move_ctx_cfg.get("compute_interval_min", 30))
        if (
            self._move_ctx_last_compute is None
            or (now - self._move_ctx_last_compute).total_seconds() >= interval_min * 60
        ):
            self._move_ctx_stats = self._compute_move_context_stats()
            self._move_ctx_last_compute = now
        return self._move_ctx_stats

    def _compute_move_context_stats(self):
        lookback_days = int(self._move_ctx_cfg.get("lookback_days", 10))
        min_window_min = int(self._move_ctx_cfg.get("min_window_minutes", 50))
        window_min = 60

        cutoff = (datetime.now(PT_TZ) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        conn = get_conn()
        rows = conn.execute(
            "SELECT date_pt, timestamp_pt, curr_price FROM gex_snapshots "
            "WHERE session_tag='RTH' AND date_pt >= ? ORDER BY date_pt ASC, timestamp_pt ASC",
            (cutoff,),
        ).fetchall()
        conn.close()

        by_date = {}
        for r in rows:
            by_date.setdefault(r["date_pt"], []).append(r)

        abs_deltas = []
        min_window_sec = min_window_min * 60
        window_sec = window_min * 60

        for _, day_rows in by_date.items():
            if len(day_rows) < 2:
                continue
            start = 0
            for end in range(len(day_rows)):
                end_ts = day_rows[end]["timestamp_pt"]
                cutoff_ts = end_ts - window_sec
                while start < end and day_rows[start]["timestamp_pt"] < cutoff_ts:
                    start += 1
                if end_ts - day_rows[start]["timestamp_pt"] < min_window_sec:
                    continue
                delta = day_rows[end]["curr_price"] - day_rows[start]["curr_price"]
                abs_deltas.append(abs(delta))

        if not abs_deltas:
            return None

        abs_deltas.sort()

        def percentile(vals, p):
            if not vals:
                return None
            idx = int(round((p / 100) * (len(vals) - 1)))
            idx = max(0, min(len(vals) - 1, idx))
            return vals[idx]

        p60 = percentile(abs_deltas, float(self._move_ctx_cfg.get("p60", 60)))
        p90 = percentile(abs_deltas, float(self._move_ctx_cfg.get("p90", 90)))

        return {
            "p60": p60,
            "p90": p90,
            "samples": len(abs_deltas),
        }

    def _compute_move_context(self, snapshots_1h, one_hour):
        if not snapshots_1h or len(snapshots_1h) < 2:
            return None, None

        stats = self._get_move_context_stats()
        if not stats or not stats.get("p60") or not stats.get("p90"):
            return None, None

        latest = snapshots_1h[-1]
        earliest = snapshots_1h[0]
        delta_60m = latest.curr_price - earliest.curr_price
        gamma_delta_bn = (latest.net_gamma - earliest.net_gamma) / 1e9
        net_gamma_bn = latest.net_gamma / 1e9
        abs_delta = abs(delta_60m)

        p60 = stats["p60"]
        p90 = stats["p90"]
        gamma_decay_bn = float(self._move_ctx_cfg.get("gamma_decay_bn", 2.0))
        slope_threshold = float(self._move_ctx_cfg.get("slope_threshold", -3.0))

        label = None
        flag = None

        if delta_60m < 0:
            severe = abs_delta >= p90
            moderate = abs_delta >= p60
            slope = one_hour.net_gamma_slope if one_hour else 0
            crash_signal = (
                net_gamma_bn < 0
                or gamma_delta_bn <= -gamma_decay_bn
                or slope <= slope_threshold
            )
            if severe and crash_signal:
                label = "CRASHING"
                flag = "move_context_crash"
            elif moderate:
                label = "CORRECTION"
                flag = "move_context_correction"
            else:
                label = "RETRACE"
                flag = "move_context_retrace"
        else:
            label = "RECOVERY" if abs_delta >= p60 else "BOUNCE"
            flag = "move_context_recovery"

        note = (
            f"Move context: {label} | 60m dPx {delta_60m:+.1f} "
            f"(p60 {p60:.1f}, p90 {p90:.1f}) | 60m dG {gamma_delta_bn:+.2f} Bn"
        )
        return note, flag

    def _maybe_fire_hourly_recap(self):
        if not self._context_alerts_enabled():
            return
        now = datetime.now(PT_TZ)
        if not is_rth_time(now):
            return

        # Only fire near the top of the hour
        if now.minute >= self._hourly_recap_window_min:
            return

        hour_key = now.strftime("%Y-%m-%d-%H")
        if self._last_hourly_recap_hour == hour_key:
            return

        snapshots_1h = get_recent_snapshots(60, session_tag="RTH")
        if len(snapshots_1h) < 2:
            return

        msg = build_hourly_recap_message(snapshots_1h)
        signal = Signal(
            signal_type=SignalType.HOURLY_RECAP,
            title="Hourly Recap",
            message=msg,
            channel="gex_engine",
            priority=SignalType.HOURLY_RECAP,
        )

        # Send once per hour window
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)
        self._last_hourly_recap_hour = hour_key

    def _maybe_fire_quiet_summary(
        self,
        snapshot: GEXSnapshot,
        fifteen_min,
        one_hour,
        advanced,
    ):
        cfg = self._quiet_cfg or {}
        if not cfg.get("enabled", True):
            return

        now = datetime.now(PT_TZ)
        if not is_rth_time(now):
            return

        interval_sec = int(cfg.get("interval_seconds", 600))
        if self._last_internal_alert_ts:
            if (now - self._last_internal_alert_ts).total_seconds() < interval_sec:
                return

        if self._last_quiet_summary_ts:
            if (now - self._last_quiet_summary_ts).total_seconds() < interval_sec:
                return

        if not self.discord.cooldown.can_fire("quiet_summary"):
            return

        signal = self.interpreter.generate_quiet_summary(
            snapshot=snapshot,
            fifteen_min=fifteen_min,
            one_hour=one_hour,
            advanced=advanced,
        )
        signal.channel = self._quiet_channel
        if self.discord.send_signal(signal):
            self._last_quiet_summary_ts = now
        self.telegram.send_signal(signal)

    def _fire_morning_checklist(self):
        """Send a one-shot 6:35 AM PT checklist for opening bias and levels."""
        if not self._context_alerts_enabled():
            return

        snapshot = get_latest_snapshot()
        if not snapshot:
            return

        now = datetime.now(PT_TZ)
        profile = self._get_day_profile(now)
        if profile:
            profile_label = profile["label"]
        else:
            g_bn = snapshot.net_gamma_bn
            open_neg_bn = float(self._day_profile_cfg.get("open_negative_bn", -4.0))
            open_pos_bn = float(self._day_profile_cfg.get("open_positive_bn", 5.0))
            if g_bn <= open_neg_bn:
                profile_label = "RISK-OFF FAST-DOWN"
            elif g_bn >= open_pos_bn:
                profile_label = "CONTROLLED UP/RANGE"
            else:
                profile_label = "MIXED/TRANSITION"

        if profile_label == "RISK-OFF FAST-DOWN":
            bias_line = "Bias: SHORT. Do not go long unless gamma flips positive."
        elif profile_label == "CONTROLLED UP/RANGE":
            bias_line = "Bias: LONG/RANGE. Watch floor for invalidation."
        else:
            bias_line = "Bias: NEUTRAL. Wait for confirmation."

        msg = (
            "**MORNING CHECKLIST**\n"
            f"Price: {snapshot.curr_price:.2f} | Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn\n"
            f"Call Wall: {snapshot.call_wall} | Put Floor: {snapshot.put_floor}\n"
            f"Day Profile: {profile_label}\n"
            f"{bias_line}\n"
            "Key levels: Floor break = trend day. Wall break = squeeze."
        )

        signal = Signal(
            signal_type=SignalType.MORNING_CHECKLIST,
            title="Morning Checklist",
            message=msg,
            channel="gex_context",
            priority=SignalType.MORNING_CHECKLIST,
            timestamp=now,
        )
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)


    # ------------------------------------------------------------------
    # Day Classifier scheduled events
    # ------------------------------------------------------------------
    def _get_prior_day_stats(self):
        """Get prior day range, EM, high, low from DB."""
        try:
            conn = get_conn()
            # Get prior trading day date
            today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
            row = conn.execute(
                "SELECT DISTINCT date_pt FROM gex_snapshots "
                "WHERE date_pt < ? AND session_tag='RTH' "
                "ORDER BY date_pt DESC LIMIT 1",
                (today,),
            ).fetchone()
            if not row:
                conn.close()
                return 0, 0, 0, 0
            prior_date = row["date_pt"]
            stats = conn.execute(
                "SELECT MAX(curr_price) as hi, MIN(curr_price) as lo "
                "FROM gex_snapshots WHERE date_pt=? AND session_tag='RTH'",
                (prior_date,),
            ).fetchone()
            conn.close()
            hi = float(stats["hi"]) if stats["hi"] else 0
            lo = float(stats["lo"]) if stats["lo"] else 0
            prior_range = hi - lo

            # Prior day EM: use prior day's anchor (day before prior)
            em_levels = self.interpreter.compute_expected_move()
            prior_em = em_levels["em_pts"] if em_levels else 0

            return prior_range, prior_em, lo, hi
        except Exception as e:
            log.error(f"_get_prior_day_stats error: {e}")
            return 0, 0, 0, 0

    def _fire_morning_read(self):
        """6:30 AM PT — Morning Read (context, always fires)."""
        snapshot = get_latest_snapshot()
        if not snapshot:
            return
        self.classifier.reset_day()  # reset at start of day

        eod = get_previous_trading_day_eod()
        em_levels = self.interpreter.compute_expected_move()
        prior_range, prior_em, _, _ = self._get_prior_day_stats()

        msg = self.classifier.morning_read(snapshot, eod, em_levels, prior_range, prior_em)
        self._send_classifier_alert("Morning Read", msg)

    def _fire_morning_bet(self):
        """7:00 AM PT — Morning Bet (conditional on first 30m ratio)."""
        snapshot = get_latest_snapshot()
        if not snapshot:
            return

        today_snaps = get_today_snapshots(session_tag="RTH")
        if not today_snaps:
            return

        eod = get_previous_trading_day_eod()
        em_levels = self.interpreter.compute_expected_move()
        prior_range, prior_em, prior_low, prior_high = self._get_prior_day_stats()

        msg = self.classifier.morning_bet(
            snapshot, today_snaps, em_levels, eod,
            prior_low, prior_high, prior_range, prior_em,
        )
        if msg:
            self._send_classifier_alert("Morning Bet", msg)

    def _fire_pin_check(self):
        """11:00 AM PT — Pin Butterfly check."""
        snapshot = get_latest_snapshot()
        if not snapshot:
            return

        msg = self.classifier.pin_butterfly(snapshot)
        if msg:
            self._send_classifier_alert("Pin Butterfly", msg)

    def _fire_morning_brief(self):
        """Generate and send morning brief at 6:00 AM PT."""
        if not self._context_alerts_enabled():
            return
        snapshot = get_latest_snapshot()
        if not snapshot:
            return

        eod = get_previous_trading_day_eod()
        overnight = compute_overnight_drift(eod, snapshot) if eod else None

        snapshots_recent = get_recent_snapshots(60)
        advanced = compute_advanced_metrics(snapshots_recent) if snapshots_recent else None

        signal = self.interpreter.generate_morning_brief(snapshot, overnight, advanced)
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)

    def _fire_final_15(self):
        """Generate and send Final 15 alert at 12:45 PM PT."""
        if not self._context_alerts_enabled():
            return
        snapshot = get_latest_snapshot()
        if not snapshot:
            return

        snapshots_recent = get_recent_snapshots(60, session_tag="RTH")
        advanced = compute_advanced_metrics(snapshots_recent, minutes_to_close=15) if snapshots_recent else None

        signal = self.interpreter.generate_final_15(snapshot, advanced)
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)

    def _fire_eod_summary(self):
        """Generate EOD summary at 1:05 PM PT + swing trade signals."""
        today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
        rth_snapshots = get_today_snapshots(session_tag="RTH")
        if not rth_snapshots:
            return

        gammas = [s.net_gamma for s in rth_snapshots]
        prices = [s.curr_price for s in rth_snapshots]

        summary = {
            "open_price": prices[0] if prices else None,
            "close_price": prices[-1] if prices else None,
            "high_gamma": max(gammas) if gammas else None,
            "low_gamma": min(gammas) if gammas else None,
            "close_gamma": gammas[-1] if gammas else None,
            "open_call_wall": rth_snapshots[0].call_wall,
            "close_call_wall": rth_snapshots[-1].call_wall,
            "open_put_floor": rth_snapshots[0].put_floor,
            "close_put_floor": rth_snapshots[-1].put_floor,
            "num_snapshots": len(rth_snapshots),
        }

        save_daily_summary(today, summary)
        log.info(f"Saved daily summary for {today} ({len(rth_snapshots)} snapshots)")

        # Fire swing trade signals
        self._fire_swing_signals(rth_snapshots)

        # Purge old data
        retention = CFG["database"]["retention_days"]
        purge_old_data(retention)

        # Reset cooldowns for next day
        self.discord.cooldown.reset_daily()
        self.telegram.cooldown.reset_daily()
        self.scheduler.reset_daily()
        self.watchdog.reset_daily()
        self.classifier.reset_day()

    def _fire_swing_signals(self, rth_snapshots):
        """Check for swing trade setups at EOD and send to Telegram.

        Swing signals (backtested on 129 days):
          LONG:
            SWING-L1: Down >50pts => 92% win, +78 avg 5d
            SWING-L2: Down >20pts + G<-8Bn => 90% win, +76 avg 5d
            SWING-S1: 2-day G>+5Bn streak => 64% win, +55 avg 5d
            SWING-S3: Up >20pts + G>+10Bn => 67% win, +46 avg 5d
        """
        import sqlite3 as _sqlite3

        if not rth_snapshots or len(rth_snapshots) < 10:
            return

        close_price = rth_snapshots[-1].curr_price
        open_price = rth_snapshots[0].curr_price
        close_gamma_bn = rth_snapshots[-1].net_gamma / 1e9
        change = close_price - open_price
        cw = rth_snapshots[-1].call_wall
        pf = rth_snapshots[-1].put_floor

        # Get yesterday's gamma close for streak detection
        try:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CFG["database"]["path"])
            conn = _sqlite3.connect(db_path)
            conn.row_factory = _sqlite3.Row
            prev_days = conn.execute(
                "SELECT date_pt, "
                "  (SELECT net_gamma/1e9 FROM gex_snapshots s2 "
                "   WHERE s2.date_pt=s1.date_pt AND s2.session_tag='RTH' "
                "   ORDER BY s2.timestamp_pt DESC LIMIT 1) as g_close, "
                "  (SELECT curr_price FROM gex_snapshots s3 "
                "   WHERE s3.date_pt=s1.date_pt AND s3.session_tag='RTH' "
                "   ORDER BY s3.timestamp_pt DESC LIMIT 1) as close_px, "
                "  (SELECT curr_price FROM gex_snapshots s4 "
                "   WHERE s4.date_pt=s1.date_pt AND s4.session_tag='RTH' "
                "   ORDER BY s4.timestamp_pt ASC LIMIT 1) as open_px "
                "FROM (SELECT DISTINCT date_pt FROM gex_snapshots "
                "      WHERE session_tag='RTH' AND date_pt < ? "
                "      ORDER BY date_pt DESC LIMIT 5) s1 "
                "ORDER BY date_pt DESC",
                (datetime.now(PT_TZ).strftime("%Y-%m-%d"),),
            ).fetchall()
            conn.close()
        except Exception as e:
            log.error(f"Swing signal DB query failed: {e}")
            prev_days = []

        prev_g_close = [dict(r) for r in prev_days] if prev_days else []

        signals_found = []

        # SWING-L1: Down >50pts today
        if change < -50:
            stop = close_price - 55
            target = close_price + 75
            signals_found.append(
                f"SWING BUY — BIG DOWN DAY [L1]\n"
                f"SPX closed {close_price:.0f} (down {abs(change):.0f} pts)\n"
                f"Gamma: {close_gamma_bn:+.1f} Bn\n"
                f"Entry: {close_price:.0f} (or tomorrow open)\n"
                f"Stop: {stop:.0f} | Target: {target:.0f}\n"
                f"Hold: 2-5 days\n"
                f"Backtest: 92% win | +78 avg | N=13\n"
                f"HIGHEST EDGE SIGNAL IN DATABASE."
            )

        # SWING-L2: Down >20pts + G < -8Bn
        if change < -20 and close_gamma_bn < -8:
            stop = close_price - 45
            target = close_price + 75
            signals_found.append(
                f"SWING BUY — DOWN + DEEP NEG GAMMA [L2]\n"
                f"SPX closed {close_price:.0f} (down {abs(change):.0f} pts) | G={close_gamma_bn:+.1f}Bn\n"
                f"Entry: {close_price:.0f} | Stop: {stop:.0f} | Target: {target:.0f}\n"
                f"Hold: 2-5 days\n"
                f"Backtest: 90% win | +76 avg | N=10"
            )

        # SWING-L3: 3+ down days (check prev 2 days)
        if change < 0 and len(prev_g_close) >= 2:
            prev_changes = [d.get("close_px", 0) - d.get("open_px", 0) for d in prev_g_close[:2]]
            if all(c < 0 for c in prev_changes):
                stop = close_price - 50
                target = close_price + 70
                signals_found.append(
                    f"SWING BUY — 3+ DOWN DAYS [L3]\n"
                    f"SPX closed {close_price:.0f} | 3rd consecutive down day\n"
                    f"Entry: {close_price:.0f} | Stop: {stop:.0f} | Target: {target:.0f}\n"
                    f"Hold: 5 days\n"
                    f"Backtest: 67% win | +71 avg | N=12"
                )

        # SWING-L4: 2-day neg gamma streak
        if close_gamma_bn < -5 and prev_g_close and prev_g_close[0].get("g_close", 0) is not None:
            if prev_g_close[0].get("g_close", 0) < -5:
                stop = close_price - 60
                target = close_price + 70
                signals_found.append(
                    f"SWING BUY — NEG GAMMA STREAK [L4]\n"
                    f"SPX {close_price:.0f} | G={close_gamma_bn:+.1f}Bn (2nd day < -5Bn)\n"
                    f"Entry: {close_price:.0f} | Stop: {stop:.0f} | Target: {target:.0f}\n"
                    f"Hold: 5 days\n"
                    f"Backtest: 78% win | +74 avg | N=9"
                )

        # SWING-S1: 3-day pos gamma streak
        if close_gamma_bn > 5 and len(prev_g_close) >= 2:
            prev_gs = [d.get("g_close", 0) for d in prev_g_close[:2]]
            if all(g is not None and g > 5 for g in prev_gs):
                stop = close_price + 30
                target = close_price - 50
                signals_found.append(
                    f"SWING SELL — POS GAMMA STREAK 3D [S1]\n"
                    f"SPX {close_price:.0f} | G={close_gamma_bn:+.1f}Bn (3rd day > +5Bn)\n"
                    f"Entry: {close_price:.0f} | Stop: {stop:.0f} | Target: {target:.0f}\n"
                    f"Hold: 5 days\n"
                    f"Backtest: 75% win | +53 avg | N=8 | MAE only 28"
                )

        # SWING-S2: 2-day pos gamma streak
        elif close_gamma_bn > 5 and prev_g_close and prev_g_close[0].get("g_close", 0) is not None:
            if prev_g_close[0].get("g_close", 0) > 5:
                stop = close_price + 35
                target = close_price - 55
                signals_found.append(
                    f"SWING SELL — POS GAMMA STREAK 2D [S2]\n"
                    f"SPX {close_price:.0f} | G={close_gamma_bn:+.1f}Bn (2nd day > +5Bn)\n"
                    f"Entry: {close_price:.0f} | Stop: {stop:.0f} | Target: {target:.0f}\n"
                    f"Hold: 5 days\n"
                    f"Backtest: 64% win | +55 avg | N=14"
                )

        # SWING-S3: Up >20pts + G > +10Bn
        if change > 20 and close_gamma_bn > 10:
            stop = close_price + 40
            target = close_price - 45
            signals_found.append(
                f"SWING SELL — UP DAY + STRONG GAMMA [S3]\n"
                f"SPX closed {close_price:.0f} (up {change:.0f} pts) | G={close_gamma_bn:+.1f}Bn\n"
                f"Entry: {close_price:.0f} | Stop: {stop:.0f} | Target: {target:.0f}\n"
                f"Hold: 2-5 days\n"
                f"Backtest: 67% win | +46 avg | N=12"
            )

        if not signals_found:
            log.info("No swing signals at EOD.")
            return

        # Build combined swing alert
        count = len(signals_found)
        header = f"EOD SWING SIGNALS ({count} active)\n{'=' * 30}\n"
        if cw:
            header += f"Walls: PF {pf} | CW {cw}\n"
        full_msg = header + "\n---\n".join(signals_found)

        signal = Signal(
            signal_type=SignalType.GAMMA_SNAP,
            title=f"Swing Signals ({count})",
            message=full_msg,
            channel="gex_trades",
            priority=SignalType.GAMMA_SNAP,
            metadata={
                "side": "SWING",
                "tier": 8,
                "tier_label": "SWING",
                "swing_count": count,
            },
        )

        # Send to both Discord and Telegram
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)
        log.info(f"Swing signals fired: {count} setups")

    def _fire_trend_review(self):
        """Run nightly trend review and write a report file."""
        try:
            path = run_trend_review()
            log.info(f"Trend review written: {path}")
        except Exception as e:
            log.error(f"Trend review failed: {e}", exc_info=True)

    def _fire_pin_forecast(self):
        """Send 12:30 PM PT pin forecast based on current GEX and history."""
        if not self._context_alerts_enabled():
            return
        snapshot = get_latest_snapshot()
        if not snapshot or snapshot.session_tag != "RTH":
            return

        msg = self._format_pin_forecast(snapshot)
        signal = Signal(
            signal_type=SignalType.PIN_FORECAST,
            title="Pin Forecast",
            message=msg,
            channel="gex_context",
            priority=SignalType.PIN_FORECAST,
        )
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)

    def _format_pin_forecast(self, snapshot: GEXSnapshot) -> str:
        cfg = CFG.get("pin_forecast", {})
        band_pts = float(cfg.get("band_pts", 5))
        lookback_days = int(cfg.get("lookback_days", 10))
        min_samples = int(cfg.get("min_samples", 5))

        # Determine max-gamma strike (pin magnet)
        positions = snapshot.top5_calls + [(s, abs(g)) for s, g in snapshot.top5_puts]
        if positions:
            pin_strike, pin_gamma = max(positions, key=lambda x: abs(x[1]))
            pin_gamma_bn = pin_gamma / 1e9
        else:
            pin_strike = 0
            pin_gamma_bn = 0.0

        # Pin probability model
        gci, gci_label = compute_gamma_concentration_index(snapshot)
        close_time = datetime.now(PT_TZ).replace(hour=13, minute=0, second=0, microsecond=0)
        minutes_to_close = max(0, (close_time - datetime.now(PT_TZ)).total_seconds() / 60)
        pin_prob = compute_pin_probability(snapshot, gci, minutes_to_close)

        # 30m trend context
        snaps_30m = get_recent_snapshots(30, session_tag="RTH")
        gamma_30m = None
        price_30m = None
        if len(snaps_30m) >= 2:
            gamma_30m = (snaps_30m[-1].net_gamma - snaps_30m[0].net_gamma) / 1e9
            price_30m = snaps_30m[-1].curr_price - snaps_30m[0].curr_price

        # Historical hit rate: close within band around 12:30 pin strike
        hist_hit_rate = None
        hist_samples = 0
        hist_hits = 0
        try:
            cutoff = (datetime.now(PT_TZ) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            conn = get_conn()
            rows = conn.execute(
                "SELECT date_pt, timestamp_pt, curr_price, top5_calls, top5_puts "
                "FROM gex_snapshots WHERE session_tag='RTH' AND date_pt >= ? "
                "ORDER BY date_pt ASC, timestamp_pt ASC",
                (cutoff,),
            ).fetchall()
            conn.close()

            by_date = {}
            for r in rows:
                by_date.setdefault(r["date_pt"], []).append(r)

            for date, day_rows in by_date.items():
                if not day_rows:
                    continue
                target_dt = PT_TZ.localize(datetime.strptime(date + " 12:30", "%Y-%m-%d %H:%M"))
                closest = min(
                    day_rows,
                    key=lambda rr: abs(rr["timestamp_pt"] - target_dt.timestamp()),
                )
                top5_calls = json.loads(closest["top5_calls"]) if closest["top5_calls"] else []
                top5_puts = json.loads(closest["top5_puts"]) if closest["top5_puts"] else []
                positions_hist = top5_calls + [(s, abs(g)) for s, g in top5_puts]
                if not positions_hist:
                    continue
                pin_strike_hist, _ = max(positions_hist, key=lambda x: abs(x[1]))
                close_px = day_rows[-1]["curr_price"]
                if close_px is None:
                    continue
                hist_samples += 1
                if (pin_strike_hist - band_pts) <= close_px <= (pin_strike_hist + band_pts):
                    hist_hits += 1

            if hist_samples >= min_samples:
                hist_hit_rate = hist_hits / hist_samples * 100
        except Exception as e:
            log.warning(f"Pin forecast history calc failed: {e}")

        # Build message
        lines = [
            f"**PIN FORECAST** | {datetime.now(PT_TZ).strftime('%I:%M %p PT')}",
            f"Pin strike: {pin_strike} | Band: {pin_strike - band_pts:.0f} - {pin_strike + band_pts:.0f} (+/-{band_pts:.0f})",
            f"Price: {snapshot.curr_price:.2f} | Net Gamma: {snapshot.net_gamma_bn:+.2f} Bn",
            f"Pin probability (model): {pin_prob:.0f}% | GCI: {gci:.0f}% ({gci_label}) | Mins to close: {minutes_to_close:.0f}",
        ]

        if gamma_30m is not None and price_30m is not None:
            strength = "building" if gamma_30m > 0 else "decaying"
            lines.append(
                f"Last 30m: dPx {price_30m:+.1f} | dG {gamma_30m:+.2f} Bn ({strength})"
            )

        if hist_hit_rate is not None:
            lines.append(
                f"Historical hit rate: {hist_hit_rate:.0f}% closes within +/-{band_pts:.0f} at 12:30 "
                f"(n={hist_samples})"
            )
        else:
            lines.append("Historical hit rate: n/a (insufficient samples)")

        return "\n".join(lines)

    def _fire_close_nowcast(self):
        """Send 12:55 PM PT close nowcast — estimated closing price and bias."""
        if not self._context_alerts_enabled():
            return
        snapshot = get_latest_snapshot()
        if not snapshot or snapshot.session_tag != "RTH":
            return

        price = snapshot.curr_price
        gamma_bn = snapshot.net_gamma / 1e9

        # Move from open
        move_from_open = 0.0
        if self._rth_open_price is not None:
            move_from_open = price - self._rth_open_price

        # Close estimate model (from Codex calibration, 126 days, MAE 3.15pts)
        delta = -0.34  # baseline mean
        bias = "NEUTRAL"
        confidence = "LOW"

        if gamma_bn <= -10:
            delta, bias, confidence = -4.42, "DOWN", "MEDIUM"
        elif move_from_open <= -40:
            delta, bias, confidence = +3.02, "BOUNCE UP", "MEDIUM"
        elif gamma_bn >= 10:
            delta, bias, confidence = -1.94, "SLIGHT DOWN", "LOW"
        elif move_from_open >= 40:
            delta, bias, confidence = +0.77, "DOWN RISK", "LOW"

        expected_close = price + delta

        msg = (
            f"CLOSE NOWCAST | 12:55 PM PT\n"
            f"SPX {price:.0f} | Expected close: ~{expected_close:.0f}\n"
            f"Bias: {bias} | Confidence: {confidence}\n"
            f"Typical band: {price + delta - 7:.0f} - {price + delta + 7:.0f} (+/-7 pts)\n"
            f"Stress band: {price + delta - 16:.0f} - {price + delta + 16:.0f} (+/-16 pts)\n"
            f"Gamma: {gamma_bn:+.1f} Bn | Move from open: {move_from_open:+.0f} pts"
        )

        signal = Signal(
            signal_type=SignalType.PIN_FORECAST,
            title="Close Nowcast",
            message=msg,
            channel="gex_context",
            priority=SignalType.PIN_FORECAST,
            timestamp=datetime.now(PT_TZ),
        )
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)


async def start_listener():
    """Start the Telethon listener and main pipeline."""
    # Load .env from configured path
    env_path = CFG["telegram"].get("env_path", "")
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_id = int(os.getenv("TG_API_ID", "0"))
    api_hash = os.getenv("TG_API_HASH", "")
    channels_cfg = CFG.get("telegram", {})
    channels = channels_cfg.get("channels") or channels_cfg.get("channel")
    if isinstance(channels, str):
        channels = [channels]

    if not api_id or not api_hash:
        log.error("TG_API_ID and TG_API_HASH must be set. Check .env file.")
        return

    session_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gex_telegram")
    # Clear stale SQLite journal/lock files to prevent "database is locked" after crashes
    for suffix in ("-journal", "-wal", "-shm"):
        lock_file = session_path + ".session" + suffix
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                log.info(f"Removed stale session lock: {lock_file}")
            except OSError:
                pass
    client = TelegramClient(session_path, api_id, api_hash)

    pipeline = GEXPipeline()

    async def data_feed_watchdog():
        interval = int(CFG.get("data_feed", {}).get("check_interval_seconds", 60))
        while True:
            await asyncio.sleep(interval)
            try:
                pipeline.check_data_feed()
                pipeline.watchdog.check_scheduled_events()
                pipeline.watchdog.check_health_reports()
                pipeline.watchdog.check_signal_flow()
            except Exception as e:
                log.error(f"Data feed watchdog error: {e}", exc_info=True)

    def _build_channel_alert(
        channel_name: str,
        direction: str | None,
        entry: float | None,
        stop: float | None,
        targets: list | None,
        pipe,
        raw_detail: str,
    ) -> str:
        """Build a Type 2 channel relay alert with GEX validation checks."""
        # Get current GEX state
        latest = get_latest_snapshot()
        if not latest:
            return f"\U0001f4e1 RELAY \u2014 {channel_name}\n{raw_detail}"

        gamma_bn = latest.net_gamma / 1e9
        price = latest.curr_price
        cw = latest.call_wall or 0
        pf = latest.put_floor or 0

        # Normalize direction
        side = None
        if direction in ("CALL", "LONG"):
            side = "LONG"
        elif direction in ("PUT", "SHORT"):
            side = "SHORT"

        # Build strike description from raw_detail
        strike_line = raw_detail.split("\n")[0] if raw_detail else ""

        # 5 GEX checks
        checks = []

        # 1. Direction alignment
        if side == "LONG" and gamma_bn < -3:
            checks.append(("\u2705", "Direction aligns (negative gamma, bullish setup)"))
        elif side == "SHORT" and gamma_bn > 3:
            checks.append(("\u2705", "Direction aligns (positive gamma, bearish setup)"))
        elif side == "LONG" and gamma_bn > 5:
            checks.append(("\u26a0\ufe0f", "Positive gamma — upside supported but may pin near resistance"))
        elif side == "SHORT" and gamma_bn < -5:
            checks.append(("\u274c", "Deep negative gamma — short squeeze risk"))
        elif side:
            checks.append(("\u26a0\ufe0f", f"Gamma {'positive' if gamma_bn > 0 else 'negative'} — neutral for this direction"))
        else:
            checks.append(("\u26a0\ufe0f", "Direction unclear from channel"))

        # 2. Strike proximity to walls
        if entry and entry > 0:
            near_pf = pf and abs(entry - pf) < 20
            near_cw = cw and abs(entry - cw) < 20
            if near_pf or near_cw:
                wall_name = "put floor support" if near_pf else "call wall resistance"
                checks.append(("\u2705", f"Strike near {wall_name}"))
            else:
                checks.append(("\u26a0\ufe0f", "Strike not near any key GEX wall"))
        else:
            checks.append(("\u26a0\ufe0f", "No entry price to check wall proximity"))

        # 3. EM position
        em = pipe.interpreter.compute_expected_move() if hasattr(pipe, 'interpreter') else None
        if em:
            em_lower = em.get("lower", 0)
            em_upper = em.get("upper", 0)
            if side == "LONG" and price <= em_lower + 10:
                checks.append(("\u2705", "Price near lower expected move — reversal zone"))
            elif side == "SHORT" and price >= em_upper - 10:
                checks.append(("\u2705", "Price near upper expected move — reversal zone"))
            elif side == "LONG" and price > em_upper:
                checks.append(("\u274c", "Price above upper EM — extended for longs"))
            elif side == "SHORT" and price < em_lower:
                checks.append(("\u274c", "Price below lower EM — extended for shorts"))
            else:
                checks.append(("\u26a0\ufe0f", "Price in mid-range of expected move"))
        else:
            checks.append(("\u26a0\ufe0f", "Expected move data unavailable"))

        # 4. VWAP alignment
        ma_vals = pipe._live_ma_cache if hasattr(pipe, '_live_ma_cache') and pipe._live_ma_cache else None
        vwap = ma_vals.get("vwap") if ma_vals else None
        if vwap and side:
            if side == "LONG" and price < vwap:
                checks.append(("\u2705", f"Below VWAP {vwap:.0f} — room to reclaim"))
            elif side == "SHORT" and price > vwap:
                checks.append(("\u2705", f"Above VWAP {vwap:.0f} — rejection zone"))
            else:
                checks.append(("\u26a0\ufe0f", f"VWAP at {vwap:.0f} — wrong side for this direction"))
        else:
            checks.append(("\u26a0\ufe0f", "VWAP data unavailable"))

        # 5. Active conflicting signals
        recent_sigs = getattr(pipe.interpreter, '_recent_signals', [])
        now = datetime.now(PT_TZ)
        cutoff = now - timedelta(minutes=10)
        opposing = [s for s, n, t in recent_sigs if t >= cutoff and s != side]
        if opposing:
            checks.append(("\u274c", f"Conflicting {opposing[0]} signal fired recently — caution"))
        else:
            same_dir = [s for s, n, t in recent_sigs if t >= cutoff and s == side]
            if same_dir:
                checks.append(("\u2705", "Engine signals agree — same direction recently"))
            else:
                checks.append(("\u26a0\ufe0f", "No recent engine signals for confirmation"))

        # Verdict logic
        greens = sum(1 for icon, _ in checks if icon == "\u2705")
        reds = sum(1 for icon, _ in checks if icon == "\u274c")
        if reds >= 2:
            verdict = "SKIP \u2014 too many conflicts"
        elif reds == 1:
            verdict = "VALID but manage tight"
        elif greens >= 3:
            verdict = "VALID \u2014 GEX confirms"
        else:
            verdict = "VALID \u2014 mixed signals, use caution"

        # Format Type 2 output — box-drawing
        lines = ["\u2501\u2501\u2501 CHANNEL RELAY \u2501\u2501\u2501"]
        lines.append(f"\U0001f4e1 {channel_name}")
        lines.append(strike_line)
        lines.append("")
        lines.append("GEX Check:")
        for icon, text in checks:
            lines.append(f"  {icon} {text}")
        lines.append("")
        lines.append(f"Verdict: {verdict}")
        lines.append("\u2501" * 18)

        return "\n".join(lines)

    # Resolve the primary GEX channel for filtering
    gex_primary_channel = channels_cfg.get("primary_gex_channel", "spotgammaraed")
    _primary_chat_id = None  # resolved on first message

    @client.on(events.NewMessage(chats=channels))
    async def handler(event):
        nonlocal _primary_chat_id
        if event.message.media:
            return
        text = event.message.text or ""
        envelope_dt = event.message.date  # UTC datetime from Telegram

        # Resolve channel identity
        chat = await event.get_chat()
        chat_username = getattr(chat, "username", "") or ""

        # Log non-primary channel messages for debugging
        if chat_username != gex_primary_channel:
            log.info(f"Channel msg [{chat_username}]: {text[:150].encode('ascii', 'replace').decode()}")

        # Resolve primary channel ID on first match
        if _primary_chat_id is None and chat_username.lower() == gex_primary_channel.lower():
            _primary_chat_id = event.chat_id
            log.info(f"Primary GEX channel resolved: {chat_username} = {_primary_chat_id}")

        if text and chat_username != gex_primary_channel:
            is_consensus_ch = chat_username in pipeline._consensus_channels
            spx_sig = parse_spx_million_message(text)

            # Debug: log parse attempts for non-primary channels
            if spx_sig:
                log.info(f"PARSED [{chat_username}]: {spx_sig.kind} dir={spx_sig.direction} entry={spx_sig.entry}")
            else:
                log.debug(f"NO PARSE [{chat_username}]: no Arabic keywords matched")

            if spx_sig:
                # Feed consensus tracker
                if is_consensus_ch:
                    pipeline.consensus.ingest(chat_username, spx_sig)
                if pipeline._intraday_alerts_enabled():
                    # Build validated alert with EM/GEX context
                    msg = _build_channel_alert(
                        chat_username, spx_sig.direction, spx_sig.entry,
                        spx_sig.stop, spx_sig.targets, pipeline,
                        format_spx_million_signal(spx_sig),
                    )
                    channel = CFG.get("spx_million", {}).get("alert_channel", "gex_relay")
                    signal = Signal(
                        signal_type=SignalType.SPX_MILLION,
                        title=f"Channel Alert — {chat_username}",
                        message=msg,
                        channel=channel,
                        priority=SignalType.SPX_MILLION,
                    )
                    pipeline.discord.send_signal(signal)
                    pipeline.telegram.send_signal(signal)
                return

            # Keyword fallback for ALL non-primary channels (not just consensus)
            call = pipeline.consensus.ingest_raw_text(chat_username, text) if is_consensus_ch else None
            direction = None
            if call:
                direction = call.direction
            else:
                # Try keyword extraction even for non-consensus channels
                from consensus_tracker import _extract_direction
                direction = _extract_direction(text)

            if direction and pipeline._intraday_alerts_enabled():
                if is_consensus_ch and call:
                    detail = f"[Keyword] {direction} detected in: {text[:200]}"
                else:
                    detail = f"[Relay] {direction} detected in: {text[:200]}"
                msg = _build_channel_alert(
                    chat_username, direction, call.entry_price if call else None,
                    None, [], pipeline,
                    detail,
                )
                signal = Signal(
                    signal_type=SignalType.SPX_MILLION,
                    title=f"Channel Alert — {chat_username}",
                    message=msg,
                    channel=CFG.get("spx_million", {}).get("alert_channel", "gex_relay"),
                    priority=SignalType.SPX_MILLION,
                )
                pipeline.discord.send_signal(signal)
                pipeline.telegram.send_signal(signal)

        # Only process GEX snapshots from the primary channel
        if _primary_chat_id is not None and event.chat_id != _primary_chat_id:
            log.debug(f"Skipping GEX parse from non-primary channel {chat_username}")
            return

        try:
            pipeline.process_message(text, envelope_dt)
        except Exception as e:
            log.error(f"Pipeline error: {e}", exc_info=True)

    log.info(f"Starting Telegram listener on channels: {channels}")
    await client.start()
    log.info("Telegram client connected. Listening for messages...")
    asyncio.create_task(data_feed_watchdog())
    await client.run_until_disconnected()
