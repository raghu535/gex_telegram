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
        self._data_lag_active: bool = False
        self._data_lag_started_at: Optional[datetime] = None
        self._data_feed_cfg = CFG.get("data_feed", {})
        self._data_feed_stale_secs = int(self._data_feed_cfg.get("stale_seconds", 300))
        self._data_feed_channel = self._data_feed_cfg.get("alert_channel", "gex_ops")
        self._flip_band_cfg = CFG.get("flip_bands", {})
        self._flip_band_cache = []
        self._flip_band_last_compute: Optional[datetime] = None
        self._move_ctx_cfg = CFG.get("move_context", {})
        self._move_ctx_stats = None
        self._move_ctx_last_compute: Optional[datetime] = None
        self._quiet_cfg = CFG.get("quiet_summary", {})
        self._quiet_channel = str(self._quiet_cfg.get("alert_channel", "market_pulse"))
        self._micro_cfg = CFG.get("micro_pulse", {})
        self._micro_channel = str(self._micro_cfg.get("alert_channel", "intraday_fast"))
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
        alerts_cfg = CFG.get("alerts", {})
        alert_mode = str(alerts_cfg.get("mode", "full")).strip().lower()
        if alert_mode not in {"full", "deep_context"}:
            log.warning(f"Unknown alerts.mode='{alert_mode}'. Falling back to 'full'.")
            alert_mode = "full"
        self._alert_mode = alert_mode
        self._setup_scheduler()

    def _intraday_alerts_enabled(self) -> bool:
        return self._alert_mode == "full"

    def _context_alerts_enabled(self) -> bool:
        return self._alert_mode in {"full", "deep_context"}

    def _setup_scheduler(self):
        scheduled = CFG.get("scheduled", {})
        self.scheduler.add_event(
            "morning_brief",
            scheduled.get("morning_brief_time", "06:00"),
            self._fire_morning_brief,
        )
        self.scheduler.add_event(
            "morning_checklist",
            scheduled.get("morning_checklist_time", "06:35"),
            self._fire_morning_checklist,
        )
        self.scheduler.add_event(
            "final_15",
            scheduled.get("final_15_time", "12:45"),
            self._fire_final_15,
        )
        self.scheduler.add_event(
            "eod_summary",
            scheduled.get("eod_summary_time", "13:05"),
            self._fire_eod_summary,
        )
        self.scheduler.add_event(
            "trend_review",
            scheduled.get("trend_review_time", "13:20"),
            self._fire_trend_review,
        )
        self.scheduler.add_event(
            "pin_forecast",
            scheduled.get("pin_forecast_time", "12:30"),
            self._fire_pin_forecast,
        )
        self.scheduler.add_event(
            "gap_alert",
            scheduled.get("gap_alert_time", "13:10"),
            self._fire_gap_alert_scheduled,
        )

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

    def process_message(self, text: str, envelope_dt: Optional[datetime] = None):
        """Main pipeline: parse -> store -> compute -> signal -> alert."""
        # Log raw message for debugging
        log.info(f"Raw message ({len(text)} chars): {text[:300]}...")
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

        # Store to SQLite
        row_id = save_snapshot(snapshot)
        log.info(
            f"Saved snapshot #{row_id}: price={snapshot.curr_price}, "
            f"gamma={snapshot.net_gamma_bn:+.2f}Bn, session={snapshot.session_tag}"
        )
        self._last_snapshot_received_at = now
        self._last_snapshot_timestamp_pt = snapshot.timestamp_pt

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
            self._maybe_fire_gap_alert(snapshot)

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

        # Evaluate all signals (every message), using cached regime/advanced when needed
        signals = self.interpreter.evaluate_all(
            snapshot=snapshot,
            fifteen_min=fifteen_min,
            one_hour=one_hour,
            overnight=None,
            advanced=advanced,
            trend=trend_dashboard,
            snapshots_1h=snapshots_1h,
        )

        if signals:
            self._last_internal_alert_ts = datetime.now(PT_TZ)
            if self._intraday_alerts_enabled():
                sent = self.discord.send_batch(signals)
                log.info(f"Sent {sent}/{len(signals)} signals to Discord.")
                # Send high-priority signals to Telegram (separate cooldowns)
                tg_sent = self.telegram.send_batch(signals)
                if tg_sent:
                    log.info(f"Sent {tg_sent}/{len(signals)} signals to Telegram.")
            else:
                log.debug(
                    "Suppressed %d intraday signals (alerts.mode=%s)",
                    len(signals),
                    self._alert_mode,
                )

        # RTH pulse (5-min heartbeat) — handled by cooldown
        if self._intraday_alerts_enabled() and self.discord.cooldown.can_fire("rth_pulse"):
            pulse = self.interpreter.generate_rth_pulse(snapshot, one_hour, advanced)
            self.discord.send_signal(pulse)

        if self._intraday_alerts_enabled():
            self._maybe_fire_micro_pulse(snapshot, fifteen_min)

        self._maybe_fire_hourly_recap()
        if self._intraday_alerts_enabled():
            self._maybe_fire_quiet_summary(snapshot, fifteen_min, one_hour, advanced)

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

    def _process_off_hours(self, snapshot: GEXSnapshot, now_ts: float):
        """Off-hours: store, compute overnight drift every 30 min."""
        drift_interval = CFG["analysis"]["overnight_drift_interval"] * 60

        if now_ts - self._last_overnight_compute >= drift_interval:
            eod = get_previous_trading_day_eod()
            if eod:
                overnight = compute_overnight_drift(eod, snapshot)
                if overnight:
                    signals = self.interpreter.evaluate_all(
                        snapshot=snapshot,
                        fifteen_min=None,
                        one_hour=None,
                        overnight=overnight,
                        advanced=None,
                        trend=None,
                    )
                    if signals and self._intraday_alerts_enabled():
                        self.discord.send_batch(signals)
                        self.telegram.send_batch(signals)

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

        channel = self._gap_alert_cfg.get("alert_channel", "daily_intel")
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
            channel="daily_intel",
            priority=SignalType.HOURLY_RECAP,
        )

        # Send once per hour window
        self.discord.send_signal(signal)
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
            channel="market_pulse",
            priority=SignalType.MORNING_CHECKLIST,
            timestamp=now,
        )
        self.discord.send_signal(signal)
        self.telegram.send_signal(signal)


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
        """Generate EOD summary at 1:05 PM PT."""
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

        # Purge old data
        retention = CFG["database"]["retention_days"]
        purge_old_data(retention)

        # Reset cooldowns for next day
        self.discord.cooldown.reset_daily()
        self.telegram.cooldown.reset_daily()
        self.scheduler.reset_daily()

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
            channel="daily_intel",
            priority=SignalType.PIN_FORECAST,
        )
        self.discord.send_signal(signal)

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
    client = TelegramClient(session_path, api_id, api_hash)

    pipeline = GEXPipeline()

    async def data_feed_watchdog():
        interval = int(CFG.get("data_feed", {}).get("check_interval_seconds", 60))
        while True:
            await asyncio.sleep(interval)
            try:
                pipeline.check_data_feed()
            except Exception as e:
                log.error(f"Data feed watchdog error: {e}", exc_info=True)

    @client.on(events.NewMessage(chats=channels))
    async def handler(event):
        if event.message.media:
            return
        text = event.message.text or ""
        envelope_dt = event.message.date  # UTC datetime from Telegram
        if text:
            spx_sig = parse_spx_million_message(text)
            if spx_sig:
                if pipeline._intraday_alerts_enabled():
                    channel = CFG.get("spx_million", {}).get("alert_channel", "market_pulse")
                    signal = Signal(
                        signal_type=SignalType.SPX_MILLION,
                        title="SPX Million",
                        message=format_spx_million_signal(spx_sig),
                        channel=channel,
                        priority=SignalType.SPX_MILLION,
                    )
                    pipeline.discord.send_signal(signal)
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
