"""
COMPREHENSIVE DAY CLASSIFICATION BACKTEST
==========================================
Builds daily feature table, classifies days by morning signals,
measures outcomes, and simulates a combined trading strategy.

Run: python full_day_backtest.py
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

PT_TZ = pytz.timezone("US/Pacific")
ET_TZ = pytz.timezone("US/Eastern")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gex_data.db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def pt_time_to_minutes(t: str) -> float:
    """Convert HH:MM:SS string to minutes since midnight."""
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1]) + (int(parts[2]) / 60 if len(parts) > 2 else 0)


def et_time_str_to_pt_minutes(et_hh: int, et_mm: int) -> float:
    """Convert ET hour:min to PT minutes-since-midnight (PT = ET - 3)."""
    pt_hh = et_hh - 3
    return pt_hh * 60 + et_mm


def nearest_snapshot_to_pt_minutes(snapshots: list, target_min: float):
    """Find snapshot nearest to target PT minutes."""
    best = None
    best_dist = float("inf")
    for s in snapshots:
        t_min = pt_time_to_minutes(s["time_pt"])
        dist = abs(t_min - target_min)
        if dist < best_dist:
            best_dist = dist
            best = s
    return best


# ─────────────────────────────────────────────────────────────
# PART 0: LOAD MARKET DATA (yfinance)
# ─────────────────────────────────────────────────────────────

def load_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load SPX and VIX daily data from yfinance."""
    import yfinance as yf

    print("  Downloading SPX data...")
    spx = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    print("  Downloading VIX data...")
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)

    # Flatten multi-level columns
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = [c[0] for c in spx.columns]
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] for c in vix.columns]

    mkt = pd.DataFrame(index=spx.index)
    mkt["spx_close"] = spx["Close"]
    mkt["spx_high"] = spx["High"]
    mkt["spx_low"] = spx["Low"]
    mkt["spx_open"] = spx["Open"]
    mkt["spx_volume"] = spx["Volume"]
    mkt["vix_close"] = vix["Close"].reindex(spx.index, method="ffill")
    mkt["vol_sma20"] = mkt["spx_volume"].rolling(20, min_periods=1).mean()
    mkt["vol_ratio"] = mkt["spx_volume"] / mkt["vol_sma20"]
    mkt.index = mkt.index.strftime("%Y-%m-%d")
    return mkt


# ─────────────────────────────────────────────────────────────
# PART 1: BUILD DAILY FEATURE TABLE
# ─────────────────────────────────────────────────────────────

# 30-min ET time marks and their PT equivalents
ET_MARKS = [
    ("0930", 9, 30), ("1000", 10, 0), ("1030", 10, 30),
    ("1100", 11, 0), ("1130", 11, 30), ("1200", 12, 0),
    ("1230", 12, 30), ("1300", 13, 0), ("1330", 13, 30),
    ("1400", 14, 0), ("1430", 14, 30), ("1500", 15, 0),
    ("1530", 15, 30), ("1600", 16, 0),
]


def classify_gamma_pattern(gamma_profile: dict, gamma_range_bn: float) -> str:
    """Classify gamma curve shape. Priority: 2 > 1 > 3 > 5 > 4."""
    g = {}
    for label, _, _ in ET_MARKS:
        g[label] = gamma_profile.get(f"gamma_{label}", None)

    # Need at least key times
    required = ["0930", "1000", "1030", "1200", "1300", "1400", "1600"]
    for r in required:
        if g.get(r) is None:
            return "MIXED"

    matches = []

    # Pattern 2 — STEADY BLEED
    if (g["1000"] < g["0930"] and g["1200"] < g["1000"]
            and g["1400"] < g["1200"]
            and (g["1400"] - g["0930"]) < -3e9):
        matches.append(2)

    # Pattern 1 — MORNING FLUSH + RECOVERY
    if (g["1030"] < g["0930"] - 3e9
            and g["1300"] > g["1030"] + 2e9):
        matches.append(1)

    # Pattern 3 — PIN BUILDUP
    if (g["1400"] > g["1200"] + 2e9
            and g["1600"] > g["1400"]
            and g["1600"] > 3e9):
        matches.append(3)

    # Pattern 5 — MORNING BUILD
    if g["1030"] > g["0930"] + 3e9:
        matches.append(5)

    # Pattern 4 — FLAT
    if gamma_range_bn < 4.0:
        matches.append(4)

    # Priority: 2 > 1 > 3 > 5 > 4
    for p in [2, 1, 3, 5, 4]:
        if p in matches:
            return str(p)

    return "MIXED"


def build_daily_features(conn: sqlite3.Connection, mkt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build one row per trading day with all features."""
    # Get all RTH days
    days = conn.execute(
        "SELECT DISTINCT date_pt FROM gex_snapshots WHERE session_tag='RTH' ORDER BY date_pt"
    ).fetchall()
    day_list = [d["date_pt"] for d in days]

    # Get sorted list of market dates for prior-day lookups
    mkt_dates = sorted(mkt.index.tolist())

    features_rows = []
    gamma_profile_rows = []

    for i, date_str in enumerate(day_list):
        # Load RTH snapshots for this day
        snaps = conn.execute(
            "SELECT time_pt, curr_price, net_gamma, total_call_gamma, total_put_gamma, "
            "call_wall, put_floor FROM gex_snapshots "
            "WHERE date_pt=? AND session_tag='RTH' ORDER BY time_pt",
            (date_str,),
        ).fetchall()
        snaps = [dict(s) for s in snaps]

        if len(snaps) < 10:
            continue

        # --- Prior day market data ---
        # Find the market trading day before this date
        prior_mkt_date = None
        for md in reversed(mkt_dates):
            if md < date_str:
                prior_mkt_date = md
                break

        if prior_mkt_date is None or prior_mkt_date not in mkt.index:
            continue

        anchor = float(mkt.loc[prior_mkt_date, "spx_close"])
        vix = float(mkt.loc[prior_mkt_date, "vix_close"])
        if pd.isna(anchor) or pd.isna(vix) or anchor == 0:
            continue

        em_pts = (vix / 16.0) / 100.0 * anchor
        em_upper = anchor + em_pts
        em_lower = anchor - em_pts

        # Volume for this day (if available)
        day_volume = float(mkt.loc[date_str, "spx_volume"]) if date_str in mkt.index else np.nan
        vol_sma20 = float(mkt.loc[date_str, "vol_sma20"]) if date_str in mkt.index else np.nan
        vol_ratio = float(mkt.loc[date_str, "vol_ratio"]) if date_str in mkt.index else np.nan

        # --- Basic price/gamma ---
        open_price = snaps[0]["curr_price"]
        day_close = snaps[-1]["curr_price"]
        all_prices = [s["curr_price"] for s in snaps]
        day_high = max(all_prices)
        day_low = min(all_prices)
        day_range = day_high - day_low

        all_gamma = [s["net_gamma"] for s in snaps]
        open_gamma = snaps[0]["net_gamma"]
        eod_gamma = snaps[-1]["net_gamma"]
        avg_gamma = np.mean(all_gamma)
        min_gamma = min(all_gamma)
        max_gamma = max(all_gamma)
        gamma_range_bn = (max_gamma - min_gamma) / 1e9
        open_spread = snaps[0]["call_wall"] - snaps[0]["put_floor"]
        gamma_drift = eod_gamma - open_gamma
        gamma_drift_bn = gamma_drift / 1e9

        # --- First 30 minutes (6:30-7:00 PT = 9:30-10:00 ET) ---
        pt_630 = 6 * 60 + 30
        pt_700 = 7 * 60
        first_30 = [s for s in snaps if pt_time_to_minutes(s["time_pt"]) <= pt_700]
        if not first_30:
            first_30 = snaps[:5]  # fallback

        first_30_high = max(s["curr_price"] for s in first_30)
        first_30_low = min(s["curr_price"] for s in first_30)
        first_30_range = first_30_high - first_30_low

        first_30_ratio = first_30_range / em_pts if em_pts > 0 else 0

        # Price at 10:00 AM ET = 7:00 AM PT
        snap_10am = nearest_snapshot_to_pt_minutes(snaps, pt_700)
        price_at_10am = snap_10am["curr_price"] if snap_10am else open_price
        gamma_at_10am = snap_10am["net_gamma"] if snap_10am else open_gamma

        first_30_direction = price_at_10am - open_price
        first_30_em_position = (price_at_10am - anchor) / em_pts if em_pts > 0 else 0

        # --- Gamma profile at 30-min intervals ---
        gamma_profile = {}
        gamma_profile_row = {"date": date_str}
        for label, et_hh, et_mm in ET_MARKS:
            pt_min = et_time_str_to_pt_minutes(et_hh, et_mm)
            snap_at = nearest_snapshot_to_pt_minutes(snaps, pt_min)
            val = snap_at["net_gamma"] if snap_at else None
            gamma_profile[f"gamma_{label}"] = val
            gamma_profile_row[f"gamma_{label}"] = val

        # Gamma 30m slope
        g0930 = gamma_profile.get("gamma_0930")
        g1000 = gamma_profile.get("gamma_1000")
        gamma_30m_slope = (g1000 - g0930) / 30.0 if g0930 is not None and g1000 is not None else 0

        # Gamma drift label
        if gamma_drift < -3e9:
            gamma_drift_label = "UNWINDING"
        elif gamma_drift > 3e9:
            gamma_drift_label = "BUILDING"
        else:
            gamma_drift_label = "STABLE"

        # Peak and trough times
        gamma_peak_idx = np.argmax(all_gamma)
        gamma_trough_idx = np.argmin(all_gamma)
        gamma_peak_time = snaps[gamma_peak_idx]["time_pt"]
        gamma_trough_time = snaps[gamma_trough_idx]["time_pt"]

        # Gamma acceleration to trough
        minutes_to_trough = pt_time_to_minutes(gamma_trough_time) - pt_time_to_minutes(snaps[0]["time_pt"])
        gamma_acceleration = (min_gamma - open_gamma) / minutes_to_trough if minutes_to_trough > 0 else 0

        # Gamma pattern classification
        gamma_pattern = classify_gamma_pattern(gamma_profile, gamma_range_bn)

        # --- Outcome ---
        actual_move = abs(day_close - anchor)
        max_move_up = (day_high - anchor) / em_pts if em_pts > 0 else 0
        max_move_down = (day_low - anchor) / em_pts if em_pts > 0 else 0
        close_em_position = (day_close - anchor) / em_pts if em_pts > 0 else 0
        close_inside_em = abs(close_em_position) <= 1.0
        intraday_breach = (max_move_up > 1.0) or (max_move_down < -1.0)
        day_range_ratio = day_range / (em_pts * 2) if em_pts > 0 else 0

        # --- Remaining move after 10:00 AM ET ---
        after_10am = [s for s in snaps if pt_time_to_minutes(s["time_pt"]) > pt_700]
        if after_10am:
            remaining_high = max(s["curr_price"] for s in after_10am)
            remaining_low = min(s["curr_price"] for s in after_10am)
        else:
            remaining_high = price_at_10am
            remaining_low = price_at_10am

        remaining_range = remaining_high - remaining_low
        remaining_move_up = remaining_high - price_at_10am
        remaining_move_down = price_at_10am - remaining_low
        remaining_max_move = max(remaining_move_up, remaining_move_down)

        row = {
            "date": date_str,
            "anchor": round(anchor, 2),
            "vix": round(vix, 2),
            "em_pts": round(em_pts, 2),
            "em_upper": round(em_upper, 2),
            "em_lower": round(em_lower, 2),
            "open_price": round(open_price, 2),
            "first_30_high": round(first_30_high, 2),
            "first_30_low": round(first_30_low, 2),
            "first_30_range": round(first_30_range, 2),
            "first_30_ratio": round(first_30_ratio, 4),
            "first_30_direction": round(first_30_direction, 2),
            "first_30_em_position": round(first_30_em_position, 4),
            "price_at_10am": round(price_at_10am, 2),
            "day_volume": day_volume,
            "vol_sma20": round(vol_sma20, 0) if not pd.isna(vol_sma20) else np.nan,
            "vol_ratio": round(vol_ratio, 4) if not pd.isna(vol_ratio) else np.nan,
            "open_gamma": open_gamma,
            "gamma_at_10am": gamma_at_10am,
            "avg_gamma": round(avg_gamma, 0),
            "min_gamma": min_gamma,
            "max_gamma": max_gamma,
            "gamma_range_bn": round(gamma_range_bn, 2),
            "open_spread": open_spread,
            "eod_gamma": eod_gamma,
            "gamma_drift": gamma_drift,
            "gamma_drift_bn": round(gamma_drift_bn, 2),
            "gamma_30m_slope": round(gamma_30m_slope, 4),
            "gamma_drift_label": gamma_drift_label,
            "gamma_peak_time": gamma_peak_time,
            "gamma_trough_time": gamma_trough_time,
            "gamma_acceleration": round(gamma_acceleration, 2),
            "gamma_pattern": gamma_pattern,
            "day_close": round(day_close, 2),
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "day_range": round(day_range, 2),
            "actual_move": round(actual_move, 2),
            "max_move_up": round(max_move_up, 4),
            "max_move_down": round(max_move_down, 4),
            "close_em_position": round(close_em_position, 4),
            "close_inside_em": close_inside_em,
            "intraday_breach": intraday_breach,
            "day_range_ratio": round(day_range_ratio, 4),
            "remaining_high": round(remaining_high, 2),
            "remaining_low": round(remaining_low, 2),
            "remaining_range": round(remaining_range, 2),
            "remaining_move_up": round(remaining_move_up, 2),
            "remaining_move_down": round(remaining_move_down, 2),
            "remaining_max_move": round(remaining_max_move, 2),
        }

        # Add call_wall at 2pm for pin trades
        snap_2pm = nearest_snapshot_to_pt_minutes(snaps, et_time_str_to_pt_minutes(14, 0))
        row["call_wall_at_2pm"] = snap_2pm["call_wall"] if snap_2pm else 0
        row["gamma_at_2pm"] = snap_2pm["net_gamma"] if snap_2pm else 0

        features_rows.append(row)
        gamma_profile_rows.append(gamma_profile_row)

    df = pd.DataFrame(features_rows)

    # --- Link next-day outcomes ---
    df["next_day_breach"] = df["intraday_breach"].shift(-1)
    df["next_day_close_inside"] = df["close_inside_em"].shift(-1)
    df["next_day_range"] = df["day_range"].shift(-1)
    df["next_day_pattern"] = df["gamma_pattern"].shift(-1)

    gamma_profiles_df = pd.DataFrame(gamma_profile_rows)
    return df, gamma_profiles_df


# ─────────────────────────────────────────────────────────────
# PART 2: DAY CLASSIFICATION AT 10:00 AM
# ─────────────────────────────────────────────────────────────

def classify_day(row: pd.Series) -> dict:
    """Classify a day using only data available by 10:00 AM ET."""
    r = first_30_ratio = row["first_30_ratio"]
    og = row["open_gamma"]
    slope = row["gamma_30m_slope"]

    # Primary group
    if r > 0.5:
        group = "A"
    elif r >= 0.3:
        group = "B"
    else:
        group = "C"

    # Subgroup (gamma)
    if og < -3e9:
        gamma_label = "1"  # negative
    elif og > 3e9:
        gamma_label = "2"  # positive
    else:
        gamma_label = "3"  # neutral

    subgroup = group + gamma_label

    # Sub-subgroup (slope)
    g0930 = row.get("gamma_0930") if "gamma_0930" in row.index else None
    g1000 = row.get("gamma_1000") if "gamma_1000" in row.index else None

    if g0930 is not None and g1000 is not None and not pd.isna(g0930) and not pd.isna(g1000):
        delta = g1000 - g0930
        if delta < -1e9:
            slope_label = "ACCELERATING"
        elif delta > 1e9:
            slope_label = "DECELERATING"
        else:
            slope_label = "FLAT"
    else:
        # Use gamma_30m_slope as fallback
        slope_bn_per_min = slope / 1e9
        if slope_bn_per_min < -1e9 / 30:
            slope_label = "ACCELERATING"
        elif slope_bn_per_min > 1e9 / 30:
            slope_label = "DECELERATING"
        else:
            slope_label = "FLAT"

    sub_subgroup = f"{subgroup}_{slope_label}"

    return {
        "group": group,
        "subgroup": subgroup,
        "slope_label": slope_label,
        "sub_subgroup": sub_subgroup,
    }


# ─────────────────────────────────────────────────────────────
# PART 3: OUTCOME ANALYSIS BY GROUP
# ─────────────────────────────────────────────────────────────

def compute_group_outcomes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute outcome stats for each group/subgroup."""

    def stats_for_mask(mask):
        sub = df[mask]
        n = len(sub)
        if n == 0:
            return None
        breach_pct = sub["intraday_breach"].mean() * 100
        close_inside_pct = sub["close_inside_em"].mean() * 100
        avg_range = sub["day_range"].mean()
        avg_remaining = sub["remaining_max_move"].mean()

        # Fade win %: reached 0.75 EM, came back inside 0.25 EM
        fade_attempts = sub[(sub["max_move_up"] > 0.75) | (sub["max_move_down"] < -0.75)]
        fade_wins = fade_attempts[abs(fade_attempts["close_em_position"]) <= 0.25]
        fade_pct = (len(fade_wins) / len(fade_attempts) * 100) if len(fade_attempts) > 0 else np.nan

        # Breakout win %: reached 0.75 EM, continued to 1.0+ EM
        bo_wins = fade_attempts[(fade_attempts["max_move_up"] > 1.0) | (fade_attempts["max_move_down"] < -1.0)]
        bo_pct = (len(bo_wins) / len(fade_attempts) * 100) if len(fade_attempts) > 0 else np.nan

        # Best trade heuristic
        if not pd.isna(fade_pct) and not pd.isna(bo_pct):
            best = "FADE" if fade_pct > bo_pct else "BREAKOUT"
        elif breach_pct > 60:
            best = "BREAKOUT"
        else:
            best = "FADE"

        return {
            "count": n,
            "pct_days": n / len(df) * 100,
            "breach_pct": breach_pct,
            "close_inside_pct": close_inside_pct,
            "avg_range": avg_range,
            "avg_remaining_move": avg_remaining,
            "fade_win_pct": fade_pct,
            "breakout_win_pct": bo_pct,
            "best_trade": best,
        }

    def percentiles_for_mask(mask):
        sub = df[mask]
        n = len(sub)
        if n < 2:
            return None
        vals = sub["remaining_max_move"]
        return {
            "count": n,
            "p25": np.percentile(vals, 25),
            "p50": np.percentile(vals, 50),
            "p75": np.percentile(vals, 75),
            "p90": np.percentile(vals, 90),
        }

    group_rows = []
    pct_rows = []

    # Groups, subgroups, sub-subgroups
    levels = []
    for g in ["A", "B", "C"]:
        levels.append(("group", g, df["group"] == g))
        for sg in [g + "1", g + "2", g + "3"]:
            levels.append(("subgroup", sg, df["subgroup"] == sg))
            for sl in ["ACCELERATING", "DECELERATING", "FLAT"]:
                ssg = f"{sg}_{sl}"
                levels.append(("sub_subgroup", ssg, df["sub_subgroup"] == ssg))

    for level_type, label, mask in levels:
        s = stats_for_mask(mask)
        if s:
            s["level"] = level_type
            s["label"] = label
            group_rows.append(s)

        p = percentiles_for_mask(mask)
        if p:
            p["label"] = label
            pct_rows.append(p)

    group_df = pd.DataFrame(group_rows)
    pct_df = pd.DataFrame(pct_rows)

    # --- Direction persistence ---
    dir_rows = []
    for level_type, label, mask in levels:
        sub = df[mask]
        if len(sub) < 3:
            continue
        down_mask = sub["first_30_direction"] < 0
        up_mask = sub["first_30_direction"] > 0
        down_sub = sub[down_mask]
        up_sub = sub[up_mask]
        # Trend continuation: if first 30 down, did price close lower than 10am?
        down_cont = (down_sub["day_close"] < down_sub["price_at_10am"]).mean() * 100 if len(down_sub) > 0 else np.nan
        up_cont = (up_sub["day_close"] > up_sub["price_at_10am"]).mean() * 100 if len(up_sub) > 0 else np.nan
        dir_rows.append({
            "label": label,
            "down_days": len(down_sub),
            "down_continuation_pct": round(down_cont, 1) if not pd.isna(down_cont) else np.nan,
            "up_days": len(up_sub),
            "up_continuation_pct": round(up_cont, 1) if not pd.isna(up_cont) else np.nan,
        })

    # --- Gamma pattern distribution ---
    pattern_rows = []
    for level_type, label, mask in levels:
        sub = df[mask]
        if len(sub) < 2:
            continue
        dist = sub["gamma_pattern"].value_counts(normalize=True) * 100
        row = {"label": label, "count": len(sub)}
        for pat in ["1", "2", "3", "4", "5", "MIXED"]:
            row[f"pattern_{pat}_pct"] = round(dist.get(pat, 0), 1)
        pattern_rows.append(row)

    pattern_dist_df = pd.DataFrame(pattern_rows)

    # --- EOD gamma → next day ---
    def eod_bucket(g):
        g_bn = g / 1e9
        if g_bn < -5:
            return "< -5Bn"
        elif g_bn < -2:
            return "-5 to -2Bn"
        elif g_bn <= 2:
            return "-2 to +2Bn"
        elif g_bn <= 5:
            return "+2 to +5Bn"
        else:
            return "> +5Bn"

    df["eod_gamma_bucket"] = df["eod_gamma"].apply(eod_bucket)
    eod_rows = []
    bucket_order = ["< -5Bn", "-5 to -2Bn", "-2 to +2Bn", "+2 to +5Bn", "> +5Bn"]
    for bucket in bucket_order:
        sub = df[df["eod_gamma_bucket"] == bucket].dropna(subset=["next_day_breach"])
        if len(sub) < 2:
            continue
        eod_rows.append({
            "eod_gamma_bucket": bucket,
            "count": len(sub),
            "next_breach_pct": round(sub["next_day_breach"].mean() * 100, 1),
            "next_close_inside_pct": round(sub["next_day_close_inside"].mean() * 100, 1),
            "next_avg_range": round(sub["next_day_range"].mean(), 1),
            "next_pattern_mode": sub["next_day_pattern"].mode().iloc[0] if len(sub) > 0 else "N/A",
        })

    eod_df = pd.DataFrame(eod_rows)

    return group_df, pct_df, pattern_dist_df, eod_df


# ─────────────────────────────────────────────────────────────
# PART 4: PREDICTIVE MODEL
# ─────────────────────────────────────────────────────────────

def build_prediction_model(df: pd.DataFrame) -> Tuple[dict, dict]:
    """Predict gamma pattern from morning features."""

    feature_cols = [
        "first_30_ratio", "first_30_em_position", "first_30_direction",
        "open_gamma_bn", "gamma_at_10am_bn", "gamma_30m_slope_bn",
        "open_spread", "vix",
    ]

    df_model = df.copy()
    df_model["open_gamma_bn"] = df_model["open_gamma"] / 1e9
    df_model["gamma_at_10am_bn"] = df_model["gamma_at_10am"] / 1e9
    df_model["gamma_30m_slope_bn"] = df_model["gamma_30m_slope"] / 1e9

    df_model = df_model.dropna(subset=feature_cols + ["gamma_pattern"])
    df_model = df_model[df_model["gamma_pattern"] != "MIXED"]  # exclude MIXED for model

    if len(df_model) < 20:
        print("  Not enough data for predictive model")
        return {}, {}

    # --- Method 1: Lookup table ---
    lookup_rows = []
    for g in df_model["group"].unique():
        for sl in df_model["slope_label"].unique():
            sub = df_model[(df_model["group"] == g) & (df_model["slope_label"] == sl)]
            if len(sub) < 2:
                continue
            mode_pattern = sub["gamma_pattern"].mode().iloc[0]
            mode_pct = (sub["gamma_pattern"] == mode_pattern).mean() * 100
            lookup_rows.append({
                "group": g,
                "slope": sl,
                "most_common_pattern": mode_pattern,
                "probability": round(mode_pct, 1),
                "count": len(sub),
            })

    # --- Method 2: Decision tree ---
    X = df_model[feature_cols].values
    y = df_model["gamma_pattern"].values

    split = int(len(df_model) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    accuracy_results = {}
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.metrics import classification_report, confusion_matrix

        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()

        accuracy_results["decision_tree"] = {
            "accuracy": round(acc * 100, 1),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "tree_text": export_text(clf, feature_names=feature_cols, max_depth=4),
        }

        # Confusion matrix
        labels = sorted(df_model["gamma_pattern"].unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        accuracy_results["confusion_matrix"] = {
            "labels": labels,
            "matrix": cm.tolist(),
        }
    except Exception as e:
        accuracy_results["decision_tree_error"] = str(e)

    # --- Method 3: Key decision rules ---
    rules = []

    # Rule 1: big open + gamma dropping
    r1_mask = (df_model["first_30_ratio"] > 0.5) & (df_model["gamma_30m_slope_bn"] < -0.05)
    if r1_mask.sum() > 2:
        r1_sub = df_model[r1_mask]
        mode = r1_sub["gamma_pattern"].mode().iloc[0]
        pct = (r1_sub["gamma_pattern"] == mode).mean() * 100
        rules.append({
            "rule": "first_30_ratio > 0.5 AND gamma_slope < -0.05Bn/min",
            "predicted_pattern": mode,
            "probability": round(pct, 1),
            "count": len(r1_sub),
        })

    # Rule 2: quiet open + positive gamma
    r2_mask = (df_model["first_30_ratio"] < 0.3) & (df_model["gamma_at_10am_bn"] > 3)
    if r2_mask.sum() > 2:
        r2_sub = df_model[r2_mask]
        mode = r2_sub["gamma_pattern"].mode().iloc[0]
        pct = (r2_sub["gamma_pattern"] == mode).mean() * 100
        rules.append({
            "rule": "first_30_ratio < 0.3 AND gamma_at_10am > +3Bn",
            "predicted_pattern": mode,
            "probability": round(pct, 1),
            "count": len(r2_sub),
        })

    # Rule 3: any open + extremely negative gamma
    r3_mask = df_model["open_gamma_bn"] < -5
    if r3_mask.sum() > 2:
        r3_sub = df_model[r3_mask]
        mode = r3_sub["gamma_pattern"].mode().iloc[0]
        pct = (r3_sub["gamma_pattern"] == mode).mean() * 100
        rules.append({
            "rule": "open_gamma < -5Bn",
            "predicted_pattern": mode,
            "probability": round(pct, 1),
            "count": len(r3_sub),
        })

    # Rule 4: high VIX + big open
    r4_mask = (df_model["vix"] > 20) & (df_model["first_30_ratio"] > 0.4)
    if r4_mask.sum() > 2:
        r4_sub = df_model[r4_mask]
        mode = r4_sub["gamma_pattern"].mode().iloc[0]
        pct = (r4_sub["gamma_pattern"] == mode).mean() * 100
        rules.append({
            "rule": "VIX > 20 AND first_30_ratio > 0.4",
            "predicted_pattern": mode,
            "probability": round(pct, 1),
            "count": len(r4_sub),
        })

    # Rule 5: gamma recovery in first 30m
    r5_mask = (df_model["gamma_30m_slope_bn"] > 0.05) & (df_model["open_gamma_bn"] < 0)
    if r5_mask.sum() > 2:
        r5_sub = df_model[r5_mask]
        mode = r5_sub["gamma_pattern"].mode().iloc[0]
        pct = (r5_sub["gamma_pattern"] == mode).mean() * 100
        rules.append({
            "rule": "gamma_slope > +0.05Bn/min AND open_gamma < 0",
            "predicted_pattern": mode,
            "probability": round(pct, 1),
            "count": len(r5_sub),
        })

    # --- THE CRITICAL QUESTION ---
    crit_mask = (
        (df_model["gamma_at_10am_bn"] - df_model["open_gamma_bn"] < -2)
        & (df_model["first_30_ratio"] > 0.5)
    )
    crit_sub = df_model[crit_mask]
    critical = {}
    if len(crit_sub) > 0:
        pattern_dist = crit_sub["gamma_pattern"].value_counts(normalize=True) * 100
        critical = {
            "description": "Gamma drops >2Bn in first 30min AND first_30_ratio > 0.5",
            "count": len(crit_sub),
            "pattern_distribution": {k: round(v, 1) for k, v in pattern_dist.items()},
            "recommendation": "RIDE (breakout)" if pattern_dist.get("2", 0) > pattern_dist.get("1", 0)
                else "FADE (flush+recover)"
        }

    prediction_rules = {
        "lookup_table": lookup_rows,
        "decision_rules": rules,
        "critical_question": critical,
    }

    return prediction_rules, accuracy_results


# ─────────────────────────────────────────────────────────────
# PART 5: STRATEGY SIMULATION
# ─────────────────────────────────────────────────────────────

SPREAD_WIDTH = 5
SPREAD_COST = 2

def simulate_trade(row, snaps_after_10am, trade_type, direction, entry, target, stop,
                   time_exit_pt_min, risk=200, reward=300, is_butterfly=False):
    """Simulate a single trade through subsequent snapshots.

    Returns: (exit_price, exit_reason, pnl_dollars)
    """
    if is_butterfly:
        # Butterfly: check close price vs call_wall
        call_wall = row.get("call_wall_at_2pm", 0)
        close_price = row["day_close"]
        distance = abs(close_price - call_wall)
        if distance <= 5:
            return close_price, "target", 1500
        else:
            return close_price, "stop", -500

    for s in snaps_after_10am:
        t_min = pt_time_to_minutes(s["time_pt"])
        px = s["curr_price"]

        # Check stop
        if direction > 0:
            if px <= stop:
                return px, "stop", -SPREAD_COST * 100
        else:
            if px >= stop:
                return px, "stop", -SPREAD_COST * 100

        # Check target
        if direction > 0:
            if px >= target:
                return px, "target", (SPREAD_WIDTH - SPREAD_COST) * 100
        else:
            if px <= target:
                return px, "target", (SPREAD_WIDTH - SPREAD_COST) * 100

        # Check time exit
        if t_min >= time_exit_pt_min:
            move_toward = (px - entry) / (target - entry) if target != entry else 0
            move_toward = max(0, min(1, move_toward))
            option_value = SPREAD_COST + (SPREAD_WIDTH - SPREAD_COST) * move_toward
            pnl = (option_value - SPREAD_COST) * 100
            return px, "time_exit", round(pnl, 2)

    # End of day exit
    px = snaps_after_10am[-1]["curr_price"] if snaps_after_10am else entry
    move_toward = (px - entry) / (target - entry) if target != entry else 0
    move_toward = max(0, min(1, move_toward))
    option_value = SPREAD_COST + (SPREAD_WIDTH - SPREAD_COST) * move_toward
    pnl = (option_value - SPREAD_COST) * 100
    return px, "eod", round(pnl, 2)


def run_strategy_simulation(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """Simulate the full trading strategy."""
    trades = []

    for idx, row in df.iterrows():
        date_str = row["date"]
        group = row["group"]
        subgroup = row["subgroup"]
        slope_label = row["slope_label"]
        og = row["open_gamma"]
        og_bn = og / 1e9
        entry = row["price_at_10am"]
        anchor = row["anchor"]
        em_pts = row["em_pts"]
        dir_sign = 1 if row["first_30_direction"] >= 0 else -1

        # Load snapshots after 10am ET (7:00 PT)
        pt_700 = 7 * 60
        snaps = conn.execute(
            "SELECT time_pt, curr_price, net_gamma, call_wall FROM gex_snapshots "
            "WHERE date_pt=? AND session_tag='RTH' ORDER BY time_pt",
            (date_str,),
        ).fetchall()
        snaps = [dict(s) for s in snaps]
        snaps_after = [s for s in snaps if pt_time_to_minutes(s["time_pt"]) > pt_700]

        if not snaps_after:
            trades.append({
                "date": date_str, "group": group, "subgroup": subgroup,
                "trade_type": "SKIP", "direction": 0, "entry": entry,
                "exit_price": entry, "exit_reason": "no_data",
                "pnl": 0, "flip_pnl": 0,
            })
            continue

        trade_type = "SKIP"
        direction = 0
        target = entry
        stop = entry
        time_exit_pt = et_time_str_to_pt_minutes(14, 0)  # 2pm ET default
        is_butterfly = False

        if group == "A" and og < -3e9 and slope_label == "ACCELERATING":
            # A1 ACCELERATING — trend day
            trade_type = "BREAKOUT"
            direction = dir_sign
            if direction < 0:
                target = min(anchor - em_pts, entry - 20)
                stop = entry + 10
            else:
                target = max(anchor + em_pts, entry + 20)
                stop = entry - 10
            time_exit_pt = et_time_str_to_pt_minutes(14, 0)

        elif group == "A" and og > 3e9:
            # A2 — big open + positive gamma → fade
            trade_type = "FADE"
            direction = -dir_sign
            target = anchor
            stop = entry + (15 * dir_sign)
            time_exit_pt = et_time_str_to_pt_minutes(14, 0)

        elif group == "A" and slope_label == "DECELERATING":
            # A with gamma recovering → fade flush
            trade_type = "FADE"
            direction = -dir_sign
            target = anchor - (0.25 * em_pts * dir_sign)
            stop = entry + (15 * dir_sign)
            time_exit_pt = et_time_str_to_pt_minutes(13, 0)

        elif group == "A":
            # A3 FLAT slope — still big open, lean breakout
            trade_type = "BREAKOUT"
            direction = dir_sign
            if direction < 0:
                target = min(anchor - em_pts, entry - 20)
                stop = entry + 10
            else:
                target = max(anchor + em_pts, entry + 20)
                stop = entry - 10
            time_exit_pt = et_time_str_to_pt_minutes(14, 0)

        elif group == "C" and og > 3e9:
            # C2 — quiet + positive gamma = pin day
            trade_type = "PIN_BUTTERFLY"
            is_butterfly = True
            # Check gamma at 2pm
            snap_2pm = nearest_snapshot_to_pt_minutes(snaps, et_time_str_to_pt_minutes(14, 0))
            if snap_2pm and snap_2pm["net_gamma"] > 3e9:
                pass  # qualify
            else:
                trade_type = "SKIP"  # doesn't qualify

        elif group == "C" and og < -3e9:
            # C1 — quiet + neg gamma, wait for fade at -0.75 EM
            trade_type = "WAIT_FADE"
            entry_level = anchor - (0.75 * em_pts) if dir_sign < 0 else anchor + (0.75 * em_pts)
            # Check if price reaches the level
            reached = False
            for s in snaps_after:
                if dir_sign < 0 and s["curr_price"] <= entry_level:
                    entry = s["curr_price"]
                    reached = True
                    break
                elif dir_sign > 0 and s["curr_price"] >= entry_level:
                    entry = s["curr_price"]
                    reached = True
                    break
            if reached:
                direction = -dir_sign
                target = anchor - (0.25 * em_pts * dir_sign)
                stop = anchor - (1.0 * em_pts * dir_sign) if dir_sign < 0 else anchor + (1.0 * em_pts * dir_sign)
                time_exit_pt = et_time_str_to_pt_minutes(15, 0)
                # Filter snaps_after to those after entry
                snaps_after = [s for s in snaps_after if s["curr_price"] != 0]  # keep all
            else:
                trade_type = "SKIP"

        elif group == "C":
            # C3 — skip
            trade_type = "SKIP"

        elif group == "B":
            if og < -5e9:
                trade_type = "FADE"
                direction = -dir_sign
                target = anchor
                stop = entry - 15 if dir_sign > 0 else entry + 15
                time_exit_pt = et_time_str_to_pt_minutes(14, 0)
            elif og > 5e9:
                trade_type = "PIN_BUTTERFLY"
                is_butterfly = True
                snap_2pm = nearest_snapshot_to_pt_minutes(snaps, et_time_str_to_pt_minutes(14, 0))
                if not (snap_2pm and snap_2pm["net_gamma"] > 3e9):
                    trade_type = "SKIP"
            else:
                trade_type = "SKIP"

        # Execute trade
        if trade_type == "SKIP":
            trades.append({
                "date": date_str, "group": group, "subgroup": subgroup,
                "trade_type": "SKIP", "direction": 0, "entry": entry,
                "exit_price": entry, "exit_reason": "skip",
                "pnl": 0, "flip_pnl": 0,
            })
            continue

        exit_px, exit_reason, pnl = simulate_trade(
            row, snaps_after, trade_type, direction, entry, target, stop,
            time_exit_pt, is_butterfly=is_butterfly,
        )

        # FLIP LOGIC: if fade hit stop, open breakout
        flip_pnl = 0
        if trade_type == "FADE" and exit_reason == "stop":
            # Open breakout in original direction
            flip_entry = exit_px
            flip_dir = dir_sign
            if flip_dir < 0:
                flip_target = flip_entry - 20
                flip_stop = flip_entry + 10
            else:
                flip_target = flip_entry + 20
                flip_stop = flip_entry - 10

            # Get remaining snaps after the stop hit
            stop_time = None
            for s in snaps_after:
                if direction > 0 and s["curr_price"] <= stop:
                    stop_time = s["time_pt"]
                    break
                elif direction < 0 and s["curr_price"] >= stop:
                    stop_time = s["time_pt"]
                    break

            remaining = snaps_after
            if stop_time:
                stop_min = pt_time_to_minutes(stop_time)
                remaining = [s for s in snaps_after if pt_time_to_minutes(s["time_pt"]) >= stop_min]

            if remaining:
                _, flip_reason, flip_pnl = simulate_trade(
                    row, remaining, "BREAKOUT", flip_dir, flip_entry,
                    flip_target, flip_stop, et_time_str_to_pt_minutes(15, 0),
                )

        trades.append({
            "date": date_str,
            "group": group,
            "subgroup": subgroup,
            "trade_type": trade_type,
            "direction": direction,
            "entry": round(entry, 2),
            "target": round(target, 2) if trade_type != "SKIP" else 0,
            "stop": round(stop, 2) if trade_type != "SKIP" else 0,
            "exit_price": round(exit_px, 2),
            "exit_reason": exit_reason,
            "pnl": round(pnl, 2),
            "flip_pnl": round(flip_pnl, 2),
            "total_pnl": round(pnl + flip_pnl, 2),
        })

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────
# PART 6: REPORTING
# ─────────────────────────────────────────────────────────────

def print_separator(char="-", width=80):
    print(char * width)


def print_summary(trades_df: pd.DataFrame, df: pd.DataFrame):
    """Print full strategy summary to console."""

    print("\n" + "=" * 80)
    print("  COMPREHENSIVE DAY CLASSIFICATION BACKTEST - RESULTS")
    print("=" * 80)

    # Date range
    print(f"\n  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"  Total trading days: {len(df)}")

    # ── GROUP OUTCOME TABLE ──
    print("\n" + "=" * 80)
    print("  PART 2-3: DAY CLASSIFICATION OUTCOMES")
    print("=" * 80)

    for g in ["A", "B", "C"]:
        sub = df[df["group"] == g]
        if len(sub) == 0:
            continue
        label = {"A": "BIG OPEN", "B": "MEDIUM OPEN", "C": "QUIET OPEN"}[g]
        print(f"\n  Group {g} - {label} ({len(sub)} days, {len(sub)/len(df)*100:.0f}%)")
        print(f"    Breach%: {sub['intraday_breach'].mean()*100:.0f}%  |  "
              f"Close inside EM: {sub['close_inside_em'].mean()*100:.0f}%  |  "
              f"Avg range: {sub['day_range'].mean():.1f} pts  |  "
              f"Avg remaining: {sub['remaining_max_move'].mean():.1f} pts")

        for sg in [g + "1", g + "2", g + "3"]:
            ss = df[df["subgroup"] == sg]
            if len(ss) == 0:
                continue
            gamma_label = {"1": "NEG GAMMA", "2": "POS GAMMA", "3": "NEUTRAL"}[sg[-1]]
            print(f"    {sg} ({gamma_label}): {len(ss)} days - "
                  f"Breach {ss['intraday_breach'].mean()*100:.0f}%, "
                  f"CloseIn {ss['close_inside_em'].mean()*100:.0f}%, "
                  f"AvgRange {ss['day_range'].mean():.1f}")

    # ── REMAINING MOVE PERCENTILES ──
    print("\n" + "-" * 80)
    print("  REMAINING MOVE AFTER 10:00 AM (percentiles, points)")
    print(f"  {'GROUP':<12} {'P25':>6} {'P50':>6} {'P75':>6} {'P90':>6}")
    print_separator()
    for g in ["A", "B", "C"]:
        sub = df[df["group"] == g]
        if len(sub) < 2:
            continue
        vals = sub["remaining_max_move"]
        print(f"  {g:<12} {np.percentile(vals,25):6.1f} {np.percentile(vals,50):6.1f} "
              f"{np.percentile(vals,75):6.1f} {np.percentile(vals,90):6.1f}")
        for sg in [g + "1", g + "2", g + "3"]:
            ss = df[df["subgroup"] == sg]
            if len(ss) < 2:
                continue
            v2 = ss["remaining_max_move"]
            print(f"    {sg:<10} {np.percentile(v2,25):6.1f} {np.percentile(v2,50):6.1f} "
                  f"{np.percentile(v2,75):6.1f} {np.percentile(v2,90):6.1f}")

    # ── DIRECTION PERSISTENCE ──
    print("\n" + "-" * 80)
    print("  DIRECTION PERSISTENCE (does first 30min predict rest of day?)")
    print(f"  {'GROUP':<12} {'Down>Lower%':>12} {'(n)':>5} {'Up>Higher%':>12} {'(n)':>5}")
    print_separator()
    for g in ["A", "B", "C"]:
        sub = df[df["group"] == g]
        if len(sub) < 3:
            continue
        dn = sub[sub["first_30_direction"] < 0]
        up = sub[sub["first_30_direction"] > 0]
        dn_pct = (dn["day_close"] < dn["price_at_10am"]).mean() * 100 if len(dn) > 0 else 0
        up_pct = (up["day_close"] > up["price_at_10am"]).mean() * 100 if len(up) > 0 else 0
        print(f"  {g:<12} {dn_pct:11.0f}% {len(dn):>4}  {up_pct:11.0f}% {len(up):>4}")

    # ── GAMMA PATTERN DISTRIBUTION ──
    print("\n" + "-" * 80)
    print("  GAMMA PATTERN DISTRIBUTION BY GROUP")
    print(f"  {'GROUP':<8} {'P1-Flush':>9} {'P2-Bleed':>9} {'P3-Pin':>9} "
          f"{'P4-Flat':>9} {'P5-Build':>9} {'MIXED':>9}")
    print_separator()
    for g in ["A", "B", "C"]:
        sub = df[df["group"] == g]
        if len(sub) == 0:
            continue
        dist = sub["gamma_pattern"].value_counts(normalize=True) * 100
        print(f"  {g:<8} {dist.get('1',0):8.0f}% {dist.get('2',0):8.0f}% "
              f"{dist.get('3',0):8.0f}% {dist.get('4',0):8.0f}% "
              f"{dist.get('5',0):8.0f}% {dist.get('MIXED',0):8.0f}%")

    # ── EOD GAMMA -> NEXT DAY ──
    print("\n" + "-" * 80)
    print("  EOD GAMMA -> NEXT DAY PREDICTIVE POWER")

    def eod_bucket(g):
        g_bn = g / 1e9
        if g_bn < -5: return "< -5Bn"
        elif g_bn < -2: return "-5 to -2Bn"
        elif g_bn <= 2: return "-2 to +2Bn"
        elif g_bn <= 5: return "+2 to +5Bn"
        else: return "> +5Bn"

    df_temp = df.copy()
    df_temp["_eod_bucket"] = df_temp["eod_gamma"].apply(eod_bucket)
    print(f"  {'EOD Gamma':<14} {'N':>4} {'NextBreach%':>12} {'NextCloseIn%':>13} {'NextRange':>10}")
    print_separator()
    for bucket in ["< -5Bn", "-5 to -2Bn", "-2 to +2Bn", "+2 to +5Bn", "> +5Bn"]:
        sub = df_temp[df_temp["_eod_bucket"] == bucket].dropna(subset=["next_day_breach"])
        if len(sub) < 2:
            continue
        print(f"  {bucket:<14} {len(sub):>4} {sub['next_day_breach'].mean()*100:11.0f}% "
              f"{sub['next_day_close_inside'].mean()*100:12.0f}% "
              f"{sub['next_day_range'].mean():9.1f}")

    # ── STRATEGY RESULTS ──
    print("\n" + "=" * 80)
    print("  PART 5-6: STRATEGY SIMULATION RESULTS")
    print("=" * 80)

    active = trades_df[trades_df["trade_type"] != "SKIP"]
    skipped = trades_df[trades_df["trade_type"] == "SKIP"]

    print(f"\n  {'TRADE TYPE':<16} {'DAYS':>5} {'WIN%':>6} {'AVG P&L':>9} {'TOTAL P&L':>11} {'SHARPE':>7}")
    print_separator()

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    all_pnls = []

    for tt in ["BREAKOUT", "FADE", "PIN_BUTTERFLY", "WAIT_FADE"]:
        sub = active[active["trade_type"] == tt]
        if len(sub) == 0:
            print(f"  {tt:<16} {'0':>5} {'--':>6} {'--':>9} {'--':>11} {'--':>7}")
            continue
        wins = (sub["total_pnl"] > 0).sum()
        win_pct = wins / len(sub) * 100
        avg_pnl = sub["total_pnl"].mean()
        sum_pnl = sub["total_pnl"].sum()
        pnls = sub["total_pnl"].values
        sharpe = (pnls.mean() / pnls.std()) * (252 ** 0.5) if pnls.std() > 0 else 0
        print(f"  {tt:<16} {len(sub):>5} {win_pct:>5.0f}% ${avg_pnl:>7.0f} ${sum_pnl:>9.0f}   {sharpe:>5.2f}")
        total_pnl += sum_pnl
        total_trades += len(sub)
        total_wins += wins
        all_pnls.extend(pnls.tolist())

    print(f"  {'SKIP':<16} {len(skipped):>5} {'--':>6} {'$0':>9} {'$0':>11} {'--':>7}")
    print_separator()

    if total_trades > 0:
        combined_pnls = np.array(all_pnls)
        combined_sharpe = (combined_pnls.mean() / combined_pnls.std()) * (252 ** 0.5) if combined_pnls.std() > 0 else 0
        print(f"  {'COMBINED':<16} {total_trades:>5} {total_wins/total_trades*100:>5.0f}% "
              f"${combined_pnls.mean():>7.0f} ${total_pnl:>9.0f}   {combined_sharpe:>5.2f}")

    # ── FLIP SAVES ──
    flips = active[active["flip_pnl"] != 0]
    if len(flips) > 0:
        print(f"\n  FLIP SAVES: {len(flips)} trades")
        flip_wins = (flips["flip_pnl"] > 0).sum()
        print(f"    Flip win%: {flip_wins/len(flips)*100:.0f}%  |  "
              f"Avg flip P&L: ${flips['flip_pnl'].mean():.0f}  |  "
              f"Total flip P&L: ${flips['flip_pnl'].sum():.0f}")
        print("\n  BLOWUP DAY DETAILS (fade stopped -> flip triggered):")
        print(f"  {'DATE':<12} {'Fade P&L':>10} {'Flip P&L':>10} {'Net P&L':>10}")
        for _, fr in flips.iterrows():
            print(f"  {fr['date']:<12} ${fr['pnl']:>9.0f} ${fr['flip_pnl']:>9.0f} ${fr['total_pnl']:>9.0f}")

    # ── MONTHLY BREAKDOWN ──
    print("\n" + "-" * 80)
    print("  MONTHLY BREAKDOWN")
    print(f"  {'MONTH':<10} {'TRADES':>7} {'WINS':>6} {'LOSSES':>7} {'SKIPS':>6} {'P&L':>10} {'WIN%':>6}")
    print_separator()

    trades_df["month"] = trades_df["date"].str[:7]
    for month in sorted(trades_df["month"].unique()):
        m = trades_df[trades_df["month"] == month]
        m_active = m[m["trade_type"] != "SKIP"]
        n_trades = len(m_active)
        n_wins = (m_active["total_pnl"] > 0).sum() if n_trades > 0 else 0
        n_losses = (m_active["total_pnl"] <= 0).sum() if n_trades > 0 else 0
        n_skips = len(m) - n_trades
        m_pnl = m_active["total_pnl"].sum() if n_trades > 0 else 0
        win_pct = f"{n_wins/n_trades*100:.0f}%" if n_trades > 0 else "--"
        print(f"  {month:<10} {n_trades:>7} {n_wins:>6} {n_losses:>7} {n_skips:>6} ${m_pnl:>8.0f} {win_pct:>6}")

    # ── OTHER METRICS ──
    print("\n" + "-" * 80)
    print("  KEY METRICS")
    print_separator()

    if total_trades > 0:
        pnl_arr = np.array(all_pnls)
        cum_pnl = np.cumsum(pnl_arr)

        # Consecutive wins/losses
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for p in pnl_arr:
            if p > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)

        # Max drawdown
        peak = cum_pnl[0]
        max_dd = 0
        for v in cum_pnl:
            peak = max(peak, v)
            dd = peak - v
            max_dd = max(max_dd, dd)

        # Profit factor
        gross_wins = pnl_arr[pnl_arr > 0].sum()
        gross_losses = abs(pnl_arr[pnl_arr <= 0].sum())
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Best / worst day
        best_idx = np.argmax(pnl_arr)
        worst_idx = np.argmin(pnl_arr)
        active_dates = active["date"].tolist()

        print(f"  Total trading days:     {len(df)}")
        print(f"  Days with trades:       {total_trades} ({total_trades/len(df)*100:.0f}%)")
        print(f"  Days skipped:           {len(skipped)} ({len(skipped)/len(df)*100:.0f}%)")
        print(f"  Overall win rate:       {total_wins/total_trades*100:.0f}%")
        print(f"  Average P&L per trade:  ${pnl_arr.mean():.0f}")
        print(f"  Max consecutive wins:   {max_win_streak}")
        print(f"  Max consecutive losses: {max_loss_streak}")
        print(f"  Max drawdown:           ${max_dd:.0f}")
        print(f"  Profit factor:          {profit_factor:.2f}")
        print(f"  Best single day:        ${pnl_arr[best_idx]:.0f} on {active_dates[best_idx]}")
        print(f"  Worst single day:       ${pnl_arr[worst_idx]:.0f} on {active_dates[worst_idx]}")

    print("\n" + "=" * 80)


# ─────────────────────────────────────────────────────────────
# PART 7: SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────

def save_all_outputs(df, gamma_profiles_df, group_df, pct_df, pattern_dist_df,
                     eod_df, prediction_rules, accuracy_results, trades_df):
    """Save all CSV, JSON, and chart files."""

    # Add gamma profile columns to df for the combined CSV
    gp_cols = [c for c in gamma_profiles_df.columns if c.startswith("gamma_")]
    for c in gp_cols:
        if c not in df.columns:
            df[c] = gamma_profiles_df[c].values if len(gamma_profiles_df) == len(df) else np.nan

    # CSV outputs
    df.to_csv(os.path.join(REPORTS_DIR, "daily_features_full.csv"), index=False)
    gamma_profiles_df.to_csv(os.path.join(REPORTS_DIR, "gamma_profiles_daily.csv"), index=False)

    # Gamma patterns
    pat_df = df[["date", "gamma_pattern", "gamma_range_bn", "gamma_drift_label",
                  "gamma_peak_time", "gamma_trough_time", "gamma_acceleration"]].copy()
    pat_df.to_csv(os.path.join(REPORTS_DIR, "gamma_patterns.csv"), index=False)

    group_df.to_csv(os.path.join(REPORTS_DIR, "group_outcomes.csv"), index=False)
    pct_df.to_csv(os.path.join(REPORTS_DIR, "remaining_move_table.csv"), index=False)
    pattern_dist_df.to_csv(os.path.join(REPORTS_DIR, "gamma_pattern_distribution.csv"), index=False)
    eod_df.to_csv(os.path.join(REPORTS_DIR, "eod_gamma_next_day.csv"), index=False)

    # JSON outputs
    with open(os.path.join(REPORTS_DIR, "pattern_prediction_rules.json"), "w") as f:
        json.dump(prediction_rules, f, indent=2, default=str)
    with open(os.path.join(REPORTS_DIR, "pattern_prediction_accuracy.json"), "w") as f:
        json.dump(accuracy_results, f, indent=2, default=str)

    # Strategy trades
    trades_df.to_csv(os.path.join(REPORTS_DIR, "strategy_trades.csv"), index=False)

    # Equity curve
    active_trades = trades_df[trades_df["trade_type"] != "SKIP"].copy()
    if len(active_trades) > 0:
        active_trades["cumulative_pnl"] = active_trades["total_pnl"].cumsum()
        eq_df = active_trades[["date", "trade_type", "total_pnl", "cumulative_pnl"]].copy()
        eq_df.to_csv(os.path.join(REPORTS_DIR, "strategy_equity_curve.csv"), index=False)

        # Chart
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
            fig.suptitle("Day Classification Strategy - Equity Curve", fontsize=14, fontweight="bold")

            dates = pd.to_datetime(active_trades["date"])
            cum_pnl = active_trades["cumulative_pnl"].values

            ax1.fill_between(dates, cum_pnl, where=(cum_pnl >= 0), alpha=0.3, color="green")
            ax1.fill_between(dates, cum_pnl, where=(cum_pnl < 0), alpha=0.3, color="red")
            ax1.plot(dates, cum_pnl, color="black", linewidth=1.5)
            ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax1.set_ylabel("Cumulative P&L ($)")
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

            # Daily P&L bars
            pnls = active_trades["total_pnl"].values
            colors = ["green" if p > 0 else "red" for p in pnls]
            ax2.bar(dates, pnls, color=colors, alpha=0.6, width=1)
            ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax2.set_ylabel("Daily P&L ($)")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

            plt.tight_layout()
            plt.savefig(os.path.join(REPORTS_DIR, "strategy_equity_curve.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved equity curve chart")
        except Exception as e:
            print(f"  Chart error (non-critical): {e}")

    # Full summary JSON
    summary = {
        "date_range": f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}",
        "total_days": len(df),
        "group_distribution": {
            g: int((df["group"] == g).sum()) for g in ["A", "B", "C"]
        },
        "pattern_distribution": df["gamma_pattern"].value_counts().to_dict(),
        "overall_breach_pct": round(df["intraday_breach"].mean() * 100, 1),
        "overall_close_inside_pct": round(df["close_inside_em"].mean() * 100, 1),
    }
    if len(active_trades) > 0:
        pnl_arr = active_trades["total_pnl"].values
        summary["strategy"] = {
            "total_trades": int(len(active_trades)),
            "skipped": int(len(trades_df) - len(active_trades)),
            "win_rate": round((pnl_arr > 0).mean() * 100, 1),
            "total_pnl": round(float(pnl_arr.sum()), 2),
            "avg_pnl": round(float(pnl_arr.mean()), 2),
            "sharpe": round(float((pnl_arr.mean() / pnl_arr.std()) * (252 ** 0.5)), 2) if pnl_arr.std() > 0 else 0,
        }

    with open(os.path.join(REPORTS_DIR, "full_backtest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  All reports saved to {REPORTS_DIR}/")
    files = [
        "daily_features_full.csv", "gamma_profiles_daily.csv", "gamma_patterns.csv",
        "group_outcomes.csv", "remaining_move_table.csv", "gamma_pattern_distribution.csv",
        "eod_gamma_next_day.csv", "pattern_prediction_rules.json",
        "pattern_prediction_accuracy.json", "strategy_trades.csv",
        "strategy_equity_curve.csv", "strategy_equity_curve.png",
        "full_backtest_summary.json",
    ]
    for f in files:
        path = os.path.join(REPORTS_DIR, f)
        exists = "OK" if os.path.exists(path) else "MISSING"
        print(f"    [{exists}] {f}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  COMPREHENSIVE DAY CLASSIFICATION BACKTEST")
    print("=" * 80)

    conn = get_conn()

    # Step 1: Load market data
    print("\n[1/7] Loading market data from yfinance...")
    # Get date range from DB
    row = conn.execute(
        "SELECT MIN(date_pt) as mn, MAX(date_pt) as mx FROM gex_snapshots WHERE session_tag='RTH'"
    ).fetchone()
    start_date = (datetime.strptime(row["mn"], "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = (datetime.strptime(row["mx"], "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
    mkt = load_market_data(start_date, end_date)
    print(f"  Market data: {len(mkt)} trading days")

    # Step 2: Build daily features
    print("\n[2/7] Building daily feature table...")
    df, gamma_profiles_df = build_daily_features(conn, mkt)
    print(f"  Built features for {len(df)} trading days")

    # Step 3: Classify days
    print("\n[3/7] Classifying days at 10:00 AM...")
    # Merge gamma profile data for classification
    for label, _, _ in ET_MARKS:
        col = f"gamma_{label}"
        if col in gamma_profiles_df.columns:
            df[col] = gamma_profiles_df[col].values

    classifications = df.apply(classify_day, axis=1, result_type="expand")
    df["group"] = classifications["group"]
    df["subgroup"] = classifications["subgroup"]
    df["slope_label"] = classifications["slope_label"]
    df["sub_subgroup"] = classifications["sub_subgroup"]

    print(f"  Group A (BIG): {(df['group']=='A').sum()} | "
          f"Group B (MED): {(df['group']=='B').sum()} | "
          f"Group C (QUIET): {(df['group']=='C').sum()}")

    # Step 4: Compute group outcomes
    print("\n[4/7] Computing group outcomes...")
    group_df, pct_df, pattern_dist_df, eod_df = compute_group_outcomes(df)

    # Step 5: Build predictive model
    print("\n[5/7] Building predictive model...")
    prediction_rules, accuracy_results = build_prediction_model(df)
    if "decision_tree" in accuracy_results:
        print(f"  Decision tree accuracy: {accuracy_results['decision_tree']['accuracy']}%")

    # Step 6: Run strategy simulation
    print("\n[6/7] Running strategy simulation...")
    trades_df = run_strategy_simulation(df, conn)
    active = trades_df[trades_df["trade_type"] != "SKIP"]
    print(f"  Simulated {len(active)} trades ({len(trades_df) - len(active)} skipped)")

    # Step 7: Save and report
    print("\n[7/7] Saving outputs and generating reports...")
    save_all_outputs(df, gamma_profiles_df, group_df, pct_df, pattern_dist_df,
                     eod_df, prediction_rules, accuracy_results, trades_df)

    # Print full console summary
    print_summary(trades_df, df)

    # Print prediction rules
    print("\n" + "=" * 80)
    print("  PART 4: PATTERN PREDICTION")
    print("=" * 80)

    if prediction_rules.get("decision_rules"):
        print("\n  TOP DECISION RULES:")
        for i, rule in enumerate(prediction_rules["decision_rules"], 1):
            print(f"    Rule {i}: IF {rule['rule']}")
            print(f"             THEN Pattern {rule['predicted_pattern']} "
                  f"({rule['probability']}% probability, n={rule['count']})")

    if prediction_rules.get("critical_question"):
        cq = prediction_rules["critical_question"]
        print(f"\n  CRITICAL QUESTION:")
        print(f"    {cq.get('description', 'N/A')}")
        print(f"    Sample size: {cq.get('count', 0)}")
        if cq.get("pattern_distribution"):
            for pat, pct in sorted(cq["pattern_distribution"].items()):
                label_map = {"1": "Flush+Recover (FADE)", "2": "Steady Bleed (RIDE)",
                             "3": "Pin Buildup", "4": "Flat", "5": "Morning Build", "MIXED": "Mixed"}
                print(f"      Pattern {pat} ({label_map.get(pat, pat)}): {pct:.0f}%")
        print(f"    Recommendation: {cq.get('recommendation', 'N/A')}")

    if "decision_tree" in accuracy_results:
        print(f"\n  DECISION TREE ({accuracy_results['decision_tree']['accuracy']}% accuracy):")
        tree_text = accuracy_results["decision_tree"]["tree_text"]
        for line in tree_text.split("\n")[:15]:
            print(f"    {line}")
        if len(tree_text.split("\n")) > 15:
            print("    ...")

    conn.close()
    print("\n  BACKTEST COMPLETE.\n")


if __name__ == "__main__":
    main()
