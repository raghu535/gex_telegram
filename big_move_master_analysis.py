"""
Master move analysis:
- Daily move distribution
- Big day predictors
- Overnight swing tests
- Intraday momentum trigger test
- Last 30 min momentum test

Inputs:
  gex_data.db (gex_snapshots, RTH)
  yfinance ^VIX

Outputs:
  reports/daily_move_distribution.csv
  reports/big_move_predictors.json
  reports/swing_overnight_trades.csv
  reports/momentum_trigger_trades.csv
  reports/last_30min_trades.csv
  reports/big_move_master_summary.json
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "gex_data.db"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def pt_min(time_str: str) -> float:
    h, m, s = [int(x) for x in time_str.split(":")]
    return h * 60 + m + s / 60.0


def et_to_pt_min(h: int, m: int = 0) -> int:
    return (h - 3) * 60 + m


def nearest_price(rows: List[dict], target_pt_min: int) -> Optional[float]:
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(r["t_pt"] - target_pt_min))
    return float(best["price"])


def nearest_row(rows: List[dict], target_pt_min: int) -> Optional[dict]:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r["t_pt"] - target_pt_min))


def max_window_directional_move(rows: List[dict], window_min: int) -> Tuple[float, Optional[float]]:
    """
    Biggest absolute move inside any rolling window of length `window_min`.
    Returned as (move_points, start_pt_min_of_best_window).
    """
    if len(rows) < 2:
        return 0.0, None
    best_move = 0.0
    best_start = None
    n = len(rows)
    j = 0
    prices = np.array([r["price"] for r in rows], dtype=float)
    times = np.array([r["t_pt"] for r in rows], dtype=float)
    for i in range(n):
        start_t = times[i]
        while j + 1 < n and times[j + 1] <= start_t + window_min:
            j += 1
        if j <= i:
            continue
        window = prices[i : j + 1]
        mv = float(window.max() - window.min())
        if mv > best_move:
            best_move = mv
            best_start = float(start_t)
    return best_move, best_start


def et_hour_bucket_from_pt(start_pt_min: Optional[float]) -> Optional[str]:
    if start_pt_min is None:
        return None
    et = start_pt_min + 180.0
    if 570 <= et < 630:
        return "9:30-10:30"
    if 630 <= et < 720:
        return "10:30-12:00"
    if 720 <= et < 840:
        return "12:00-2:00"
    if 840 <= et < 900:
        return "2:00-3:00"
    if 900 <= et <= 960:
        return "3:00-4:00"
    return "outside"


def load_snapshots() -> Dict[str, List[dict]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT date_pt, time_pt, curr_price, net_gamma, call_wall, put_floor
        FROM gex_snapshots
        WHERE session_tag='RTH'
        ORDER BY date_pt, time_pt
        """
    ).fetchall()
    conn.close()
    out: Dict[str, List[dict]] = {}
    for r in rows:
        d = r["date_pt"]
        out.setdefault(d, []).append(
            {
                "date": d,
                "time_pt": r["time_pt"],
                "t_pt": pt_min(r["time_pt"]),
                "price": float(r["curr_price"]),
                "gamma": float(r["net_gamma"]),
                "call_wall": float(r["call_wall"]),
                "put_floor": float(r["put_floor"]),
            }
        )
    return out


def load_vix(min_date: str, max_date: str) -> pd.DataFrame:
    start = pd.Timestamp(min_date) - pd.Timedelta(days=10)
    end = pd.Timestamp(max_date) + pd.Timedelta(days=5)
    vix = yf.download("^VIX", start=start.date(), end=end.date(), progress=False, auto_adjust=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] for c in vix.columns]
    vix = vix[["Open", "Close"]].copy()
    vix.index = vix.index.tz_localize(None)
    vix["date"] = vix.index.strftime("%Y-%m-%d")
    return vix.reset_index(drop=True)


def prev_vix_close(vix_df: pd.DataFrame, date_str: str) -> Optional[float]:
    sub = vix_df[vix_df["date"] < date_str]
    if sub.empty:
        return None
    return float(sub.iloc[-1]["Close"])


def vix_open(vix_df: pd.DataFrame, date_str: str) -> Optional[float]:
    sub = vix_df[vix_df["date"] == date_str]
    if sub.empty:
        return None
    return float(sub.iloc[0]["Open"])


def vix_close(vix_df: pd.DataFrame, date_str: str) -> Optional[float]:
    sub = vix_df[vix_df["date"] == date_str]
    if sub.empty:
        return None
    return float(sub.iloc[0]["Close"])


@dataclass
class DailyRow:
    date: str
    open_price: float
    close_price: float
    day_high: float
    day_low: float
    day_range: float
    max_2hr_move: float
    max_1hr_move: float
    max_30min_move: float
    max_1hr_start_pt: Optional[float]
    max_1hr_bucket: Optional[str]
    last_hour_move: float
    last_30min_move: float
    morning_move: float
    afternoon_move: float
    price_10am: float
    price_11am: float
    price_1pm: float
    price_2pm: float
    price_230pm: float
    price_330pm: float
    open_gamma: float
    eod_gamma: float
    gamma_330: float
    open_spread: float
    call_wall_330: float
    put_floor_330: float
    first30_high: float
    first30_low: float
    first30_range: float
    first30_direction: float
    anchor: Optional[float]
    em_pts: Optional[float]
    em_upper: Optional[float]
    em_lower: Optional[float]
    em_pos_10am: Optional[float]
    vix_prev_close: Optional[float]
    vix_open_today: Optional[float]
    overnight_gap: Optional[float]
    prior_day_range: Optional[float]
    prior_day_open: Optional[float]
    prior_day_close: Optional[float]
    prior_day_high: Optional[float]
    prior_day_low: Optional[float]
    prior_day_em_pts: Optional[float]
    prior_day_eod_gamma: Optional[float]
    prior_day_compressed: Optional[int]
    prior_day_close_em_pos: Optional[float]
    first30_ratio: Optional[float]
    first30_sweep_prior_level: Optional[int]


def build_daily_table(snap_map: Dict[str, List[dict]], vix_df: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(snap_map.keys())
    rows: List[DailyRow] = []
    date_to_partial: Dict[str, Dict[str, float]] = {}

    for date_str in dates:
        day = snap_map[date_str]
        prices = [r["price"] for r in day]
        open_p = float(day[0]["price"])
        close_p = float(day[-1]["price"])
        high_p = float(max(prices))
        low_p = float(min(prices))
        day_range = high_p - low_p

        max2h, _ = max_window_directional_move(day, 120)
        max1h, max1h_start = max_window_directional_move(day, 60)
        max30, _ = max_window_directional_move(day, 30)

        p10 = nearest_price(day, et_to_pt_min(10, 0))
        p11 = nearest_price(day, et_to_pt_min(11, 0))
        p1 = nearest_price(day, et_to_pt_min(13, 0))
        p2 = nearest_price(day, et_to_pt_min(14, 0))
        p230 = nearest_price(day, et_to_pt_min(14, 30))
        p3 = nearest_price(day, et_to_pt_min(15, 0))
        p330 = nearest_price(day, et_to_pt_min(15, 30))

        r330 = nearest_row(day, et_to_pt_min(15, 30))

        f30 = [r for r in day if r["t_pt"] <= et_to_pt_min(10, 0)]
        if not f30:
            f30 = day[:1]
        f30_high = float(max(r["price"] for r in f30))
        f30_low = float(min(r["price"] for r in f30))
        f30_range = f30_high - f30_low
        f30_direction = (p10 - open_p) if p10 is not None else np.nan

        vix_prev = prev_vix_close(vix_df, date_str)
        vix_o = vix_open(vix_df, date_str)

        # prior day references
        idx = dates.index(date_str)
        if idx > 0:
            prior_date = dates[idx - 1]
            prior_day = date_to_partial[prior_date]
            anchor = prior_day["close_price"]
            overnight_gap = open_p - anchor
            prior_day_range = prior_day["day_range"]
            prior_open = prior_day["open_price"]
            prior_close = prior_day["close_price"]
            prior_high = prior_day["day_high"]
            prior_low = prior_day["day_low"]
            prior_eod_gamma = prior_day["eod_gamma"]
            prior_em_pts = prior_day.get("em_pts")
            prior_close_em_pos = prior_day.get("close_em_pos")
        else:
            anchor = None
            overnight_gap = None
            prior_day_range = None
            prior_open = None
            prior_close = None
            prior_high = None
            prior_low = None
            prior_eod_gamma = None
            prior_em_pts = None
            prior_close_em_pos = None

        if anchor is not None and vix_prev is not None:
            em = (vix_prev / 16.0) / 100.0 * anchor
            em_u = anchor + em
            em_l = anchor - em
            em_pos_10 = (p10 - anchor) / em if (p10 is not None and em > 0) else np.nan
            f30_ratio = f30_range / em if em > 0 else np.nan
            close_em_pos = (close_p - anchor) / em if em > 0 else np.nan
        else:
            em = em_u = em_l = em_pos_10 = f30_ratio = close_em_pos = np.nan

        prior_compressed = (
            int(prior_day_range < (0.5 * prior_em_pts))
            if (prior_day_range is not None and prior_em_pts is not None and prior_em_pts > 0)
            else None
        )

        swept = (
            int((f30_high > prior_high) or (f30_low < prior_low))
            if (prior_high is not None and prior_low is not None)
            else None
        )

        row = DailyRow(
            date=date_str,
            open_price=open_p,
            close_price=close_p,
            day_high=high_p,
            day_low=low_p,
            day_range=day_range,
            max_2hr_move=max2h,
            max_1hr_move=max1h,
            max_30min_move=max30,
            max_1hr_start_pt=max1h_start,
            max_1hr_bucket=et_hour_bucket_from_pt(max1h_start),
            last_hour_move=abs((p3 if p3 is not None else close_p) - close_p),
            last_30min_move=abs((p330 if p330 is not None else close_p) - close_p),
            morning_move=abs(open_p - (p11 if p11 is not None else open_p)),
            afternoon_move=abs((p1 if p1 is not None else close_p) - close_p),
            price_10am=float(p10 if p10 is not None else open_p),
            price_11am=float(p11 if p11 is not None else open_p),
            price_1pm=float(p1 if p1 is not None else close_p),
            price_2pm=float(p2 if p2 is not None else close_p),
            price_230pm=float(p230 if p230 is not None else close_p),
            price_330pm=float(p330 if p330 is not None else close_p),
            open_gamma=float(day[0]["gamma"]),
            eod_gamma=float(day[-1]["gamma"]),
            gamma_330=float(r330["gamma"] if r330 is not None else day[-1]["gamma"]),
            open_spread=float(day[0]["call_wall"] - day[0]["put_floor"]),
            call_wall_330=float(r330["call_wall"] if r330 is not None else day[-1]["call_wall"]),
            put_floor_330=float(r330["put_floor"] if r330 is not None else day[-1]["put_floor"]),
            first30_high=f30_high,
            first30_low=f30_low,
            first30_range=f30_range,
            first30_direction=float(f30_direction),
            anchor=float(anchor) if anchor is not None else np.nan,
            em_pts=float(em) if not np.isnan(em) else np.nan,
            em_upper=float(em_u) if not np.isnan(em_u) else np.nan,
            em_lower=float(em_l) if not np.isnan(em_l) else np.nan,
            em_pos_10am=float(em_pos_10) if not np.isnan(em_pos_10) else np.nan,
            vix_prev_close=float(vix_prev) if vix_prev is not None else np.nan,
            vix_open_today=float(vix_o) if vix_o is not None else np.nan,
            overnight_gap=float(overnight_gap) if overnight_gap is not None else np.nan,
            prior_day_range=float(prior_day_range) if prior_day_range is not None else np.nan,
            prior_day_open=float(prior_open) if prior_open is not None else np.nan,
            prior_day_close=float(prior_close) if prior_close is not None else np.nan,
            prior_day_high=float(prior_high) if prior_high is not None else np.nan,
            prior_day_low=float(prior_low) if prior_low is not None else np.nan,
            prior_day_em_pts=float(prior_em_pts) if prior_em_pts is not None else np.nan,
            prior_day_eod_gamma=float(prior_eod_gamma) if prior_eod_gamma is not None else np.nan,
            prior_day_compressed=prior_compressed,
            prior_day_close_em_pos=float(prior_close_em_pos) if prior_close_em_pos is not None else np.nan,
            first30_ratio=float(f30_ratio) if not np.isnan(f30_ratio) else np.nan,
            first30_sweep_prior_level=swept,
        )
        rows.append(row)
        date_to_partial[date_str] = {
            "open_price": open_p,
            "close_price": close_p,
            "day_high": high_p,
            "day_low": low_p,
            "day_range": day_range,
            "eod_gamma": float(day[-1]["gamma"]),
            "em_pts": float(em) if not np.isnan(em) else None,
            "close_em_pos": float(close_em_pos) if not np.isnan(close_em_pos) else None,
        }

    return pd.DataFrame([r.__dict__ for r in rows])


def analysis_1_distribution(daily: pd.DataFrame) -> dict:
    n = len(daily)
    c_range_100 = int((daily["day_range"] > 100).sum())
    c_range_50 = int((daily["day_range"] > 50).sum())
    c_range_30 = int((daily["day_range"] > 30).sum())
    c_2h_50 = int((daily["max_2hr_move"] > 50).sum())
    c_2h_30 = int((daily["max_2hr_move"] > 30).sum())
    c_1h_20 = int((daily["max_1hr_move"] > 20).sum())
    c_last30_15 = int((daily["last_30min_move"] > 15).sum())
    c_last1h_20 = int((daily["last_hour_move"] > 20).sum())

    hour_dist = daily["max_1hr_bucket"].value_counts(dropna=False).to_dict()

    summary_line = (
        f"On {100.0 * c_range_50 / n:.1f}% of days there is a 50+ pt move somewhere in the day. "
        f"On {100.0 * c_2h_30 / n:.1f}% there is a 30+ pt 2-hour move. "
        f"On {100.0 * c_last1h_20 / n:.1f}% the last hour moves 20+ pts."
    )

    return {
        "total_days": int(n),
        "counts": {
            "day_range_gt_100": c_range_100,
            "day_range_gt_50": c_range_50,
            "day_range_gt_30": c_range_30,
            "max_2hr_move_gt_50": c_2h_50,
            "max_2hr_move_gt_30": c_2h_30,
            "max_1hr_move_gt_20": c_1h_20,
            "last_30min_move_gt_15": c_last30_15,
            "last_hour_move_gt_20": c_last1h_20,
        },
        "hour_bucket_distribution": hour_dist,
        "summary_line": summary_line,
    }


def _group_name(day_range: float) -> str:
    if day_range > 50:
        return "BIG"
    if day_range >= 30:
        return "MEDIUM"
    return "SMALL"


def hit_rate(mask: pd.Series, outcome: pd.Series) -> dict:
    m = mask.fillna(False)
    n = int(m.sum())
    if n == 0:
        return {"instances": 0, "hit_rate": None}
    return {"instances": n, "hit_rate": float(outcome[m].mean())}


def analysis_2_predictors(daily: pd.DataFrame) -> dict:
    d = daily.copy()
    d["size_group"] = d["day_range"].apply(_group_name)
    d["is_big"] = d["day_range"] > 50

    grp_stats = {}
    for g in ["BIG", "MEDIUM", "SMALL"]:
        s = d[d["size_group"] == g]
        if s.empty:
            continue
        grp_stats[g] = {
            "count": int(len(s)),
            "morning_state": {
                "avg_open_gamma_bn": float(s["open_gamma"].mean() / 1e9),
                "avg_vix": float(s["vix_prev_close"].mean()),
                "avg_open_spread": float(s["open_spread"].mean()),
                "avg_overnight_gap": float(s["overnight_gap"].mean()),
                "avg_prior_day_range": float(s["prior_day_range"].mean()),
                "avg_prior_day_eod_gamma_bn": float(s["prior_day_eod_gamma"].mean() / 1e9),
            },
            "prior_day_state": {
                "avg_prior_eod_gamma_bn": float(s["prior_day_eod_gamma"].mean() / 1e9),
                "avg_prior_day_range": float(s["prior_day_range"].mean()),
                "avg_prior_day_close_em_pos": float(s["prior_day_close_em_pos"].mean()),
                "prior_day_compressed_rate": float(s["prior_day_compressed"].fillna(0).mean()),
            },
        }

    # Night-before predictors
    p1 = hit_rate(d["prior_day_eod_gamma"] < -5e9, d["is_big"])
    p2 = hit_rate(d["prior_day_compressed"] == 1, d["is_big"])
    p3 = hit_rate(d["vix_prev_close"] > 20, d["is_big"])
    p4 = hit_rate((d["prior_day_eod_gamma"] < -5e9) & (d["prior_day_compressed"] == 1), d["is_big"])

    # 10am predictors
    p5 = hit_rate(d["first30_ratio"] > 0.3, d["is_big"])
    p6 = hit_rate((d["first30_ratio"] > 0.3) & (d["open_gamma"] < -3e9), d["is_big"])
    p7 = hit_rate(d["first30_sweep_prior_level"] == 1, d["is_big"])

    return {
        "group_stats": grp_stats,
        "night_before_predictors": {
            "prior_eod_gamma_lt_neg5bn": p1,
            "prior_day_compressed": p2,
            "vix_prev_close_gt_20": p3,
            "prior_eod_gamma_lt_neg5bn_and_compressed": p4,
        },
        "ten_am_predictors": {
            "first30_ratio_gt_0p3": p5,
            "first30_ratio_gt_0p3_and_open_gamma_lt_neg3bn": p6,
            "first30_swept_prior_high_or_low": p7,
        },
    }


def option_cost_base(vix: float, spx: float) -> float:
    return (vix / 16.0) / 100.0 * spx * 0.4


def analysis_3_swing(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy().reset_index(drop=True)
    out = []

    for i in range(1, len(d)):
        prev = d.iloc[i - 1]
        cur = d.iloc[i]
        if pd.isna(prev["close_price"]) or pd.isna(cur["price_2pm"]):
            continue
        anchor = float(prev["close_price"])
        exit_px = float(cur["price_2pm"])
        vix_c = vix_close_global.get(prev["date"])
        if vix_c is None or np.isnan(vix_c):
            continue
        base = option_cost_base(float(vix_c), anchor)
        if base <= 0:
            continue
        call_pay = max(0.0, exit_px - anchor)
        put_pay = max(0.0, anchor - exit_px)
        abs_pay = max(call_pay, put_pay)
        next_day_range = float(cur["day_range"])

        # A: every night straddle
        cost_a = 2.0 * base
        pnl_a = abs_pay - cost_a
        out.append(
            {
                "setup": "Swing A",
                "signal_date": prev["date"],
                "trade_date": cur["date"],
                "direction": "STRADDLE",
                "anchor": anchor,
                "exit_price_2pm": exit_px,
                "cost": cost_a,
                "payout": abs_pay,
                "pnl": pnl_a,
                "return_pct": (pnl_a / cost_a) * 100.0 if cost_a > 0 else np.nan,
                "win": int(next_day_range > 2.0 * cost_a),
            }
        )

        # B/C: eod gamma extreme negative
        if prev["eod_gamma"] < -5e9:
            pnl_b = call_pay - base
            out.append(
                {
                    "setup": "Swing B",
                    "signal_date": prev["date"],
                    "trade_date": cur["date"],
                    "direction": "CALL",
                    "anchor": anchor,
                    "exit_price_2pm": exit_px,
                    "cost": base,
                    "payout": call_pay,
                    "pnl": pnl_b,
                    "return_pct": (pnl_b / base) * 100.0 if base > 0 else np.nan,
                    "win": int(pnl_b > 0),
                }
            )
            pnl_c = put_pay - base
            out.append(
                {
                    "setup": "Swing C",
                    "signal_date": prev["date"],
                    "trade_date": cur["date"],
                    "direction": "PUT",
                    "anchor": anchor,
                    "exit_price_2pm": exit_px,
                    "cost": base,
                    "payout": put_pay,
                    "pnl": pnl_c,
                    "return_pct": (pnl_c / base) * 100.0 if base > 0 else np.nan,
                    "win": int(pnl_c > 0),
                }
            )

        # D: prior day compressed
        if (not pd.isna(prev["em_pts"])) and prev["em_pts"] > 0 and prev["day_range"] < (0.5 * prev["em_pts"]):
            cost_d = 2.0 * base
            pnl_d = abs_pay - cost_d
            out.append(
                {
                    "setup": "Swing D",
                    "signal_date": prev["date"],
                    "trade_date": cur["date"],
                    "direction": "STRADDLE",
                    "anchor": anchor,
                    "exit_price_2pm": exit_px,
                    "cost": cost_d,
                    "payout": abs_pay,
                    "pnl": pnl_d,
                    "return_pct": (pnl_d / cost_d) * 100.0 if cost_d > 0 else np.nan,
                    "win": int(pnl_d > 0),
                }
            )

        # E: prior day direction continuation
        if not pd.isna(prev["open_price"]) and not pd.isna(prev["close_price"]) and prev["close_price"] != prev["open_price"]:
            if prev["close_price"] > prev["open_price"]:
                payout = call_pay
                direction = "CALL"
            else:
                payout = put_pay
                direction = "PUT"
            pnl_e = payout - base
            out.append(
                {
                    "setup": "Swing E",
                    "signal_date": prev["date"],
                    "trade_date": cur["date"],
                    "direction": direction,
                    "anchor": anchor,
                    "exit_price_2pm": exit_px,
                    "cost": base,
                    "payout": payout,
                    "pnl": pnl_e,
                    "return_pct": (pnl_e / base) * 100.0 if base > 0 else np.nan,
                    "win": int(pnl_e > 0),
                }
            )

    return pd.DataFrame(out)


def gamma_regime(g: float) -> str:
    if g < -2e9:
        return "neg"
    if g > 2e9:
        return "pos"
    return "neutral"


def tod_bucket(t_pt: float) -> str:
    et = t_pt + 180.0
    if et < 660:
        return "morning"
    if et < 840:
        return "midday"
    return "afternoon"


def option_cost_intraday(t_pt: float) -> float:
    et = t_pt + 180.0
    if et < 660:
        return 5.0
    if et < 840:
        return 4.0
    return 3.0

def analysis_4_momentum(daily: pd.DataFrame, snap_map: Dict[str, List[dict]]) -> pd.DataFrame:
    by_date = {r["date"]: r for _, r in daily.iterrows()}
    dates = sorted(snap_map.keys())
    out = []

    for idx, date_str in enumerate(dates):
        if idx == 0:
            continue
        day = snap_map[date_str]
        if len(day) < 10:
            continue
        prev_date = dates[idx - 1]
        prev = by_date.get(prev_date)
        if prev is None:
            continue
        prior_hi = float(prev["day_high"])
        prior_lo = float(prev["day_low"])
        open_px = float(day[0]["price"])

        armed = True
        for j in range(1, len(day)):
            cur = day[j]
            t = cur["t_pt"]
            if t > et_to_pt_min(15, 30):
                break

            t_back = t - 15.0
            i_back = None
            for k in range(j - 1, -1, -1):
                if day[k]["t_pt"] <= t_back:
                    i_back = k
                    break
            if i_back is None:
                continue

            delta = float(cur["price"] - day[i_back]["price"])
            cond = abs(delta) >= 15.0
            if not cond:
                armed = True
                continue
            if not armed:
                continue
            armed = False

            direction = 1 if delta > 0 else -1
            entry_px = float(cur["price"])
            entry_t = float(cur["t_pt"])
            g = float(cur["gamma"])

            # continuation/reversal measurements
            fut = day[j + 1 :]
            fut30 = [r for r in fut if r["t_pt"] <= entry_t + 30.0]
            fut60 = [r for r in fut if r["t_pt"] <= entry_t + 60.0]
            fut_close = fut
            if not fut_close:
                continue
            dir_move30 = max([direction * (r["price"] - entry_px) for r in fut30], default=0.0)
            dir_move60 = max([direction * (r["price"] - entry_px) for r in fut60], default=0.0)
            dir_move_close = max([direction * (r["price"] - entry_px) for r in fut_close], default=0.0)
            adverse60 = max([-direction * (r["price"] - entry_px) for r in fut60], default=0.0)

            # trade sim
            t_limit = min(entry_t + 60.0, et_to_pt_min(15, 30))
            sim = [r for r in fut if r["t_pt"] <= t_limit]
            if not sim:
                continue
            stop_px = entry_px - 8.0 * direction
            best_ext = entry_px
            last_new_t = entry_t
            exit_row = sim[-1]
            exit_reason = "time"
            for r in sim:
                px = float(r["price"])
                tt = float(r["t_pt"])
                # stop
                if (direction == 1 and px <= stop_px) or (direction == -1 and px >= stop_px):
                    exit_row = r
                    exit_reason = "stop"
                    break
                # new extreme in trade direction
                if (direction == 1 and px > best_ext) or (direction == -1 and px < best_ext):
                    best_ext = px
                    last_new_t = tt
                # exhaustion: no new extreme for 10 minutes
                if tt - last_new_t >= 10.0:
                    exit_row = r
                    exit_reason = "exhaust"
                    break

            exit_px = float(exit_row["price"])
            cost = option_cost_intraday(entry_t)
            if exit_reason == "stop":
                exit_val = 0.5 * cost
            else:
                move = direction * (exit_px - entry_px)
                exit_val = max(0.2, cost + 2.5 * move)
            pnl = exit_val - cost
            ret = (pnl / cost) * 100.0 if cost > 0 else np.nan

            # context splits
            open_to_trigger = entry_px - open_px
            with_trend = int((open_to_trigger == 0) or (math.copysign(1, open_to_trigger) == direction))
            so_far = day[: j + 1]
            hi_so_far = max(r["price"] for r in so_far)
            lo_so_far = min(r["price"] for r in so_far)
            swept_prior = int((hi_so_far >= prior_hi) or (lo_so_far <= prior_lo))

            out.append(
                {
                    "date": date_str,
                    "trigger_time_pt": cur["time_pt"],
                    "trigger_time_et": f"{int((entry_t+180)//60):02d}:{int((entry_t+180)%60):02d}",
                    "direction": "UP" if direction == 1 else "DOWN",
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "exit_reason": exit_reason,
                    "net_gamma": g,
                    "gamma_regime": gamma_regime(g),
                    "time_bucket": tod_bucket(entry_t),
                    "with_trend": with_trend,
                    "swept_prior_level_before_trigger": swept_prior,
                    "continue_15": int(dir_move60 >= 15.0),
                    "continue_30": int(dir_move60 >= 30.0),
                    "reverse_8": int(adverse60 >= 8.0),
                    "next30_dir_move": float(dir_move30),
                    "next60_dir_move": float(dir_move60),
                    "to_close_dir_move": float(dir_move_close),
                    "cost": float(cost),
                    "exit_value": float(exit_val),
                    "pnl": float(pnl),
                    "return_pct": float(ret),
                    "moonshot_3_to_50": int((cost <= 5.0) and (exit_val >= 50.0)),
                    "loser_3_to_1": int((cost <= 3.5) and (exit_val <= 1.0)),
                    "win": int(pnl > 0),
                    "setup": "Momentum 15pt",
                }
            )

    return pd.DataFrame(out)


def analysis_5_last30(daily: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in daily.iterrows():
        p330 = float(r["price_330pm"])
        p230 = float(r["price_230pm"])
        close = float(r["close_price"])
        trend = p330 - p230
        if trend == 0:
            continue
        direction = 1 if trend > 0 else -1
        move = direction * (close - p330)
        cont = int(move > 0)
        em = float(r["em_pts"]) if not pd.isna(r["em_pts"]) else np.nan
        anchor = float(r["anchor"]) if not pd.isna(r["anchor"]) else np.nan
        em_pos = ((p330 - anchor) / em) if (not np.isnan(em) and em > 0) else np.nan
        g = float(r["gamma_330"])
        cost = min(2.0, max(0.5, 1.0 + (0.4 * abs(em_pos) if not np.isnan(em_pos) else 0.0)))
        exit_val = max(0.1, cost + 4.0 * move)
        pnl = exit_val - cost
        ret = (pnl / cost) * 100.0 if cost > 0 else np.nan
        out.append(
            {
                "date": r["date"],
                "direction": "UP" if direction == 1 else "DOWN",
                "price_230": p230,
                "entry_330": p330,
                "close": close,
                "last30_underlying_move": move,
                "continuation": cont,
                "net_gamma_330": g,
                "gamma_regime": gamma_regime(g),
                "price_vs_call_wall": p330 - float(r["call_wall_330"]),
                "price_vs_anchor": (p330 - anchor) if not np.isnan(anchor) else np.nan,
                "em_position_330": em_pos,
                "cost": cost,
                "exit_value": exit_val,
                "pnl": pnl,
                "return_pct": ret,
                "win": int(pnl > 0),
                "setup": "Last 30min",
            }
        )
    return pd.DataFrame(out)


def summarize_setup(df: pd.DataFrame, setup_name: str) -> dict:
    s = df[df["setup"] == setup_name] if "setup" in df.columns else df
    if s.empty:
        return {
            "trades": 0,
            "win_pct": None,
            "avg_return_pct": None,
            "total_pnl_contract": 0.0,
            "best_trade": None,
            "worst_trade": None,
            "sharpe_like": None,
        }
    pnl = s["pnl"].astype(float)
    sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else None
    best_i = int(pnl.idxmax())
    worst_i = int(pnl.idxmin())
    return {
        "trades": int(len(s)),
        "win_pct": float(s["win"].mean() * 100.0),
        "avg_return_pct": float(s["return_pct"].mean()),
        "total_pnl_contract": float(pnl.sum()),
        "best_trade": {
            "date": str(s.loc[best_i, "trade_date"] if "trade_date" in s.columns else s.loc[best_i, "date"]),
            "pnl": float(s.loc[best_i, "pnl"]),
        },
        "worst_trade": {
            "date": str(s.loc[worst_i, "trade_date"] if "trade_date" in s.columns else s.loc[worst_i, "date"]),
            "pnl": float(s.loc[worst_i, "pnl"]),
        },
        "sharpe_like": sharpe,
    }


def build_master_summary(
    a1: dict,
    a2: dict,
    swing: pd.DataFrame,
    momentum: pd.DataFrame,
    last30: pd.DataFrame,
) -> dict:
    summary_table = {
        "Swing A": summarize_setup(swing, "Swing A"),
        "Swing B": summarize_setup(swing, "Swing B"),
        "Swing C": summarize_setup(swing, "Swing C"),
        "Swing D": summarize_setup(swing, "Swing D"),
        "Swing E": summarize_setup(swing, "Swing E"),
        "Momentum 15pt": summarize_setup(momentum, "Momentum 15pt"),
        "Last 30min": summarize_setup(last30, "Last 30min"),
    }

    # risk-adjusted winner
    sharpe_items = [
        (k, v.get("sharpe_like"))
        for k, v in summary_table.items()
        if v.get("sharpe_like") is not None
    ]
    sharpe_items = [x for x in sharpe_items if not np.isnan(x[1])]
    best_risk_adj = max(sharpe_items, key=lambda x: x[1])[0] if sharpe_items else None

    moonshot_count = int(momentum["moonshot_3_to_50"].sum()) if not momentum.empty else 0
    loser_3to1_count = int(momentum["loser_3_to_1"].sum()) if not momentum.empty else 0

    place_walk = None
    # Use swing setups only for "place and walk away".
    swing_sharpes = [
        (k, summary_table[k]["sharpe_like"])
        for k in ["Swing A", "Swing B", "Swing C", "Swing D", "Swing E"]
        if summary_table[k]["sharpe_like"] is not None
    ]
    swing_sharpes = [x for x in swing_sharpes if not np.isnan(x[1])]
    if swing_sharpes:
        place_walk = max(swing_sharpes, key=lambda x: x[1])[0]

    # Momentum split stats
    momentum_split = {}
    if not momentum.empty:
        for col in ["gamma_regime", "time_bucket", "with_trend", "swept_prior_level_before_trigger"]:
            momentum_split[col] = (
                momentum.groupby(col)
                .agg(
                    trades=("pnl", "size"),
                    win_pct=("win", lambda x: float(np.mean(x) * 100.0)),
                    avg_pnl=("pnl", "mean"),
                    avg_return_pct=("return_pct", "mean"),
                    cont15=("continue_15", "mean"),
                    cont30=("continue_30", "mean"),
                )
                .reset_index()
                .to_dict(orient="records")
            )

    # Last 30 split by gamma
    last30_split = []
    if not last30.empty:
        last30_split = (
            last30.groupby("gamma_regime")
            .agg(
                trades=("pnl", "size"),
                continuation_rate=("continuation", "mean"),
                avg_last30_move=("last30_underlying_move", "mean"),
                avg_pnl=("pnl", "mean"),
                win_pct=("win", lambda x: float(np.mean(x) * 100.0)),
            )
            .reset_index()
            .to_dict(orient="records")
        )

    return {
        "analysis_1": a1,
        "analysis_2": a2,
        "summary_table": summary_table,
        "momentum": {
            "moonshot_3_to_50_count": moonshot_count,
            "loser_3_to_1_count": loser_3to1_count,
            "split_stats": momentum_split,
        },
        "last_30min": {
            "overall_continuation_rate": float(last30["continuation"].mean()) if not last30.empty else None,
            "avg_last30_move": float(last30["last30_underlying_move"].mean()) if not last30.empty else None,
            "split_by_gamma": last30_split,
        },
        "best_risk_adjusted_setup": best_risk_adj,
        "best_place_and_walk_setup": place_walk,
        "notes": {
            "option_pricing_model": "Synthetic heuristic model based on underlying move and time-to-close; not real option marks.",
            "data_scope": "RTH snapshots only from gex_snapshots + VIX from yfinance.",
        },
    }


def print_compact(summary: dict) -> None:
    a1 = summary["analysis_1"]
    print("\n" + "=" * 72)
    print("ANALYSIS 1 QUICK STATS")
    print("=" * 72)
    print(a1["summary_line"])

    print("\nPredictors (hit rates for BIG day > 50 pts):")
    nb = summary["analysis_2"]["night_before_predictors"]
    am = summary["analysis_2"]["ten_am_predictors"]
    for k, v in nb.items():
        hr = None if v["hit_rate"] is None else v["hit_rate"] * 100.0
        print(f"  Night {k}: n={v['instances']}, hit={hr:.1f}%" if hr is not None else f"  Night {k}: n=0")
    for k, v in am.items():
        hr = None if v["hit_rate"] is None else v["hit_rate"] * 100.0
        print(f"  10am  {k}: n={v['instances']}, hit={hr:.1f}%" if hr is not None else f"  10am  {k}: n=0")

    print("\nSUMMARY TABLE")
    print("SETUP         | TRADES | WIN% | AVG RETURN% | TOTAL PNL/CONTRACT")
    order = ["Swing A", "Swing B", "Swing C", "Swing D", "Swing E", "Momentum 15pt", "Last 30min"]
    for name in order:
        row = summary["summary_table"][name]
        if row["trades"] == 0:
            print(f"{name:<13} | {0:>6} |   n/a |       n/a   | {0:>18.2f}")
            continue
        print(
            f"{name:<13} | {row['trades']:>6} | {row['win_pct']:>4.1f}% | "
            f"{row['avg_return_pct']:>10.1f}% | {row['total_pnl_contract']:>18.2f}"
        )

    print("\nWhich setup captures $3->$50 trades?")
    print(f"  Momentum 15pt moonshots: {summary['momentum']['moonshot_3_to_50_count']}")
    print("\nBest risk-adjusted setup:")
    print(f"  {summary['best_risk_adjusted_setup']}")

    print("\nBest place-and-walk-away style:")
    print(f"  {summary['best_place_and_walk_setup']}")


# Global VIX cache keyed by date string for swing simulation.
vix_close_global: Dict[str, float] = {}


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing database: {DB_PATH}")

    snap_map = load_snapshots()
    if not snap_map:
        raise RuntimeError("No RTH snapshots found in gex_snapshots.")

    dates = sorted(snap_map.keys())
    vix_df = load_vix(dates[0], dates[-1])
    global vix_close_global
    vix_close_global = {
        str(r["date"]): float(r["Close"])
        for _, r in vix_df.iterrows()
        if not pd.isna(r["Close"])
    }

    daily = build_daily_table(snap_map, vix_df)
    daily.to_csv(REPORTS_DIR / "daily_move_distribution.csv", index=False)

    a1 = analysis_1_distribution(daily)
    a2 = analysis_2_predictors(daily)
    with open(REPORTS_DIR / "big_move_predictors.json", "w", encoding="utf-8") as f:
        json.dump({"analysis_1": a1, "analysis_2": a2}, f, indent=2)

    swing = analysis_3_swing(daily)
    swing.to_csv(REPORTS_DIR / "swing_overnight_trades.csv", index=False)

    momentum = analysis_4_momentum(daily, snap_map)
    momentum.to_csv(REPORTS_DIR / "momentum_trigger_trades.csv", index=False)

    last30 = analysis_5_last30(daily)
    last30.to_csv(REPORTS_DIR / "last_30min_trades.csv", index=False)

    summary = build_master_summary(a1, a2, swing, momentum, last30)
    with open(REPORTS_DIR / "big_move_master_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_compact(summary)
    print("\nSaved files:")
    for p in [
        "daily_move_distribution.csv",
        "big_move_predictors.json",
        "swing_overnight_trades.csv",
        "momentum_trigger_trades.csv",
        "last_30min_trades.csv",
        "big_move_master_summary.json",
    ]:
        print(f"  reports/{p}")


if __name__ == "__main__":
    main()
