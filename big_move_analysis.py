"""
Big Move Analysis — 5 analyses across all RTH data.
  1. Daily move distribution
  2. Big move predictors
  3. Swing/overnight option simulation
  4. Intraday momentum triggers
  5. Last 30 minutes EOD momentum
"""

import sqlite3, json, os, statistics, warnings
from collections import defaultdict
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

DB = "gex_data.db"
REPORTS = "reports"
os.makedirs(REPORTS, exist_ok=True)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def pt_min(t):
    """time_pt string -> minutes since midnight PT"""
    p = t.split(":")
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 60.0

def et2pt(h, m=0):
    """ET hour,min -> PT minutes since midnight"""
    return (h - 3) * 60 + m

def nearest(snaps, target_min):
    best, bd = None, 1e9
    for s in snaps:
        d = abs(s["min"] - target_min)
        if d < bd:
            bd, best = d, s
    return best

def fmt(v, w=8):
    if isinstance(v, float):
        return f"{v:>{w}.1f}"
    return f"{str(v):>{w}}"

def pct(n, d):
    return round(100 * n / d, 1) if d else 0.0

# ---------------------------------------------------------------------------
# data load
# ---------------------------------------------------------------------------
def load_data():
    print("=" * 70)
    print("  BIG MOVE ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM gex_snapshots WHERE session_tag='RTH' ORDER BY date_pt, time_pt"
    ).fetchall()
    conn.close()

    # group by date
    by_date = defaultdict(list)
    for r in rows:
        d = dict(r)
        d["min"] = pt_min(d["time_pt"])
        by_date[d["date_pt"]].append(d)

    # VIX + SPX from yfinance
    import yfinance as yf
    dates = sorted(by_date.keys())
    start = dates[0]
    end = pd.Timestamp(dates[-1]) + pd.Timedelta(days=5)
    spx = yf.download("^GSPC", start=start, end=end.strftime("%Y-%m-%d"), progress=False)
    vix = yf.download("^VIX", start=start, end=end.strftime("%Y-%m-%d"), progress=False)

    # flatten multi-level columns
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = [c[0] if isinstance(c, tuple) else c for c in spx.columns]
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]

    spx.index = pd.to_datetime(spx.index).strftime("%Y-%m-%d")
    vix.index = pd.to_datetime(vix.index).strftime("%Y-%m-%d")

    spx_dict = {}
    for dt in spx.index:
        spx_dict[dt] = {
            "open": float(spx.loc[dt, "Open"]),
            "high": float(spx.loc[dt, "High"]),
            "low": float(spx.loc[dt, "Low"]),
            "close": float(spx.loc[dt, "Close"]),
        }
    vix_dict = {}
    for dt in vix.index:
        vix_dict[dt] = float(vix.loc[dt, "Close"])

    print(f"  {len(by_date)} trading days, {sum(len(v) for v in by_date.values())} RTH snapshots")
    return by_date, spx_dict, vix_dict


# ---------------------------------------------------------------------------
# ANALYSIS 1: Daily Move Distribution
# ---------------------------------------------------------------------------
def analysis_1(by_date):
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: HOW BIG DO DAYS ACTUALLY MOVE?")
    print("=" * 70)

    rows_out = []

    for dt in sorted(by_date):
        snaps = by_date[dt]
        if len(snaps) < 10:
            continue

        prices = [s["curr_price"] for s in snaps]
        mins = [s["min"] for s in snaps]
        n = len(prices)

        day_high = max(prices)
        day_low = min(prices)
        day_range = day_high - day_low
        day_open = prices[0]
        day_close = prices[-1]

        # rolling window moves
        def max_rolling_move(window_min):
            best = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if mins[j] - mins[i] > window_min + 2:
                        break
                    if mins[j] - mins[i] <= window_min:
                        move = abs(prices[j] - prices[i])
                        if move > best:
                            best = move
            return best

        max_2hr = max_rolling_move(120)
        max_1hr = max_rolling_move(60)
        max_30m = max_rolling_move(30)

        # specific time moves
        s_11am = nearest(snaps, et2pt(11, 0))
        s_1pm = nearest(snaps, et2pt(13, 0))
        s_3pm = nearest(snaps, et2pt(15, 0))
        s_330 = nearest(snaps, et2pt(15, 30))

        morning_move = abs(day_open - s_11am["curr_price"]) if s_11am else 0
        afternoon_move = abs(s_1pm["curr_price"] - day_close) if s_1pm else 0
        last_hour_move = abs(s_3pm["curr_price"] - day_close) if s_3pm else 0
        last_30min_move = abs(s_330["curr_price"] - day_close) if s_330 else 0

        # which hour had max 1hr move
        best_hour_start = 0
        best_hour_move = 0
        for i in range(n):
            for j in range(i + 1, n):
                if mins[j] - mins[i] > 62:
                    break
                if mins[j] - mins[i] <= 60:
                    move = abs(prices[j] - prices[i])
                    if move > best_hour_move:
                        best_hour_move = move
                        best_hour_start = mins[i]

        # convert PT to ET bucket
        et_start = best_hour_start + 180  # PT -> ET in minutes
        if et_start < 630:
            bucket = "9:30-10:30"
        elif et_start < 720:
            bucket = "10:30-12"
        elif et_start < 840:
            bucket = "12-2"
        elif et_start < 900:
            bucket = "2-3"
        else:
            bucket = "3-4"

        rows_out.append({
            "date": dt,
            "day_range": round(day_range, 1),
            "max_2hr_move": round(max_2hr, 1),
            "max_1hr_move": round(max_1hr, 1),
            "max_30min_move": round(max_30m, 1),
            "morning_move": round(morning_move, 1),
            "afternoon_move": round(afternoon_move, 1),
            "last_hour_move": round(last_hour_move, 1),
            "last_30min_move": round(last_30min_move, 1),
            "max_1hr_bucket": bucket,
        })

    df = pd.DataFrame(rows_out)

    # thresholds
    th = [
        ("day_range > 100", (df["day_range"] > 100).sum()),
        ("day_range > 50", (df["day_range"] > 50).sum()),
        ("day_range > 30", (df["day_range"] > 30).sum()),
        ("max_2hr > 50", (df["max_2hr_move"] > 50).sum()),
        ("max_2hr > 30", (df["max_2hr_move"] > 30).sum()),
        ("max_1hr > 20", (df["max_1hr_move"] > 20).sum()),
        ("last_30min > 15", (df["last_30min_move"] > 15).sum()),
    ]
    n_days = len(df)
    print(f"\n  Distribution ({n_days} days):")
    for label, cnt in th:
        print(f"    {label:25s}: {cnt:3d} days ({pct(cnt, n_days):5.1f}%)")

    # bucket distribution
    print(f"\n  When does the biggest 1hr move happen?")
    buckets = df["max_1hr_bucket"].value_counts()
    for b in ["9:30-10:30", "10:30-12", "12-2", "2-3", "3-4"]:
        cnt = buckets.get(b, 0)
        print(f"    {b:12s}: {cnt:3d} days ({pct(cnt, n_days):5.1f}%)")

    # averages
    print(f"\n  Averages:")
    for col in ["day_range", "max_2hr_move", "max_1hr_move", "max_30min_move",
                 "morning_move", "afternoon_move", "last_hour_move", "last_30min_move"]:
        print(f"    {col:20s}: {df[col].mean():6.1f} pts  (median {df[col].median():5.1f})")

    d50 = pct((df["day_range"] > 50).sum(), n_days)
    d30_2h = pct((df["max_2hr_move"] > 30).sum(), n_days)
    d20_lh = pct((df["last_hour_move"] > 20).sum(), n_days)
    print(f"\n  On {d50}% of days there is a 50+ pt day range.")
    print(f"  On {d30_2h}% there is a 30+ pt 2-hour move.")
    print(f"  On {d20_lh}% the last hour moves 20+ pts.")

    df.to_csv(os.path.join(REPORTS, "daily_move_distribution.csv"), index=False)
    print(f"  Saved daily_move_distribution.csv")

    return df


# ---------------------------------------------------------------------------
# ANALYSIS 2: What Predicts Big Moves?
# ---------------------------------------------------------------------------
def analysis_2(by_date, spx_dict, vix_dict, move_df):
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: WHAT PREDICTS BIG MOVES?")
    print("=" * 70)

    dates = sorted(by_date.keys())
    day_features = {}

    for i, dt in enumerate(dates):
        snaps = by_date[dt]
        if len(snaps) < 10:
            continue

        prices = [s["curr_price"] for s in snaps]
        gammas = [s["net_gamma"] for s in snaps]

        open_gamma = gammas[0] / 1e9
        day_close = prices[-1]
        day_open = prices[0]
        day_range = max(prices) - min(prices)

        vix_val = vix_dict.get(dt, 18.0)
        anchor = spx_dict[dt]["close"] if dt in spx_dict else day_close

        # call_wall - put_floor spread
        open_spread = (snaps[0]["call_wall"] - snaps[0]["put_floor"]) if snaps[0]["call_wall"] and snaps[0]["put_floor"] else 0

        # overnight gap
        prior_close = None
        prior_range = None
        prior_eod_gamma = None
        prior_em_pos = None
        prior_compressed = False
        if i > 0:
            prev_dt = dates[i - 1]
            prev_snaps = by_date[prev_dt]
            prior_close = prev_snaps[-1]["curr_price"]
            prev_prices = [s["curr_price"] for s in prev_snaps]
            prior_range = max(prev_prices) - min(prev_prices)
            prior_eod_gamma = prev_snaps[-1]["net_gamma"] / 1e9

            # prior day EM
            prev_vix = vix_dict.get(prev_dt, 18.0)
            prev_anchor = spx_dict.get(prev_dt, {}).get("close", prior_close)
            prev_em = (prev_vix / 16) / 100 * prev_anchor
            if prev_em > 0:
                prior_em_pos = (prev_snaps[-1]["curr_price"] - prev_anchor) / prev_em
                prior_compressed = prior_range < 0.5 * prev_em

        gap = (day_open - prior_close) if prior_close else 0

        # first 30 min
        s_10am = nearest(snaps, et2pt(10, 0))
        first_30 = [s for s in snaps if s["min"] <= et2pt(10, 0)]
        if first_30:
            f30_high = max(s["curr_price"] for s in first_30)
            f30_low = min(s["curr_price"] for s in first_30)
            f30_range = f30_high - f30_low
        else:
            f30_range = 0
        em_pts = (vix_val / 16) / 100 * anchor if anchor else 0
        first_30_ratio = f30_range / em_pts if em_pts > 0 else 0

        # gamma at 10am
        gamma_10am = s_10am["net_gamma"] / 1e9 if s_10am else open_gamma

        # did price sweep prior day high or low in first 30?
        swept_prior = False
        if i > 0:
            prev_snaps = by_date[dates[i - 1]]
            prev_prices = [s["curr_price"] for s in prev_snaps]
            prev_high = max(prev_prices)
            prev_low = min(prev_prices)
            for s in first_30:
                if s["curr_price"] > prev_high or s["curr_price"] < prev_low:
                    swept_prior = True
                    break

        day_features[dt] = {
            "date": dt,
            "day_range": day_range,
            "open_gamma": open_gamma,
            "vix": vix_val,
            "open_spread": open_spread,
            "gap": gap,
            "prior_range": prior_range,
            "prior_eod_gamma": prior_eod_gamma,
            "prior_em_pos": prior_em_pos,
            "prior_compressed": prior_compressed,
            "first_30_ratio": first_30_ratio,
            "gamma_10am": gamma_10am,
            "swept_prior": swept_prior,
        }

    feat_df = pd.DataFrame(day_features.values())

    # classify
    feat_df["size"] = feat_df["day_range"].apply(
        lambda x: "BIG" if x > 50 else ("MEDIUM" if x > 30 else "SMALL")
    )

    print(f"\n  Day size distribution:")
    for sz in ["BIG", "MEDIUM", "SMALL"]:
        cnt = (feat_df["size"] == sz).sum()
        print(f"    {sz:8s}: {cnt:3d} days ({pct(cnt, len(feat_df)):5.1f}%)")

    # morning state by group
    print(f"\n  Morning state by group:")
    print(f"  {'Group':8s} {'OpenGamma':>10s} {'VIX':>8s} {'Spread':>8s} {'Gap':>8s} {'PriorRng':>10s} {'PrEODGam':>10s}")
    for sz in ["BIG", "MEDIUM", "SMALL"]:
        g = feat_df[feat_df["size"] == sz]
        print(f"  {sz:8s} {g['open_gamma'].mean():10.1f} {g['vix'].mean():8.1f} {g['open_spread'].mean():8.0f} "
              f"{g['gap'].mean():8.1f} {g['prior_range'].mean():10.1f} {g['prior_eod_gamma'].mean():10.1f}")

    # prior day EOD state
    print(f"\n  Prior day EOD state:")
    print(f"  {'Group':8s} {'PrEODGam':>10s} {'PrRange':>10s} {'PrEMPos':>10s} {'PrCompress%':>12s}")
    for sz in ["BIG", "MEDIUM", "SMALL"]:
        g = feat_df[feat_df["size"] == sz]
        comp_pct = pct(g["prior_compressed"].sum(), len(g))
        pr_em = g["prior_em_pos"].dropna().mean() if len(g["prior_em_pos"].dropna()) > 0 else 0
        print(f"  {sz:8s} {g['prior_eod_gamma'].mean():10.1f} {g['prior_range'].mean():10.1f} "
              f"{pr_em:10.2f} {comp_pct:12.1f}%")

    # predictors
    print(f"\n  Predictor hit rates for BIG days (range > 50 pts):")
    n_big = (feat_df["size"] == "BIG").sum()

    predictors = []

    def test_pred(name, mask):
        triggered = mask.sum()
        if triggered == 0:
            print(f"    {name:45s}: 0 triggered")
            predictors.append({"predictor": name, "triggered": 0, "big_hits": 0, "hit_rate": 0, "precision": 0})
            return
        hits = (mask & (feat_df["size"] == "BIG")).sum()
        hr = pct(hits, triggered)
        recall = pct(hits, n_big) if n_big > 0 else 0
        print(f"    {name:45s}: {triggered:3d} triggered, {hits:2d} big -> precision {hr:5.1f}%, recall {recall:5.1f}%")
        predictors.append({"predictor": name, "triggered": int(triggered), "big_hits": int(hits),
                           "hit_rate": hr, "precision": hr, "recall": recall})

    test_pred("prior eod_gamma < -5Bn",
              feat_df["prior_eod_gamma"].fillna(0) < -5)
    test_pred("prior range < 0.5 EM (compressed)",
              feat_df["prior_compressed"])
    test_pred("VIX > 20",
              feat_df["vix"] > 20)
    test_pred("eod_gamma < -5Bn AND compressed",
              (feat_df["prior_eod_gamma"].fillna(0) < -5) & feat_df["prior_compressed"])
    test_pred("first_30_ratio > 0.3 (at 10am)",
              feat_df["first_30_ratio"] > 0.3)
    test_pred("first_30_ratio > 0.3 AND gamma < -3Bn",
              (feat_df["first_30_ratio"] > 0.3) & (feat_df["gamma_10am"] < -3))
    test_pred("swept prior day high/low in first 30",
              feat_df["swept_prior"])

    pred_json = {"day_counts": {"BIG": int(n_big),
                                "MEDIUM": int((feat_df["size"] == "MEDIUM").sum()),
                                "SMALL": int((feat_df["size"] == "SMALL").sum())},
                 "predictors": predictors}
    with open(os.path.join(REPORTS, "big_move_predictors.json"), "w") as f:
        json.dump(pred_json, f, indent=2)
    print(f"  Saved big_move_predictors.json")

    return feat_df


# ---------------------------------------------------------------------------
# ANALYSIS 3: Swing / Overnight Positioning
# ---------------------------------------------------------------------------
def analysis_3(by_date, spx_dict, vix_dict):
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: SWING / OVERNIGHT POSITIONING")
    print("=" * 70)

    dates = sorted(by_date.keys())
    all_trades = []

    for i in range(1, len(dates)):
        prev_dt = dates[i - 1]
        dt = dates[i]

        prev_snaps = by_date[prev_dt]
        snaps = by_date[dt]
        if len(snaps) < 10 or len(prev_snaps) < 10:
            continue

        prev_prices = [s["curr_price"] for s in prev_snaps]
        prev_close = prev_prices[-1]
        prev_open = prev_prices[0]
        prev_range = max(prev_prices) - min(prev_prices)
        prev_eod_gamma = prev_snaps[-1]["net_gamma"] / 1e9

        vix_val = vix_dict.get(prev_dt, 18.0)
        anchor = prev_close
        em_pts = (vix_val / 16) / 100 * anchor

        # option cost estimate
        atm_cost = (vix_val / 16) / 100 * anchor * 0.4
        straddle_cost = 2 * atm_cost

        # next day price at 2pm ET = 11:00 PT
        s_2pm = nearest(snaps, et2pt(14, 0))
        if not s_2pm:
            continue
        price_2pm = s_2pm["curr_price"]
        day_close = snaps[-1]["curr_price"]

        day_prices = [s["curr_price"] for s in snaps]
        day_high = max(day_prices)
        day_low = min(day_prices)
        day_range = day_high - day_low

        prior_compressed = prev_range < 0.5 * em_pts

        trade_row = {
            "date": dt,
            "prev_date": prev_dt,
            "anchor": round(anchor, 2),
            "vix": round(vix_val, 1),
            "atm_cost": round(atm_cost, 2),
            "straddle_cost": round(straddle_cost, 2),
            "price_2pm": round(price_2pm, 2),
            "day_close": round(day_close, 2),
            "day_range": round(day_range, 1),
            "prev_eod_gamma": round(prev_eod_gamma, 1),
            "prev_range": round(prev_range, 1),
            "prev_compressed": prior_compressed,
            "prev_direction": "UP" if prev_close > prev_open else "DOWN",
        }

        move_up = max(0, price_2pm - anchor)
        move_down = max(0, anchor - price_2pm)

        # SWING A: straddle every night
        call_val = max(0, price_2pm - anchor)
        put_val = max(0, anchor - price_2pm)
        straddle_pnl = max(call_val, put_val) - straddle_cost
        trade_row["swing_a_pnl"] = round(straddle_pnl, 2)
        trade_row["swing_a_win"] = 1 if straddle_pnl > 0 else 0

        # SWING B: buy call if eod_gamma < -5Bn (mean reversion bounce)
        if prev_eod_gamma < -5:
            pnl = move_up - atm_cost
            trade_row["swing_b_pnl"] = round(pnl, 2)
            trade_row["swing_b_win"] = 1 if pnl > 0 else 0
        else:
            trade_row["swing_b_pnl"] = None
            trade_row["swing_b_win"] = None

        # SWING C: buy put if eod_gamma < -5Bn (continuation selling)
        if prev_eod_gamma < -5:
            pnl = move_down - atm_cost
            trade_row["swing_c_pnl"] = round(pnl, 2)
            trade_row["swing_c_win"] = 1 if pnl > 0 else 0
        else:
            trade_row["swing_c_pnl"] = None
            trade_row["swing_c_win"] = None

        # SWING D: straddle if prior compressed
        if prior_compressed:
            straddle_pnl_d = max(call_val, put_val) - straddle_cost
            trade_row["swing_d_pnl"] = round(straddle_pnl_d, 2)
            trade_row["swing_d_win"] = 1 if straddle_pnl_d > 0 else 0
        else:
            trade_row["swing_d_pnl"] = None
            trade_row["swing_d_win"] = None

        # SWING E: direction of prior day close vs open
        if prev_close < prev_open:
            # bearish -> buy put
            pnl = move_down - atm_cost
        else:
            # bullish -> buy call
            pnl = move_up - atm_cost
        trade_row["swing_e_pnl"] = round(pnl, 2)
        trade_row["swing_e_win"] = 1 if pnl > 0 else 0

        all_trades.append(trade_row)

    df = pd.DataFrame(all_trades)

    print(f"\n  {'SETUP':12s} {'TRADES':>7s} {'WIN%':>7s} {'AVG_RET%':>9s} {'TOT_PNL':>9s} {'BEST':>9s} {'WORST':>9s}")
    print("  " + "-" * 65)

    summary_setups = []
    for label, col in [("Swing A", "swing_a"), ("Swing B", "swing_b"),
                       ("Swing C", "swing_c"), ("Swing D", "swing_d"),
                       ("Swing E", "swing_e")]:
        pnl_col = f"{col}_pnl"
        win_col = f"{col}_win"
        valid = df[df[pnl_col].notna()]
        if len(valid) == 0:
            print(f"  {label:12s} {'0':>7s}")
            summary_setups.append({"setup": label, "trades": 0})
            continue
        trades = len(valid)
        wins = int(valid[win_col].sum())
        wr = pct(wins, trades)
        avg_pnl = valid[pnl_col].mean()
        total = valid[pnl_col].sum()
        best = valid[pnl_col].max()
        worst = valid[pnl_col].min()
        # avg return % relative to cost
        cost_col = "straddle_cost" if label == "Swing A" or label == "Swing D" else "atm_cost"
        avg_ret = (valid[pnl_col] / valid[cost_col] * 100).mean()
        print(f"  {label:12s} {trades:7d} {wr:6.1f}% {avg_ret:8.1f}% {total:9.1f} {best:9.1f} {worst:9.1f}")
        summary_setups.append({
            "setup": label, "trades": trades, "win_rate": wr,
            "avg_return_pct": round(avg_ret, 1), "total_pnl": round(total, 1),
            "best": round(best, 1), "worst": round(worst, 1),
        })

    df.to_csv(os.path.join(REPORTS, "swing_overnight_trades.csv"), index=False)
    print(f"\n  Saved swing_overnight_trades.csv")

    return summary_setups


# ---------------------------------------------------------------------------
# ANALYSIS 4: Intraday Momentum Triggers
# ---------------------------------------------------------------------------
def analysis_4(by_date, vix_dict, spx_dict):
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: INTRADAY MOMENTUM TRIGGERS")
    print("=" * 70)

    TRIGGER_PTS = 15
    LOOKBACK_MIN = 15
    STOP_PTS = 8
    TIME_EXIT_MIN = 60
    COOLDOWN_MIN = 30

    dates = sorted(by_date.keys())
    triggers = []

    for dt in dates:
        snaps = by_date[dt]
        if len(snaps) < 20:
            continue

        prices = [s["curr_price"] for s in snaps]
        mins_arr = [s["min"] for s in snaps]
        n = len(snaps)

        # prior day high/low for sweep check
        idx = dates.index(dt)
        prev_high, prev_low = None, None
        if idx > 0:
            prev_snaps = by_date[dates[idx - 1]]
            prev_prices = [s["curr_price"] for s in prev_snaps]
            prev_high = max(prev_prices)
            prev_low = min(prev_prices)

        last_trigger_min = -999

        for i in range(1, n):
            if mins_arr[i] - last_trigger_min < COOLDOWN_MIN:
                continue
            # no triggers after 3:30 PM ET = 12:30 PT
            if mins_arr[i] > et2pt(15, 30):
                break

            # look back ~15 min
            lookback_start = mins_arr[i] - LOOKBACK_MIN
            for j in range(max(0, i - 20), i):
                if mins_arr[j] >= lookback_start:
                    move = prices[i] - prices[j]
                    if abs(move) >= TRIGGER_PTS:
                        direction = "UP" if move > 0 else "DOWN"
                        trigger_price = prices[i]
                        trigger_min = mins_arr[i]
                        trigger_gamma = snaps[i]["net_gamma"] / 1e9
                        last_trigger_min = trigger_min

                        # open-to-trigger direction
                        trend_dir = "UP" if trigger_price > prices[0] else "DOWN"
                        with_trend = (direction == trend_dir)

                        # sweep check
                        swept = False
                        if prev_high and prev_low:
                            if trigger_price > prev_high or trigger_price < prev_low:
                                swept = True

                        # measure continuation
                        future_prices = []
                        for k in range(i + 1, n):
                            future_prices.append((mins_arr[k], prices[k]))

                        if not future_prices:
                            break

                        # continuation check
                        if direction == "UP":
                            max_continue = max(p for _, p in future_prices) - trigger_price
                            max_reverse = trigger_price - min(p for _, p in future_prices)
                        else:
                            max_continue = trigger_price - min(p for _, p in future_prices)
                            max_reverse = max(p for _, p in future_prices) - trigger_price

                        continue_15 = max_continue >= 15
                        continue_30 = max_continue >= 30

                        # next 30/60 min moves
                        move_30 = 0
                        move_60 = 0
                        move_to_close = 0
                        for t_min, p in future_prices:
                            delta = t_min - trigger_min
                            if direction == "UP":
                                m = p - trigger_price
                            else:
                                m = trigger_price - p
                            if delta <= 30 and m > move_30:
                                move_30 = m
                            if delta <= 60 and m > move_60:
                                move_60 = m
                        # to close
                        close_p = prices[-1]
                        if direction == "UP":
                            move_to_close = close_p - trigger_price
                        else:
                            move_to_close = trigger_price - close_p

                        # simulate 0DTE option trade
                        # cost estimate: ~$4 for ATM 0DTE midday
                        option_cost = 4.0
                        exit_pnl = None
                        exit_reason = None
                        exit_time = None

                        for k in range(i + 1, n):
                            t_k = mins_arr[k]
                            p_k = prices[k]

                            # check stop: 8pt reversal
                            if direction == "UP":
                                reversal = trigger_price - p_k
                            else:
                                reversal = p_k - trigger_price

                            if reversal >= STOP_PTS:
                                exit_pnl = -option_cost * 0.5  # lose ~50%
                                exit_reason = "stop"
                                exit_time = t_k
                                break

                            # check time exit: 1hr or 3:30pm
                            if t_k - trigger_min >= TIME_EXIT_MIN or t_k >= et2pt(15, 30):
                                if direction == "UP":
                                    move_at_exit = p_k - trigger_price
                                else:
                                    move_at_exit = trigger_price - p_k
                                # rough option value: intrinsic only
                                exit_pnl = max(move_at_exit * 0.5, -option_cost) - option_cost * 0.3
                                exit_reason = "time_exit"
                                exit_time = t_k
                                break

                            # check exhaustion: no new high/low for ~10 min
                            if k > i + 5:  # need at least 5 snaps
                                recent = [prices[kk] for kk in range(max(i+1, k-5), k+1)]
                                if direction == "UP":
                                    curr_high = max(recent)
                                    if p_k < curr_high - 3:  # failed to extend
                                        move_at_exit = p_k - trigger_price
                                        exit_pnl = max(move_at_exit * 0.5, -option_cost) - option_cost * 0.3
                                        exit_reason = "exhaustion"
                                        exit_time = t_k
                                        break
                                else:
                                    curr_low = min(recent)
                                    if p_k > curr_low + 3:
                                        move_at_exit = trigger_price - p_k
                                        exit_pnl = max(move_at_exit * 0.5, -option_cost) - option_cost * 0.3
                                        exit_reason = "exhaustion"
                                        exit_time = t_k
                                        break

                        if exit_pnl is None:
                            # EOD exit
                            if direction == "UP":
                                move_at_exit = prices[-1] - trigger_price
                            else:
                                move_at_exit = trigger_price - prices[-1]
                            exit_pnl = max(move_at_exit * 0.5, -option_cost) - option_cost * 0.3
                            exit_reason = "eod"
                            exit_time = mins_arr[-1]

                        # time bucket
                        et_trig = trigger_min + 180
                        if et_trig < 630:
                            time_bucket = "9:30-10:30"
                        elif et_trig < 720:
                            time_bucket = "10:30-12"
                        elif et_trig < 840:
                            time_bucket = "12-2"
                        elif et_trig < 900:
                            time_bucket = "2-3"
                        else:
                            time_bucket = "3-4"

                        gamma_regime = "POS" if trigger_gamma > 3 else ("NEG" if trigger_gamma < -3 else "NEUTRAL")

                        triggers.append({
                            "date": dt,
                            "trigger_time_pt": snaps[i]["time_pt"],
                            "direction": direction,
                            "trigger_price": round(trigger_price, 2),
                            "trigger_move": round(abs(move), 1),
                            "gamma": round(trigger_gamma, 1),
                            "gamma_regime": gamma_regime,
                            "time_bucket": time_bucket,
                            "with_trend": with_trend,
                            "swept_prior": swept,
                            "continue_15": continue_15,
                            "continue_30": continue_30,
                            "max_continue": round(max_continue, 1),
                            "max_reverse": round(max_reverse, 1),
                            "move_30min": round(move_30, 1),
                            "move_60min": round(move_60, 1),
                            "move_to_close": round(move_to_close, 1),
                            "option_cost": option_cost,
                            "option_pnl": round(exit_pnl, 2),
                            "exit_reason": exit_reason,
                            "option_win": 1 if exit_pnl > 0 else 0,
                            "option_ret_pct": round(exit_pnl / option_cost * 100, 1),
                        })
                        break  # only first trigger in lookback window

        # end of date loop

    df = pd.DataFrame(triggers)
    n_trig = len(df)
    print(f"\n  Total triggers: {n_trig}")

    if n_trig == 0:
        print("  No triggers found!")
        return []

    # continuation stats
    print(f"\n  After 15pt trigger:")
    print(f"    Continues 15+ pts: {df['continue_15'].sum():3d} ({pct(df['continue_15'].sum(), n_trig):.1f}%)")
    print(f"    Continues 30+ pts: {df['continue_30'].sum():3d} ({pct(df['continue_30'].sum(), n_trig):.1f}%)")
    print(f"    Avg max continuation: {df['max_continue'].mean():.1f} pts")
    print(f"    Avg max reversal:     {df['max_reverse'].mean():.1f} pts")
    print(f"    Avg move next 30min:  {df['move_30min'].mean():.1f} pts")
    print(f"    Avg move next 60min:  {df['move_60min'].mean():.1f} pts")
    print(f"    Avg move to close:    {df['move_to_close'].mean():.1f} pts")

    # split by gamma regime
    print(f"\n  By gamma regime:")
    for regime in ["NEG", "NEUTRAL", "POS"]:
        g = df[df["gamma_regime"] == regime]
        if len(g) == 0:
            continue
        print(f"    {regime:8s}: {len(g):3d} triggers, continue_15 {pct(g['continue_15'].sum(), len(g)):5.1f}%, "
              f"avg_continue {g['max_continue'].mean():5.1f}, avg_reverse {g['max_reverse'].mean():5.1f}")

    # split by time
    print(f"\n  By time of day:")
    for bucket in ["9:30-10:30", "10:30-12", "12-2", "2-3", "3-4"]:
        g = df[df["time_bucket"] == bucket]
        if len(g) == 0:
            continue
        print(f"    {bucket:12s}: {len(g):3d} triggers, continue_15 {pct(g['continue_15'].sum(), len(g)):5.1f}%, "
              f"avg_continue {g['max_continue'].mean():5.1f}")

    # with vs counter trend
    print(f"\n  With trend vs counter-trend:")
    for label, val in [("With trend", True), ("Counter", False)]:
        g = df[df["with_trend"] == val]
        if len(g) == 0:
            continue
        print(f"    {label:12s}: {len(g):3d} triggers, continue_15 {pct(g['continue_15'].sum(), len(g)):5.1f}%, "
              f"avg_continue {g['max_continue'].mean():5.1f}")

    # swept prior
    print(f"\n  After sweep of prior day level:")
    for label, val in [("Swept", True), ("Not swept", False)]:
        g = df[df["swept_prior"] == val]
        if len(g) == 0:
            continue
        print(f"    {label:12s}: {len(g):3d} triggers, continue_15 {pct(g['continue_15'].sum(), len(g)):5.1f}%, "
              f"avg_continue {g['max_continue'].mean():5.1f}")

    # option trade results
    print(f"\n  0DTE Option Trade Simulation:")
    wins = df["option_win"].sum()
    print(f"    Trades:     {n_trig}")
    print(f"    Win rate:   {wins}/{n_trig} ({pct(wins, n_trig):.1f}%)")
    print(f"    Avg P&L:    ${df['option_pnl'].mean():.2f}")
    print(f"    Total P&L:  ${df['option_pnl'].sum():.2f}")
    print(f"    Avg return:  {df['option_ret_pct'].mean():.1f}%")
    print(f"    Best trade: ${df['option_pnl'].max():.2f} ({df['option_ret_pct'].max():.0f}%)")
    print(f"    Worst trade: ${df['option_pnl'].min():.2f}")

    # big winners ($3->$50 type)
    big_wins = df[df["option_ret_pct"] > 200]
    print(f"\n    Trades with >200% return: {len(big_wins)}")
    if len(big_wins) > 0:
        for _, r in big_wins.iterrows():
            print(f"      {r['date']} {r['direction']} at {r['trigger_time_pt']} -> {r['option_ret_pct']:.0f}% (${r['option_pnl']:.2f})")

    # exit reasons
    print(f"\n    Exit reasons:")
    for reason in df["exit_reason"].value_counts().index:
        g = df[df["exit_reason"] == reason]
        print(f"      {reason:12s}: {len(g):3d} trades, avg P&L ${g['option_pnl'].mean():.2f}")

    df.to_csv(os.path.join(REPORTS, "momentum_trigger_trades.csv"), index=False)
    print(f"\n  Saved momentum_trigger_trades.csv")

    return [{"setup": "Momentum 15pt", "trades": n_trig,
             "win_rate": round(pct(wins, n_trig), 1),
             "avg_return_pct": round(df["option_ret_pct"].mean(), 1),
             "total_pnl": round(df["option_pnl"].sum(), 1),
             "best": round(df["option_pnl"].max(), 1),
             "worst": round(df["option_pnl"].min(), 1)}]


# ---------------------------------------------------------------------------
# ANALYSIS 5: Last 30 Minutes EOD Momentum
# ---------------------------------------------------------------------------
def analysis_5(by_date, vix_dict):
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: LAST 30 MINUTES - EOD MOMENTUM")
    print("=" * 70)

    trades = []
    dates = sorted(by_date.keys())

    for dt in dates:
        snaps = by_date[dt]
        if len(snaps) < 20:
            continue

        # 3:30 PM ET = 12:30 PT
        s_330 = nearest(snaps, et2pt(15, 30))
        # 2:30 PM ET = 11:30 PT (1hr before 3:30)
        s_230 = nearest(snaps, et2pt(14, 30))
        # close
        s_close = snaps[-1]

        if not s_330 or not s_230:
            continue

        # check we have a snap near 3:30
        if abs(s_330["min"] - et2pt(15, 30)) > 5:
            continue

        price_330 = s_330["curr_price"]
        price_230 = s_230["curr_price"]
        price_close = s_close["curr_price"]

        # last 1hr trend direction
        last_1hr_dir = "UP" if price_330 > price_230 else "DOWN"
        last_1hr_move = price_330 - price_230

        # state at 3:30
        gamma_330 = s_330["net_gamma"] / 1e9
        call_wall = s_330["call_wall"]
        put_floor = s_330["put_floor"]

        vix_val = vix_dict.get(dt, 18.0)
        anchor = snaps[0]["curr_price"]  # day open as anchor
        em_pts = (vix_val / 16) / 100 * anchor
        em_pos = (price_330 - anchor) / em_pts if em_pts > 0 else 0

        # trade: buy ATM 0DTE in direction of last 1hr trend
        # option cost: very cheap, $0.50-2.00
        option_cost = 1.50  # midpoint estimate

        if last_1hr_dir == "UP":
            # buy call
            intrinsic_close = max(0, price_close - price_330)
            trade_pnl = intrinsic_close * 0.7 - option_cost  # 0.7 for delta < 1 on 0DTE
        else:
            # buy put
            intrinsic_close = max(0, price_330 - price_close)
            trade_pnl = intrinsic_close * 0.7 - option_cost

        last_30_move = price_close - price_330
        continued = (last_1hr_dir == "UP" and last_30_move > 0) or \
                    (last_1hr_dir == "DOWN" and last_30_move < 0)

        trades.append({
            "date": dt,
            "price_330": round(price_330, 2),
            "price_close": round(price_close, 2),
            "last_1hr_dir": last_1hr_dir,
            "last_1hr_move": round(last_1hr_move, 1),
            "last_30_move": round(last_30_move, 1),
            "abs_last_30": round(abs(last_30_move), 1),
            "gamma_330": round(gamma_330, 1),
            "gamma_sign": "NEG" if gamma_330 < 0 else "POS",
            "call_wall": call_wall,
            "em_pos": round(em_pos, 2),
            "continued": continued,
            "option_cost": option_cost,
            "option_pnl": round(trade_pnl, 2),
            "option_win": 1 if trade_pnl > 0 else 0,
            "option_ret_pct": round(trade_pnl / option_cost * 100, 1),
        })

    df = pd.DataFrame(trades)
    n = len(df)

    cont = df["continued"].sum()
    print(f"\n  Total days with 3:30pm data: {n}")
    print(f"  Last 30min trend continues:  {cont}/{n} ({pct(cont, n):.1f}%)")
    print(f"  Average last 30min move:     {df['abs_last_30'].mean():.1f} pts")

    # split by gamma
    print(f"\n  By gamma at 3:30pm:")
    for sign in ["NEG", "POS"]:
        g = df[df["gamma_sign"] == sign]
        if len(g) == 0:
            continue
        c = g["continued"].sum()
        print(f"    {sign:5s} gamma: {len(g):3d} days, continues {c}/{len(g)} ({pct(c, len(g)):.1f}%), "
              f"avg move {g['abs_last_30'].mean():.1f} pts")

    # trade results
    wins = df["option_win"].sum()
    print(f"\n  0DTE Trade at 3:30pm:")
    print(f"    Trades:     {n}")
    print(f"    Win rate:   {wins}/{n} ({pct(wins, n):.1f}%)")
    print(f"    Avg P&L:    ${df['option_pnl'].mean():.2f}")
    print(f"    Total P&L:  ${df['option_pnl'].sum():.2f}")
    print(f"    Avg return:  {df['option_ret_pct'].mean():.1f}%")
    print(f"    Best:       ${df['option_pnl'].max():.2f} ({df['option_ret_pct'].max():.0f}%)")
    print(f"    Worst:      ${df['option_pnl'].min():.2f}")

    # by gamma sign
    print(f"\n  Trade results by gamma:")
    for sign in ["NEG", "POS"]:
        g = df[df["gamma_sign"] == sign]
        if len(g) == 0:
            continue
        w = g["option_win"].sum()
        print(f"    {sign:5s}: {len(g):3d} trades, win {pct(w, len(g)):.1f}%, "
              f"avg P&L ${g['option_pnl'].mean():.2f}, total ${g['option_pnl'].sum():.2f}")

    df.to_csv(os.path.join(REPORTS, "last_30min_trades.csv"), index=False)
    print(f"\n  Saved last_30min_trades.csv")

    return [{"setup": "Last 30min", "trades": n,
             "win_rate": round(pct(wins, n), 1),
             "avg_return_pct": round(df["option_ret_pct"].mean(), 1),
             "total_pnl": round(df["option_pnl"].sum(), 1),
             "best": round(df["option_pnl"].max(), 1),
             "worst": round(df["option_pnl"].min(), 1)}]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    by_date, spx_dict, vix_dict = load_data()

    # Analysis 1
    move_df = analysis_1(by_date)

    # Analysis 2
    feat_df = analysis_2(by_date, spx_dict, vix_dict, move_df)

    # Analysis 3
    swing_results = analysis_3(by_date, spx_dict, vix_dict)

    # Analysis 4
    momentum_results = analysis_4(by_date, vix_dict, spx_dict)

    # Analysis 5
    eod_results = analysis_5(by_date, vix_dict)

    # Summary Table
    print("\n" + "=" * 70)
    print("  MASTER SUMMARY TABLE")
    print("=" * 70)
    all_setups = swing_results + momentum_results + eod_results
    print(f"\n  {'SETUP':16s} {'TRADES':>7s} {'WIN%':>7s} {'AVG_RET%':>9s} {'TOT_PNL':>10s}")
    print("  " + "-" * 55)
    for s in all_setups:
        if s.get("trades", 0) == 0:
            continue
        print(f"  {s['setup']:16s} {s['trades']:7d} {s.get('win_rate', 0):6.1f}% "
              f"{s.get('avg_return_pct', 0):8.1f}% {s.get('total_pnl', 0):10.1f}")

    # identify best setups
    viable = [s for s in all_setups if s.get("trades", 0) >= 5 and s.get("total_pnl", 0) > 0]
    if viable:
        best_rar = max(viable, key=lambda x: x.get("avg_return_pct", 0))
        best_pnl = max(viable, key=lambda x: x.get("total_pnl", 0))
        print(f"\n  Best risk-adjusted return: {best_rar['setup']} ({best_rar['avg_return_pct']}%)")
        print(f"  Best total P&L: {best_pnl['setup']} (${best_pnl['total_pnl']})")
    else:
        print(f"\n  No viable setups with positive P&L found.")

    # save master summary
    master = {
        "total_days": len(by_date),
        "setups": all_setups,
    }
    with open(os.path.join(REPORTS, "big_move_master_summary.json"), "w") as f:
        json.dump(master, f, indent=2, default=str)
    print(f"\n  Saved big_move_master_summary.json")
    print(f"\n  ANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()
