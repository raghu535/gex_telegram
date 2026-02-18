"""
backtest_combined.py - Backtest GEX GAMMA_SNAP signals with SPY/VIX context overlays.
"""
import sqlite3, pandas as pd, numpy as np, yfinance as yf

DB_PATH = r"C:\Code\gex_telegram\gex_data.db"

print("=" * 80)
print("STEP 1: Downloading SPY and VIX daily data from yfinance")
print("=" * 80)

spy_df = yf.download("SPY", start="2026-01-10", end="2026-02-19", progress=False)
vix_df = yf.download("^VIX", start="2026-01-10", end="2026-02-19", progress=False)

if isinstance(spy_df.columns, pd.MultiIndex):
    spy_df.columns = spy_df.columns.get_level_values(0)
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = vix_df.columns.get_level_values(0)

print(f"  SPY rows: {len(spy_df)},  VIX rows: {len(vix_df)}")
if spy_df.empty or vix_df.empty:
    print("  WARNING: yfinance returned no data -- date range may be in the future.")

print("\nSTEP 2: Computing daily features (VIX, RSI-14, returns)")
print("-" * 80)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

daily = pd.DataFrame(index=spy_df.index)
daily["spy_close"] = spy_df["Close"]
daily["spy_return"] = spy_df["Close"].pct_change() * 100
daily["spy_5d_return"] = spy_df["Close"].pct_change(5) * 100
daily["spy_rsi14"] = compute_rsi(spy_df["Close"], 14)
daily["spy_up_day"] = (spy_df["Close"] >= spy_df["Open"]).astype(int)
daily["vix_close"] = vix_df["Close"].reindex(daily.index)
daily.index = pd.to_datetime(daily.index).strftime("%Y-%m-%d")
print(daily.to_string())
print()

print("STEP 3: Loading RTH GEX snapshots from gex_data.db")
print("-" * 80)

conn = sqlite3.connect(DB_PATH)
query = "SELECT date_pt, timestamp_pt, curr_price, net_gamma, put_floor, call_wall FROM gex_snapshots WHERE session_tag = 'RTH' ORDER BY date_pt, timestamp_pt"
gex = pd.read_sql_query(query, conn)
conn.close()
gex = gex.drop_duplicates(subset=["date_pt", "timestamp_pt"]).reset_index(drop=True)
print(f'  Loaded {len(gex)} unique RTH snapshots across {gex["date_pt"].nunique()} days')

print("\nSTEP 4: Computing gamma_bn, velocities, and 30-min lookahead")
print("-" * 80)

gex["gamma_bn"] = gex["net_gamma"] / 1e9
results = []

for date, grp in gex.groupby("date_pt"):
    grp = grp.sort_values("timestamp_pt").reset_index(drop=True)
    n = len(grp)
    ts_arr = grp["timestamp_pt"].values
    price_arr = grp["curr_price"].values
    gamma_arr = grp["gamma_bn"].values
    for i in range(n):
        row = grp.iloc[i].to_dict()
        ts_now = ts_arr[i]
        lb20_idx = np.where((ts_arr >= ts_now - 20*60) & (ts_arr <= ts_now))[0]
        if len(lb20_idx) >= 5:
            row["p_vel_20m"] = price_arr[i] - price_arr[lb20_idx[0]]
            row["g_vel_20m"] = gamma_arr[i] - gamma_arr[lb20_idx[0]]
        else:
            row["p_vel_20m"] = np.nan
            row["g_vel_20m"] = np.nan
        lb15_idx = np.where((ts_arr >= ts_now - 15*60) & (ts_arr <= ts_now))[0]
        if len(lb15_idx) >= 4:
            row["p_vel_15m"] = price_arr[i] - price_arr[lb15_idx[0]]
            row["g_vel_15m"] = gamma_arr[i] - gamma_arr[lb15_idx[0]]
        else:
            row["p_vel_15m"] = np.nan
            row["g_vel_15m"] = np.nan
        la30_idx = np.where((ts_arr > ts_now) & (ts_arr <= ts_now + 30*60))[0]
        if len(la30_idx) >= 5:
            moves = price_arr[la30_idx] - price_arr[i]
            row["max_up_30m"] = float(np.max(moves))
            row["max_down_30m"] = float(np.min(moves))
            row["close_30m"] = float(moves[-1])
        else:
            row["max_up_30m"] = np.nan
            row["max_down_30m"] = np.nan
            row["close_30m"] = np.nan
        results.append(row)

df = pd.DataFrame(results)
print(f'  Total feature rows: {len(df)}')
print(f'  Rows with valid lookahead: {df["close_30m"].notna().sum()}')
print(f'  Rows with valid 20m lookback: {df["p_vel_20m"].notna().sum()}')

print("\nSTEP 5: Tagging each snapshot with daily VIX, RSI, SPY context")
print("-" * 80)

for col in ["vix_close", "spy_rsi14", "spy_return", "spy_5d_return", "spy_up_day"]:
    if col in daily.columns:
        df[col] = df["date_pt"].map(daily[col].to_dict())
    else:
        df[col] = np.nan

matched = df["vix_close"].notna().sum()
print(f"  Snapshots with daily context: {matched} / {len(df)}")

print("\n" + "=" * 80)
print("STEP 6: SIGNAL BACKTEST RESULTS")
print("=" * 80)

valid = df.dropna(subset=["close_30m"]).copy()

signals = [
    ("T1: gamma<-5 & pvel20<-15",  "LONG",  lambda d: (d["gamma_bn"] < -5)  & (d["p_vel_20m"] < -15)),
    ("T2: gamma<-5 & pvel20<-10",  "LONG",  lambda d: (d["gamma_bn"] < -5)  & (d["p_vel_20m"] < -10)),
    ("T3: gamma<-10 & pvel20<-5",  "LONG",  lambda d: (d["gamma_bn"] < -10) & (d["p_vel_20m"] < -5)),
    ("T4: gamma<-15",              "LONG",  lambda d: (d["gamma_bn"] < -15)),
    ("S1: pvel15>=10 & gvel15<=-1","SHORT", lambda d: (d["p_vel_15m"] >= 10) & (d["g_vel_15m"] <= -1)),
]

context_filters = [
    ("Base (no filter)",           lambda d: pd.Series(True, index=d.index)),
    ("VIX > 20",                   lambda d: d["vix_close"] > 20),
    ("VIX > 25",                   lambda d: d["vix_close"] > 25),
    ("RSI < 40",                   lambda d: d["spy_rsi14"] < 40),
    ("SPY down day",               lambda d: d["spy_up_day"] == 0),
    ("VIX>20 & RSI<40",           lambda d: (d["vix_close"] > 20) & (d["spy_rsi14"] < 40)),
]

def compute_stats(subset, direction, is_base=False):
    n = len(subset)
    if n == 0:
        return {"N": 0, "wp": "  N/A", "avg_net": "    N/A", "avg_fav": "    N/A", "avg_adv": "    N/A"}
    if direction == "LONG":
        win = (subset["close_30m"] > 3).sum()
        fav = subset["max_up_30m"].mean()
        adv = subset["max_down_30m"].mean()
        net = subset["close_30m"].mean()
    else:
        win = (subset["close_30m"] < -3).sum()
        fav = -subset["max_down_30m"].mean()
        adv = -subset["max_up_30m"].mean()
        net = -subset["close_30m"].mean()
    stats = {"N": n, "wp": f"{100*win/n:5.1f}", "avg_net": f"{net:7.2f}"}
    if is_base:
        stats["avg_fav"] = f"{fav:7.2f}"
        stats["avg_adv"] = f"{adv:7.2f}"
    return stats

for sig_name, direction, sig_func in signals:
    uline = "_" * 70
    print(f"\n{uline}")
    print(f"  SIGNAL: {sig_name}  [{direction}]")
    wl = "> +3" if direction == "LONG" else "< -3"
    print(f"  Win threshold: close_30m {wl} SPX pts")
    print(f"{uline}")
    try:
        sig_mask = sig_func(valid).fillna(False)
    except Exception as e:
        print(f"  Error evaluating signal: {e}")
        continue
    sig_data = valid[sig_mask]
    for j, (ctx_name, ctx_func) in enumerate(context_filters):
        is_base = (j == 0)
        try:
            ctx_mask = ctx_func(sig_data).fillna(False)
        except Exception:
            ctx_mask = pd.Series(False, index=sig_data.index)
        subset = sig_data[ctx_mask]
        stats = compute_stats(subset, direction, is_base=is_base)
        label = chr(ord("a") + j) + ") " + ctx_name
        if is_base:
            line = f'  {label:30s}  N={stats["N"]:>5d}   win%={stats["wp"]}   avg_fav={stats["avg_fav"]}   avg_adv={stats["avg_adv"]}   avg_net={stats["avg_net"]}'
        else:
            line = f'  {label:30s}  N={stats["N"]:>5d}   win%={stats["wp"]}   avg_net={stats["avg_net"]}'
        print(line)

print("\n" + "=" * 80)
print("STEP 7: DAILY CONTEXT TABLE")
print("=" * 80)

daily_pnl = {}
for sig_name, direction, sig_func in signals:
    try:
        mask = sig_func(valid).fillna(False)
    except Exception:
        continue
    sig_rows = valid[mask]
    for date, grp in sig_rows.groupby("date_pt"):
        if date not in daily_pnl:
            daily_pnl[date] = {"n_signals": 0, "total_net": 0.0}
        if direction == "LONG":
            net = grp["close_30m"].sum()
        else:
            net = -grp["close_30m"].sum()
        daily_pnl[date]["n_signals"] += len(grp)
        daily_pnl[date]["total_net"] += net

dates_in_gex = sorted(df["date_pt"].unique())
hdr_parts = [f"{'Date':>12s}", f"{'VIX':>6s}", f"{'RSI14':>6s}", f"{'SPY Ret':>9s}"]
hdr_parts += [f"{'5d Ret':>8s}", f"{'Up/Dn':>5s}", f"{'#Sigs':>6s}", f"{'Net PnL':>8s}"]
hdr = "  ".join(hdr_parts)
print(f"\n{hdr}")
print("-" * 80)

for d in dates_in_gex:
    if d in daily.index:
        vix = daily.loc[d, "vix_close"]
        rsi = daily.loc[d, "spy_rsi14"]
        ret = daily.loc[d, "spy_return"]
        ret5 = daily.loc[d, "spy_5d_return"]
        updn = "UP" if daily.loc[d, "spy_up_day"] == 1 else "DN"
    else:
        vix = rsi = ret = ret5 = np.nan
        updn = "N/A"
    pnl_info = daily_pnl.get(d, {"n_signals": 0, "total_net": 0.0})
    vix_s = f"{vix:6.2f}" if pd.notna(vix) else "   N/A"
    rsi_s = f"{rsi:6.1f}" if pd.notna(rsi) else "   N/A"
    ret_s = f"{ret:8.2f}%" if pd.notna(ret) else "     N/A"
    ret5_s = f"{ret5:7.2f}%" if pd.notna(ret5) else "    N/A"
    nsig = pnl_info["n_signals"]
    tnet = pnl_info["total_net"]
    print(f"{d:>12s}  {vix_s}  {rsi_s}  {ret_s}  {ret5_s}  {updn:>5s}  {nsig:>6d}  {tnet:>8.1f}")

print("\n" + "=" * 80)
print("Backtest complete.")
print("=" * 80)
