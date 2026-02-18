"""
Test intraday SPY data from yfinance for EMA and VWAP computation.
"""

import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

SEPARATOR = "=" * 80

# ──────────────────────────────────────────────────────────────────────────────
# 1. Pull intraday data
# ──────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("1) DOWNLOADING INTRADAY SPY DATA")
print(SEPARATOR)

print("\n>> SPY 1-min (period=1d) ...")
spy_1m = yf.download("SPY", period="1d", interval="1m")

print(f"\n>> SPY 5-min (period=5d) ...")
spy_5m = yf.download("SPY", period="5d", interval="5m")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Inspect the data
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("2) DATA INSPECTION")
print(SEPARATOR)

for label, df in [("SPY 1-min", spy_1m), ("SPY 5-min", spy_5m)]:
    print(f"\n--- {label} ---")
    if df.empty:
        print("  ** DataFrame is EMPTY (market may be closed / no data)")
        continue
    print(f"  Shape       : {df.shape}")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Index dtype : {df.index.dtype}")
    print(f"  Time range  : {df.index[0]}  -->  {df.index[-1]}")
    print(f"  Row count   : {len(df)}")
    print(f"\n  First 3 rows:\n{df.head(3)}")
    print(f"\n  Last 3 rows:\n{df.tail(3)}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Compute EMA & VWAP on 5-min data
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("3) EMA & VWAP CALCULATIONS (5-min SPY)")
print(SEPARATOR)

if spy_5m.empty:
    print("\n  ** No 5-min data available -- skipping calculations.")
else:
    # Flatten multi-level columns if yfinance returns them
    if isinstance(spy_5m.columns, pd.MultiIndex):
        spy_5m.columns = [c[0] if isinstance(c, tuple) else c for c in spy_5m.columns]

    close = spy_5m["Close"]
    high = spy_5m["High"]
    low = spy_5m["Low"]
    volume = spy_5m["Volume"]

    # EMA(9) and EMA(21)
    spy_5m["EMA9"] = close.ewm(span=9, adjust=False).mean()
    spy_5m["EMA21"] = close.ewm(span=21, adjust=False).mean()

    # Session VWAP: reset each trading day
    # Typical price = (H + L + C) / 3
    spy_5m["TypicalPrice"] = (high + low + close) / 3.0
    spy_5m["TP_x_Vol"] = spy_5m["TypicalPrice"] * volume

    # Group by trading date to reset VWAP each session
    spy_5m["Date"] = spy_5m.index.date
    spy_5m["CumTPVol"] = spy_5m.groupby("Date")["TP_x_Vol"].cumsum()
    spy_5m["CumVol"] = spy_5m.groupby("Date")["Volume"].cumsum()
    spy_5m["VWAP"] = spy_5m["CumTPVol"] / spy_5m["CumVol"]

    # Current values
    latest = spy_5m.iloc[-1]
    cur_price = latest["Close"]
    cur_vwap = latest["VWAP"]
    cur_ema9 = latest["EMA9"]
    cur_ema21 = latest["EMA21"]

    vwap_dist = cur_price - cur_vwap
    vwap_pct = (vwap_dist / cur_vwap) * 100
    vwap_side = "ABOVE" if vwap_dist > 0 else "BELOW"

    ema_state = "BULLISH (EMA9 > EMA21)" if cur_ema9 > cur_ema21 else "BEARISH (EMA9 < EMA21)"

    print(f"\n  Latest bar timestamp : {spy_5m.index[-1]}")
    print(f"  Current Close        : {cur_price:.2f}")
    print(f"  EMA(9)               : {cur_ema9:.2f}")
    print(f"  EMA(21)              : {cur_ema21:.2f}")
    print(f"  EMA crossover state  : {ema_state}")
    print(f"  Session VWAP         : {cur_vwap:.2f}")
    print(f"  Price vs VWAP        : {vwap_side} by ${abs(vwap_dist):.2f} ({abs(vwap_pct):.3f}%)")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Last 20 rows with indicators
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("5) LAST 20 ROWS -- 5-min SPY with EMA9, EMA21, VWAP")
    print(SEPARATOR)

    display_cols = ["Open", "High", "Low", "Close", "Volume", "EMA9", "EMA21", "VWAP"]
    print(spy_5m[display_cols].tail(20).to_string())

# ──────────────────────────────────────────────────────────────────────────────
# 4. Test ^GSPC (SPX) intraday
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("4) TESTING ^GSPC (SPX) 1-MIN INTRADAY")
print(SEPARATOR)

print("\n>> Downloading ^GSPC 1-min (period=1d) ...")
spx_1m = yf.download("^GSPC", period="1d", interval="1m")

if spx_1m.empty:
    print("  ** ^GSPC 1-min returned EMPTY -- SPX intraday may not be available via yfinance.")
    print("  ** Trying period=5d, interval=5m as fallback ...")
    spx_5m = yf.download("^GSPC", period="5d", interval="5m")
    if spx_5m.empty:
        print("  ** ^GSPC 5-min also EMPTY.")
    else:
        if isinstance(spx_5m.columns, pd.MultiIndex):
            spx_5m.columns = [c[0] if isinstance(c, tuple) else c for c in spx_5m.columns]
        print(f"  ^GSPC 5-min shape: {spx_5m.shape}, range: {spx_5m.index[0]} -> {spx_5m.index[-1]}")
        print(f"  Last close: {spx_5m['Close'].iloc[-1]:.2f}")
else:
    if isinstance(spx_1m.columns, pd.MultiIndex):
        spx_1m.columns = [c[0] if isinstance(c, tuple) else c for c in spx_1m.columns]
    print(f"  ^GSPC 1-min shape: {spx_1m.shape}, range: {spx_1m.index[0]} -> {spx_1m.index[-1]}")
    print(f"  Last close: {spx_1m['Close'].iloc[-1]:.2f}")
    print(f"\n  First 3 rows:\n{spx_1m.head(3)}")

print(f"\n{SEPARATOR}")
print("DONE -- intraday data test complete.")
print(SEPARATOR)
