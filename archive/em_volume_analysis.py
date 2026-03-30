"""
SPX Expected Move vs Actual Move + Volume/Gamma Correlation Analysis
====================================================================

Usage:
  python em_volume_analysis.py
"""

import sqlite3
from datetime import timedelta
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("yfinance not installed. Using gex_data.db only.")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Skipping charts.")


def load_from_gex_db(db_path="gex_data.db"):
    """Load daily OHLC + gamma summary from RTH snapshots."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT date_pt,
               MIN(time_pt) as first_time,
               MAX(time_pt) as last_time,
               (SELECT curr_price FROM gex_snapshots s2
                WHERE s2.date_pt = s.date_pt AND s2.session_tag = 'RTH'
                ORDER BY s2.time_pt ASC LIMIT 1) as day_open,
               (SELECT curr_price FROM gex_snapshots s3
                WHERE s3.date_pt = s.date_pt AND s3.session_tag = 'RTH'
                ORDER BY s3.time_pt DESC LIMIT 1) as day_close,
               MAX(curr_price) as day_high,
               MIN(curr_price) as day_low,
               COUNT(*) as snapshot_count,
               AVG(net_gamma) as avg_gamma,
               MIN(net_gamma) as min_gamma,
               MAX(net_gamma) as max_gamma
        FROM gex_snapshots s
        WHERE session_tag = 'RTH'
        GROUP BY date_pt
        ORDER BY date_pt
        """,
        conn,
    )
    conn.close()
    df["date_pt"] = pd.to_datetime(df["date_pt"])
    return df


def load_vix_and_volume(start_date, end_date):
    """Pull VIX closes and SPX volume from yfinance."""
    if not HAS_YFINANCE:
        return None, None
    spx = yf.Ticker("^GSPC")
    spx_hist = spx.history(start=start_date, end=end_date + timedelta(days=5))
    vix = yf.Ticker("^VIX")
    vix_hist = vix.history(
        start=start_date - timedelta(days=5), end=end_date + timedelta(days=5)
    )
    return spx_hist, vix_hist


def compute_expected_move_analysis(daily_df, spx_hist=None, vix_hist=None):
    """Compute EM behavior and breakout/inside stats per day."""
    results = []

    for i in range(1, len(daily_df)):
        today = daily_df.iloc[i]
        yesterday = daily_df.iloc[i - 1]
        today_date = today["date_pt"]
        today_day = today_date.date()

        vix_close = None
        if vix_hist is not None:
            vix_before = vix_hist[pd.Series(vix_hist.index.date, index=vix_hist.index) < today_day]
            if len(vix_before) > 0:
                vix_close = vix_before["Close"].iloc[-1]
        if vix_close is None:
            continue

        volume = None
        if spx_hist is not None:
            spx_today = spx_hist[pd.Series(spx_hist.index.date, index=spx_hist.index) == today_day]
            if len(spx_today) > 0:
                volume = spx_today["Volume"].iloc[0]

        anchor = yesterday["day_close"]
        em_pct = (vix_close / 16.0) / 100.0
        em_pts = anchor * em_pct
        em_upper = anchor + em_pts
        em_lower = anchor - em_pts

        actual_close = today["day_close"]
        actual_high = today["day_high"]
        actual_low = today["day_low"]
        actual_move = abs(actual_close - anchor)
        actual_range = actual_high - actual_low

        close_inside = (actual_close >= em_lower) and (actual_close <= em_upper)
        high_breached = actual_high > em_upper
        low_breached = actual_low < em_lower
        intraday_breach = high_breached or low_breached

        em_position = (actual_close - anchor) / em_pts if em_pts > 0 else 0
        max_em_up = (actual_high - anchor) / em_pts if em_pts > 0 else 0
        max_em_down = (actual_low - anchor) / em_pts if em_pts > 0 else 0

        avg_gamma = today["avg_gamma"] if "avg_gamma" in today.index else None
        min_gamma = today["min_gamma"] if "min_gamma" in today.index else None

        results.append(
            {
                "date": today_date,
                "anchor": round(anchor, 2),
                "vix": round(vix_close, 2),
                "em_pts": round(em_pts, 2),
                "em_upper": round(em_upper, 2),
                "em_lower": round(em_lower, 2),
                "actual_close": round(actual_close, 2),
                "actual_high": round(actual_high, 2),
                "actual_low": round(actual_low, 2),
                "actual_move": round(actual_move, 2),
                "actual_range": round(actual_range, 2),
                "em_position": round(em_position, 3),
                "max_em_up": round(max_em_up, 3),
                "max_em_down": round(max_em_down, 3),
                "close_inside": close_inside,
                "intraday_breach": intraday_breach,
                "high_breached": high_breached,
                "low_breached": low_breached,
                "volume": volume,
                "avg_gamma": avg_gamma,
                "min_gamma": min_gamma,
                "move_ratio": round(actual_move / em_pts, 3) if em_pts > 0 else 0,
                "range_ratio": round(actual_range / (em_pts * 2), 3) if em_pts > 0 else 0,
            }
        )

    return pd.DataFrame(results)


def analyze_volume_correlation(df):
    if df["volume"].isna().all():
        print("\nNo volume data available. Skipping volume analysis.")
        return

    df = df.dropna(subset=["volume"]).copy()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"]
    df = df.dropna(subset=["vol_sma20"])

    print("\n" + "=" * 60)
    print("VOLUME vs EXPECTED MOVE ANALYSIS")
    print("=" * 60)

    high_vol = df[df["vol_ratio"] > 1.2]
    normal_vol = df[(df["vol_ratio"] >= 0.8) & (df["vol_ratio"] <= 1.2)]
    low_vol = df[df["vol_ratio"] < 0.8]

    print(f"\nHigh Volume Days (>1.2x avg): {len(high_vol)}")
    if len(high_vol) > 0:
        print(f"  Close inside EM: {high_vol['close_inside'].mean()*100:.1f}%")
        print(f"  Intraday breach: {high_vol['intraday_breach'].mean()*100:.1f}%")
        print(f"  Avg move ratio:  {high_vol['move_ratio'].mean():.3f}")
        print(f"  Avg range ratio: {high_vol['range_ratio'].mean():.3f}")

    print(f"\nNormal Volume Days (0.8-1.2x avg): {len(normal_vol)}")
    if len(normal_vol) > 0:
        print(f"  Close inside EM: {normal_vol['close_inside'].mean()*100:.1f}%")
        print(f"  Intraday breach: {normal_vol['intraday_breach'].mean()*100:.1f}%")
        print(f"  Avg move ratio:  {normal_vol['move_ratio'].mean():.3f}")
        print(f"  Avg range ratio: {normal_vol['range_ratio'].mean():.3f}")

    print(f"\nLow Volume Days (<0.8x avg): {len(low_vol)}")
    if len(low_vol) > 0:
        print(f"  Close inside EM: {low_vol['close_inside'].mean()*100:.1f}%")
        print(f"  Intraday breach: {low_vol['intraday_breach'].mean()*100:.1f}%")
        print(f"  Avg move ratio:  {low_vol['move_ratio'].mean():.3f}")
        print(f"  Avg range ratio: {low_vol['range_ratio'].mean():.3f}")

    print(f"\n{'='*60}")
    print("ANTI-GRAVITY TEST: Volume Predicts Breakout?")
    print(f"{'='*60}")

    corr_move = df["vol_ratio"].corr(df["move_ratio"])
    corr_breach = df["vol_ratio"].corr(df["intraday_breach"].astype(float))

    print(f"\nCorrelation: Volume ratio vs Move ratio:      {corr_move:.3f}")
    print(f"Correlation: Volume ratio vs Intraday breach: {corr_breach:.3f}")

    if abs(corr_breach) > 0.3:
        print("Meaningful correlation: volume helps predict breakouts.")
    elif abs(corr_breach) > 0.15:
        print("Weak correlation: volume adds limited signal.")
    else:
        print("No significant correlation: volume is weak for breakout prediction.")

    print(f"\n{'='*60}")
    print("VOLUME THRESHOLD SCAN")
    print(f"{'='*60}")
    for threshold in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        above = df[df["vol_ratio"] > threshold]
        below = df[df["vol_ratio"] <= threshold]
        if len(above) > 3 and len(below) > 3:
            above_breach = above["intraday_breach"].mean() * 100
            below_breach = below["intraday_breach"].mean() * 100
            above_inside = above["close_inside"].mean() * 100
            below_inside = below["close_inside"].mean() * 100
            print(
                f"  Vol > {threshold:.1f}x: breach {above_breach:.0f}%, inside {above_inside:.0f}% ({len(above)} days)"
            )
            print(
                f"  Vol <= {threshold:.1f}x: breach {below_breach:.0f}%, inside {below_inside:.0f}% ({len(below)} days)"
            )
            print(f"   Spread: {above_breach - below_breach:+.0f}% breach difference\n")

    return df


def analyze_gamma_correlation(df):
    if df["avg_gamma"].isna().all():
        print("\nNo gamma data available. Skipping gamma analysis.")
        return

    df_g = df.dropna(subset=["avg_gamma"]).copy()
    if len(df_g) < 5:
        print("\nNot enough gamma data for analysis.")
        return

    print(f"\n{'='*60}")
    print("GAMMA vs EXPECTED MOVE ANALYSIS")
    print(f"{'='*60}")

    neg_gamma = df_g[df_g["avg_gamma"] < -2e9]
    neutral = df_g[(df_g["avg_gamma"] >= -2e9) & (df_g["avg_gamma"] <= 2e9)]
    pos_gamma = df_g[df_g["avg_gamma"] > 2e9]

    print(f"\nNegative Gamma Days (<-2Bn avg): {len(neg_gamma)}")
    if len(neg_gamma) > 0:
        print(f"  Close inside EM: {neg_gamma['close_inside'].mean()*100:.1f}%")
        print(f"  Intraday breach: {neg_gamma['intraday_breach'].mean()*100:.1f}%")
        print(f"  Avg move ratio:  {neg_gamma['move_ratio'].mean():.3f}")
        print(f"  Avg range ratio: {neg_gamma['range_ratio'].mean():.3f}")

    print(f"\nNeutral Gamma Days (-2 to +2Bn): {len(neutral)}")
    if len(neutral) > 0:
        print(f"  Close inside EM: {neutral['close_inside'].mean()*100:.1f}%")
        print(f"  Intraday breach: {neutral['intraday_breach'].mean()*100:.1f}%")
        print(f"  Avg move ratio:  {neutral['move_ratio'].mean():.3f}")

    print(f"\nPositive Gamma Days (>+2Bn avg): {len(pos_gamma)}")
    if len(pos_gamma) > 0:
        print(f"  Close inside EM: {pos_gamma['close_inside'].mean()*100:.1f}%")
        print(f"  Intraday breach: {pos_gamma['intraday_breach'].mean()*100:.1f}%")
        print(f"  Avg move ratio:  {pos_gamma['move_ratio'].mean():.3f}")

    corr = df_g["avg_gamma"].corr(df_g["move_ratio"])
    print(f"\nCorrelation: Avg gamma vs Move ratio: {corr:.3f}")

    if corr < -0.2:
        print("Negative correlation: more negative gamma tends to larger moves.")
    else:
        print("Weak/no gamma correlation in this sample.")

    print(f"\n{'='*60}")
    print("EM FADE WIN RATE SPLIT BY GAMMA")
    print(f"{'='*60}")
    fade_triggered = df_g[df_g["max_em_down"] <= -0.75].copy()
    if len(fade_triggered) > 3:
        fade_triggered["fade_won"] = fade_triggered["em_position"] > -0.5
        neg_fade = fade_triggered[fade_triggered["avg_gamma"] < -2e9]
        pos_fade = fade_triggered[fade_triggered["avg_gamma"] > 0]

        print(f"\nDays fade triggered (reached -0.75 EM): {len(fade_triggered)}")
        print(f"  Overall fade win rate: {fade_triggered['fade_won'].mean()*100:.1f}%")
        if len(neg_fade) > 0:
            print(
                f"  Negative gamma fade win rate: {neg_fade['fade_won'].mean()*100:.1f}% ({len(neg_fade)} days)"
            )
        if len(pos_fade) > 0:
            print(
                f"  Positive gamma fade win rate: {pos_fade['fade_won'].mean()*100:.1f}% ({len(pos_fade)} days)"
            )
        if len(neg_fade) > 0 and len(pos_fade) > 0:
            diff = (neg_fade["fade_won"].mean() - pos_fade["fade_won"].mean()) * 100
            print(f"  Difference: {diff:+.1f}%")


def generate_charts(df, output_dir="reports"):
    if not HAS_MATPLOTLIB:
        print("\nmatplotlib not available. Skipping charts.")
        return

    Path(output_dir).mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("SPX Expected Move Analysis", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    colors = ["green" if x else "red" for x in df["close_inside"]]
    ax.bar(df["date"], df["em_position"], color=colors, alpha=0.7, width=1)
    ax.axhline(y=0.75, color="red", linestyle="--", alpha=0.5, label="0.75 fade")
    ax.axhline(y=-0.75, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=1.0, color="darkred", linestyle="-", alpha=0.5, label="1.0 EM")
    ax.axhline(y=-1.0, color="darkred", linestyle="-", alpha=0.5)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_ylabel("EM Position at Close")
    ax.set_title("Daily Close EM Position")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)

    ax = axes[0, 1]
    ax.hist(df["move_ratio"], bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(x=1.0, color="red", linestyle="--", label="1.0 EM")
    ax.axvline(x=0.75, color="orange", linestyle="--", label="0.75 fade")
    ax.set_xlabel("Actual Move / Expected Move")
    ax.set_ylabel("Days")
    ax.set_title("Move Ratio Distribution")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(df["em_pts"], df["actual_range"], alpha=0.5, c="steelblue", s=30)
    max_val = max(df["em_pts"].max(), df["actual_range"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val * 2], "r--", alpha=0.3, label="Range=2xEM")
    ax.plot([0, max_val], [0, max_val], "gray", alpha=0.3, label="Range=EM")
    ax.set_xlabel("Expected Move (pts)")
    ax.set_ylabel("Actual Range (pts)")
    ax.set_title("Expected Move vs Intraday Range")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if len(df) > 10:
        df_sorted = df.sort_values("date")
        rolling_inside = df_sorted["close_inside"].rolling(10, min_periods=5).mean() * 100
        ax.plot(
            df_sorted["date"],
            rolling_inside,
            color="steelblue",
            linewidth=2,
            label="10d % inside",
        )
        ax2 = ax.twinx()
        ax2.plot(df_sorted["date"], df_sorted["vix"], color="orange", alpha=0.5, linewidth=1, label="VIX")
        ax2.set_ylabel("VIX", color="orange")
        ax.set_ylabel("% Close Inside EM")
        ax.set_title("EM Accuracy Over Time")
        ax.axhline(y=68, color="gray", linestyle="--", alpha=0.3, label="68% theory")
        ax.legend(loc="lower left", fontsize=8)
        ax2.legend(loc="lower right", fontsize=8)
    ax.tick_params(axis="x", rotation=45)

    ax = axes[2, 0]
    if not df["volume"].isna().all():
        df_v = df.dropna(subset=["volume"]).copy()
        df_v["vol_sma20"] = df_v["volume"].rolling(20).mean()
        df_v = df_v.dropna(subset=["vol_sma20"])
        df_v["vol_ratio"] = df_v["volume"] / df_v["vol_sma20"]
        colors = ["red" if b else "green" for b in df_v["intraday_breach"]]
        ax.scatter(df_v["vol_ratio"], df_v["move_ratio"], c=colors, alpha=0.5, s=30)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Volume Ratio (20d avg)")
        ax.set_ylabel("Move Ratio")
        ax.set_title("Volume vs Move Ratio")
    else:
        ax.text(0.5, 0.5, "No volume data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[2, 1]
    if not df["avg_gamma"].isna().all():
        df_g = df.dropna(subset=["avg_gamma"]).copy()
        colors = ["red" if not i else "green" for i in df_g["close_inside"]]
        ax.scatter(df_g["avg_gamma"] / 1e9, df_g["move_ratio"], c=colors, alpha=0.5, s=30)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Avg Net Gamma (Bn)")
        ax.set_ylabel("Move Ratio")
        ax.set_title("Gamma vs Move Ratio")
    else:
        ax.text(0.5, 0.5, "No gamma data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    chart_path = f"{output_dir}/em_analysis_charts.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCharts saved: {chart_path}")


def main():
    print("=" * 60)
    print("SPX EXPECTED MOVE vs ACTUAL + VOLUME/GAMMA ANALYSIS")
    print("=" * 60)

    db_path = "gex_data.db"
    if not Path(db_path).exists():
        print(f"{db_path} not found. Run in gex_telegram directory.")
        return

    print(f"\nLoading from {db_path}...")
    daily_df = load_from_gex_db(db_path)
    print(
        f"  Loaded {len(daily_df)} trading days: {daily_df['date_pt'].min().date()} to {daily_df['date_pt'].max().date()}"
    )

    start_date = daily_df["date_pt"].min()
    end_date = daily_df["date_pt"].max()
    spx_hist = None
    vix_hist = None
    if HAS_YFINANCE:
        print("Loading VIX + volume from yfinance...")
        spx_hist, vix_hist = load_vix_and_volume(start_date, end_date)
        if vix_hist is not None:
            print(f"  VIX data rows: {len(vix_hist)}")
        if spx_hist is not None:
            print(f"  SPX volume rows: {len(spx_hist)}")

    print("\nComputing expected move analysis...")
    results = compute_expected_move_analysis(daily_df, spx_hist, vix_hist)
    if len(results) == 0:
        print("No results. Check VIX data availability.")
        return

    Path("reports").mkdir(exist_ok=True)
    results_path = Path("reports/em_daily_analysis.csv")
    results.to_csv(results_path, index=False)
    print(f"  Raw data saved: {results_path}")

    print(f"\n{'='*60}")
    print("EXPECTED MOVE ACCURACY")
    print(f"{'='*60}")
    total = len(results)
    inside = int(results["close_inside"].sum())
    outside = total - inside
    breach_intraday = int(results["intraday_breach"].sum())

    print(f"\nTotal trading days analyzed: {total}")
    print(f"Close inside EM:   {inside} ({inside/total*100:.1f}%)")
    print(f"Close outside EM:  {outside} ({outside/total*100:.1f}%)")
    print(f"Intraday breach:   {breach_intraday} ({breach_intraday/total*100:.1f}%)")
    print("Theoretical (1 sigma): ~68%")

    print(f"\nAvg expected move:  {results['em_pts'].mean():.1f} pts")
    print(f"Avg actual move:    {results['actual_move'].mean():.1f} pts")
    print(f"Avg move ratio:     {results['move_ratio'].mean():.3f}")
    print(f"Avg VIX:            {results['vix'].mean():.2f}")

    print(f"\n{'='*60}")
    print("BREAKDOWN BY VIX LEVEL")
    print(f"{'='*60}")
    for vix_low, vix_high, label in [
        (0, 15, "Low VIX (<15)"),
        (15, 20, "Medium VIX (15-20)"),
        (20, 30, "High VIX (20-30)"),
        (30, 100, "Very High VIX (30+)"),
    ]:
        subset = results[(results["vix"] >= vix_low) & (results["vix"] < vix_high)]
        if len(subset) > 0:
            print(f"\n{label}: {len(subset)} days")
            print(f"  Close inside EM: {subset['close_inside'].mean()*100:.1f}%")
            print(f"  Avg move ratio:  {subset['move_ratio'].mean():.3f}")
            print(f"  Avg range ratio: {subset['range_ratio'].mean():.3f}")

    print(f"\n{'='*60}")
    print("BREACH ANALYSIS")
    print(f"{'='*60}")
    breached = results[results["intraday_breach"]]
    if len(breached) > 0:
        came_back = breached[breached["close_inside"]]
        stayed_out = breached[~breached["close_inside"]]
        print(f"\nDays that breached EM intraday: {len(breached)}")
        print(f"  Came back inside by close: {len(came_back)} ({len(came_back)/len(breached)*100:.1f}%)")
        print(f"  Stayed outside at close:   {len(stayed_out)} ({len(stayed_out)/len(breached)*100:.1f}%)")
        print(f"\n{len(came_back)/len(breached)*100:.0f}% of intraday breaches came back inside.")

    analyze_volume_correlation(results)
    analyze_gamma_correlation(results)
    generate_charts(results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    breach_pct = breach_intraday / total * 100
    fade_success = 0.0
    breached = results[results["intraday_breach"]]
    if len(breached) > 0:
        fade_success = breached["close_inside"].mean() * 100
    print(
        f"""
EM holds at close: {inside/total*100:.0f}% of days (theory: 68%)
Intraday breach:   {breach_pct:.0f}% of days
Breach -> fade win: {fade_success:.0f}% of breaches come back inside

Use volume/gamma sections above to assess filter edge.
"""
    )


if __name__ == "__main__":
    main()
