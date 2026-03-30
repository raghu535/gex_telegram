#!/usr/bin/env python3
"""MA Retrace + GEX Backtest: Which MA does price retrace to, and what happens after?"""

import sqlite3
import pandas as pd
import numpy as np

DB = r"C:\Code\gex_telegram\gex_data.db"

def load_data():
    conn = sqlite3.connect(DB)
    df = pd.read_sql('''
        SELECT m.date_pt, m.time_pt, m.timestamp_pt, m.curr_price,
               m.ma_1m_20, m.ma_1m_50, m.ma_1m_200,
               m.ma_3m_20, m.ma_3m_50, m.ma_3m_200,
               m.ma_5m_20, m.ma_5m_50, m.ma_5m_200,
               m.ma_15m_20, m.ma_15m_50, m.ma_15m_200,
               m.ma_30m_20, m.ma_30m_50, m.ma_30m_200,
               m.ma_60m_20, m.ma_60m_50, m.ma_60m_200,
               m.ma_d_20, m.ma_d_50, m.ma_d_200,
               g.net_gamma
        FROM gex_ma_features m
        JOIN gex_snapshots g ON m.date_pt = g.date_pt
            AND abs(m.timestamp_pt - g.timestamp_pt) < 5
            AND g.session_tag = 'RTH'
        WHERE m.date_pt >= '2025-10-01'
        ORDER BY m.date_pt, m.timestamp_pt
    ''', conn)
    conn.close()
    MA_COLS = [c for c in df.columns if c.startswith('ma_')]
    for c in MA_COLS + ['curr_price', 'net_gamma']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df, MA_COLS


def detect_touches(df, MA_COLS):
    touch_pts = 3.0
    min_prior_dist = 8.0
    lookback = 5
    results = []

    for date, grp in df.groupby('date_pt'):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        prices = grp['curr_price'].values.astype(float)
        for ma_col in MA_COLS:
            ma_vals = grp[ma_col].values.astype(float)
            if np.all(np.isnan(ma_vals)):
                continue
            dists = prices - ma_vals
            last_touch_i = -999
            for i in range(lookback, n):
                if np.isnan(dists[i]) or abs(dists[i]) > touch_pts:
                    continue
                if i - last_touch_i < 10:
                    continue
                prior = dists[max(0, i - lookback):i]
                prior = prior[~np.isnan(prior)]
                if len(prior) == 0 or np.max(np.abs(prior)) < min_prior_dist:
                    continue
                last_touch_i = i
                avg_prior = np.mean(prior)
                retrace_from = "ABOVE" if avg_prior > 0 else "BELOW"
                r15 = prices[min(i + 8, n - 1)] - prices[i] if i + 8 < n else np.nan
                r30 = prices[min(i + 15, n - 1)] - prices[i] if i + 15 < n else np.nan
                r_close = prices[n - 1] - prices[i]
                fwd = prices[i + 1:min(i + 31, n)]
                mfe_up = float(np.max(fwd) - prices[i]) if len(fwd) > 0 else 0.0
                mfe_dn = float(prices[i] - np.min(fwd)) if len(fwd) > 0 else 0.0
                gamma_bn = float(grp['net_gamma'].iloc[i]) / 1e9
                results.append({
                    'date': date, 'ma': ma_col, 'retrace_from': retrace_from,
                    'gamma_bn': gamma_bn,
                    'r15': r15, 'r30': r30, 'r_close': r_close,
                    'mfe_up': mfe_up, 'mfe_dn': mfe_dn,
                })
    return pd.DataFrame(results)


def gex_bucket(g):
    if g < -10:
        return 'G < -10'
    if g < -5:
        return 'G -10 to -5'
    if g < 0:
        return 'G -5 to 0'
    if g < 5:
        return 'G 0 to +5'
    if g < 10:
        return 'G +5 to +10'
    if g < 20:
        return 'G +10 to +20'
    return 'G > +20'


def edge_long(row):
    if row['win'] >= 60 and row['r15'] > 3:
        return 'STRONG'
    if row['win'] >= 50 and row['r15'] > 2:
        return 'GOOD'
    if row['win'] >= 45:
        return 'OK'
    return 'SKIP'


def edge_short(row):
    if row['win'] >= 60 and row['r15'] < -3:
        return 'STRONG'
    if row['win'] >= 50 and row['r15'] < -2:
        return 'GOOD'
    if row['win'] >= 45:
        return 'OK'
    return 'SKIP'


def main():
    print("Loading data...")
    df, MA_COLS = load_data()
    print(f"Loaded {len(df)} rows, {df['date_pt'].nunique()} days")
    print()

    print("Detecting MA touches...")
    res = detect_touches(df, MA_COLS)
    print(f"Total touches: {len(res)}, Unique dates: {res['date'].nunique()}")
    res['gex'] = res['gamma_bn'].apply(gex_bucket)

    # ================================================================
    # SECTION 1: Clean trader card — MA retrace results
    # ================================================================
    print()
    print("=" * 90)
    print("  MA RETRACE BOUNCE CARD")
    print("  Touch = price within 3pts of MA, was 8+ pts away in prior 5 bars")
    print("  Win = r15 moves >2pts in expected direction. N >= 15.")
    print("=" * 90)

    for direction, side_label in [('ABOVE', 'LONG'), ('BELOW', 'SHORT')]:
        sub = res[res['retrace_from'] == direction].copy()
        if direction == 'ABOVE':
            sub['win'] = sub['r15'] > 2
            desc = "Price was ABOVE MA, retraced DOWN to it -> bounce UP?"
        else:
            sub['win'] = sub['r15'] < -2
            desc = "Price was BELOW MA, rallied UP to it -> fade DOWN?"

        print(f"\n--- {side_label}: {desc} ---")
        hdr = f"{'MA':<12} {'N':>5} {'Win%':>6} {'Avg r15':>8} {'Avg r30':>8} {'MFE fav':>8} {'MFE adv':>8} {'Edge':>8}"
        print(hdr)
        print("-" * len(hdr))

        agg = sub.groupby('ma').agg(
            N=('win', 'count'),
            win_pct=('win', 'mean'),
            r15=('r15', 'mean'),
            r30=('r30', 'mean'),
            mfe_up=('mfe_up', 'mean'),
            mfe_dn=('mfe_dn', 'mean'),
        ).reset_index()
        agg = agg[agg['N'] >= 15].copy()
        agg['win_pct'] = agg['win_pct'] * 100

        if direction == 'ABOVE':
            agg['mfe_fav'] = agg['mfe_up']
            agg['mfe_adv'] = agg['mfe_dn']
            agg['edge'] = agg.apply(
                lambda r: 'STRONG' if r['win_pct'] >= 48 and r['r15'] > 3 else
                'GOOD' if r['win_pct'] >= 45 and r['r15'] > 2 else
                'OK' if r['r15'] > 0.5 else 'WEAK', axis=1)
            agg = agg.sort_values('r15', ascending=False)
        else:
            agg['mfe_fav'] = agg['mfe_dn']
            agg['mfe_adv'] = agg['mfe_up']
            agg['edge'] = agg.apply(
                lambda r: 'STRONG' if r['win_pct'] >= 48 and r['r15'] < -3 else
                'GOOD' if r['win_pct'] >= 45 and r['r15'] < -2 else
                'OK' if r['r15'] < -0.5 else 'WEAK', axis=1)
            agg = agg.sort_values('r15', ascending=True)

        for _, r in agg.iterrows():
            ma_name = r['ma'].replace('ma_', '')
            print(f"{ma_name:<12} {int(r['N']):>5} {r['win_pct']:>5.1f}% {r['r15']:>+7.1f} {r['r30']:>+7.1f} {r['mfe_fav']:>7.1f} {r['mfe_adv']:>7.1f} {r['edge']:>8}")

    # ================================================================
    # SECTION 2: GEX overlay on top MAs
    # ================================================================
    print()
    print("=" * 90)
    print("  GEX OVERLAY: Top MAs broken down by gamma bucket (N >= 5)")
    print("=" * 90)

    top_long = ['ma_15m_50', 'ma_3m_200', 'ma_30m_20', 'ma_5m_20', 'ma_60m_20']
    top_short = ['ma_30m_200', 'ma_60m_50', 'ma_15m_200', 'ma_d_20', 'ma_15m_50']

    gex_order = ['G < -10', 'G -10 to -5', 'G -5 to 0', 'G 0 to +5', 'G +5 to +10', 'G +10 to +20', 'G > +20']

    print("\n--- LONG: retrace down to MA then bounce UP ---")
    print(f"{'MA':<10} {'GEX':<16} {'N':>4} {'Win%':>6} {'r15':>7} {'r30':>7} {'MFE+':>6} {'MFE-':>6}")
    print("-" * 65)
    sub_l = res[(res['retrace_from'] == 'ABOVE') & (res['ma'].isin(top_long))].copy()
    sub_l['win'] = sub_l['r15'] > 2
    for ma in top_long:
        ms = sub_l[sub_l['ma'] == ma]
        for gex in gex_order:
            g = ms[ms['gex'] == gex]
            if len(g) < 5:
                continue
            w = g['win'].mean() * 100
            print(f"{ma.replace('ma_', ''):<10} {gex:<16} {len(g):>4} {w:>5.0f}% {g['r15'].mean():>+6.1f} {g['r30'].mean():>+6.1f} {g['mfe_up'].mean():>5.1f} {g['mfe_dn'].mean():>5.1f}")
        print()

    print("\n--- SHORT: retrace up to MA then continue DOWN ---")
    print(f"{'MA':<10} {'GEX':<16} {'N':>4} {'Win%':>6} {'r15':>7} {'r30':>7} {'MFE+':>6} {'MFE-':>6}")
    print("-" * 65)
    sub_s = res[(res['retrace_from'] == 'BELOW') & (res['ma'].isin(top_short))].copy()
    sub_s['win'] = sub_s['r15'] < -2
    for ma in top_short:
        ms = sub_s[sub_s['ma'] == ma]
        for gex in gex_order:
            g = ms[ms['gex'] == gex]
            if len(g) < 5:
                continue
            w = g['win'].mean() * 100
            print(f"{ma.replace('ma_', ''):<10} {gex:<16} {len(g):>4} {w:>5.0f}% {g['r15'].mean():>+6.1f} {g['r30'].mean():>+6.1f} {g['mfe_up'].mean():>5.1f} {g['mfe_dn'].mean():>5.1f}")
        print()

    # ================================================================
    # SECTION 3: FINAL TRADER CARD — Best combos only
    # ================================================================
    print()
    print("=" * 90)
    print("  FINAL CARD: BEST MA + GEX COMBOS (N >= 8, STRONG/GOOD only)")
    print("=" * 90)

    # LONG
    print("\n--- LONG after retrace to MA ---")
    hdr = f"{'MA + GEX':<32} {'N':>4} {'Win%':>6} {'r15':>8} {'r30':>8} {'Edge':<8}"
    print(hdr)
    print("-" * len(hdr))
    all_long = res[res['retrace_from'] == 'ABOVE'].copy()
    all_long['win'] = (all_long['r15'] > 2).astype(float)
    grp = all_long.groupby(['ma', 'gex']).agg(
        N=('win', 'count'), win=('win', 'mean'), r15=('r15', 'mean'), r30=('r30', 'mean')
    ).reset_index()
    grp = grp[grp['N'] >= 8].copy()
    grp['win'] = grp['win'] * 100
    grp['edge'] = grp.apply(edge_long, axis=1)
    grp = grp[grp['edge'].isin(['STRONG', 'GOOD'])].sort_values('win', ascending=False)
    for _, r in grp.iterrows():
        label = f"{r['ma'].replace('ma_', '')} + {r['gex']}"
        print(f"{label:<32} {int(r['N']):>4} {r['win']:>5.0f}% {r['r15']:>+7.1f} {r['r30']:>+7.1f} {r['edge']:<8}")

    # SHORT
    print("\n--- SHORT after retrace to MA ---")
    print(hdr)
    print("-" * len(hdr))
    all_short = res[res['retrace_from'] == 'BELOW'].copy()
    all_short['win'] = (all_short['r15'] < -2).astype(float)
    grp = all_short.groupby(['ma', 'gex']).agg(
        N=('win', 'count'), win=('win', 'mean'), r15=('r15', 'mean'), r30=('r30', 'mean')
    ).reset_index()
    grp = grp[grp['N'] >= 8].copy()
    grp['win'] = grp['win'] * 100
    grp['edge'] = grp.apply(edge_short, axis=1)
    grp = grp[grp['edge'].isin(['STRONG', 'GOOD'])].sort_values('win', ascending=False)
    for _, r in grp.iterrows():
        label = f"{r['ma'].replace('ma_', '')} + {r['gex']}"
        print(f"{label:<32} {int(r['N']):>4} {r['win']:>5.0f}% {r['r15']:>+7.1f} {r['r30']:>+7.1f} {r['edge']:<8}")


if __name__ == '__main__':
    main()
