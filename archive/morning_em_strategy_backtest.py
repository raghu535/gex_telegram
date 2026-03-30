from __future__ import annotations
import json, sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

DB='gex_data.db'
RISK=500.0
REP=Path('reports'); REP.mkdir(exist_ok=True)
F30_START=time(6,30); F30_END=time(7,0); CUTOFF_A=time(11,0)

def tdt(ts): return datetime.fromtimestamp(ts)
def thms(s): h,m,sec=map(int,s.split(':')); return time(h,m,sec)
def clip(v,lo,hi): return max(lo,min(hi,v))
def pct(a,b): return None if b==0 else (a/b)*100.0

def load_rows():
    conn=sqlite3.connect(DB); conn.row_factory=sqlite3.Row
    rows=[dict(r) for r in conn.execute("""
    SELECT timestamp_pt,date_pt,time_pt,session_tag,curr_price,net_gamma,call_wall,put_floor
    FROM gex_snapshots WHERE session_tag='RTH' ORDER BY date_pt,timestamp_pt
    """).fetchall()]
    conn.close()
    seen=set(); out=[]
    for r in rows:
        k=(r['timestamp_pt'],r['curr_price'],r['net_gamma'],r['call_wall'],r['put_floor'])
        if k in seen: continue
        seen.add(k); r['dt']=tdt(r['timestamp_pt']); r['t']=thms(r['time_pt']); out.append(r)
    return out

def yfin_maps(start_day,end_day):
    s=datetime.strptime(start_day,'%Y-%m-%d')-timedelta(days=10)
    e=datetime.strptime(end_day,'%Y-%m-%d')+timedelta(days=10)
    vix=yf.Ticker('^VIX').history(start=s,end=e)
    spx=yf.Ticker('^GSPC').history(start=s,end=e)
    vix_m={idx.date().isoformat():float(row['Close']) for idx,row in vix.iterrows()} if len(vix)>0 else {}
    vol_m={idx.date().isoformat():float(row['Volume']) for idx,row in spx.iterrows()} if len(spx)>0 else {}
    return vix_m,vol_m

def prev_val(day, mp):
    dd=datetime.strptime(day,'%Y-%m-%d').date()
    ks=[k for k in mp if datetime.strptime(k,'%Y-%m-%d').date()<dd]
    if not ks: return None
    return mp[max(ks)]

def classify(row):
    r=row['first_30_ratio']; g=row['open_gamma']/1e9 if pd.notna(row['open_gamma']) else np.nan
    if pd.isna(r): return (None,None)
    grp='A' if r>0.5 else ('B' if r>=0.3 else 'C')
    sub='1' if (not pd.isna(g) and g<-3) else ('2' if (not pd.isna(g) and g>3) else '3')
    return grp, f'{grp}{sub}'

def build_features(rows):
    by_day={}
    for r in rows: by_day.setdefault(r['date_pt'],[]).append(r)
    for d in by_day: by_day[d].sort(key=lambda x:x['timestamp_pt'])
    days=sorted(by_day.keys())
    if not days: return pd.DataFrame(), by_day
    vix_m, vol_m = yfin_maps(days[0], days[-1])
    close_m={d:dr[-1]['curr_price'] for d,dr in by_day.items() if dr}

    rec=[]
    for i,d in enumerate(days):
        dr=by_day[d]; op=dr[0]; open_px=float(op['curr_price'])
        day_close=float(dr[-1]['curr_price']); day_high=max(float(x['curr_price']) for x in dr); day_low=min(float(x['curr_price']) for x in dr)
        day_range=day_high-day_low

        prev=days[i-1] if i>0 else None
        anchor=close_m.get(prev) if prev else None
        vix=prev_val(d,vix_m)
        em=((vix/16)/100*anchor) if (vix is not None and anchor is not None) else None
        em_u=(anchor+em) if (anchor is not None and em is not None) else None
        em_l=(anchor-em) if (anchor is not None and em is not None) else None

        f30=[x for x in dr if F30_START<=x['t']<=F30_END]
        f30h=max(float(x['curr_price']) for x in f30) if f30 else np.nan
        f30l=min(float(x['curr_price']) for x in f30) if f30 else np.nan
        f30r=(f30h-f30l) if pd.notna(f30h) and pd.notna(f30l) else np.nan

        tgt_dt=datetime.combine(dr[0]['dt'].date(), F30_END)
        r10=min(dr, key=lambda x: abs((x['dt']-tgt_dt).total_seconds()))
        p10=float(r10['curr_price']); g10=float(r10['net_gamma'])

        f30dir=p10-open_px
        f30ratio=(f30r/em) if (em is not None and em>0 and pd.notna(f30r)) else np.nan
        f30em=((p10-anchor)/em) if (anchor is not None and em is not None and em>0) else np.nan

        op_g=float(op['net_gamma']); avg_g=float(np.mean([x['net_gamma'] for x in dr])); min_g=float(min(x['net_gamma'] for x in dr)); max_g=float(max(x['net_gamma'] for x in dr))
        g_rng=max_g-min_g
        op_sp=float((op['call_wall'] or 0)-(op['put_floor'] or 0))

        act_mv=abs(day_close-anchor) if anchor is not None else np.nan
        max_up=((day_high-anchor)/em) if (anchor is not None and em is not None and em>0) else np.nan
        max_dn=((day_low-anchor)/em) if (anchor is not None and em is not None and em>0) else np.nan
        close_pos=((day_close-anchor)/em) if (anchor is not None and em is not None and em>0) else np.nan
        inside=(abs(close_pos)<=1.0) if pd.notna(close_pos) else np.nan
        breach=((max_up>1.0) or (max_dn<-1.0)) if pd.notna(max_up) and pd.notna(max_dn) else np.nan
        day_rr=(day_range/(em*2)) if (em is not None and em>0) else np.nan

        post10=[x for x in dr if x['dt']>r10['dt']]
        if post10:
            rh=max(float(x['curr_price']) for x in post10); rl=min(float(x['curr_price']) for x in post10)
        else:
            rh=p10; rl=p10
        rr=rh-rl; rm_up=rh-p10; rm_dn=p10-rl; rm=max(rm_up,rm_dn)

        rec.append({
            'date':d,'anchor':anchor,'vix':vix,'em_pts':em,'em_upper':em_u,'em_lower':em_l,
            'open_price':open_px,'first_30_high':f30h,'first_30_low':f30l,'first_30_range':f30r,
            'first_30_ratio':f30ratio,'first_30_direction':f30dir,'first_30_em_position':f30em,
            'day_volume':vol_m.get(d),'open_gamma':op_g,'avg_gamma':avg_g,'min_gamma':min_g,'max_gamma':max_g,
            'gamma_range':g_rng,'open_spread':op_sp,'gamma_at_10am':g10,'day_close':day_close,'day_high':day_high,
            'day_low':day_low,'day_range':day_range,'actual_move':act_mv,'max_move_up':max_up,'max_move_down':max_dn,
            'close_em_position':close_pos,'close_inside_em':inside,'intraday_breach':breach,'day_range_ratio':day_rr,
            'price_at_10am':p10,'remaining_high':rh,'remaining_low':rl,'remaining_range':rr,'remaining_move_up':rm_up,
            'remaining_move_down':rm_dn,'remaining_max_move':rm,'ts_10am':r10['timestamp_pt']
        })

    df=pd.DataFrame(rec).sort_values('date').reset_index(drop=True)
    df['vol_sma20']=df['day_volume'].rolling(20,min_periods=20).mean()
    df['vol_ratio']=df['day_volume']/df['vol_sma20']
    gs=df.apply(classify,axis=1,result_type='expand')
    df['group']=gs[0]; df['subgroup']=gs[1]
    return df, by_day

def aggregate(df,mask):
    s=df[mask].copy(); n=len(s)
    if n==0: return {'days':0}
    up=s[s['first_30_direction']>0]; dn=s[s['first_30_direction']<0]
    pred=s[(s['first_30_direction']*(s['day_close']-s['price_at_10am']))>0]
    dn_t=s[s['max_move_down']<=-0.75]; up_t=s[s['max_move_up']>=0.75]
    dn_f=dn_t[dn_t['close_em_position']>=-0.25]; dn_b=dn_t[dn_t['close_em_position']<=-1.0]
    up_f=up_t[up_t['close_em_position']<=0.25]; up_b=up_t[up_t['close_em_position']>=1.0]
    trig=len(dn_t)+len(up_t)
    return {
        'days':n,'close_inside_em_pct':pct(s['close_inside_em'].sum(),n),'intraday_breach_pct':pct(s['intraday_breach'].sum(),n),
        'avg_close_em_position':float(s['close_em_position'].mean()),'avg_max_move_up':float(s['max_move_up'].mean()),
        'avg_max_move_down':float(s['max_move_down'].mean()),'avg_day_range_pts':float(s['day_range'].mean()),
        'avg_day_range_ratio':float(s['day_range_ratio'].mean()),'avg_remaining_max_move_pts':float(s['remaining_max_move'].mean()),
        'max_remaining_max_move_pts':float(s['remaining_max_move'].max()),'avg_remaining_range_pts':float(s['remaining_range'].mean()),
        'down_persistence_pct':pct((dn['day_close']<dn['price_at_10am']).sum(),len(dn)),
        'up_persistence_pct':pct((up['day_close']>up['price_at_10am']).sum(),len(up)),
        'direction_predictive_pct':pct(len(pred),len(s[s['first_30_direction']!=0])),
        'down_trigger_days':len(dn_t),'down_fade_win_pct':pct(len(dn_f),len(dn_t)),'down_breakout_win_pct':pct(len(dn_b),len(dn_t)),
        'up_trigger_days':len(up_t),'up_fade_win_pct':pct(len(up_f),len(up_t)),'up_breakout_win_pct':pct(len(up_b),len(up_t)),
        'fade_win_pct_all_triggers':pct(len(dn_f)+len(up_f),trig),'breakout_win_pct_all_triggers':pct(len(dn_b)+len(up_b),trig),
        'remaining_p25':float(s['remaining_max_move'].quantile(0.25)),'remaining_p50':float(s['remaining_max_move'].quantile(0.5)),
        'remaining_p75':float(s['remaining_max_move'].quantile(0.75)),
    }

@dataclass
class Trade:
    date:str; group:str; subgroup:str; trade_type:str; side:str
    entry_time:str; exit_time:str; entry_price:float; exit_price:float
    target_price:float; stop_price:float; reason:str
    pnl_units:float; pnl_dollars:float; win:int

def entry_idx(day_rows,ts10):
    return min(range(len(day_rows)), key=lambda i: abs(day_rows[i]['timestamp_pt']-ts10))

def sim_long(day_rows,i,target,stop,cutoff=None):
    for j in range(i+1,len(day_rows)):
        r=day_rows[j]; p=float(r['curr_price'])
        if p<=stop: return r,'stop'
        if p>=target: return r,'target'
        if cutoff is not None and r['t']>=cutoff: return r,'time'
    return day_rows[-1],'eod'

def sim_short(day_rows,i,target,stop,cutoff=None):
    for j in range(i+1,len(day_rows)):
        r=day_rows[j]; p=float(r['curr_price'])
        if p>=stop: return r,'stop'
        if p<=target: return r,'target'
        if cutoff is not None and r['t']>=cutoff: return r,'time'
    return day_rows[-1],'eod'

def pnl_units(side,reason,entry,exit_px,target):
    if reason=='target': return 1.0
    if reason=='stop': return -1.0
    if side=='LONG':
        den=target-entry
        den=den if abs(den)>1e-9 else 1e-9
        return clip((exit_px-entry)/den,-1.0,1.0)
    den=entry-target
    den=den if abs(den)>1e-9 else 1e-9
    return clip((entry-exit_px)/den,-1.0,1.0)

def simulate(df, by_day):
    out=[]
    for _,row in df.iterrows():
        sg=row['subgroup']
        if sg in [None,'B1','B2','B3','A3','C3']: continue
        d=row['date']; dr=by_day.get(d)
        if not dr: continue
        if pd.isna(row['em_pts']) or row['em_pts']<=0 or pd.isna(row['anchor']): continue
        i=entry_idx(dr,float(row['ts_10am'])); er=dr[i]; ep=float(er['curr_price'])
        fdir=float(row['first_30_direction']); em=float(row['em_pts']); anc=float(row['anchor'])
        eu=float(row['em_upper']); el=float(row['em_lower'])

        if sg=='A1':
            if fdir==0: continue
            if fdir>0:
                side='LONG'; tgt=max(eu, ep+0.25*em); stop=ep-0.5*abs(fdir)
                if stop>=ep: stop=ep-0.2*em
                xr,reason=sim_long(dr,i,tgt,stop,CUTOFF_A); xp=float(xr['curr_price']); u=pnl_units(side,reason,ep,xp,tgt)
            else:
                side='SHORT'; tgt=min(el, ep-0.25*em); stop=ep+0.5*abs(fdir)
                if stop<=ep: stop=ep+0.2*em
                xr,reason=sim_short(dr,i,tgt,stop,CUTOFF_A); xp=float(xr['curr_price']); u=pnl_units(side,reason,ep,xp,tgt)
            out.append(Trade(d,'A',sg,'A1_CONTINUATION',side,er['dt'].strftime('%H:%M:%S'),xr['dt'].strftime('%H:%M:%S'),ep,xp,tgt,stop,reason,u,u*RISK,1 if u>0 else 0))
            continue

        if sg=='A2':
            if fdir==0: continue
            if fdir>0:
                side='SHORT'; tgt=anc if anc<ep else (ep-0.25*em); stop=max(eu, ep+0.2*em)
                xr,reason=sim_short(dr,i,tgt,stop,CUTOFF_A); xp=float(xr['curr_price']); u=pnl_units(side,reason,ep,xp,tgt)
            else:
                side='LONG'; tgt=anc if anc>ep else (ep+0.25*em); stop=min(el, ep-0.2*em)
                xr,reason=sim_long(dr,i,tgt,stop,CUTOFF_A); xp=float(xr['curr_price']); u=pnl_units(side,reason,ep,xp,tgt)
            out.append(Trade(d,'A',sg,'A2_FADE',side,er['dt'].strftime('%H:%M:%S'),xr['dt'].strftime('%H:%M:%S'),ep,xp,tgt,stop,reason,u,u*RISK,1 if u>0 else 0))
            continue

        if sg=='C2':
            post=[x for x in dr if x['dt']>er['dt']]
            if not post: post=[er]
            hi=max(float(x['curr_price']) for x in post); lo=min(float(x['curr_price']) for x in post)
            m=max((hi-anc)/em, -(lo-anc)/em)
            if m<=0.85: u,reason=1.0,'inside_condor'
            elif m>=1.10: u,reason=-1.0,'breach_wings'
            else:
                frac=(m-0.85)/0.25; u=1.0-2.0*frac; reason='partial_breach'
            xr=dr[-1]; xp=float(xr['curr_price'])
            out.append(Trade(d,'C',sg,'C2_IRON_CONDOR','NEUTRAL',er['dt'].strftime('%H:%M:%S'),xr['dt'].strftime('%H:%M:%S'),ep,xp,anc,np.nan,reason,u,u*RISK,1 if u>0 else 0))
            continue

        if sg=='C1':
            trig=None
            for j in range(i,len(dr)):
                p=float(dr[j]['curr_price']); empos=(p-anc)/em
                if empos<=-0.75: trig=j; break
            if trig is None: continue
            tr=dr[trig]; tep=float(tr['curr_price'])
            tgt=anc-0.25*em
            if tgt<=tep: tgt=tep+0.25*em
            stop=anc-1.0*em
            xr,reason=sim_long(dr,trig,tgt,stop,None); xp=float(xr['curr_price']); u=pnl_units('LONG',reason,tep,xp,tgt)
            out.append(Trade(d,'C',sg,'C1_LATE_EM_FADE','LONG',tr['dt'].strftime('%H:%M:%S'),xr['dt'].strftime('%H:%M:%S'),tep,xp,tgt,stop,reason,u,u*RISK,1 if u>0 else 0))
            continue
    return out

def max_dd(series):
    peak=series.cummax(); dd=peak-series
    return float(dd.max()) if len(dd) else 0.0

def max_cons_losses(pnls):
    cur=best=0
    for p in pnls:
        if p<0: cur+=1; best=max(best,cur)
        else: cur=0
    return best

def main():
    rows=load_rows()
    if not rows: raise SystemExit('No RTH rows')
    df,by_day=build_features(rows)
    if df.empty: raise SystemExit('No features built')

    # Step1 output
    p_daily=REP/'daily_features_full.csv'; df.to_csv(p_daily,index=False)

    valid=df[df['subgroup'].notna()].copy(); total=len(valid)

    # Step3 outcomes
    out=[]
    for g in ['A','B','C']:
        m=aggregate(valid, valid['group']==g)
        if m['days']==0: continue
        m.update({'level':'GROUP','code':g,'freq_pct_total':pct(m['days'],total)}); out.append(m)
    for sg in ['A1','A2','A3','B1','B2','B3','C1','C2','C3']:
        m=aggregate(valid, valid['subgroup']==sg)
        if m['days']==0: continue
        m.update({'level':'SUBGROUP','code':sg,'freq_pct_total':pct(m['days'],total)}); out.append(m)
    go=pd.DataFrame(out)
    cols=['level','code','days','freq_pct_total','close_inside_em_pct','intraday_breach_pct','avg_close_em_position',
          'avg_max_move_up','avg_max_move_down','avg_day_range_pts','avg_day_range_ratio','avg_remaining_max_move_pts',
          'max_remaining_max_move_pts','avg_remaining_range_pts','down_persistence_pct','up_persistence_pct',
          'direction_predictive_pct','down_trigger_days','down_fade_win_pct','down_breakout_win_pct','up_trigger_days',
          'up_fade_win_pct','up_breakout_win_pct','fade_win_pct_all_triggers','breakout_win_pct_all_triggers',
          'remaining_p25','remaining_p50','remaining_p75']
    go=go[cols]
    p_go=REP/'group_outcomes.csv'; go.to_csv(p_go,index=False)

    # Step4 remaining lookup
    rem=[]
    for sg in ['A1','A2','A3','B1','B2','B3','C1','C2','C3']:
        s=valid[valid['subgroup']==sg]
        if len(s)==0: continue
        dn=s[s['max_move_down']<=-0.75]; up=s[s['max_move_up']>=0.75]
        dn_f=dn[dn['close_em_position']>=-0.25]; up_f=up[up['close_em_position']<=0.25]
        dn_b=dn[dn['close_em_position']<=-1.0]; up_b=up[up['close_em_position']>=1.0]
        trig=len(dn)+len(up)
        rem.append({'group':sg,'days':len(s),'avg_remaining_move_pts':float(s['remaining_max_move'].mean()),
                    'fade_win_pct':pct(len(dn_f)+len(up_f),trig),'breakout_win_pct':pct(len(dn_b)+len(up_b),trig),
                    'remaining_move_p25':float(s['remaining_max_move'].quantile(0.25)),
                    'remaining_move_p50':float(s['remaining_max_move'].quantile(0.5)),
                    'remaining_move_p75':float(s['remaining_max_move'].quantile(0.75))})
    rem_df=pd.DataFrame(rem); p_rem=REP/'remaining_move_table.csv'; rem_df.to_csv(p_rem,index=False)

    # Step5+6 strategy
    trades=simulate(valid,by_day)
    tdf=pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(columns=['date','group','subgroup','trade_type','side','entry_time','exit_time','entry_price','exit_price','target_price','stop_price','reason','pnl_units','pnl_dollars','win'])
    if not tdf.empty: tdf=tdf.sort_values('date')
    p_tr=REP/'strategy_trades.csv'; tdf.to_csv(p_tr,index=False)

    day_pnl=tdf.groupby('date')['pnl_dollars'].sum().to_dict() if not tdf.empty else {}
    eq=[]; cum=0.0
    for _,r in valid.sort_values('date').iterrows():
        d=r['date']; pnl=float(day_pnl.get(d,0.0)); took=1 if d in day_pnl else 0
        cum+=pnl
        eq.append({'date':d,'group':r['group'],'subgroup':r['subgroup'],'trade_taken':took,'daily_pnl_dollars':pnl,'cumulative_pnl_dollars':cum})
    eq_df=pd.DataFrame(eq); p_eq=REP/'strategy_equity_curve.csv'; eq_df.to_csv(p_eq,index=False)

    monthly=[]
    if not tdf.empty:
        tdf['month']=tdf['date'].str.slice(0,7)
        monthly=(tdf.groupby('month').agg(trades=('pnl_dollars','count'),win_rate_pct=('win',lambda x:float(np.mean(x)*100.0)),pnl_dollars=('pnl_dollars','sum'),avg_pnl_dollars=('pnl_dollars','mean')).reset_index().to_dict(orient='records'))

    traded=int(eq_df['trade_taken'].sum()) if not eq_df.empty else 0
    skipped=len(valid)-traded
    winr=float(tdf['win'].mean()*100.0) if not tdf.empty else 0.0
    total_pnl=float(tdf['pnl_dollars'].sum()) if not tdf.empty else 0.0
    std=float(eq_df['daily_pnl_dollars'].std(ddof=0)) if not eq_df.empty else 0.0
    mean=float(eq_df['daily_pnl_dollars'].mean()) if not eq_df.empty else 0.0
    sharpe=(mean/std) if std>0 else 0.0
    mdd=max_dd(eq_df['cumulative_pnl_dollars']) if not eq_df.empty else 0.0
    mcl=max_cons_losses(list(tdf['pnl_dollars'])) if not tdf.empty else 0
    best=tdf.loc[tdf['pnl_dollars'].idxmax()].to_dict() if not tdf.empty else None
    worst=tdf.loc[tdf['pnl_dollars'].idxmin()].to_dict() if not tdf.empty else None

    perf={}
    if not tdf.empty:
        for k,s in tdf.groupby('subgroup'):
            perf[k]={'trades':int(len(s)),'win_rate_pct':float(s['win'].mean()*100.0),'avg_pnl_dollars':float(s['pnl_dollars'].mean()),'total_pnl_dollars':float(s['pnl_dollars'].sum())}

    summary={'data_scope':{'db_path':DB,'session_tag':'RTH','date_min':str(valid['date'].min()) if not valid.empty else None,'date_max':str(valid['date'].max()) if not valid.empty else None,'days_in_features':int(len(df)),'days_classified':int(len(valid))},
             'outputs':{'daily_features_full_csv':str(p_daily),'group_outcomes_csv':str(p_go),'remaining_move_table_csv':str(p_rem),'strategy_trades_csv':str(p_tr),'strategy_equity_curve_csv':str(p_eq),'full_backtest_summary_json':str(REP/'full_backtest_summary.json')},
             'strategy':{'risk_per_trade_dollars':RISK,'days_traded':traded,'days_skipped':skipped,'overall_win_rate_pct':winr,'total_pnl_dollars':total_pnl,'daily_mean_pnl_dollars':mean,'daily_std_pnl_dollars':std,'sharpe_daily_mean_over_std':sharpe,'max_drawdown_dollars':mdd,'max_consecutive_losses':mcl,'best_trade':best,'worst_trade':worst,'group_performance':perf,'monthly_breakdown':monthly}}
    p_sum=REP/'full_backtest_summary.json'; p_sum.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
