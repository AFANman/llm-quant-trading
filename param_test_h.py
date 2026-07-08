"""H: 冷却2天, 上涨底仓2万, 跌>3%加仓, 1×ATR卖出, [30k,20k,10k]"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators

def run(df, init_cash=100000, base_pos=20000, max_add=3, adds=[30000,20000,10000], cooldown=2, dip_thr=-0.03, bounce_mult=1.0):
    df = compute_indicators(df)
    cash = init_cash - base_pos; position = base_pos
    add_count=0; add_costs=[]; add_used=[]; cd=0; trades=[]; equity=[]
    for i in range(60, len(df)):
        r=df.iloc[i]; p=df.iloc[i-1]
        if position>0 and p['close']>0: position*=(r['close']/p['close'])
        if cd>0: cd-=1
        pct=r.get('pct_change',0); ma20=r.get('ma20',0) if not pd.isna(r.get('ma20',0)) else 0; ma60=r.get('ma60',0) if not pd.isna(r.get('ma60',0)) else 0
        is_down=ma20>0 and ma60>0 and ma20<ma60
        equity.append({'day':i,'equity':cash+position,'close':r['close']})
        if is_down:
            if pct is not None and not pd.isna(pct) and pct<dip_thr and add_count<max_add and cd==0 and cash>=adds[add_count]:
                amt=adds[add_count]; cash-=amt; position+=amt; add_costs.append(r['close']); add_used.append(amt); add_count+=1; cd=cooldown
                trades.append({'day':i,'action':'DIP_ADD','price':r['close'],'amount':amt,'pos':position,'n':add_count})
            elif add_count>0 and add_costs:
                avg=np.mean(add_costs); atr=r.get('atr',0) if not pd.isna(r.get('atr',0)) else 0
                rsi=r.get('rsi',50) if not pd.isna(r.get('rsi',50)) else 50
                if (atr>0 and r['close']>avg+bounce_mult*atr) or rsi>70:
                    sell=min(sum(add_used),max(0,position-base_pos))
                    if sell>100: cash+=sell; position-=sell; trades.append({'day':i,'action':'BOUNCE','price':r['close'],'amount':sell,'pos':position}); add_count=0; add_costs=[]; add_used=[]
    eq=pd.DataFrame(equity); peak=eq['equity'].cummax(); dd=((eq['equity']-peak)/peak).min()
    final=cash+position; tr=(final-init_cash)/init_cash; br=(df.iloc[-1]['close']/df.iloc[60]['close']-1)
    dip=[t for t in trades if 'ADD' in t['action']]; bn=[t for t in trades if 'BOUNCE' in t['action']]
    return {'ret':tr,'bench':br,'alpha':tr-br,'mdd':dd,'trades':len(trades),'dip':len(dip),'bn':len(bn),'final':final}

tests=[('600519','贵州茅台(下跌)','20220101','20221231'),('515880','科创50(上涨)','20230101','20241231'),('159915','创业板(震荡)','20210101','20221231'),('510300','沪深300(综合)','20200101','20241231')]
print("H: 冷却2天 | 底仓2万 | 跌>3%加仓 | 1×ATR卖 | [30k,20k,10k]")
print("="*80)
alphas=[]; mdds=[]; tcs=[]
for c,n,s,e in tests:
    try:
        df=fetch_stock_data(c,s,e); r=run(df)
        alphas.append(r['alpha']); mdds.append(r['mdd']); tcs.append(r['trades'])
        print(f"{n:<20} Alpha:{r['alpha']:>+8.2%}  回撤:{r['mdd']:>8.2%}  交易:{r['trades']:>3}")
    except Exception as ex: print(f"{n:<20} 错误: {ex}")
print(f"{'─'*80}\n平均Alpha:{np.mean(alphas):>+.2%} | 平均回撤:{np.mean(mdds):.2%} | 平均交易:{np.mean(tcs):.1f}")
