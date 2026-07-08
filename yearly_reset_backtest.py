"""
科创50 每年重置10万 — 正确方式：拉完整数据，按自然年切片回测
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd
import numpy as np

COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.001
SLIPPAGE = 0.001

def calc_buy_cost(a): return max(a*COMMISSION_RATE, MIN_COMMISSION) + a*SLIPPAGE
def calc_sell_cost(a): return max(a*COMMISSION_RATE, MIN_COMMISSION) + a*STAMP_TAX + a*SLIPPAGE

def backtest_segment(df_seg, df_preheat, init_cash=100000, base_amt=8000,
                     dip_pct=0.07, tp_pct=0.08, mults=None, max_add=4):
    """对一段数据跑回测，df_preheat用于计算指标"""
    if mults is None: mults = [1,3,5,7]
    
    # 合并preheat + seg，计算指标
    combined = pd.concat([df_preheat, df_seg]).drop_duplicates(subset='date', keep='last')
    combined = combined.sort_values('date').reset_index(drop=True)
    combined = compute_indicators(combined)
    
    # 找到seg开始的index
    seg_start = combined[combined['date'] == df_seg.iloc[0]['date']].index[0]
    
    cash = init_cash; shares = 0; ti = 0; ac = 0; avg = 0; tc = 0
    rounds = 0; profits = []; log = []
    eq_list = []
    
    for i in range(seg_start, len(combined)):
        row = combined.iloc[i]; price = row['close']; date = row['date']
        
        if i == seg_start and shares == 0:
            c = calc_buy_cost(base_amt); tc += c; cash -= base_amt+c
            shares = base_amt/price; avg = price; ti = base_amt; ac = 0
            log.append(f"    {str(date)[:10]} 建仓 {base_amt}元@{price:.3f}")
        
        elif shares > 0:
            pnl = price/avg - 1
            if ac > 0 and pnl >= tp_pct:
                sv = shares*price; sc = calc_sell_cost(sv); tc += sc
                net = sv - sc; profit = net - ti
                profits.append(profit); rounds += 1
                log.append(f"    {str(date)[:10]} 止盈 投入{ti:,}→卖{sv:,.0f} 利润{profit:+,.0f}")
                cash += net; shares = 0; ti = 0; ac = 0; avg = 0
                if i < len(combined)-5 and cash >= base_amt:
                    c = calc_buy_cost(base_amt); tc += c; cash -= base_amt+c
                    shares = base_amt/price; avg = price; ti = base_amt; ac = 0
                    log.append(f"    {str(date)[:10]} 新建仓 {base_amt}元@{price:.3f}")
            elif pnl <= -dip_pct*(ac+1) and ac < max_add:
                amt = base_amt*mults[ac]
                if cash >= amt:
                    c = calc_buy_cost(amt); tc += c; cash -= amt+c
                    ns = amt/price; ov = avg*shares; shares += ns
                    avg = (ov+amt)/shares; ti += amt; ac += 1
                    log.append(f"    {str(date)[:10]} 加仓{ac} {amt:,}元@{price:.3f}")
        
        fp = shares*price if shares > 0 else 0
        eq_list.append(cash + fp)
    
    eq = pd.Series(eq_list)
    peak = eq.cummax()
    mdd = ((eq-peak)/peak).min() if len(eq) > 1 else 0
    fp = shares*combined.iloc[-1]['close'] if shares > 0 else 0
    fe = cash + fp
    ur = fp - ti if shares > 0 else 0
    ret = (fe - init_cash) / init_cash
    bench = combined.iloc[-1]['close'] / combined.iloc[seg_start]['close'] - 1
    dr = eq.pct_change().dropna()
    rf = 0.02/252
    sharpe = (dr.mean()-rf)/dr.std()*np.sqrt(252) if dr.std()>0 and len(dr)>10 else 0
    
    return {
        'fe': fe, 'ret': ret, 'bench': bench, 'alpha': ret-bench,
        'mdd': mdd, 'sharpe': sharpe, 'tc': tc, 'rounds': rounds,
        'profits': profits, 'ur': ur, 'log': log,
        'sp': combined.iloc[seg_start]['close'], 'ep': combined.iloc[-1]['close'],
        'sd': str(combined.iloc[seg_start]['date'])[:10],
        'ed': str(combined.iloc[-1]['date'])[:10],
    }

# ── 拉取3年+预热期数据 ──
df_all = fetch_stock_data('515880', '20210101', '20260708')
df_all['date'] = pd.to_datetime(df_all['date'])
print(f"数据: {len(df_all)}行, {str(df_all['date'].min())[:10]} → {str(df_all['date'].max())[:10]}")

# 半年度切分
segments = [
    ('2023H2', '2023-07-03', '2023-12-29'),
    ('2024H1', '2024-01-02', '2024-06-28'),
    ('2024H2', '2024-07-01', '2024-12-31'),
    ('2025H1', '2025-01-02', '2025-06-30'),
    ('2025H2', '2025-07-01', '2025-12-31'),
    ('2026H1', '2026-01-02', '2026-07-08'),
]

print(f"\n科创50 ETF (515880) 每半年重置10万 — 稳健马丁格尔")
print(f"参数: 底仓8000 | 间距7% | 止盈8% | 翻倍1-3-5-7")
print(f"{'='*110}\n")

all_rets = []; all_benches = []; all_alphas = []; all_pnls = []; all_mdds = []

for name, start_s, end_s in segments:
    start_dt = pd.to_datetime(start_s)
    end_dt = pd.to_datetime(end_s)
    
    seg = df_all[(df_all['date'] >= start_dt) & (df_all['date'] <= end_dt)].copy()
    preheat = df_all[(df_all['date'] >= pd.to_datetime('2022-01-01')) & (df_all['date'] < start_dt)].copy()
    
    if len(seg) < 20:
        print(f"  {name}: 数据不足（{len(seg)}行），跳过")
        continue
    
    r = backtest_segment(seg, preheat)
    pnl = r['ret'] * 100000
    all_rets.append(r['ret'])
    all_benches.append(r['bench'])
    all_alphas.append(r['alpha'])
    all_pnls.append(pnl)
    all_mdds.append(r['mdd'])
    
    print(f"  {name}（{r['sd']} → {r['ed']}，{len(seg)}天）")
    print(f"    价格: {r['sp']:.3f} → {r['ep']:.3f}（{r['bench']:+.2%}）")
    print(f"    策略: {r['ret']:>+7.2%}（{pnl:+,.0f}元）| Alpha: {r['alpha']:>+7.2%} | 回撤: {r['mdd']:>6.2%} | 夏普: {r['sharpe']:>5.2f}")
    print(f"    {r['rounds']}轮止盈 | 已实现{sum(r['profits']):+,.0f}元 | 浮盈{r['ur']:+,.0f}元 | 成本{r['tc']:.0f}元")
    for t in r['log']:
        print(t)
    print()

print(f"{'='*110}")
print(f"汇总（每半年重置10万）")
print(f"{'─'*110}")
print(f"  时段数:       {len(all_rets)}个半年")
print(f"  平均收益:     {np.mean(all_rets):>+8.2%}（{np.mean(all_pnls):+,.0f}元/半年）")
print(f"  平均Alpha:    {np.mean(all_alphas):>+8.2%}")
print(f"  平均回撤:     {np.mean(all_mdds):>8.2%}")
print(f"  正Alpha率:    {sum(1 for a in all_alphas if a>0)}/{len(all_alphas)}")
print(f"  正收益率:     {sum(1 for p in all_pnls if p>0)}/{len(all_pnls)}")
print(f"  累计盈亏:     {sum(all_pnls):>+10,.0f}元（每半年独立10万）")
print()

print(f"  逐半年明细:")
print(f"  {'时段':<10} {'策略收益':>10} {'盈亏金额':>10} {'基准':>10} {'Alpha':>10} {'回撤':>10}")
print(f"  {'─'*60}")
for i, (name, _, _) in enumerate(segments[:len(all_rets)]):
    print(f"  {name:<10} {all_rets[i]:>+10.2%} {all_pnls[i]:>+10,.0f} {all_benches[i]:>+10.2%} {all_alphas[i]:>+10.2%} {all_mdds[i]:>10.2%}")
print(f"  {'─'*60}")
print(f"  {'合计':<10} {'':>10} {sum(all_pnls):>+10,.0f} {'':>10} {'':>10}")
print(f"{'='*110}")
