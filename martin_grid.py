"""
传统马丁格尔参数网格搜索 — 科创50
底仓: 2000, 3000, 5000, 8000
间距: 3%, 5%, 7%, 10%
止盈: 5%, 8%, 10%, 15%
翻倍: 1-2-4-8, 1-1.5-2-3, 1-3-5-7
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd, numpy as np

def strat_martin(df, base_amt=3000, dip_pct=0.05, tp_pct=0.10, mults=None, max_add=4, init_cash=100000):
    df = compute_indicators(df)
    if mults is None:
        mults = [1, 2, 4, 8][:max_add]
    
    cash = init_cash - base_amt
    position = base_amt
    cost_basis = df.iloc[60]['close']
    add_count = 0
    equity = []

    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i - 1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        equity.append(cash + position)

        if position > 0:
            pnl = r['close'] / cost_basis - 1
            if pnl >= tp_pct:
                cash += position; position = 0; add_count = 0
                if i < len(df) - 5:
                    if cash >= base_amt:
                        cash -= base_amt; position = base_amt; cost_basis = r['close']
            elif pnl <= -dip_pct * (add_count + 1) and add_count < max_add and add_count < len(mults):
                amt = base_amt * mults[add_count]
                if cash >= amt:
                    cash -= amt; position += amt; add_count += 1

    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = eq.iloc[-1]
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    n_days = len(eq)
    ann_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    return {'ret': ret, 'bench': bench, 'alpha': ret - bench, 'mdd': dd, 'sharpe': sharpe, 'calmar': calmar}

base_amounts = [2000, 3000, 5000, 8000]
dip_pcts = [0.03, 0.05, 0.07, 0.10]
tp_pcts = [0.05, 0.08, 0.10, 0.15]
mult_sets = {
    '1-2-4-8': [1, 2, 4, 8],
    '1-1.5-2-3': [1, 1.5, 2, 3],
    '1-3-5-7': [1, 3, 5, 7],
}

years = [
    ('2023', '20220701', '20231231'),
    ('2024', '20230701', '20241231'),
    ('2025H1', '20240701', '20250701'),
    ('2026H1', '20250701', '20260702'),
]

dfs = {}
for yr, s, e in years:
    dfs[yr] = fetch_stock_data('515880', s, e)

results = []

for base in base_amounts:
    for dip in dip_pcts:
        for tp in tp_pcts:
            for mname, mults in mult_sets.items():
                alphas = []; sharpes = []; calmars = []; mdds = []
                for yr, s, e in years:
                    r = strat_martin(dfs[yr], base_amt=base, dip_pct=dip, tp_pct=tp, mults=mults)
                    alphas.append(r['alpha']); sharpes.append(r['sharpe'])
                    calmars.append(r['calmar']); mdds.append(r['mdd'])
                
                results.append({
                    'base': base, 'dip': dip, 'tp': tp, 'mults': mname,
                    'avg_alpha': np.mean(alphas), 'avg_sharpe': np.mean(sharpes),
                    'avg_calmar': np.mean(calmars), 'avg_mdd': np.mean(mdds),
                    'alphas': alphas,
                })

# Alpha最优
results.sort(key=lambda x: x['avg_alpha'], reverse=True)
print(f"参数网格搜索：共{len(results)}种组合")
print(f"\n{'='*130}")
print(f"按平均Alpha排序 Top 20:")
print(f"{'─'*130}")
hdr = f"{'#':<3} {'底仓':>5} {'间距':>5} {'止盈':>5} {'翻倍':>10} {'平均Alpha':>10} {'平均夏普':>8} {'平均卡尔马':>8} {'平均回撤':>8} | {'2023':>7} {'2024':>7} {'25H1':>7} {'26H1':>7}"
print(hdr)
print(f"{'─'*130}")
for i, r in enumerate(results[:20]):
    a = r['alphas']
    line = f"{i+1:<3} {r['base']:>5} {r['dip']:>4.0%} {r['tp']:>4.0%} {r['mults']:>10} {r['avg_alpha']:>+9.2%} {r['avg_sharpe']:>7.2f} {r['avg_calmar']:>7.2f} {r['avg_mdd']:>7.2%} | {a[0]:>+6.1%} {a[1]:>+6.1%} {a[2]:>+6.1%} {a[3]:>+6.1%}"
    print(line)

# 卡尔马最优
print(f"\n{'='*130}")
print(f"按卡尔马排序 Top 10:")
print(f"{'─'*130}")
print(hdr)
print(f"{'─'*130}")
results.sort(key=lambda x: x['avg_calmar'], reverse=True)
for i, r in enumerate(results[:10]):
    a = r['alphas']
    line = f"{i+1:<3} {r['base']:>5} {r['dip']:>4.0%} {r['tp']:>4.0%} {r['mults']:>10} {r['avg_alpha']:>+9.2%} {r['avg_sharpe']:>7.2f} {r['avg_calmar']:>7.2f} {r['avg_mdd']:>7.2%} | {a[0]:>+6.1%} {a[1]:>+6.1%} {a[2]:>+6.1%} {a[3]:>+6.1%}"
    print(line)

# 夏普最优
print(f"\n{'='*130}")
print(f"按夏普排序 Top 10:")
print(f"{'─'*130}")
print(hdr)
print(f"{'─'*130}")
results.sort(key=lambda x: x['avg_sharpe'], reverse=True)
for i, r in enumerate(results[:10]):
    a = r['alphas']
    line = f"{i+1:<3} {r['base']:>5} {r['dip']:>4.0%} {r['tp']:>4.0%} {r['mults']:>10} {r['avg_alpha']:>+9.2%} {r['avg_sharpe']:>7.2f} {r['avg_calmar']:>7.2f} {r['avg_mdd']:>7.2%} | {a[0]:>+6.1%} {a[1]:>+6.1%} {a[2]:>+6.1%} {a[3]:>+6.1%}"
    print(line)

# 回撤最小
print(f"\n{'='*130}")
print(f"按回撤最小排序 Top 10:")
print(f"{'─'*130}")
print(hdr)
print(f"{'─'*130}")
results.sort(key=lambda x: x['avg_mdd'], reverse=True)  # mdd is negative, less negative = better
for i, r in enumerate(results[:10]):
    a = r['alphas']
    line = f"{i+1:<3} {r['base']:>5} {r['dip']:>4.0%} {r['tp']:>4.0%} {r['mults']:>10} {r['avg_alpha']:>+9.2%} {r['avg_sharpe']:>7.2f} {r['avg_calmar']:>7.2f} {r['avg_mdd']:>7.2%} | {a[0]:>+6.1%} {a[1]:>+6.1%} {a[2]:>+6.1%} {a[3]:>+6.1%}"
    print(line)
