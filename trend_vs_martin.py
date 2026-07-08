"""
浮盈加仓 vs 马丁格尔 对比
逻辑：底仓买入，涨X%加仓一次，跌X%止损
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd, numpy as np

def trend_strategy(df, base_amt=30000, add_pct=0.05, add_amt=20000, 
                   stop_loss=0.15, max_add=3, init_cash=100000):
    """
    浮盈加仓策略
    base_amt: 底仓
    add_pct: 涨多少触发加仓
    add_amt: 每次加仓金额
    stop_loss: 止损线（从最高价回撤）
    max_add: 最多加仓次数
    """
    df = compute_indicators(df)
    
    cash = init_cash - base_amt
    position = base_amt
    cost_basis = df.iloc[60]['close']
    add_count = 0
    highest_price = df.iloc[60]['close']
    equity = []
    
    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']
        
        # 更新持仓市值
        if prev['close'] > 0 and position > 0:
            position *= (price / prev['close'])
        
        # 更新最高价
        if price > highest_price:
            highest_price = price
        
        equity.append(cash + position)
        
        # 止损（从最高价回撤超过阈值）
        drawdown = (highest_price - price) / highest_price
        if drawdown >= stop_loss and position > 0:
            cash += position
            position = 0
            add_count = 0
            continue
        
        # 浮盈加仓
        gain = (price / cost_basis - 1) if cost_basis > 0 else 0
        if gain >= add_pct * (add_count + 1) and add_count < max_add:
            if cash >= add_amt:
                cash -= add_amt
                position += add_amt
                add_count += 1
                # 更新成本
                total_invested = base_amt + add_amt * add_count
                # 简化：成本不变，只看盈亏
    
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = eq.iloc[-1]
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    ann_ret = (1 + ret) ** (252 / max(len(eq), 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    
    return {'ret': ret, 'bench': bench, 'alpha': ret - bench, 'mdd': dd, 
            'sharpe': sharpe, 'calmar': calmar}

def martin_strategy(df, base_amt=8000, dip_pct=0.07, tp_pct=0.08, 
                    mults=None, max_add=4, init_cash=100000):
    """马丁格尔（稳健版）"""
    df = compute_indicators(df)
    if mults is None:
        mults = [1, 3, 5, 7]
    
    cash = init_cash - base_amt
    position = base_amt
    cost_basis = df.iloc[60]['close']
    add_count = 0
    equity = []
    
    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        
        if position > 0 and prev['close'] > 0:
            position *= (row['close'] / prev['close'])
        equity.append(cash + position)
        
        if position > 0:
            pnl = row['close'] / cost_basis - 1
            if pnl >= tp_pct:
                cash += position
                position = 0
                add_count = 0
                if i < len(df) - 5:
                    if cash >= base_amt:
                        cash -= base_amt
                        position = base_amt
                        cost_basis = row['close']
            elif pnl <= -dip_pct * (add_count + 1) and add_count < max_add:
                amt = base_amt * mults[add_count]
                if cash >= amt:
                    cash -= amt
                    position += amt
                    add_count += 1
    
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = eq.iloc[-1]
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    ann_ret = (1 + ret) ** (252 / max(len(eq), 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    
    return {'ret': ret, 'bench': bench, 'alpha': ret - bench, 'mdd': dd,
            'sharpe': sharpe, 'calmar': calmar}

# 参数组合
trend_params = [
    {'base_amt': 20000, 'add_pct': 0.03, 'add_amt': 15000, 'stop_loss': 0.10, 'max_add': 4},
    {'base_amt': 30000, 'add_pct': 0.05, 'add_amt': 20000, 'stop_loss': 0.15, 'max_add': 3},
    {'base_amt': 40000, 'add_pct': 0.08, 'add_amt': 25000, 'stop_loss': 0.20, 'max_add': 2},
]

years = [
    ('2021-2022', '20210101', '20221231'),
    ('2023', '20220701', '20231231'),
    ('2024', '20230701', '20241231'),
    ('2025H1', '20240701', '20250701'),
    ('2026H1', '20250701', '20260702'),
]

dfs = {}
for yr, s, e in years:
    dfs[yr] = fetch_stock_data('515880', s, e)

print(f"科创50 策略对比")
print(f"{'='*130}")
print(f"\n{'策略':<50} {'平均Alpha':>10} {'平均夏普':>10} {'平均卡尔马':>10} {'平均回撤':>10}")
print(f"{'─'*130}")

# 马丁格尔
martin_results = []
for yr in dfs:
    r = martin_strategy(dfs[yr])
    martin_results.append(r)

avg_alpha = np.mean([r['alpha'] for r in martin_results])
avg_sharpe = np.mean([r['sharpe'] for r in martin_results])
avg_calmar = np.mean([r['calmar'] for r in martin_results])
avg_mdd = np.mean([r['mdd'] for r in martin_results])
print(f"{'马丁格尔(8k/7%/8%/1-3-5-7)':<50} {avg_alpha:>+9.2%} {avg_sharpe:>9.2f} {avg_calmar:>9.2f} {avg_mdd:>9.2%}")

# 浮盈加仓
for i, params in enumerate(trend_params):
    results = []
    for yr in dfs:
        r = trend_strategy(dfs[yr], **params)
        results.append(r)
    
    avg_alpha = np.mean([r['alpha'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    avg_calmar = np.mean([r['calmar'] for r in results])
    avg_mdd = np.mean([r['mdd'] for r in results])
    
    desc = f"浮盈加仓(底仓{params['base_amt']//1000}k/涨{params['add_pct']:.0%}加{params['add_amt']//1000}k/止损{params['stop_loss']:.0%})"
    print(f"{desc:<50} {avg_alpha:>+9.2%} {avg_sharpe:>9.2f} {avg_calmar:>9.2f} {avg_mdd:>9.2%}")

# 分年度详细对比
print(f"\n{'='*130}")
print(f"\n分年度对比")
print(f"{'─'*130}")
print(f"{'年份':<10} {'基准':>8} {'马丁格尔':>10} {'浮盈A':>10} {'浮盈B':>10} {'浮盈C':>10}")
print(f"{'─'*130}")

for yr in years:
    yr_name = yr[0]
    bench = dfs[yr_name].iloc[-1]['close'] / dfs[yr_name].iloc[60]['close'] - 1
    
    m_r = martin_strategy(dfs[yr_name])
    t_r = [trend_strategy(dfs[yr_name], **p) for p in trend_params]
    
    print(f"{yr_name:<10} {bench:>+7.2%} {m_r['alpha']:>+9.2%} {t_r[0]['alpha']:>+9.2%} {t_r[1]['alpha']:>+9.2%} {t_r[2]['alpha']:>+9.2%}")

# 回撤对比
print(f"\n{'='*130}")
print(f"\n最大回撤对比")
print(f"{'─'*130}")
print(f"{'年份':<10} {'马丁格尔':>10} {'浮盈A':>10} {'浮盈B':>10} {'浮盈C':>10}")
print(f"{'─'*130}")

for yr in years:
    yr_name = yr[0]
    m_r = martin_strategy(dfs[yr_name])
    t_r = [trend_strategy(dfs[yr_name], **p) for p in trend_params]
    print(f"{yr_name:<10} {m_r['mdd']:>9.2%} {t_r[0]['mdd']:>9.2%} {t_r[1]['mdd']:>9.2%} {t_r[2]['mdd']:>9.2%}")
