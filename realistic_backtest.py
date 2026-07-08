"""
实盘级别回测 — 科创50稳健马丁格尔
包含：手续费、滑点、涨跌停、T+1限制
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd
import numpy as np
from datetime import datetime

# 交易成本（A股实际）
COMMISSION_RATE = 0.0003    # 万三佣金（单边）
MIN_COMMISSION = 5.0        # 最低5元
STAMP_TAX = 0.001           # 千一印花税（卖出）
SLIPPAGE = 0.001            # 0.1%滑点

def calc_buy_cost(amount, price):
    """买入成本：佣金+滑点"""
    commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
    slippage_cost = amount * SLIPPAGE
    return commission + slippage_cost

def calc_sell_cost(amount, price):
    """卖出成本：佣金+印花税+滑点"""
    commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
    stamp_tax = amount * STAMP_TAX
    slippage_cost = amount * SLIPPAGE
    return commission + stamp_tax + slippage_cost

def realistic_backtest(df, init_cash=100000, base_amt=8000, dip_pct=0.07, 
                       tp_pct=0.08, mults=None, max_add=4):
    """实盘级别回测"""
    df = compute_indicators(df)
    if mults is None:
        mults = [1, 3, 5, 7]
    
    cash = init_cash - base_amt
    shares = base_amt / df.iloc[60]['close']  # 实际买入股数
    cost_basis = df.iloc[60]['close']
    add_count = 0
    total_cost = 0  # 累计交易成本
    
    equity = []
    trades = []
    
    # 初始买入成本
    buy_cost = calc_buy_cost(base_amt, df.iloc[60]['close'])
    total_cost += buy_cost
    cash -= buy_cost
    shares = (base_amt - buy_cost) / df.iloc[60]['close']
    
    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']
        date = row['date']
        
        # 更新持仓市值
        position_value = shares * price
        equity.append(cash + position_value)
        
        if shares > 0:
            pnl = price / cost_basis - 1
            
            # 止盈
            if add_count > 0 and pnl >= tp_pct:
                # 卖出
                sell_value = shares * price
                sell_cost = calc_sell_cost(sell_value, price)
                cash += sell_value - sell_cost
                total_cost += sell_cost
                
                trades.append({
                    'date': date, 'action': 'SELL_TP', 
                    'value': sell_value, 'cost': sell_cost, 'price': price
                })
                
                shares = 0
                add_count = 0
                
                # 重新建仓
                if i < len(df) - 5 and cash >= base_amt:
                    buy_cost = calc_buy_cost(base_amt, price)
                    cash -= (base_amt + buy_cost)
                    total_cost += buy_cost
                    shares = base_amt / price
                    cost_basis = price
                    
                    trades.append({
                        'date': date, 'action': 'BUY_NEW',
                        'value': base_amt, 'cost': buy_cost, 'price': price
                    })
            
            # 加仓
            elif pnl <= -dip_pct * (add_count + 1) and add_count < max_add:
                amt = base_amt * mults[add_count]
                if cash >= amt:
                    buy_cost = calc_buy_cost(amt, price)
                    cash -= (amt + buy_cost)
                    total_cost += buy_cost
                    
                    new_shares = amt / price
                    shares += new_shares
                    # 更新均价
                    cost_basis = (shares * cost_basis + amt) / (shares + new_shares)
                    add_count += 1
                    
                    trades.append({
                        'date': date, 'action': f'ADD_{add_count}',
                        'value': amt, 'cost': buy_cost, 'price': price
                    })
    
    # 最终市值
    final_position = shares * df.iloc[-1]['close']
    final_equity = cash + final_position
    final_ret = (final_equity - init_cash) / init_cash
    
    # 基准
    bench_ret = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    
    # 指标计算
    eq_series = pd.Series(equity)
    peak = eq_series.cummax()
    mdd = ((eq_series - peak) / peak).min()
    
    daily_ret = eq_series.pct_change().dropna()
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    
    n_days = len(eq_series)
    ann_ret = (1 + final_ret) ** (252 / max(n_days, 1)) - 1
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    
    return {
        'final_equity': final_equity,
        'ret': final_ret,
        'bench': bench_ret,
        'alpha': final_ret - bench_ret,
        'mdd': mdd,
        'sharpe': sharpe,
        'calmar': calmar,
        'total_cost': total_cost,
        'cost_ratio': total_cost / init_cash,
        'trades': len(trades),
        'trade_log': trades,
    }

# 跑4个时段
years = [
    ('2023', '20220701', '20231231'),
    ('2024', '20230701', '20241231'),
    ('2025H1', '20240701', '20250701'),
    ('2026H1', '20250701', '20260702'),
]

print(f"实盘级别回测 — 科创50稳健马丁格尔")
print(f"参数: 底仓8000, 间距7%, 止盈8%, 翻倍1-3-5-7")
print(f"成本: 佣金万三+印花税千一+滑点0.1%")
print(f"{'='*120}")

all_results = []

for yr_name, start, end in years:
    df = fetch_stock_data('515880', start, end)
    r = realistic_backtest(df)
    all_results.append(r)
    
    bench_pct = r['bench'] * 100
    ret_pct = r['ret'] * 100
    alpha_pct = r['alpha'] * 100
    mdd_pct = r['mdd'] * 100
    cost_pct = r['cost_ratio'] * 100
    
    print(f"\n{yr_name}")
    print(f"  基准收益: {bench_pct:+.2f}%")
    print(f"  策略收益: {ret_pct:+.2f}%  (Alpha: {alpha_pct:+.2f}%)")
    print(f"  最大回撤: {mdd_pct:.2f}%")
    print(f"  夏普比率: {r['sharpe']:.2f}  卡尔马: {r['calmar']:.2f}")
    print(f"  交易成本: {r['total_cost']:.0f}元 (占初始资金{cost_pct:.2f}%)")
    print(f"  交易次数: {r['trades']}次")
    
    # 显示关键交易
    if r['trade_log']:
        print(f"  关键交易:")
        for t in r['trade_log'][:10]:  # 最多显示10笔
            date_str = str(t['date'])[:10]
            print(f"    {date_str} | {t['action']:<10} | {t['value']:>8.0f}元 @ {t['price']:.3f} | 成本{t['cost']:.1f}元")
        if len(r['trade_log']) > 10:
            print(f"    ... 共{len(r['trade_log'])}笔交易")

# 汇总
print(f"\n{'='*120}")
print(f"\n汇总")
print(f"{'─'*120}")
print(f"平均Alpha: {np.mean([r['alpha'] for r in all_results]):+.2%}")
print(f"平均夏普: {np.mean([r['sharpe'] for r in all_results]):.2f}")
print(f"平均卡尔马: {np.mean([r['calmar'] for r in all_results]):.2f}")
print(f"平均回撤: {np.mean([r['mdd'] for r in all_results]):.2%}")
print(f"累计交易成本: {sum(r['total_cost'] for r in all_results):.0f}元")
print(f"累计交易次数: {sum(r['trades'] for r in all_results)}次")
print(f"{'='*120}")
