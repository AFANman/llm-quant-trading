"""
最近一个月模拟盘回放 — 科创50 稳健马丁格尔
从一个月前开始逐日执行，输出交易记录和最终状态
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
from datetime import datetime, timedelta

CODE = '515880'
INIT_CASH = 100000
BASE_AMT = 8000
DIP_PCT = 0.07
TP_PCT = 0.08
MULTS = [1, 3, 5, 7]
MAX_ADD = 4

# 拉取最近半年数据（留60天给指标预热，只回看最后一个月）
start = (datetime.now() - timedelta(days=200)).strftime('%Y%m%d')
end = datetime.now().strftime('%Y%m%d')
df = fetch_stock_data(CODE, start, end)
print(f"原始数据: {len(df)} 行")
df = compute_indicators(df)
print(f"计算指标后: {len(df)} 行")
if len(df) == 0:
    print("错误: 无数据")
    sys.exit(1)

# 找到一个月前的位置（约22个交易日）
n_days_back = min(22, len(df) - 1)
start_idx = max(61, len(df) - n_days_back)
if start_idx >= len(df):
    start_idx = len(df) - 2

print(f"回测区间: {df.iloc[start_idx]['date']} → {df.iloc[-1]['date']}")
print(f"共 {len(df) - start_idx} 个交易日")
print(f"{'='*100}")

# 初始化
cash = INIT_CASH - BASE_AMT
position_value = BASE_AMT
cost_basis = df.iloc[start_idx]['close']
add_count = 0
peak_since_buy = df.iloc[start_idx]['close']
trades = []

print(f"Day 0: {df.iloc[start_idx]['date']} | 初始建仓 {BASE_AMT}元 @ {cost_basis:.4f}")
print(f"  现金: {cash:.0f} | 持仓: {position_value:.0f} | 总值: {INIT_CASH:.0f}")
print(f"{'─'*100}")
print(f"{'日期':<12} {'价格':>8} {'操作':<12} {'金额':>10} {'现金':>10} {'持仓':>10} {'总值':>10} {'盈亏%':>8}")
print(f"{'─'*100}")

for i in range(start_idx + 1, len(df)):
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    price = row['close']
    date = row['date']
    
    # 更新持仓市值
    if prev['close'] > 0 and position_value > 0:
        position_value = position_value * (price / prev['close'])
    
    if price > peak_since_buy:
        peak_since_buy = price
    
    pnl_pct = (price / cost_basis - 1) if cost_basis > 0 else 0
    total = cash + position_value
    total_pnl = (total - INIT_CASH) / INIT_CASH * 100
    
    action = 'HOLD'
    amount = 0
    note = ''
    
    # 止盈
    if add_count > 0 and pnl_pct >= TP_PCT:
        sell_value = position_value - BASE_AMT
        if sell_value > 100:
            cash += sell_value
            position_value = BASE_AMT
            add_count = 0
            cost_basis = price
            peak_since_buy = price
            action = 'SELL_TP'
            amount = sell_value
            note = f'止盈+{pnl_pct:.1%}'
    
    # 加仓
    elif add_count < MAX_ADD:
        dip = -pnl_pct
        if dip >= DIP_PCT * (add_count + 1):
            amt = BASE_AMT * MULTS[add_count]
            if cash >= amt:
                cash -= amt
                position_value += amt
                add_count += 1
                action = 'ADD'
                amount = amt
                note = f'加仓第{add_count}次 跌{dip:.1%}'
    
    # 打印
    if action != 'HOLD' or i == start_idx + 1 or i == len(df) - 1:
        op_str = action
        if action == 'ADD':
            op_str = f'ADD({add_count})'
        total = cash + position_value
        total_pnl = (total - INIT_CASH) / INIT_CASH * 100
        print(f"{date:<12} {price:>7.4f} {op_str:<12} {amount:>9.0f} {cash:>9.0f} {position_value:>9.0f} {total:>9.0f} {total_pnl:>+7.2f}%")
    
    if action != 'HOLD':
        trades.append({'date': date, 'action': action, 'amount': amount, 'price': price, 'note': note})

# 最终状态
total = cash + position_value
total_pnl = (total - INIT_CASH) / INIT_CASH * 100
bench_pnl = (df.iloc[-1]['close'] / df.iloc[start_idx]['close'] - 1) * 100
alpha = total_pnl - bench_pnl

print(f"{'='*100}")
print(f"\n最终状态:")
print(f"  现金: {cash:.0f}")
print(f"  持仓: {position_value:.0f}")
print(f"  总值: {total:.0f}")
print(f"  累计盈亏: {total_pnl:+.2f}%")
print(f"  基准盈亏: {bench_pnl:+.2f}%")
print(f"  Alpha: {alpha:+.2f}%")
print(f"  加仓次数: {add_count}")
print(f"  交易次数: {len(trades)}")

print(f"\n交易记录:")
print(f"{'─'*80}")
for t in trades:
    print(f"  {t['date']} | {t['action']:<10} | {t['amount']:>8.0f}元 @ {t['price']:.4f} | {t['note']}")

# 回撤计算
equity = [INIT_CASH]
pos = BASE_AMT
c = INIT_CASH - BASE_AMT
cb = df.iloc[start_idx]['close']
ac = 0
for i in range(start_idx + 1, len(df)):
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    price = row['close']
    if prev['close'] > 0 and pos > 0:
        pos = pos * (price / prev['close'])
    
    pnl = (price / cb - 1) if cb > 0 else 0
    # 重放交易逻辑
    if ac > 0 and pnl >= TP_PCT:
        sv = pos - BASE_AMT
        if sv > 100:
            c += sv
            pos = BASE_AMT
            ac = 0
            cb = price
    elif ac < MAX_ADD:
        dip = -pnl
        if dip >= DIP_PCT * (ac + 1):
            amt = BASE_AMT * MULTS[ac]
            if c >= amt:
                c -= amt
                pos += amt
                ac += 1
    
    equity.append(c + pos)

import numpy as np
eq = np.array(equity)
peak = np.maximum.accumulate(eq)
dd = ((eq - peak) / peak)
max_dd = dd.min()
print(f"\n  最大回撤: {max_dd:.2%}")

# 逐日收益算夏普
daily_ret = np.diff(eq) / eq[:-1]
rf = 0.02 / 252
sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
print(f"  夏普比率: {sharpe:.2f}")
