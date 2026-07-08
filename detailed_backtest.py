"""
详细交易记录回测 — 严格计算每笔交易的真实利润
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd
from datetime import datetime

# 交易成本
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.001
SLIPPAGE = 0.001

def calc_buy_cost(amount):
    commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
    slippage_cost = amount * SLIPPAGE
    return commission + slippage_cost

def calc_sell_cost(amount):
    commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
    stamp_tax = amount * STAMP_TAX
    slippage_cost = amount * SLIPPAGE
    return commission + stamp_tax + slippage_cost

def detailed_backtest(df, init_cash=100000, base_amt=8000, dip_pct=0.07, 
                      tp_pct=0.08, mults=None, max_add=4):
    """详细记录每笔交易"""
    df = compute_indicators(df)
    if mults is None:
        mults = [1, 3, 5, 7]
    
    cash = init_cash
    shares = 0  # 持仓股数
    total_invested = 0  # 当前持仓的总投入成本
    add_count = 0
    
    detailed_trades = []
    current_position = None  # 当前持仓记录
    
    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']
        date = row['date']
        
        # 初始建仓
        if i == 60 and shares == 0:
            buy_cost = calc_buy_cost(base_amt)
            cash -= (base_amt + buy_cost)
            shares = base_amt / price
            total_invested = base_amt
            
            current_position = {
                'open_date': date,
                'open_price': price,
                'shares': shares,
                'total_invested': base_amt,
                'avg_cost': price,
                'add_count': 0,
                'trades': [{
                    'date': date, 'action': 'BUY_BASE', 
                    'price': price, 'amount': base_amt, 'shares': shares,
                    'cost': buy_cost
                }]
            }
            add_count = 0
        
        elif shares > 0:
            pnl = price / current_position['avg_cost'] - 1
            
            # 止盈
            if add_count > 0 and pnl >= tp_pct:
                sell_value = shares * price
                sell_cost = calc_sell_cost(sell_value)
                net_proceeds = sell_value - sell_cost
                
                # 计算利润
                profit = net_proceeds - total_invested
                profit_pct = profit / total_invested * 100
                
                # 记录完整交易
                current_position['close_date'] = date
                current_position['close_price'] = price
                current_position['sell_value'] = sell_value
                current_position['sell_cost'] = sell_cost
                current_position['net_proceeds'] = net_proceeds
                current_position['profit'] = profit
                current_position['profit_pct'] = profit_pct
                current_position['actual_pnl'] = pnl * 100
                
                detailed_trades.append(current_position.copy())
                
                cash += net_proceeds
                shares = 0
                total_invested = 0
                add_count = 0
                current_position = None
                
                # 重新建仓
                if i < len(df) - 5 and cash >= base_amt:
                    buy_cost = calc_buy_cost(base_amt)
                    cash -= (base_amt + buy_cost)
                    shares = base_amt / price
                    total_invested = base_amt
                    
                    current_position = {
                        'open_date': date,
                        'open_price': price,
                        'shares': shares,
                        'total_invested': base_amt,
                        'avg_cost': price,
                        'add_count': 0,
                        'trades': [{
                            'date': date, 'action': 'BUY_NEW',
                            'price': price, 'amount': base_amt, 'shares': shares,
                            'cost': buy_cost
                        }]
                    }
            
            # 加仓
            elif pnl <= -dip_pct * (add_count + 1) and add_count < max_add:
                amt = base_amt * mults[add_count]
                if cash >= amt:
                    buy_cost = calc_buy_cost(amt)
                    cash -= (amt + buy_cost)
                    
                    new_shares = amt / price
                    shares += new_shares
                    
                    # 更新均价
                    old_cost = current_position['avg_cost'] * current_position['shares']
                    new_cost = amt
                    current_position['shares'] = shares
                    current_position['avg_cost'] = (old_cost + new_cost) / shares
                    current_position['total_invested'] += amt
                    current_position['add_count'] += 1
                    
                    current_position['trades'].append({
                        'date': date, 'action': f'ADD_{add_count + 1}',
                        'price': price, 'amount': amt, 'shares': new_shares,
                        'cost': buy_cost
                    })
                    
                    add_count += 1
                    total_invested = current_position['total_invested']
    
    # 如果还有持仓，计算未实现盈亏
    if shares > 0 and current_position:
        final_price = df.iloc[-1]['close']
        unrealized_value = shares * final_price
        unrealized_pnl = unrealized_value - total_invested
        unrealized_pct = (final_price / current_position['avg_cost'] - 1) * 100
        
        current_position['status'] = 'OPEN'
        current_position['unrealized_value'] = unrealized_value
        current_position['unrealized_pnl'] = unrealized_pnl
        current_position['unrealized_pct'] = unrealized_pct
        detailed_trades.append(current_position)
    
    return detailed_trades

# 跑2026H1（交易最多的时段）
tp_pct = 0.08
df = fetch_stock_data('515880', '20250701', '20260702')
trades = detailed_backtest(df)

print(f"2026H1 详细交易记录 — 科创50稳健马丁格尔")
print(f"参数: 底仓8000, 间距7%, 止盈8%, 翻倍1-3-5-7")
print(f"{'='*150}\n")

total_profit = 0
total_cost = 0

for idx, t in enumerate(trades, 1):
    status = t.get('status', 'CLOSED')
    open_date = str(t['open_date'])[:10]
    close_date = str(t.get('close_date', '持仓中'))[:10]
    
    print(f"第{idx}笔交易 [{status}]")
    print(f"  开仓: {open_date} @ {t['open_price']:.3f}")
    
    if status == 'CLOSED':
        print(f"  平仓: {close_date} @ {t['close_price']:.3f}")
    else:
        print(f"  当前: 持仓中")
    
    print(f"  持仓天数: {t.get('close_date', df.iloc[-1]['date']) - t['open_date']}")
    print(f"  加仓次数: {t['add_count']}")
    print(f"\n  交易明细:")
    
    for trade in t['trades']:
        date_str = str(trade['date'])[:10]
        print(f"    {date_str} | {trade['action']:<10} | {trade['amount']:>8.0f}元 @ {trade['price']:.3f} | "
              f"买入{trade['shares']:.2f}股 | 成本{trade['cost']:.1f}元")
        total_cost += trade['cost']
    
    print(f"\n  持仓汇总:")
    print(f"    总投入: {t['total_invested']:.0f}元")
    print(f"    持仓股数: {t['shares']:.2f}股")
    print(f"    平均成本: {t['avg_cost']:.3f}元")
    
    if status == 'CLOSED':
        print(f"    卖出市值: {t['sell_value']:.0f}元")
        print(f"    卖出成本: {t['sell_cost']:.1f}元")
        print(f"    净收入: {t['net_proceeds']:.0f}元")
        print(f"    实际涨幅: {t['actual_pnl']:.2f}% (目标{tp_pct*100:.0f}%)")
        print(f"    **利润: {t['profit']:.0f}元 ({t['profit_pct']:.2f}%)**")
        total_profit += t['profit']
    else:
        print(f"    当前市值: {t['unrealized_value']:.0f}元")
        print(f"    当前涨幅: {t['unrealized_pct']:.2f}%")
        print(f"    **未实现盈亏: {t['unrealized_pnl']:.0f}元**")
        total_profit += t['unrealized_pnl']
    
    print(f"{'─'*150}\n")

print(f"\n{'='*150}")
print(f"汇总")
print(f"{'─'*150}")
print(f"交易笔数: {len(trades)}")
print(f"累计利润: {total_profit:.0f}元")
print(f"累计成本: {total_cost:.0f}元")
print(f"净收益: {total_profit - total_cost:.0f}元")
print(f"收益率: {(total_profit - total_cost) / 100000 * 100:.2f}%")
print(f"{'='*150}")
