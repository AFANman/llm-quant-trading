"""
马丁格尔变体策略回测
下跌行情：底仓5000，单日跌>5%加仓（最多3次：10000→5000→2500），反弹1.5×ATR卖出
上涨行情：MA20>MA60时加仓到15000，跌破MA20减仓回5000
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators

def martingale_backtest(df: pd.DataFrame, base_position=5000, max_add=3, add_amounts=[10000, 5000, 2500]):
    """
    马丁格尔变体回测
    """
    df = compute_indicators(df)
    
    # 状态变量
    position = base_position  # 当前持仓金额
    cash = 50000  # 初始资金
    add_count = 0  # 已加仓次数
    add_prices = []  # 加仓价格记录
    trades = []
    equity_curve = []
    
    for i in range(60, len(df)):  # 从第60天开始（等MA60计算出来）
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        date = df.index[i]
        
        # 当前权益
        current_equity = cash + position * (row['close'] / prev_row['close'] - 1) if position > 0 else cash
        equity_curve.append({'date': date, 'equity': current_equity, 'position': position})
        
        # 下跌行情逻辑：单日跌>5%触发加仓
        pct_change = row['pct_change']
        if pct_change < -0.05 and add_count < max_add:
            # 加仓
            add_amount = add_amounts[add_count]
            if cash >= add_amount:
                cash -= add_amount
                position += add_amount
                add_count += 1
                add_prices.append(row['close'])
                trades.append({
                    'date': date,
                    'action': 'ADD',
                    'price': row['close'],
                    'amount': add_amount,
                    'total_position': position
                })
        
        # 反弹卖出逻辑：反弹1.5×ATR
        elif add_count > 0 and row['atr'] > 0:
            bounce_threshold = add_prices[-1] * (1 + 1.5 * row['atr'] / add_prices[-1])
            if row['close'] > bounce_threshold:
                # 卖出加仓部分
                sell_amount = sum(add_amounts[:add_count])
                if position >= sell_amount:
                    cash += sell_amount
                    position -= sell_amount
                    trades.append({
                        'date': date,
                        'action': 'SELL_BOUNCE',
                        'price': row['close'],
                        'amount': sell_amount,
                        'remaining_position': position
                    })
                    add_count = 0
                    add_prices = []
        
        # 上涨行情逻辑：MA20>MA60时加仓
        elif row['ma20'] > row['ma60'] and position < 15000 and cash >= 5000:
            # 趋势加仓
            add_amount = min(10000, cash)
            cash -= add_amount
            position += add_amount
            trades.append({
                'date': date,
                'action': 'TREND_ADD',
                'price': row['close'],
                'amount': add_amount,
                'total_position': position
            })
        
        # 趋势结束减仓：跌破MA20
        elif row['close'] < row['ma20'] and position > base_position:
            # 减仓回底仓
            sell_amount = position - base_position
            cash += sell_amount
            position = base_position
            trades.append({
                'date': date,
                'action': 'TREND_EXIT',
                'price': row['close'],
                'amount': sell_amount,
                'remaining_position': position
            })
            add_count = 0
            add_prices = []
    
    # 计算最终权益
    final_equity = cash + position * (df.iloc[-1]['close'] / df.iloc[60]['close'])
    total_return = (final_equity - 50000) / 50000
    
    # 基准收益（买入持有）
    benchmark_return = (df.iloc[-1]['close'] / df.iloc[60]['close'] - 1)
    
    return {
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'alpha': total_return - benchmark_return,
        'trades': trades,
        'equity_curve': pd.DataFrame(equity_curve),
        'final_position': position,
        'final_cash': cash
    }

# 测试三个市场
test_cases = [
    ('600519', '贵州茅台（下跌市）', '20220101', '20221231'),
    ('515880', '科创50（上涨市）', '20230101', '20241231'),
    ('159915', '创业板（震荡市）', '20210101', '20221231')
]

print("=" * 80)
print("马丁格尔变体策略回测")
print("=" * 80)

for code, name, start, end in test_cases:
    print(f"\n【{name}】{start}-{end}")
    print("-" * 80)
    
    try:
        df = fetch_stock_data(code, start, end)
        result = martingale_backtest(df)
        
        print(f"策略收益: {result['total_return']:.2%}")
        print(f"基准收益: {result['benchmark_return']:.2%}")
        print(f"Alpha: {result['alpha']:.2%}")
        print(f"交易次数: {len(result['trades'])}")
        print(f"最终持仓: {result['final_position']:.0f}")
        print(f"最终现金: {result['final_cash']:.0f}")
        
        if result['trades']:
            print(f"\n交易记录（前5笔）:")
            for t in result['trades'][:5]:
                print(f"  {t['date'].strftime('%Y-%m-%d')} | {t['action']:12s} | 价格:{t['price']:.2f} | 金额:{t['amount']:.0f}")
    except Exception as e:
        print(f"错误: {e}")

print("\n" + "=" * 80)
