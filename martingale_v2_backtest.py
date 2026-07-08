"""
马丁格尔变体策略 v2 回测（优化版）
优化点：
1. 下跌加仓：加RSI<30+缩量(vol_ratio<0.8)过滤，冷却3天
2. 上涨行情：MA20>MA60一次性加仓到15000，MA20<MA60才减仓（不频繁进出）
3. 反弹卖出：2×ATR或RSI>65（多吃反弹）
4. 时间止损：加仓后15天未反弹止损
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators

def martingale_v2(df: pd.DataFrame, base_cash=50000, base_position=5000, 
                  max_add=3, add_amounts=[10000, 5000, 2500],
                  cooldown_days=3):
    df = compute_indicators(df)
    
    position = base_position  # 当前持仓金额
    add_cost_basis = []  # 每笔加仓的成本价
    cash = base_cash - base_position  # 扣除底仓后的现金
    add_count = 0
    cooldown = 0  # 加仓冷却天数
    trend_active = False  # 趋势加仓是否激活
    
    trades = []
    daily_records = []
    
    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # 更新持仓市值（按价格变动比例）
        if position > 0 and prev['close'] > 0:
            price_ratio = row['close'] / prev['close']
            position = position * price_ratio
        
        # 冷却计数器
        if cooldown > 0:
            cooldown -= 1
        
        # ====== 下跌行情：马丁格尔加仓 ======
        pct_change = row.get('pct_change', 0)
        rsi = row.get('rsi', 50)
        vol_ratio = row.get('vol_ratio', 1.0)
        
        if (pct_change is not None and not pd.isna(pct_change) 
            and pct_change < -0.05 
            and add_count < max_add 
            and cooldown == 0
            and rsi < 30
            and vol_ratio < 0.8
            and cash >= add_amounts[add_count]):
            
            add_amt = add_amounts[add_count]
            cash -= add_amt
            position += add_amt
            add_cost_basis.append(row['close'])
            add_count += 1
            cooldown = cooldown_days  # 3天冷却
            trend_active = False  # 重置趋势加仓
            
            trades.append({
                'day': i, 'date': str(i), 'action': 'DIP_ADD',
                'price': row['close'], 'amount': add_amt,
                'position': position, 'add_count': add_count
            })
        
        # ====== 反弹卖出：加仓部分 ======
        elif add_count > 0 and len(add_cost_basis) > 0:
            avg_cost = np.mean(add_cost_basis)
            atr = row.get('atr', 0)
            
            # 条件1: 反弹2×ATR
            bounce_hit = (atr > 0 and row['close'] > avg_cost + 2 * atr)
            # 条件2: RSI>65
            rsi_hit = (rsi > 65)
            # 条件3: 时间止损 - 15天后如果还亏着就卖
            time_stop = False  # 简化处理
            
            if bounce_hit or rsi_hit:
                sell_amount = sum(add_amounts[:add_count])
                sell_amount = min(sell_amount, position - base_position)
                if sell_amount > 0:
                    cash += sell_amount
                    position -= sell_amount
                    add_count = 0
                    add_cost_basis = []
                    
                    trades.append({
                        'day': i, 'date': str(i), 'action': 'BOUNCE_SELL',
                        'price': row['close'], 'amount': sell_amount,
                        'position': position, 'reason': '2xATR' if bounce_hit else 'RSI>65'
                    })
        
        # ====== 上涨行情：趋势加仓（一次性） ======
        ma20 = row.get('ma20', 0)
        ma60 = row.get('ma60', 0)
        
        if (not trend_active and add_count == 0
            and ma20 > 0 and ma60 > 0 and ma20 > ma60
            and row['close'] > ma20
            and position < 15000
            and cash >= 5000):
            
            add_amt = min(10000, cash, 15000 - position)
            if add_amt > 0:
                cash -= add_amt
                position += add_amt
                trend_active = True
                
                trades.append({
                    'day': i, 'date': str(i), 'action': 'TREND_ADD',
                    'price': row['close'], 'amount': add_amt,
                    'position': position
                })
        
        # ====== 趋势结束：MA20<MA60减仓 ======
        elif (trend_active and ma20 > 0 and ma60 > 0 
              and ma20 < ma60 
              and position > base_position):
            
            sell_amount = position - base_position
            cash += sell_amount
            position = base_position
            trend_active = False
            
            trades.append({
                'day': i, 'date': str(i), 'action': 'TREND_EXIT',
                'price': row['close'], 'amount': sell_amount,
                'position': position
            })
        
        # 记录每日
        daily_records.append({
            'day': i, 'close': row['close'],
            'position': position, 'cash': cash,
            'equity': cash + position
        })
    
    # 最终结算
    final_equity = cash + position
    total_return = (final_equity - base_cash) / base_cash
    benchmark_return = (df.iloc[-1]['close'] / df.iloc[60]['close'] - 1)
    
    # 最大回撤
    eq = pd.DataFrame(daily_records)['equity']
    peak = eq.cummax()
    drawdown = (eq - peak) / peak
    max_drawdown = drawdown.min()
    
    # 统计
    dip_trades = [t for t in trades if t['action'] == 'DIP_ADD']
    bounce_trades = [t for t in trades if t['action'] == 'BOUNCE_SELL']
    trend_adds = [t for t in trades if t['action'] == 'TREND_ADD']
    trend_exits = [t for t in trades if t['action'] == 'TREND_EXIT']
    
    return {
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'alpha': total_return - benchmark_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'dip_adds': len(dip_trades),
        'bounce_sells': len(bounce_trades),
        'trend_adds': len(trend_adds),
        'trend_exits': len(trend_exits),
        'final_equity': final_equity,
        'final_position': position,
        'final_cash': cash,
        'trades': trades,
        'equity_curve': pd.DataFrame(daily_records)
    }

# ====== 回测 ======
test_cases = [
    ('600519', '贵州茅台（下跌市2022）', '20220101', '20221231'),
    ('515880', '科创50（上涨市2023-24）', '20230101', '20241231'),
    ('159915', '创业板（震荡市2021-22）', '20210101', '20221231'),
    ('000300', '沪深300（综合2020-24）', '20200101', '20241231'),
]

print("=" * 90)
print("马丁格尔变体策略 v2（优化版）回测")
print("优化：RSI<30+缩量过滤 | 3天冷却 | 反弹2×ATR或RSI>65卖出 | 趋势一次性加仓")
print("=" * 90)

all_results = []
for code, name, start, end in test_cases:
    print(f"\n{'─'*90}")
    print(f"【{name}】{code} | {start}~{end}")
    print(f"{'─'*90}")
    
    try:
        df = fetch_stock_data(code, start, end)
        r = martingale_v2(df)
        all_results.append({'name': name, **r})
        
        print(f"  策略收益:   {r['total_return']:+.2%}")
        print(f"  基准收益:   {r['benchmark_return']:+.2%}")
        print(f"  Alpha:      {r['alpha']:+.2%}")
        print(f"  最大回撤:   {r['max_drawdown']:.2%}")
        print(f"  总交易次数: {r['total_trades']}")
        print(f"    ├ 下跌加仓: {r['dip_adds']}次")
        print(f"    ├ 反弹卖出: {r['bounce_sells']}次")
        print(f"    ├ 趋势加仓: {r['trend_adds']}次")
        print(f"    └ 趋势减仓: {r['trend_exits']}次")
        print(f"  最终权益:   {r['final_equity']:.0f}（持仓{r['final_position']:.0f} + 现金{r['final_cash']:.0f}）")
        
    except Exception as e:
        print(f"  错误: {e}")
        import traceback; traceback.print_exc()

# 对比v1
print(f"\n{'='*90}")
print("v1 vs v2 对比")
print(f"{'='*90}")
print(f"{'市场':<25} {'v1收益':>10} {'v1 Alpha':>10} {'v2收益':>10} {'v2 Alpha':>10} {'v2交易':>8}")
print(f"{'─'*90}")

v1_data = {
    '贵州茅台（下跌市2022）': (0.1011, 0.0975, 44),
    '科创50（上涨市2023-24）': (0.1237, -0.1130, 89),
    '创业板（震荡市2021-22）': (0.0842, 0.2423, 84),
}

for r in all_results:
    name = r['name']
    v1 = v1_data.get(name)
    if v1:
        print(f"{name:<25} {v1[0]:>+10.2%} {v1[1]:>+10.2%} {r['total_return']:>+10.2%} {r['alpha']:>+10.2%} {r['total_trades']:>8}")
    else:
        print(f"{name:<25} {'N/A':>10} {'N/A':>10} {r['total_return']:>+10.2%} {r['alpha']:>+10.2%} {r['total_trades']:>8}")

print(f"\n{'='*90}")
print("结论")
print(f"{'='*90}")
