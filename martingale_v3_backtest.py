"""
马丁格尔变体 v3 — 修复v2过严问题
改动：
1. 下跌加仓：跌>3% OR RSI<30（放宽），冷却5天
2. 反弹卖出：RSI>60（放宽），或反弹回加仓价就卖
3. 趋势加仓：MA20>MA60 一次性加仓，MA20<MA60 才减仓（不频繁进出）
4. 加 510300（沪深300ETF）作为第4个测试
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators

def martingale_v3(df, init_cash=50000, base_pos=5000,
                  max_add=3, add_amounts=[10000, 5000, 2500],
                  cooldown=5):
    df = compute_indicators(df)
    n = len(df)
    
    cash = init_cash - base_pos  # 底仓已投入
    position = base_pos
    add_count = 0
    add_costs = []  # 加仓成本价
    add_amounts_used = []
    trend_on = False
    cd = 0  # 冷却计数
    
    trades = []
    equity = []
    
    for i in range(60, n):
        r = df.iloc[i]
        p = df.iloc[i-1]
        
        # 持仓市值更新
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        
        if cd > 0:
            cd -= 1
        
        pct = r.get('pct_change', 0)
        rsi = r.get('rsi', 50) if not pd.isna(r.get('rsi', 50)) else 50
        ma20 = r.get('ma20', 0) if not pd.isna(r.get('ma20', 0)) else 0
        ma60 = r.get('ma60', 0) if not pd.isna(r.get('ma60', 0)) else 0
        
        total_eq = cash + position
        equity.append({'day': i, 'close': r['close'], 'equity': total_eq, 
                       'position': position, 'cash': cash})
        
        # ===== 下跌加仓（放宽条件） =====
        if (pct is not None and not pd.isna(pct) 
            and (pct < -0.03 or rsi < 30)  # 跌>3% OR RSI<30
            and add_count < max_add 
            and cd == 0
            and cash >= add_amounts[add_count]):
            
            amt = add_amounts[add_count]
            cash -= amt
            position += amt
            add_costs.append(r['close'])
            add_amounts_used.append(amt)
            add_count += 1
            cd = cooldown
            
            trades.append({'day': i, 'action': 'DIP_ADD', 'price': r['close'],
                          'amount': amt, 'pos': position, 'add_n': add_count})
        
        # ===== 反弹卖出加仓部分 =====
        elif add_count > 0 and len(add_costs) > 0:
            avg_cost = np.mean(add_costs)
            atr = r.get('atr', 0) if not pd.isna(r.get('atr', 0)) else 0
            
            # 反弹回成本价 OR RSI>60 OR 反弹2×ATR
            bounce_to_cost = r['close'] >= avg_cost
            bounce_atr = (atr > 0 and r['close'] > avg_cost + 2 * atr)
            rsi_exit = rsi > 60
            
            if bounce_to_cost or bounce_atr or rsi_exit:
                sell_amt = sum(add_amounts_used)
                sell_amt = min(sell_amt, max(0, position - base_pos))
                if sell_amt > 100:
                    cash += sell_amt
                    position -= sell_amt
                    reason = '回本' if bounce_to_cost else ('2xATR' if bounce_atr else 'RSI>60')
                    trades.append({'day': i, 'action': 'BOUNCE_SELL', 'price': r['close'],
                                  'amount': sell_amt, 'pos': position, 'reason': reason})
                    add_count = 0
                    add_costs = []
                    add_amounts_used = []
        
        # ===== 趋势加仓（一次性） =====
        if (not trend_on and add_count == 0
            and ma20 > 0 and ma60 > 0 and ma20 > ma60
            and position < 12000
            and cash >= 5000):
            
            amt = min(7000, cash, 12000 - position)
            if amt > 0:
                cash -= amt
                position += amt
                trend_on = True
                trades.append({'day': i, 'action': 'TREND_ADD', 'price': r['close'],
                              'amount': amt, 'pos': position})
        
        # ===== 趋势减仓（MA20<MA60才减，一次性） =====
        elif (trend_on and ma20 > 0 and ma60 > 0 
              and ma20 < ma60
              and position > base_pos + 500):
            
            sell_amt = position - base_pos
            cash += sell_amt
            position = base_pos
            trend_on = False
            trades.append({'day': i, 'action': 'TREND_EXIT', 'price': r['close'],
                          'amount': sell_amt, 'pos': position})
    
    # 最终
    final_eq = cash + position
    total_ret = (final_eq - init_cash) / init_cash
    bench_ret = (df.iloc[-1]['close'] / df.iloc[60]['close'] - 1)
    
    eq_df = pd.DataFrame(equity)
    peak = eq_df['equity'].cummax()
    max_dd = ((eq_df['equity'] - peak) / peak).min()
    
    dip = [t for t in trades if t['action'] == 'DIP_ADD']
    bounce = [t for t in trades if t['action'] == 'BOUNCE_SELL']
    tadd = [t for t in trades if t['action'] == 'TREND_ADD']
    texit = [t for t in trades if t['action'] == 'TREND_EXIT']
    
    return {
        'total_return': total_ret, 'benchmark_return': bench_ret,
        'alpha': total_ret - bench_ret, 'max_drawdown': max_dd,
        'total_trades': len(trades),
        'dip_adds': len(dip), 'bounce_sells': len(bounce),
        'trend_adds': len(tadd), 'trend_exits': len(texit),
        'final_equity': final_eq, 'trades': trades
    }

# ====== 回测 ======
tests = [
    ('600519', '贵州茅台(下跌2022)', '20220101', '20221231'),
    ('515880', '科创50(上涨2023-24)', '20230101', '20241231'),
    ('159915', '创业板(震荡2021-22)', '20210101', '20221231'),
    ('510300', '沪深300(综合2020-24)', '20200101', '20241231'),
]

print("=" * 90)
print("马丁格尔变体 v3 回测")
print("优化：跌>3% OR RSI<30加仓 | 冷却5天 | 反弹回本/RSI>60卖 | 趋势一次性加减仓")
print("=" * 90)

results = []
for code, name, s, e in tests:
    print(f"\n{'─'*90}")
    print(f"【{name}】{code} | {s}~{e}")
    print(f"{'─'*90}")
    try:
        df = fetch_stock_data(code, s, e)
        r = martingale_v3(df)
        results.append({'name': name, **r})
        
        print(f"  策略收益:   {r['total_return']:+.2%}")
        print(f"  基准收益:   {r['benchmark_return']:+.2%}")
        print(f"  Alpha:      {r['alpha']:+.2%}")
        print(f"  最大回撤:   {r['max_drawdown']:.2%}")
        print(f"  交易次数:   {r['total_trades']}（下跌加仓{r['dip_adds']} + 反弹卖{r['bounce_sells']} + 趋势加{r['trend_adds']} + 趋势减{r['trend_exits']}）")
        print(f"  最终权益:   {r['final_equity']:.0f}")
        
        if r['trades']:
            print(f"\n  交易明细:")
            for t in r['trades']:
                extra = t.get('reason', t.get('add_n', ''))
                print(f"    Day{t['day']:>4d} | {t['action']:>12s} | {t['price']:>8.3f} | {t['amount']:>8.0f} | pos:{t['pos']:>8.0f} | {extra}")
    except Exception as ex:
        print(f"  错误: {ex}")

# v1 vs v2 vs v3
print(f"\n{'='*90}")
print("v1 → v2 → v3 对比")
print(f"{'='*90}")
print(f"{'市场':<25} {'v1 Alpha':>10} {'v2 Alpha':>10} {'v3 Alpha':>10} {'v3回撤':>10} {'v3交易':>8}")
print(f"{'─'*90}")

v1 = {'贵州茅台(下跌2022)': 0.0975, '科创50(上涨2023-24)': -0.1130, '创业板(震荡2021-22)': 0.2423}
v2 = {'贵州茅台(下跌2022)': 0.0084, '科创50(上涨2023-24)': -0.2322, '创业板(震荡2021-22)': 0.1473}

for r in results:
    n = r['name']
    a1 = v1.get(n, None)
    a2 = v2.get(n, None)
    print(f"{n:<25} {a1:>+10.2%} {a2:>+10.2%} {r['alpha']:>+10.2%} {r['max_drawdown']:>10.2%} {r['total_trades']:>8}")
