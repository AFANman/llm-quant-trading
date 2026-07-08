"""
参数测试 G — 马丁格尔自适应变体
参数：冷却=0天，上涨底仓=15000，加仓触发=跌>2%，反弹卖出=1.5×ATR，
      加仓金额=[25000,15000,10000]，最多加仓3次
逻辑：MA20<MA60=下跌→马丁格尔加仓；MA20>=MA60=上涨→持有不动。
初始现金=100000-15000=85000，position=15000。
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators


def martingale_param_g(df, init_cash=100000, base_pos=15000,
                       max_add=3, add_amounts=[25000, 15000, 10000],
                       cooldown=0, dip_threshold=-0.02, atr_multiplier=1.5):
    """
    马丁格尔自适应策略 - 参数G
    
    参数:
        init_cash: 初始总资金 (默认100000)
        base_pos: 上涨底仓 (默认15000)
        max_add: 最多加仓次数 (默认3)
        add_amounts: 每次加仓金额列表 (默认[25000,15000,10000])
        cooldown: 冷却天数 (默认0)
        dip_threshold: 加仓触发跌幅阈值 (默认-0.02即跌>2%)
        atr_multiplier: 反弹卖出ATR倍数 (默认1.5)
    """
    df = compute_indicators(df)
    n = len(df)
    
    cash = init_cash - base_pos  # 85000
    position = base_pos          # 15000
    add_count = 0
    add_costs = []
    add_amounts_used = []
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
        ma20 = r.get('ma20', 0) if not pd.isna(r.get('ma20', 0)) else 0
        ma60 = r.get('ma60', 0) if not pd.isna(r.get('ma60', 0)) else 0
        
        # 环境判断
        is_downtrend = (ma20 > 0 and ma60 > 0 and ma20 < ma60)
        is_uptrend = (ma20 > 0 and ma60 > 0 and ma20 >= ma60)
        
        total_eq = cash + position
        equity.append({'day': i, 'close': r['close'], 'equity': total_eq,
                       'position': position, 'cash': cash,
                       'regime': 'down' if is_downtrend else 'up'})
        
        # ===== 下跌环境：马丁格尔加仓 =====
        if is_downtrend:
            if (pct is not None and not pd.isna(pct)
                and pct < dip_threshold   # 跌>2%
                and add_count < max_add
                and cd == 0
                and cash >= add_amounts[add_count]):
                
                amt = add_amounts[add_count]
                cash -= amt
                position += amt
                add_costs.append(r['close'])
                add_amounts_used.append(amt)
                add_count += 1
                cd = cooldown  # 0天冷却
                
                trades.append({'day': i, 'action': 'DIP_ADD', 'price': r['close'],
                              'amount': amt, 'pos': position, 'add_n': add_count})
            
            # 反弹卖出加仓部分
            elif add_count > 0 and len(add_costs) > 0:
                avg_cost = np.mean(add_costs)
                atr = r.get('atr', 0) if not pd.isna(r.get('atr', 0)) else 0
                
                # 反弹1.5×ATR卖出
                bounce_atr = (atr > 0 and r['close'] > avg_cost + atr_multiplier * atr)
                
                if bounce_atr:
                    sell_amt = sum(add_amounts_used)
                    sell_amt = min(sell_amt, max(0, position - base_pos))
                    if sell_amt > 100:
                        cash += sell_amt
                        position -= sell_amt
                        trades.append({'day': i, 'action': 'BOUNCE_SELL', 'price': r['close'],
                                      'amount': sell_amt, 'pos': position,
                                      'reason': f'{atr_multiplier}xATR'})
                        add_count = 0
                        add_costs = []
                        add_amounts_used = []
        
        # ===== 上涨环境：持有不动 =====
        elif is_uptrend:
            pass
    
    # 最终
    final_eq = cash + position
    total_ret = (final_eq - init_cash) / init_cash
    bench_ret = (df.iloc[-1]['close'] / df.iloc[60]['close'] - 1)
    
    eq_df = pd.DataFrame(equity)
    peak = eq_df['equity'].cummax()
    max_dd = ((eq_df['equity'] - peak) / peak).min()
    
    dip = [t for t in trades if t['action'] == 'DIP_ADD']
    bounce = [t for t in trades if t['action'] == 'BOUNCE_SELL']
    
    return {
        'total_return': total_ret, 'benchmark_return': bench_ret,
        'alpha': total_ret - bench_ret, 'max_drawdown': max_dd,
        'total_trades': len(trades),
        'dip_adds': len(dip), 'bounce_sells': len(bounce),
        'final_equity': final_eq, 'trades': trades
    }


# ====== 回测 ======
tests = [
    ('600519', '贵州茅台(下跌2022)', '20220101', '20221231'),
    ('515880', '科创50(2023-24)', '20230101', '20241231'),
    ('159915', '创业板(震荡2021-22)', '20210101', '20221231'),
    ('510300', '沪深300(综合2020-24)', '20200101', '20241231'),
]

print("=" * 90)
print("参数测试 G — 马丁格尔自适应")
print("参数: 冷却=0天 | 底仓=15000 | 加仓触发=跌>2% | 反弹卖出=1.5×ATR")
print("      加仓金额=[25000,15000,10000] | 最多加仓3次")
print("逻辑: MA20<MA60=下跌→马丁格尔加仓 | MA20>=MA60=上涨→持有不动")
print("初始: 现金=85000, 持仓=15000, 总计=100000")
print("=" * 90)

results = []
for code, name, s, e in tests:
    print(f"\n{'─'*90}")
    print(f"【{name}】{code} | {s}~{e}")
    print(f"{'─'*90}")
    try:
        df = fetch_stock_data(code, s, e)
        r = martingale_param_g(df)
        results.append({'name': name, 'code': code, **r})
        
        print(f"  策略收益:   {r['total_return']:+.2%}")
        print(f"  基准收益:   {r['benchmark_return']:+.2%}")
        print(f"  Alpha:      {r['alpha']:+.2%}")
        print(f"  最大回撤:   {r['max_drawdown']:.2%}")
        print(f"  交易次数:   {r['total_trades']}（加仓{r['dip_adds']} + 反弹卖{r['bounce_sells']}）")
        print(f"  最终权益:   {r['final_equity']:.0f}")
        
        if r['trades']:
            print(f"\n  交易明细:")
            for t in r['trades']:
                extra = t.get('reason', t.get('add_n', ''))
                print(f"    Day{t['day']:>4d} | {t['action']:>12s} | {t['price']:>8.3f} | {t['amount']:>8.0f} | pos:{t['pos']:>8.0f} | {extra}")
    except Exception as ex:
        print(f"  错误: {ex}")
        import traceback
        traceback.print_exc()

# ====== 汇总 ======
print(f"\n\n{'='*90}")
print("汇总 — 参数测试G 各市场表现")
print(f"{'='*90}")
print(f"{'市场':<25} {'策略收益':>10} {'基准收益':>10} {'Alpha':>10} {'最大回撤':>10} {'交易次数':>8}")
print(f"{'─'*90}")

total_alpha = 0
total_trades = 0
for r in results:
    print(f"{r['name']:<25} {r['total_return']:>+10.2%} {r['benchmark_return']:>+10.2%} {r['alpha']:>+10.2%} {r['max_drawdown']:>10.2%} {r['total_trades']:>8}")
    total_alpha += r['alpha']
    total_trades += r['total_trades']

print(f"{'─'*90}")
avg_alpha = total_alpha / len(results) if results else 0
print(f"{'平均/合计':<25} {'':>10} {'':>10} {avg_alpha:>+10.2%} {'':>10} {total_trades:>8}")
print(f"{'='*90}")
