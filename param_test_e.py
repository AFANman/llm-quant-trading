"""
参数测试 E — 马丁格尔自适应回测
参数：冷却=3天，上涨底仓=25000，加仓触发=跌>5%，反弹卖出=1.5×ATR，加仓金额=[20000,10000,5000]

逻辑：
- MA20<MA60=下跌环境→马丁格尔加仓
- MA20>=MA60=上涨环境→持有不动
- 初始现金=100000-25000，position=25000
- 跌>5%时加仓（冷却3天），反弹1.5×ATR或RSI>70卖出加仓部分
- 最多加仓3次
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators


def martingale_param_e(df, init_cash=100000, base_pos=25000,
                        max_add=3, add_amounts=[20000, 10000, 5000],
                        cooldown=3):
    """
    马丁格尔自适应策略（参数E）
    冷却=3天, 上涨底仓=25000, 加仓触发=跌>5%, 反弹卖出=1.5×ATR, 加仓金额=[20000,10000,5000]
    """
    df = compute_indicators(df)
    n = len(df)

    cash = init_cash - base_pos  # 100000 - 25000 = 75000
    position = base_pos          # 25000
    add_count = 0
    add_costs = []
    add_amounts_used = []
    cd = 0  # 冷却计数

    trades = []
    equity = []

    for i in range(60, n):
        r = df.iloc[i]
        p = df.iloc[i - 1]

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
        equity.append({
            'day': i, 'close': r['close'], 'equity': total_eq,
            'position': position, 'cash': cash,
            'regime': 'down' if is_downtrend else 'up'
        })

        # ===== 下跌环境：马丁格尔加仓 =====
        if is_downtrend:
            if (pct is not None and not pd.isna(pct)
                    and pct < -0.05  # 跌>5%
                    and add_count < max_add
                    and cd == 0
                    and cash >= add_amounts[add_count]):

                amt = add_amounts[add_count]
                cash -= amt
                position += amt
                add_costs.append(r['close'])
                add_amounts_used.append(amt)
                add_count += 1
                cd = cooldown  # 3天冷却

                trades.append({
                    'day': i, 'action': 'DIP_ADD', 'price': r['close'],
                    'amount': amt, 'pos': position, 'add_n': add_count
                })

            # 反弹卖出加仓部分
            elif add_count > 0 and len(add_costs) > 0:
                avg_cost = np.mean(add_costs)
                atr = r.get('atr', 0) if not pd.isna(r.get('atr', 0)) else 0

                # 反弹1.5×ATR OR RSI>70
                bounce_atr = (atr > 0 and r['close'] > avg_cost + 1.5 * atr)
                rsi = r.get('rsi', 50) if not pd.isna(r.get('rsi', 50)) else 50
                rsi_exit = rsi > 70

                if bounce_atr or rsi_exit:
                    sell_amt = sum(add_amounts_used)
                    sell_amt = min(sell_amt, max(0, position - base_pos))
                    if sell_amt > 100:
                        cash += sell_amt
                        position -= sell_amt
                        reason = '1.5xATR' if bounce_atr else 'RSI>70'
                        trades.append({
                            'day': i, 'action': 'BOUNCE_SELL', 'price': r['close'],
                            'amount': sell_amt, 'pos': position, 'reason': reason
                        })
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
    ('515880', '科创50(上涨2023-24)', '20230101', '20241231'),
    ('159915', '创业板(震荡2021-22)', '20210101', '20221231'),
    ('510300', '沪深300(综合2020-24)', '20200101', '20241231'),
]

print("=" * 90)
print("参数测试 E — 马丁格尔自适应回测")
print("参数：冷却=3天 | 上涨底仓=25000 | 加仓触发=跌>5% | 反弹卖出=1.5×ATR")
print("加仓金额=[20000,10000,5000] | 总仓位=100000")
print("逻辑：MA20<MA60=下跌→马丁格尔加仓 | MA20>=MA60=上涨→底仓持有不动")
print("=" * 90)

results = []
for code, name, s, e in tests:
    print(f"\n{'─' * 90}")
    print(f"【{name}】{code} | {s}~{e}")
    print(f"{'─' * 90}")
    try:
        df = fetch_stock_data(code, s, e)
        r = martingale_param_e(df)
        results.append({'name': name, **r})

        print(f"  策略收益:   {r['total_return']:+.2%}")
        print(f"  基准收益:   {r['benchmark_return']:+.2%}")
        print(f"  Alpha:      {r['alpha']:+.2%}")
        print(f"  最大回撤:   {r['max_drawdown']:.2%}")
        print(f"  交易次数:   {r['total_trades']}（下跌加仓{r['dip_adds']} + 反弹卖{r['bounce_sells']}）")
        print(f"  最终权益:   {r['final_equity']:.0f}")

        if r['trades']:
            print(f"\n  交易明细:")
            for t in r['trades']:
                extra = t.get('reason', t.get('add_n', ''))
                print(f"    Day{t['day']:>4d} | {t['action']:>12s} | {t['price']:>8.3f} | {t['amount']:>8.0f} | pos:{t['pos']:>8.0f} | {extra}")
    except Exception as ex:
        print(f"  错误: {ex}")

# ====== 汇总 ======
print(f"\n{'=' * 90}")
print("汇总对比")
print(f"{'=' * 90}")
print(f"{'市场':<25} {'策略收益':>10} {'基准收益':>10} {'Alpha':>10} {'最大回撤':>10} {'交易次数':>8}")
print(f"{'─' * 90}")

for r in results:
    print(f"{r['name']:<25} {r['total_return']:>+10.2%} {r['benchmark_return']:>+10.2%} {r['alpha']:>+10.2%} {r['max_drawdown']:>10.2%} {r['total_trades']:>8}")

# 总结
print(f"\n{'=' * 90}")
if results:
    avg_alpha = np.mean([r['alpha'] for r in results])
    avg_dd = np.mean([r['max_drawdown'] for r in results])
    avg_trades = np.mean([r['total_trades'] for r in results])
    positive_alpha = sum(1 for r in results if r['alpha'] > 0)
    print(f"总结：参数E(冷却3天/底仓25000/加仓[20000,10000,5000]/反弹1.5×ATR) | "
          f"平均Alpha={avg_alpha:+.2%} | 平均回撤={avg_dd:.2%} | "
          f"平均交易={avg_trades:.0f}次 | {positive_alpha}/{len(results)}个市场跑赢基准")
else:
    print("总结：无有效回测结果")
print(f"{'=' * 90}")
