"""
科创50 3年全量回测 — 10万总仓，看最终到手多少钱
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd
import numpy as np

COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.001
SLIPPAGE = 0.001

def calc_buy_cost(amount):
    return max(amount * COMMISSION_RATE, MIN_COMMISSION) + amount * SLIPPAGE

def calc_sell_cost(amount):
    return max(amount * COMMISSION_RATE, MIN_COMMISSION) + amount * STAMP_TAX + amount * SLIPPAGE

def full_backtest(df, init_cash=100000, base_amt=8000, dip_pct=0.07,
                  tp_pct=0.08, mults=None, max_add=4):
    df = compute_indicators(df)
    if mults is None:
        mults = [1, 3, 5, 7]

    cash = init_cash
    shares = 0
    total_invested = 0
    avg_cost = 0
    add_count = 0
    equity = []
    trade_log = []
    total_cost = 0
    round_trips = 0  # 完整交易轮次（开仓→止盈算一次）
    round_profits = []

    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']
        date = row['date']

        # 初始建仓
        if i == 60 and shares == 0:
            cost = calc_buy_cost(base_amt)
            total_cost += cost
            cash -= (base_amt + cost)
            shares = base_amt / price
            avg_cost = price
            total_invested = base_amt
            add_count = 0

            trade_log.append({
                'date': date, 'action': 'BUY_BASE',
                'price': price, 'amount': base_amt, 'shares': shares,
                'cost': cost, 'cash_after': cash
            })

        elif shares > 0:
            pnl = price / avg_cost - 1

            # 止盈
            if add_count > 0 and pnl >= tp_pct:
                sell_value = shares * price
                sell_cost = calc_sell_cost(sell_value)
                total_cost += sell_cost
                net = sell_value - sell_cost

                profit = net - total_invested
                round_profits.append(profit)
                round_trips += 1

                trade_log.append({
                    'date': date, 'action': 'SELL_TP',
                    'price': price, 'amount': sell_value, 'shares': shares,
                    'cost': sell_cost, 'cash_after': cash + net,
                    'invested': total_invested, 'profit': profit,
                    'round': round_trips
                })

                cash += net
                shares = 0
                total_invested = 0
                add_count = 0
                avg_cost = 0

                # 重新建仓
                if i < len(df) - 5 and cash >= base_amt:
                    cost = calc_buy_cost(base_amt)
                    total_cost += cost
                    cash -= (base_amt + cost)
                    shares = base_amt / price
                    avg_cost = price
                    total_invested = base_amt
                    add_count = 0

                    trade_log.append({
                        'date': date, 'action': 'BUY_NEW',
                        'price': price, 'amount': base_amt, 'shares': shares,
                        'cost': cost, 'cash_after': cash
                    })

            # 加仓
            elif pnl <= -dip_pct * (add_count + 1) and add_count < max_add:
                amt = base_amt * mults[add_count]
                if cash >= amt:
                    cost = calc_buy_cost(amt)
                    total_cost += cost
                    cash -= (amt + cost)

                    new_shares = amt / price
                    old_value = avg_cost * shares
                    shares += new_shares
                    avg_cost = (old_value + amt) / shares
                    total_invested += amt
                    add_count += 1

                    trade_log.append({
                        'date': date, 'action': f'ADD_{add_count}',
                        'price': price, 'amount': amt, 'shares': new_shares,
                        'cost': cost, 'cash_after': cash
                    })

        # 记录每日净值
        position_value = shares * price
        equity.append(cash + position_value)

    # 最终状态
    final_price = df.iloc[-1]['close']
    final_position = shares * final_price
    final_equity = cash + final_position
    unrealized = final_position - total_invested if shares > 0 else 0

    eq = pd.Series(equity)
    peak = eq.cummax()
    mdd = ((eq - peak) / peak).min()
    bench_ret = final_price / df.iloc[60]['close'] - 1
    strategy_ret = (final_equity - init_cash) / init_cash

    daily_ret = eq.pct_change().dropna()
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    n = len(eq)
    ann_ret = (1 + strategy_ret) ** (252 / max(n, 1)) - 1
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

    return {
        'init_cash': init_cash,
        'final_equity': final_equity,
        'strategy_ret': strategy_ret,
        'bench_ret': bench_ret,
        'alpha': strategy_ret - bench_ret,
        'mdd': mdd,
        'sharpe': sharpe,
        'calmar': calmar,
        'total_cost': total_cost,
        'round_trips': round_trips,
        'round_profits': round_profits,
        'unrealized': unrealized,
        'cash': cash,
        'position_value': final_position,
        'trade_log': trade_log,
        'start_date': str(df.iloc[60]['date'])[:10],
        'end_date': str(df.iloc[-1]['date'])[:10],
        'n_days': len(eq),
    }

# ── 主回测 ──
df = fetch_stock_data('515880', '20220701', '20260708')
r = full_backtest(df)

print(f"科创50 ETF (515880) 3年回测 — 稳健马丁格尔")
print(f"回测区间: {r['start_date']} → {r['end_date']} ({r['n_days']}个交易日)")
print(f"参数: 底仓8000, 间距7%, 止盈8%, 翻倍1-3-5-7")
print(f"成本: 佣金万三 + 印花税千一 + 滑点0.1%")
print(f"{'='*80}")

print(f"\n资金变化:")
print(f"  初始资金:   100,000元")
print(f"  最终资金:   {r['final_equity']:,.0f}元")
print(f"    现金:     {r['cash']:,.0f}元")
print(f"    持仓:     {r['position_value']:,.0f}元")
print(f"  策略收益:   {r['strategy_ret']:+.2%} ({r['final_equity']-100000:+,.0f}元)")
print(f"  基准收益:   {r['bench_ret']:+.2%}")
print(f"  Alpha:      {r['alpha']:+.2%}")

print(f"\n风险指标:")
print(f"  最大回撤:   {r['mdd']:.2%}")
print(f"  夏普比率:   {r['sharpe']:.2f}")
print(f"  卡尔马比率: {r['calmar']:.2f}")
print(f"  累计成本:   {r['total_cost']:.0f}元")

print(f"\n交易统计:")
print(f"  完整轮次:   {r['round_trips']}次（开仓→止盈→清仓）")
print(f"  已实现利润: {sum(r['round_profits']):+,.0f}元")
print(f"  未实现盈亏: {r['unrealized']:+,.0f}元")

if r['round_profits']:
    print(f"  每轮平均利润: {np.mean(r['round_profits']):+,.0f}元")
    print(f"  每轮最大利润: {max(r['round_profits']):+,.0f}元")
    print(f"  每轮最小利润: {min(r['round_profits']):+,.0f}元")

print(f"\n{'='*80}")
print(f"\n逐笔交易记录:")
print(f"{'─'*80}")

for t in r['trade_log']:
    date_str = str(t['date'])[:10]
    action = t['action']

    if 'SELL' in action:
        print(f"  {date_str} | {action:<10} | 投入{t['invested']:>7,.0f}元 → "
              f"卖出{t['amount']:>7,.0f}元 @ {t['price']:.3f} | "
              f"利润{t['profit']:>+6,.0f}元 | 第{t['round']}轮")
    else:
        print(f"  {date_str} | {action:<10} | {t['amount']:>7,.0f}元 @ {t['price']:.3f} | "
              f"现金{t['cash_after']:>8,.0f}元")

print(f"\n{'='*80}")
print(f"\n分年度收益:")
print(f"{'─'*80}")

# 分年度
year_ranges = [
    ('2023H2', '20230701', '20231231'),
    ('2024H1', '20240101', '20240701'),
    ('2024H2', '20240701', '20250101'),
    ('2025H1', '20250101', '20250701'),
    ('2026H1', '20250701', '20260708'),
]

for yr, s, e in year_ranges:
    yr_df = fetch_stock_data('515880', s, e)
    yr_r = full_backtest(yr_df)
    print(f"  {yr}: 策略{yr_r['strategy_ret']:>+7.2%} | 基准{yr_r['bench_ret']:>+7.2%} | "
          f"Alpha{yr_r['alpha']:>+7.2%} | 回撤{yr_r['mdd']:>6.2%} | {yr_r['round_trips']}轮")

print(f"\n{'='*80}")
