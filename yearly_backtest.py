"""
科创50 3年回测 — 按年切片看每年真实收益
连续运行，不重置资金
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

def run_backtest(df, init_cash=100000, base_amt=8000, dip_pct=0.07,
                 tp_pct=0.08, mults=None, max_add=4):
    df = compute_indicators(df)
    if mults is None:
        mults = [1, 3, 5, 7]

    cash = init_cash
    shares = 0
    total_invested = 0
    avg_cost = 0
    add_count = 0
    total_cost = 0
    round_trips = 0
    round_profits = []

    daily_records = []

    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']
        date = row['date']
        action = ''
        trade_detail = ''

        if i == 60 and shares == 0:
            cost = calc_buy_cost(base_amt)
            total_cost += cost
            cash -= (base_amt + cost)
            shares = base_amt / price
            avg_cost = price
            total_invested = base_amt
            add_count = 0
            action = '建仓'
            trade_detail = f'{base_amt}元@{price:.3f}'

        elif shares > 0:
            pnl = price / avg_cost - 1

            if add_count > 0 and pnl >= tp_pct:
                sell_value = shares * price
                sell_cost = calc_sell_cost(sell_value)
                total_cost += sell_cost
                net = sell_value - sell_cost
                profit = net - total_invested
                round_profits.append(profit)
                round_trips += 1
                action = '止盈'
                trade_detail = f'投入{total_invested:,.0f}→卖{sell_value:,.0f} 利润{profit:+,.0f}'

                cash += net
                shares = 0
                total_invested = 0
                add_count = 0
                avg_cost = 0

                if i < len(df) - 5 and cash >= base_amt:
                    cost = calc_buy_cost(base_amt)
                    total_cost += cost
                    cash -= (base_amt + cost)
                    shares = base_amt / price
                    avg_cost = price
                    total_invested = base_amt
                    add_count = 0
                    action += '+新建仓'

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
                    action = f'加仓{add_count}'
                    trade_detail = f'{amt:,}元@{price:.3f}'

        position_value = shares * price
        equity = cash + position_value

        daily_records.append({
            'date': date,
            'price': price,
            'cash': cash,
            'position': position_value,
            'equity': equity,
            'shares': shares,
            'avg_cost': avg_cost,
            'add_count': add_count,
            'total_invested': total_invested,
            'action': action,
            'trade_detail': trade_detail,
        })

    return daily_records, round_profits, round_trips, total_cost

# ── 主程序 ──
df = fetch_stock_data('515880', '20220701', '20260708')
records, round_profits, round_trips, total_cost = run_backtest(df)

rdf = pd.DataFrame(records)
rdf['date'] = pd.to_datetime(rdf['date'])

print(f"科创50 ETF (515880) 3年回测 — 稳健马丁格尔")
print(f"参数: 底仓8000 | 间距7% | 止盈8% | 翻倍1-3-5-7 | 总仓10万")
print(f"{'='*100}")

# 按年度切分
years = [
    ('2023H2', '2023-07-01', '2024-01-01'),
    ('2024H1', '2024-01-01', '2024-07-01'),
    ('2024H2', '2024-07-01', '2025-01-01'),
    ('2025H1', '2025-01-01', '2025-07-01'),
    ('2026H1', '2025-07-01', '2026-07-08'),
]

# 也用完整年份
full_years = [
    ('2024', '2024-01-01', '2025-01-01'),
    ('2025', '2025-01-01', '2026-01-01'),
    ('2026H1', '2026-01-01', '2026-07-08'),
]

print(f"\n{'─'*100}")
print(f"按年度净值变化（连续运行，资金滚动）")
print(f"{'─'*100}")
print(f"{'年份':<10} {'期初净值':>10} {'期末净值':>10} {'当年收益':>10} {'盈亏金额':>10} {'基准收益':>10} {'Alpha':>10} {'当年回撤':>10}")
print(f"{'─'*100}")

prev_end_equity = 100000

for yr, start_s, end_s in years:
    start_dt = pd.to_datetime(start_s)
    end_dt = pd.to_datetime(end_s)

    yr_data = rdf[(rdf['date'] >= start_dt) & (rdf['date'] < end_dt)]
    if len(yr_data) == 0:
        continue

    start_eq = yr_data.iloc[0]['equity']
    end_eq = yr_data.iloc[-1]['equity']
    yr_ret = (end_eq - start_eq) / start_eq
    yr_pnl = end_eq - start_eq

    # 基准
    start_price = yr_data.iloc[0]['price']
    end_price = yr_data.iloc[-1]['price']
    bench_ret = (end_price - start_price) / start_price

    # 年内回撤
    eq_series = yr_data['equity']
    peak = eq_series.cummax()
    yr_mdd = ((eq_series - peak) / peak).min()

    alpha = yr_ret - bench_ret
    print(f"  {yr:<10} {start_eq:>9,.0f} {end_eq:>9,.0f} {yr_ret:>+9.2%} {yr_pnl:>+9,.0f} {bench_ret:>+9.2%} {alpha:>+9.2%} {yr_mdd:>9.2%}")

print(f"{'─'*100}")

# 完整年份
print(f"\n{'─'*100}")
print(f"按完整自然年")
print(f"{'─'*100}")
print(f"{'年份':<10} {'期初净值':>10} {'期末净值':>10} {'当年收益':>10} {'盈亏金额':>10} {'基准收益':>10} {'Alpha':>10} {'当年回撤':>10}")
print(f"{'─'*100}")

for yr, start_s, end_s in full_years:
    start_dt = pd.to_datetime(start_s)
    end_dt = pd.to_datetime(end_s)

    yr_data = rdf[(rdf['date'] >= start_dt) & (rdf['date'] < end_dt)]
    if len(yr_data) == 0:
        continue

    start_eq = yr_data.iloc[0]['equity']
    end_eq = yr_data.iloc[-1]['equity']
    yr_ret = (end_eq - start_eq) / start_eq
    yr_pnl = end_eq - start_eq

    start_price = yr_data.iloc[0]['price']
    end_price = yr_data.iloc[-1]['price']
    bench_ret = (end_price - start_price) / start_price

    eq_series = yr_data['equity']
    peak = eq_series.cummax()
    yr_mdd = ((eq_series - peak) / peak).min()

    alpha = yr_ret - bench_ret
    print(f"  {yr:<10} {start_eq:>9,.0f} {end_eq:>9,.0f} {yr_ret:>+9.2%} {yr_pnl:>+9,.0f} {bench_ret:>+9.2%} {alpha:>+9.2%} {yr_mdd:>9.2%}")

print(f"{'─'*100}")

# 总账
print(f"\n{'='*100}")
print(f"总账")
print(f"{'─'*100}")
print(f"  初始: 100,000元 → 最终: {rdf.iloc[-1]['equity']:,.0f}元")
print(f"  3年累计收益: {(rdf.iloc[-1]['equity']-100000)/100000:+.2%} ({rdf.iloc[-1]['equity']-100000:+,.0f}元)")
bench_total = (rdf.iloc[-1]['price'] / rdf.iloc[0]['price'] - 1)
print(f"  3年基准收益: {bench_total:+.2%}")
print(f"  3年Alpha:    {(rdf.iloc[-1]['equity']-100000)/100000 - bench_total:+.2%}")
print(f"  完成轮次: {round_trips}轮 | 已实现利润: {sum(round_profits):+,.0f}元")
print(f"  交易成本: {total_cost:.0f}元")

# 交易明细
print(f"\n{'='*100}")
print(f"交易明细")
print(f"{'─'*100}")

for _, r in rdf[rdf['action'] != ''].iterrows():
    d = str(r['date'])[:10]
    print(f"  {d} | {r['action']:<12} | {r['trade_detail']}")

print(f"{'='*100}")
