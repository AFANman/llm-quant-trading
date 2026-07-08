"""
三组参数泛化测试 — 多市场多时段
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd, numpy as np

def strat_martin(df, base_amt=3000, dip_pct=0.05, tp_pct=0.10, mults=None, max_add=4, init_cash=100000):
    df = compute_indicators(df)
    if mults is None:
        mults = [1, 2, 4, 8][:max_add]
    cash = init_cash - base_amt
    position = base_amt
    cost_basis = df.iloc[60]['close']
    add_count = 0
    equity = []
    trades = 0

    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i - 1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        equity.append(cash + position)
        if position > 0:
            pnl = r['close'] / cost_basis - 1
            if pnl >= tp_pct:
                cash += position; position = 0; add_count = 0; trades += 1
                if i < len(df) - 5:
                    if cash >= base_amt:
                        cash -= base_amt; position = base_amt; cost_basis = r['close']
            elif pnl <= -dip_pct * (add_count + 1) and add_count < max_add and add_count < len(mults):
                amt = base_amt * mults[add_count]
                if cash >= amt:
                    cash -= amt; position += amt; add_count += 1

    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = eq.iloc[-1]
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    n_days = len(eq)
    ann_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    return {'ret': ret, 'bench': bench, 'alpha': ret - bench, 'mdd': dd,
            'sharpe': sharpe, 'calmar': calmar, 'final': final, 'trades': trades}

configs = {
    '原版(3k/5%/10%/1-2-4-8)': {'base_amt': 3000, 'dip_pct': 0.05, 'tp_pct': 0.10, 'mults': [1, 2, 4, 8]},
    '进攻(5k/5%/8%/1-3-5-7)':  {'base_amt': 5000, 'dip_pct': 0.05, 'tp_pct': 0.08, 'mults': [1, 3, 5, 7]},
    '稳健(8k/7%/8%/1-3-5-7)':  {'base_amt': 8000, 'dip_pct': 0.07, 'tp_pct': 0.08, 'mults': [1, 3, 5, 7]},
}

markets = [
    ('科创50',    '515880', [
        ('2021-2022', '20210101', '20221231'),
        ('2023',      '20220701', '20231231'),
        ('2024',      '20230701', '20241231'),
        ('2025H1',    '20240701', '20250701'),
        ('2026H1',    '20250701', '20260702'),
    ]),
    ('沪深300',   '510300', [
        ('2020-2021', '20200101', '20211231'),
        ('2022-2023', '20220101', '20231231'),
        ('2024-2025', '20230701', '20250701'),
        ('2025-2026', '20240701', '20260702'),
    ]),
    ('贵州茅台',  '600519', [
        ('2020-2021', '20200101', '20211231'),
        ('2022',      '20220101', '20221231'),
        ('2023-2024', '20230101', '20241231'),
        ('2025-2026', '20240701', '20260702'),
    ]),
    ('创业板',    '159915', [
        ('2020-2021', '20200101', '20211231'),
        ('2022',      '20220101', '20221231'),
        ('2023-2024', '20230101', '20241231'),
        ('2025-2026', '20240701', '20260702'),
    ]),
    ('中证500',   '510500', [
        ('2020-2021', '20200101', '20211231'),
        ('2022',      '20220101', '20221231'),
        ('2023-2024', '20230101', '20241231'),
        ('2025-2026', '20240701', '20260702'),
    ]),
]

for mkt_name, code, periods in markets:
    print(f"\n{'='*100}")
    print(f"  {mkt_name}（{code}）")
    print(f"{'='*100}")

    all_results = {name: [] for name in configs}

    for period_name, start, end in periods:
        try:
            df = fetch_stock_data(code, start, end)
            if len(df) < 70:
                print(f"  {period_name}: 数据不足，跳过")
                continue
            bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
            print(f"\n  {period_name}（基准: {bench:+.2%}）")
            print(f"  {'策略':<30} {'收益':>8} {'Alpha':>8} {'回撤':>8} {'夏普':>6} {'卡尔马':>6}")
            print(f"  {'─'*75}")

            for name, cfg in configs.items():
                r = strat_martin(df, **cfg)
                all_results[name].append(r)
                print(f"  {name:<30} {r['ret']:>+7.2%} {r['alpha']:>+7.2%} {r['mdd']:>7.2%} {r['sharpe']:>5.2f} {r['calmar']:>5.2f}")

        except Exception as ex:
            print(f"  {period_name}: 错误 {ex}")

    # 汇总
    if any(len(v) > 0 for v in all_results.values()):
        n_periods = max(len(v) for v in all_results.values())
        print(f"\n  {mkt_name} 汇总（{n_periods}个时段平均）")
        print(f"  {'─'*85}")
        print(f"  {'策略':<30} {'平均Alpha':>10} {'平均夏普':>10} {'平均卡尔马':>10} {'平均回撤':>10} {'正Alpha率':>10}")
        print(f"  {'─'*85}")
        for name, rs in all_results.items():
            if rs:
                avg_a = np.mean([r['alpha'] for r in rs])
                avg_s = np.mean([r['sharpe'] for r in rs])
                avg_c = np.mean([r['calmar'] for r in rs])
                avg_m = np.mean([r['mdd'] for r in rs])
                pos_rate = sum(1 for r in rs if r['alpha'] > 0) / len(rs)
                print(f"  {name:<30} {avg_a:>+9.2%} {avg_s:>9.2f} {avg_c:>9.2f} {avg_m:>9.2%} {pos_rate:>9.0%}")

# 全局汇总
print(f"\n\n{'='*100}")
print(f"全局汇总（所有市场×所有时段）")
print(f"{'='*100}")
print(f"{'策略':<30} {'平均Alpha':>10} {'平均夏普':>10} {'平均卡尔马':>10} {'平均回撤':>10} {'正Alpha率':>10}")
print(f"{'─'*85}")
for name in configs:
    all_r = []
    for mkt_name, code, periods in markets:
        for period_name, start, end in periods:
            try:
                df = fetch_stock_data(code, start, end)
                if len(df) < 70:
                    continue
                r = strat_martin(df, **configs[name])
                all_r.append(r)
            except:
                pass
    if all_r:
        avg_a = np.mean([r['alpha'] for r in all_r])
        avg_s = np.mean([r['sharpe'] for r in all_r])
        avg_c = np.mean([r['calmar'] for r in all_r])
        avg_m = np.mean([r['mdd'] for r in all_r])
        pos_rate = sum(1 for r in all_r if r['alpha'] > 0) / len(all_r)
        print(f"{name:<30} {avg_a:>+9.2%} {avg_s:>9.2f} {avg_c:>9.2f} {avg_m:>9.2%} {pos_rate:>9.0%} ({sum(1 for r in all_r if r['alpha']>0)}/{len(all_r)})")
print(f"{'='*100}")
