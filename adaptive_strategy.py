"""
环境自适应策略 — 牛市/跌市自动切换
上涨趋势(MA20>MA60): 底仓40%-50%，持有为主
下跌趋势(MA20<MA60): 马丁格尔(8k底仓+7%间距+8%止盈+1-3-5-7)
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd, numpy as np

def adaptive_strategy(df, init_cash=100000,
                      # 牛市参数
                      bull_base_pct=0.40,  # 底仓占总资产40%
                      bull_add_pct=0.10,   # 涨10%加仓一次
                      bull_add_pct_amt=0.15,  # 每次加仓15%
                      bull_stop_loss=0.20,    # 从最高价回撤20%止损
                      # 熊市参数（马丁格尔）
                      bear_base=8000,
                      bear_dip=0.07,
                      bear_tp=0.08,
                      bear_mults=None,
                      bear_max_add=4):
    df = compute_indicators(df)
    if bear_mults is None:
        bear_mults = [1, 3, 5, 7]

    cash = init_cash
    position = 0
    cost_basis = 0
    add_count = 0
    highest_since_buy = 0
    regime = 'neutral'
    equity = []
    trades = 0
    regime_changes = []

    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']

        # 判断趋势
        ma20 = row.get('ma20', None) or (df['close'].iloc[max(0,i-19):i+1].mean())
        ma60 = row.get('ma60', None) or (df['close'].iloc[max(0,i-59):i+1].mean())

        prev_regime = regime
        if ma20 > ma60:
            regime = 'bull'
        else:
            regime = 'bear'

        if regime != prev_regime:
            regime_changes.append((i - 60, regime))

        # 更新持仓市值
        if prev['close'] > 0 and position > 0:
            position *= (price / prev['close'])

        if price > highest_since_buy:
            highest_since_buy = price

        equity.append(cash + position)

        if regime == 'bear':
            # ── 熊市：马丁格尔 ──
            if position == 0:
                # 清仓后重新建仓
                if cash >= bear_base:
                    cash -= bear_base
                    position = bear_base
                    cost_basis = price
                    highest_since_buy = price
                    add_count = 0
            elif position > 0:
                pnl = price / cost_basis - 1 if cost_basis > 0 else 0
                # 止盈
                if add_count > 0 and pnl >= bear_tp:
                    cash += position
                    position = 0
                    add_count = 0
                    trades += 1
                # 加仓
                elif pnl <= -bear_dip * (add_count + 1) and add_count < bear_max_add:
                    amt = bear_base * bear_mults[add_count]
                    if cash >= amt:
                        cash -= amt
                        position += amt
                        add_count += 1

        elif regime == 'bull':
            # ── 牛市：大底仓持有 ──
            total_val = cash + position
            if position == 0:
                # 空仓→建仓40%
                buy_amt = total_val * bull_base_pct
                if cash >= buy_amt:
                    cash -= buy_amt
                    position = buy_amt
                    cost_basis = price
                    highest_since_buy = price
                    add_count = 0
            elif position > 0:
                # 止损
                drawdown = (highest_since_buy - price) / highest_since_buy
                if drawdown >= bull_stop_loss:
                    cash += position
                    position = 0
                    add_count = 0
                    trades += 1
                # 浮盈加仓（涨10%加15%）
                elif cost_basis > 0:
                    gain = price / cost_basis - 1
                    if gain >= bull_add_pct * (add_count + 1) and add_count < 3:
                        add_amt = total_val * bull_add_pct_amt
                        if cash >= add_amt:
                            cash -= add_amt
                            position += add_amt
                            add_count += 1

    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = eq.iloc[-1]
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    n = len(eq)
    ann_ret = (1 + ret) ** (252 / max(n, 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0

    return {
        'ret': ret, 'bench': bench, 'alpha': ret - bench, 'mdd': dd,
        'sharpe': sharpe, 'calmar': calmar, 'trades': trades,
        'regime_changes': len(regime_changes), 'final': final,
    }

def martin_only(df, init_cash=100000):
    """纯马丁格尔（稳健版）"""
    df = compute_indicators(df)
    cash = init_cash - 8000
    position = 8000
    cost_basis = df.iloc[60]['close']
    add_count = 0
    equity = []
    mults = [1, 3, 5, 7]

    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        if position > 0 and prev['close'] > 0:
            position *= (row['close'] / prev['close'])
        equity.append(cash + position)
        if position > 0:
            pnl = row['close'] / cost_basis - 1
            if pnl >= 0.08:
                cash += position; position = 0; add_count = 0
                if i < len(df) - 5 and cash >= 8000:
                    cash -= 8000; position = 8000; cost_basis = row['close']
            elif pnl <= -0.07 * (add_count + 1) and add_count < 4:
                amt = 8000 * mults[add_count]
                if cash >= amt:
                    cash -= amt; position += amt; add_count += 1

    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = eq.iloc[-1]
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    ann_ret = (1 + ret) ** (252 / max(len(eq), 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    return {'ret': ret, 'bench': bench, 'alpha': ret - bench, 'mdd': dd,
            'sharpe': sharpe, 'calmar': calmar}

def buy_hold(df, init_cash=100000):
    """买入持有"""
    df = compute_indicators(df)
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    return {'ret': bench, 'bench': bench, 'alpha': 0, 'mdd': 0, 'sharpe': 0, 'calmar': 0}

# ── 参数组合 ──
adaptive_configs = {
    '自适应A(牛40%底/熊马丁)': {
        'bull_base_pct': 0.40, 'bull_add_pct': 0.10, 'bull_add_pct_amt': 0.15, 'bull_stop_loss': 0.20,
    },
    '自适应B(牛30%底/熊马丁)': {
        'bull_base_pct': 0.30, 'bull_add_pct': 0.08, 'bull_add_pct_amt': 0.10, 'bull_stop_loss': 0.15,
    },
    '自适应C(牛50%底/熊马丁)': {
        'bull_base_pct': 0.50, 'bull_add_pct': 0.12, 'bull_add_pct_amt': 0.20, 'bull_stop_loss': 0.25,
    },
}

years = [
    ('2021-2022', '20210101', '20221231'),
    ('2023', '20220701', '20231231'),
    ('2024', '20230701', '20241231'),
    ('2025H1', '20240701', '20250701'),
    ('2026H1', '20250701', '20260702'),
]

dfs = {}
for yr, s, e in years:
    dfs[yr] = fetch_stock_data('515880', s, e)

# 分年度对比
print(f"环境自适应策略 vs 纯马丁格尔 vs 买入持有 — 科创50")
print(f"{'='*140}")

for yr_name in [y[0] for y in years]:
    df = dfs[yr_name]
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    m = martin_only(df)
    bh = {'alpha': 0, 'ret': bench, 'mdd': 0, 'sharpe': 0}

    print(f"\n  {yr_name}（基准: {bench:+.2%}）")
    print(f"  {'策略':<35} {'收益':>8} {'Alpha':>8} {'回撤':>8} {'夏普':>6} {'卡尔马':>6} {'切换次数':>8}")
    print(f"  {'─'*95}")
    print(f"  {'纯马丁格尔(8k/7%/8%)':<35} {m['ret']:>+7.2%} {m['alpha']:>+7.2%} {m['mdd']:>7.2%} {m['sharpe']:>5.2f} {m['calmar']:>5.2f} {'—':>8}")
    print(f"  {'买入持有':<35} {bench:>+7.2%} {0:>+7.2%} {'—':>8} {'—':>6} {'—':>6} {'—':>8}")

    for name, cfg in adaptive_configs.items():
        r = adaptive_strategy(df, **cfg)
        print(f"  {name:<35} {r['ret']:>+7.2%} {r['alpha']:>+7.2%} {r['mdd']:>7.2%} {r['sharpe']:>5.2f} {r['calmar']:>5.2f} {r['regime_changes']:>8}")

# 汇总
print(f"\n{'='*140}")
print(f"\n汇总（5个时段平均）")
print(f"{'─'*140}")
print(f"{'策略':<35} {'平均Alpha':>10} {'平均夏普':>10} {'平均卡尔马':>10} {'平均回撤':>10} {'正Alpha率':>10}")
print(f"{'─'*140}")

# 纯马丁
m_rs = [martin_only(dfs[yr[0]]) for yr in years]
print(f"{'纯马丁格尔':<35} {np.mean([r['alpha'] for r in m_rs]):>+9.2%} {np.mean([r['sharpe'] for r in m_rs]):>9.2f} {np.mean([r['calmar'] for r in m_rs]):>9.2f} {np.mean([r['mdd'] for r in m_rs]):>9.2%} {sum(1 for r in m_rs if r['alpha']>0)/len(m_rs):>9.0%}")

# 自适应
for name, cfg in adaptive_configs.items():
    rs = [adaptive_strategy(dfs[yr[0]], **cfg) for yr in years]
    pos = sum(1 for r in rs if r['alpha'] > 0)
    print(f"{name:<35} {np.mean([r['alpha'] for r in rs]):>+9.2%} {np.mean([r['sharpe'] for r in rs]):>9.2f} {np.mean([r['calmar'] for r in rs]):>9.2f} {np.mean([r['mdd'] for r in rs]):>9.2%} {pos/len(rs):>9.0%}")
