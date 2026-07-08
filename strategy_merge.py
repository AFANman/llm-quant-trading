"""
策略合并+优化实验 — 科创50
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd, numpy as np

def calc_metrics(equity_list, init_cash, df):
    eq = pd.Series(equity_list)
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
    down = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() - rf) / down.std() * np.sqrt(252) if len(down) > 0 and down.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    return {'ret': ret, 'bench': bench, 'alpha': ret - bench, 'mdd': dd,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'final': final}


# ============================================================
# 基准：传统马丁格尔（原版）
# ============================================================
def strat_classic_martin(df, init_cash=100000):
    df = compute_indicators(df)
    base_amt = 3000
    mults = [1, 2, 4, 8]
    dip = -0.05
    tp = 0.10
    cash = init_cash - base_amt
    position = base_amt
    cost_basis = df.iloc[60]['close']
    add_count = 0
    equity = []

    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i - 1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        equity.append(cash + position)

        if position > 0:
            pnl = r['close'] / cost_basis - 1
            if pnl >= tp:
                cash += position; position = 0; add_count = 0
                if i < len(df) - 5:
                    buy = base_amt
                    if cash >= buy:
                        cash -= buy; position = buy; cost_basis = r['close']
            elif pnl <= dip * (add_count + 1) and add_count < 4:
                amt = base_amt * mults[add_count]
                if cash >= amt:
                    cash -= amt; position += amt; add_count += 1

    return calc_metrics(equity, init_cash, df)


# ============================================================
# V1：基础合并 — 翻倍加仓 + 累计跌幅触发 + ATR止盈
# ============================================================
def strat_v1_merge(df, init_cash=100000):
    df = compute_indicators(df)
    base_amt = 3000
    mults = [1, 2, 4, 8]  # 翻倍
    tiers = [(-0.12, 3), (-0.08, 2), (-0.05, 1), (-0.02, 0)]  # 累计跌幅→mults索引
    bounce_mult = 1.0  # ATR止盈倍数

    cash = init_cash - base_amt; position = base_amt
    cost_basis = df.iloc[60]['close']
    peak_price = df.iloc[60]['close']
    used_tiers = set()
    add_round = 0  # 当前用了第几个mult
    equity = []

    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i - 1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        equity.append(cash + position)

        if r['close'] > peak_price:
            peak_price = r['close']
        cum_drop = r['close'] / peak_price - 1

        ma20 = r.get('ma20', 0) if not pd.isna(r.get('ma20', 0)) else 0
        ma60 = r.get('ma60', 0) if not pd.isna(r.get('ma60', 0)) else 0
        is_down = ma20 > 0 and ma60 > 0 and ma20 < ma60

        # 加仓：累计跌幅触发 + 翻倍金额
        if is_down and add_round < 4:
            pct = r.get('pct_change', 0)
            if pct is not None and not pd.isna(pct) and pct < -0.01:
                for thr, mult_idx in tiers:
                    if cum_drop <= thr and mult_idx not in used_tiers:
                        amt = base_amt * mults[mult_idx]
                        if cash >= amt:
                            cash -= amt; position += amt
                            used_tiers.add(mult_idx)
                            add_round = max(add_round, mult_idx + 1)
                        break

        # 止盈：1×ATR反弹
        if add_round > 0:
            atr = r.get('atr', 0) if not pd.isna(r.get('atr', 0)) else 0
            rsi = r.get('rsi', 50) if not pd.isna(r.get('rsi', 50)) else 50
            if position > base_amt and atr > 0 and r['close'] > cost_basis + bounce_mult * atr or rsi > 70:
                sell = min(position - base_amt, position * 0.9)
                if sell > 100:
                    cash += sell; position -= sell
                    add_round = 0; used_tiers = set()
                    peak_price = r['close']; cost_basis = r['close']

    return calc_metrics(equity, init_cash, df)


# ============================================================
# V2：牛市增强 — 上涨环境底仓2万 + 下跌环境3000翻倍
# ============================================================
def strat_v2_bull_boost(df, init_cash=100000):
    df = compute_indicators(df)
    base_small = 3000   # 下跌底仓
    base_big = 20000    # 上涨底仓
    mults = [1, 2, 4, 8]
    tiers = [(-0.12, 3), (-0.08, 2), (-0.05, 1), (-0.02, 0)]
    bounce_mult = 1.0

    cash = init_cash - base_small; position = base_small
    cost_basis = df.iloc[60]['close']
    peak_price = df.iloc[60]['close']
    used_tiers = set(); add_round = 0
    equity = []

    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i - 1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        equity.append(cash + position)

        if r['close'] > peak_price:
            peak_price = r['close']
        cum_drop = r['close'] / peak_price - 1

        ma20 = r.get('ma20', 0) if not pd.isna(r.get('ma20', 0)) else 0
        ma60 = r.get('ma60', 0) if not pd.isna(r.get('ma60', 0)) else 0
        is_down = ma20 > 0 and ma60 > 0 and ma20 < ma60
        is_up = ma20 > 0 and ma60 > 0 and ma20 >= ma60

        # 牛市增强：上涨环境加仓到2万
        if is_up and position < base_big and cash >= (base_big - position):
            buy = base_big - position
            cash -= buy; position += buy

        # 熊市减仓：下跌环境且无加仓时减到底仓
        if is_down and add_round == 0 and position > base_small + 1000:
            sell = position - base_small
            cash += sell; position -= sell

        # 加仓
        if is_down and add_round < 4:
            pct = r.get('pct_change', 0)
            if pct is not None and not pd.isna(pct) and pct < -0.01:
                for thr, mult_idx in tiers:
                    if cum_drop <= thr and mult_idx not in used_tiers:
                        amt = base_small * mults[mult_idx]
                        if cash >= amt:
                            cash -= amt; position += amt
                            used_tiers.add(mult_idx)
                            add_round = max(add_round, mult_idx + 1)
                        break

        # 止盈
        if add_round > 0:
            atr = r.get('atr', 0) if not pd.isna(r.get('atr', 0)) else 0
            rsi = r.get('rsi', 50) if not pd.isna(r.get('rsi', 50)) else 50
            if position > base_small and atr > 0 and (r['close'] > cost_basis + bounce_mult * atr or rsi > 70):
                sell = min(position - base_small, position * 0.9)
                if sell > 100:
                    cash += sell; position -= sell
                    add_round = 0; used_tiers = set()
                    peak_price = r['close']; cost_basis = r['close']

    return calc_metrics(equity, init_cash, df)


# ============================================================
# V3：分档止盈 — 涨5%卖一半加仓，涨10%全卖
# ============================================================
def strat_v3_tiered_tp(df, init_cash=100000):
    df = compute_indicators(df)
    base_amt = 3000
    mults = [1, 2, 4, 8]
    tiers = [(-0.12, 3), (-0.08, 2), (-0.05, 1), (-0.02, 0)]
    tp1 = 0.05  # 第一档止盈5%
    tp2 = 0.10  # 第二档止盈10%

    cash = init_cash - base_amt; position = base_amt
    cost_basis = df.iloc[60]['close']
    peak_price = df.iloc[60]['close']
    used_tiers = set(); add_round = 0
    tp1_done = False
    equity = []

    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i - 1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        equity.append(cash + position)

        if r['close'] > peak_price:
            peak_price = r['close']
        cum_drop = r['close'] / peak_price - 1

        ma20 = r.get('ma20', 0) if not pd.isna(r.get('ma20', 0)) else 0
        ma60 = r.get('ma60', 0) if not pd.isna(r.get('ma60', 0)) else 0
        is_down = ma20 > 0 and ma60 > 0 and ma20 < ma60

        # 加仓
        if is_down and add_round < 4:
            pct = r.get('pct_change', 0)
            if pct is not None and not pd.isna(pct) and pct < -0.01:
                for thr, mult_idx in tiers:
                    if cum_drop <= thr and mult_idx not in used_tiers:
                        amt = base_amt * mults[mult_idx]
                        if cash >= amt:
                            cash -= amt; position += amt
                            used_tiers.add(mult_idx)
                            add_round = max(add_round, mult_idx + 1)
                        break

        # 分档止盈
        if add_round > 0 and position > base_amt:
            pnl = r['close'] / cost_basis - 1

            if pnl >= tp2:
                # 第二档：全卖加仓部分
                sell = min(position - base_amt, position * 0.9)
                if sell > 100:
                    cash += sell; position -= sell
                    add_round = 0; used_tiers = set(); tp1_done = False
                    peak_price = r['close']; cost_basis = r['close']

            elif pnl >= tp1 and not tp1_done:
                # 第一档：卖一半加仓
                add_pos = position - base_amt
                sell = add_pos * 0.5
                if sell > 100:
                    cash += sell; position -= sell
                    tp1_done = True

    return calc_metrics(equity, init_cash, df)


# ============================================================
# V4：趋势过滤 — MA60之上才做马丁，之下空仓
# ============================================================
def strat_v4_trend_filter(df, init_cash=100000):
    df = compute_indicators(df)
    base_amt = 3000
    mults = [1, 2, 4, 8]
    tiers = [(-0.12, 3), (-0.08, 2), (-0.05, 1), (-0.02, 0)]
    bounce_mult = 1.0

    cash = init_cash; position = 0
    cost_basis = 0; peak_price = 0
    used_tiers = set(); add_round = 0
    equity = []

    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i - 1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        equity.append(cash + position)

        ma60 = r.get('ma60', 0) if not pd.isna(r.get('ma60', 0)) else 0
        ma20 = r.get('ma20', 0) if not pd.isna(r.get('ma20', 0)) else 0
        above_ma60 = ma60 > 0 and r['close'] > ma60
        is_down = ma20 > 0 and ma60 > 0 and ma20 < ma60

        # 空仓且在MA60之上→建仓
        if position < 100 and above_ma60 and cash >= base_amt:
            cash -= base_amt; position = base_amt
            cost_basis = r['close']; peak_price = r['close']
            used_tiers = set(); add_round = 0

        # 跌破MA60且无加仓→清仓
        if not above_ma60 and add_round == 0 and position > 0:
            cash += position; position = 0

        if position > 0:
            if r['close'] > peak_price:
                peak_price = r['close']
            cum_drop = r['close'] / peak_price - 1

            # 加仓
            if is_down and add_round < 4:
                pct = r.get('pct_change', 0)
                if pct is not None and not pd.isna(pct) and pct < -0.01:
                    for thr, mult_idx in tiers:
                        if cum_drop <= thr and mult_idx not in used_tiers:
                            amt = base_amt * mults[mult_idx]
                            if cash >= amt:
                                cash -= amt; position += amt
                                used_tiers.add(mult_idx)
                                add_round = max(add_round, mult_idx + 1)
                            break

            # ATR止盈
            if add_round > 0:
                atr = r.get('atr', 0) if not pd.isna(r.get('atr', 0)) else 0
                rsi = r.get('rsi', 50) if not pd.isna(r.get('rsi', 50)) else 50
                if position > base_amt and atr > 0 and (r['close'] > cost_basis + bounce_mult * atr or rsi > 70):
                    sell = min(position - base_amt, position * 0.9)
                    if sell > 100:
                        cash += sell; position -= sell
                        add_round = 0; used_tiers = set()
                        peak_price = r['close']; cost_basis = r['close']

    return calc_metrics(equity, init_cash, df)


# ============================================================
# V5：终极合并 — 牛市底仓2万 + 翻倍加仓 + 累计跌幅 + ATR止盈 + 分档止盈
# ============================================================
def strat_v5_ultimate(df, init_cash=100000):
    df = compute_indicators(df)
    base_small = 3000
    base_big = 20000
    mults = [1, 2, 4, 8]
    tiers = [(-0.12, 3), (-0.08, 2), (-0.05, 1), (-0.02, 0)]
    tp1 = 0.05
    tp2 = 0.10

    cash = init_cash - base_small; position = base_small
    cost_basis = df.iloc[60]['close']
    peak_price = df.iloc[60]['close']
    used_tiers = set(); add_round = 0; tp1_done = False
    equity = []

    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i - 1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        equity.append(cash + position)

        if r['close'] > peak_price:
            peak_price = r['close']
        cum_drop = r['close'] / peak_price - 1

        ma20 = r.get('ma20', 0) if not pd.isna(r.get('ma20', 0)) else 0
        ma60 = r.get('ma60', 0) if not pd.isna(r.get('ma60', 0)) else 0
        is_down = ma20 > 0 and ma60 > 0 and ma20 < ma60
        is_up = ma20 > 0 and ma60 > 0 and ma20 >= ma60

        # 牛市增强
        if is_up and add_round == 0 and position < base_big and cash >= (base_big - position):
            buy = base_big - position
            cash -= buy; position += buy

        # 熊市缩仓
        if is_down and add_round == 0 and position > base_small + 1000:
            sell = position - base_small
            cash += sell; position -= sell

        # 加仓
        if is_down and add_round < 4:
            pct = r.get('pct_change', 0)
            if pct is not None and not pd.isna(pct) and pct < -0.01:
                for thr, mult_idx in tiers:
                    if cum_drop <= thr and mult_idx not in used_tiers:
                        amt = base_small * mults[mult_idx]
                        if cash >= amt:
                            cash -= amt; position += amt
                            used_tiers.add(mult_idx)
                            add_round = max(add_round, mult_idx + 1)
                        break

        # 分档止盈
        if add_round > 0 and position > base_small:
            pnl = r['close'] / cost_basis - 1

            if pnl >= tp2:
                sell = min(position - base_small, position * 0.9)
                if sell > 100:
                    cash += sell; position -= sell
                    add_round = 0; used_tiers = set(); tp1_done = False
                    peak_price = r['close']; cost_basis = r['close']
            elif pnl >= tp1 and not tp1_done:
                add_pos = position - base_small
                sell = add_pos * 0.5
                if sell > 100:
                    cash += sell; position -= sell
                    tp1_done = True

    return calc_metrics(equity, init_cash, df)


# ============================================================
# 跑回测
# ============================================================
strategies = {
    '传统马丁(原版)': strat_classic_martin,
    'V1 基础合并': strat_v1_merge,
    'V2 牛市增强': strat_v2_bull_boost,
    'V3 分档止盈': strat_v3_tiered_tp,
    'V4 趋势过滤': strat_v4_trend_filter,
    'V5 终极合并': strat_v5_ultimate,
}

years = [
    ('2023', '20220701', '20231231'),
    ('2024', '20230701', '20241231'),
    ('2025H1', '20240701', '20250701'),
    ('2026H1', '20250701', '20260702'),
]

for yr, s, e in years:
    df = fetch_stock_data('515880', s, e)
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1

    print(f"\n{'='*95}")
    print(f"科创50 {yr}（基准: {bench:+.2%}）")
    print(f"{'='*95}")
    print(f"{'策略':<20} {'收益':>8} {'Alpha':>8} {'回撤':>8} {'夏普':>6} {'卡尔马':>6} {'索提诺':>6}")
    print(f"{'─'*70}")

    results = []
    for name, func in strategies.items():
        try:
            r = func(df)
            results.append((name, r))
            print(f"{name:<20} {r['ret']:>+7.2%} {r['alpha']:>+7.2%} {r['mdd']:>7.2%} {r['sharpe']:>5.2f} {r['calmar']:>5.2f} {r['sortino']:>5.2f}")
        except Exception as ex:
            print(f"{name:<20}  错误: {ex}")

    results.sort(key=lambda x: x[1]['alpha'], reverse=True)
    ranking = ' > '.join([f"{n}({r['alpha']:+.2%})" for n, r in results])
    print(f"\n  Alpha排名: {ranking}")

# 汇总
print(f"\n{'='*95}")
print(f"\n全部时段汇总")
print(f"{'='*95}")
print(f"{'策略':<20} {'平均Alpha':>10} {'平均夏普':>10} {'平均卡尔马':>10} {'平均回撤':>10} {'平均索提诺':>10}")
print(f"{'─'*75}")

for name, func in strategies.items():
    alphas = []; sharpes = []; calmars = []; mdds = []; sortinos = []
    for yr, s, e in years:
        df = fetch_stock_data('515880', s, e)
        try:
            r = func(df)
            alphas.append(r['alpha']); sharpes.append(r['sharpe'])
            calmars.append(r['calmar']); mdds.append(r['mdd']); sortinos.append(r['sortino'])
        except:
            pass
    if alphas:
        print(f"{name:<20} {np.mean(alphas):>+9.2%} {np.mean(sharpes):>9.2f} {np.mean(calmars):>9.2f} {np.mean(mdds):>9.2%} {np.mean(sortinos):>9.2f}")

print(f"{'='*95}")
