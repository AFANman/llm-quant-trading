"""
6策略对比回测 — 科创50
1. 网格交易
2. 双均线+RSI择时
3. 传统马丁格尔翻倍
4. 网格+趋势复合
5. 你的4档分档策略
6. 买入持有（基准）
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd, numpy as np

# ============================================================
# 策略1: 网格交易
# ============================================================
def strategy_grid(df, init_cash=100000):
    """网格交易：等距2%格子，每格3000元，底仓30%"""
    df = compute_indicators(df)
    grid_pct = 0.02  # 网格间距2%
    grid_amount = 3000  # 每格金额
    base_pct = 0.30  # 底仓30%
    
    base_pos = init_cash * base_pct
    cash = init_cash - base_pos
    position = base_pos
    last_grid_price = df.iloc[60]['close']
    equity = []
    
    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i-1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        
        price = r['close']
        equity.append(cash + position)
        
        # 价格下跌触发买入格
        if price <= last_grid_price * (1 - grid_pct):
            if cash >= grid_amount:
                cash -= grid_amount
                position += grid_amount
                last_grid_price = price
        
        # 价格上涨触发卖出格（只卖网格部分，不卖底仓）
        elif price >= last_grid_price * (1 + grid_pct):
            sell = min(grid_amount, max(0, position - base_pos))
            if sell > 100:
                cash += sell
                position -= sell
                last_grid_price = price
    
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = cash + position
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    n_days = len(eq)
    ann_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    
    return {'ret': ret, 'bench': bench, 'alpha': ret-bench, 'mdd': dd, 'sharpe': sharpe, 'calmar': calmar}


# ============================================================
# 策略2: 双均线+RSI择时
# ============================================================
def strategy_ma_rsi(df, init_cash=100000):
    """双均线+RSI：MA5/MA20交叉+RSI超买超卖"""
    df = compute_indicators(df)
    # 额外计算MA5
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20_calc'] = df['close'].rolling(20).mean()
    
    cash = init_cash
    position = 0
    equity = []
    stop_loss_price = 0
    
    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i-1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        
        ma5 = r.get('ma5', 0) if not pd.isna(r.get('ma5', 0)) else 0
        ma20 = r.get('ma20_calc', 0) if not pd.isna(r.get('ma20_calc', 0)) else 0
        pma5 = p.get('ma5', 0) if not pd.isna(p.get('ma5', 0)) else 0
        pma20 = p.get('ma20_calc', 0) if not pd.isna(p.get('ma20_calc', 0)) else 0
        rsi = r.get('rsi', 50) if not pd.isna(r.get('rsi', 50)) else 50
        
        equity.append(cash + position)
        
        # 买入：MA5上穿MA20 且 RSI<60
        if ma5 > 0 and ma20 > 0 and pma5 <= pma20 and ma5 > ma20 and rsi < 60:
            if cash > 1000:
                buy = cash * 0.8  # 80%仓位买入
                cash -= buy
                position += buy
                stop_loss_price = r['close'] * 0.92  # -8%止损
        
        # 卖出：MA5下穿MA20 或 RSI>75
        elif position > 0 and ma5 > 0 and ma20 > 0:
            if (pma5 >= pma20 and ma5 < ma20) or rsi > 75:
                cash += position
                position = 0
                stop_loss_price = 0
        
        # 止损
        elif position > 0 and stop_loss_price > 0 and r['close'] < stop_loss_price:
            cash += position
            position = 0
            stop_loss_price = 0
    
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = cash + position
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    n_days = len(eq)
    ann_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    
    return {'ret': ret, 'bench': bench, 'alpha': ret-bench, 'mdd': dd, 'sharpe': sharpe, 'calmar': calmar}


# ============================================================
# 策略3: 传统马丁格尔翻倍加仓
# ============================================================
def strategy_martingale_classic(df, init_cash=100000):
    """传统马丁格尔：跌5%翻倍加仓，+10%止盈"""
    df = compute_indicators(df)
    base_amount = 3000  # 基础金额
    multipliers = [1, 2, 4, 8]  # 翻倍
    dip_interval = -0.05  # 每跌5%触发
    take_profit = 0.10  # +10%止盈
    max_rounds = 4
    
    cash = init_cash
    position = 0
    equity = []
    
    # 建仓
    cash -= base_amount
    position = base_amount
    cost_basis = df.iloc[60]['close']
    add_count = 0
    
    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i-1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        
        equity.append(cash + position)
        
        if position > 0:
            pnl_pct = r['close'] / cost_basis - 1
            
            # 止盈：+10%全部卖出
            if pnl_pct >= take_profit:
                cash += position
                position = 0
                add_count = 0
                # 重新建仓
                if i < len(df) - 5:
                    cash -= base_amount
                    position = base_amount
                    cost_basis = r['close']
            
            # 加仓：跌5%翻倍
            elif pnl_pct <= dip_interval * (add_count + 1) and add_count < max_rounds:
                amt = base_amount * multipliers[add_count]
                if cash >= amt:
                    cash -= amt
                    position += amt
                    add_count += 1
    
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = cash + position
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    n_days = len(eq)
    ann_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    
    return {'ret': ret, 'bench': bench, 'alpha': ret-bench, 'mdd': dd, 'sharpe': sharpe, 'calmar': calmar}


# ============================================================
# 策略4: 网格+趋势复合
# ============================================================
def strategy_grid_trend(df, init_cash=100000):
    """网格+趋势：MA60判断趋势，向上偏多，向下偏空"""
    df = compute_indicators(df)
    df['ma60_calc'] = df['close'].rolling(60).mean()
    
    grid_pct = 0.025  # 2.5%格子
    grid_amount = 3000
    base_pct = 0.30
    max_pct = 0.80
    
    base_pos = init_cash * base_pct
    cash = init_cash - base_pos
    position = base_pos
    last_grid_price = df.iloc[60]['close']
    equity = []
    
    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i-1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        
        price = r['close']
        ma60 = r.get('ma60_calc', 0) if not pd.isna(r.get('ma60_calc', 0)) else 0
        is_uptrend = ma60 > 0 and price > ma60
        
        equity.append(cash + position)
        
        # 下跌触发买入
        if price <= last_grid_price * (1 - grid_pct):
            if is_uptrend:
                buy_amt = grid_amount * 2  # 偏多：买2份
            else:
                buy_amt = grid_amount  # 偏空：买1份
            
            total_pos = position + buy_amt
            if cash >= buy_amt and total_pos <= init_cash * max_pct:
                cash -= buy_amt
                position += buy_amt
                last_grid_price = price
        
        # 上涨触发卖出
        elif price >= last_grid_price * (1 + grid_pct):
            if is_uptrend:
                sell_amt = grid_amount  # 偏多：卖1份
            else:
                sell_amt = grid_amount * 2  # 偏空：卖2份
            
            sell = min(sell_amt, max(0, position - base_pos))
            if sell > 100:
                cash += sell
                position -= sell
                last_grid_price = price
    
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = cash + position
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    n_days = len(eq)
    ann_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    
    return {'ret': ret, 'bench': bench, 'alpha': ret-bench, 'mdd': dd, 'sharpe': sharpe, 'calmar': calmar}


# ============================================================
# 策略5: 你的4档分档策略
# ============================================================
def strategy_4tier(df, init_cash=100000):
    """4档累计跌幅分档：2%加5k/5%加10k/8%加20k/12%加30k"""
    df = compute_indicators(df)
    tiers = [(-0.12, 30000), (-0.08, 20000), (-0.05, 10000), (-0.02, 5000)]
    base_pos = 20000
    bounce_mult = 1.0
    
    cash = init_cash - base_pos; position = base_pos
    add_costs = []; add_used = []; used_tiers = set()
    peak_price = df.iloc[60]['close']
    equity = []
    
    for i in range(60, len(df)):
        r = df.iloc[i]; p = df.iloc[i-1]
        if position > 0 and p['close'] > 0:
            position *= (r['close'] / p['close'])
        
        pct = r.get('pct_change', 0)
        ma20 = r.get('ma20', 0) if not pd.isna(r.get('ma20', 0)) else 0
        ma60 = r.get('ma60', 0) if not pd.isna(r.get('ma60', 0)) else 0
        is_down = ma20 > 0 and ma60 > 0 and ma20 < ma60
        
        equity.append(cash + position)
        
        if r['close'] > peak_price: peak_price = r['close']
        cum_drop = (r['close'] / peak_price - 1)
        
        if is_down:
            if pct is not None and not pd.isna(pct) and pct < -0.01:
                for thr, amt in tiers:
                    if cum_drop <= thr and thr not in used_tiers and cash >= amt:
                        cash -= amt; position += amt
                        add_costs.append(r['close']); add_used.append(amt)
                        used_tiers.add(thr)
                        break
            
            if len(add_costs) > 0:
                avg = np.mean(add_costs)
                atr = r.get('atr', 0) if not pd.isna(r.get('atr', 0)) else 0
                rsi = r.get('rsi', 50) if not pd.isna(r.get('rsi', 50)) else 50
                if (atr > 0 and r['close'] > avg + bounce_mult * atr) or rsi > 70:
                    sell = min(sum(add_used), max(0, position - base_pos))
                    if sell > 100:
                        cash += sell; position -= sell
                        add_costs = []; add_used = []; used_tiers = set()
                        peak_price = r['close']
    
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    final = cash + position
    ret = (final - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    daily_ret = eq.pct_change().dropna()
    n_days = len(eq)
    ann_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    
    return {'ret': ret, 'bench': bench, 'alpha': ret-bench, 'mdd': dd, 'sharpe': sharpe, 'calmar': calmar}


# ============================================================
# 策略6: 买入持有
# ============================================================
def strategy_buy_hold(df, init_cash=100000):
    df = compute_indicators(df)
    prices = df.iloc[60:]['close'].values
    ret = prices[-1] / prices[0] - 1
    peak = np.maximum.accumulate(prices)
    dd = ((prices - peak) / peak).min()
    daily_ret = pd.Series(prices).pct_change().dropna()
    n_days = len(prices)
    ann_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    
    return {'ret': ret, 'bench': ret, 'alpha': 0, 'mdd': dd, 'sharpe': sharpe, 'calmar': calmar}


# ============================================================
# 主程序
# ============================================================
strategies = {
    '买入持有': strategy_buy_hold,
    '网格交易': strategy_grid,
    '双均线+RSI': strategy_ma_rsi,
    '传统马丁格尔': strategy_martingale_classic,
    '网格+趋势': strategy_grid_trend,
    '4档分档(你的)': strategy_4tier,
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
    
    print(f"\n{'='*100}")
    print(f"科创50 {yr}（基准: {bench:+.2%}）")
    print(f"{'='*100}")
    print(f"{'策略':<18} {'收益':>8} {'Alpha':>8} {'回撤':>8} {'夏普':>6} {'卡尔马':>6}")
    print(f"{'─'*60}")
    
    results = []
    for name, func in strategies.items():
        try:
            r = func(df)
            results.append((name, r))
            print(f"{name:<18} {r['ret']:>+7.2%} {r['alpha']:>+7.2%} {r['mdd']:>7.2%} {r['sharpe']:>5.2f} {r['calmar']:>5.2f}")
        except Exception as ex:
            print(f"{name:<18}  错误: {ex}")
    
    # 排名
    results.sort(key=lambda x: x[1]['alpha'], reverse=True)
    ranking = ' > '.join([f"{n}({r['alpha']:+.2%})" for n, r in results])
    print(f"\n  Alpha排名: {ranking}")

print(f"\n{'='*100}")

# 汇总
print(f"\n全部时段汇总（平均Alpha、平均夏普、平均卡尔马）")
print(f"{'='*100}")
print(f"{'策略':<18} {'平均Alpha':>10} {'平均夏普':>10} {'平均卡尔马':>10} {'平均回撤':>10}")
print(f"{'─'*65}")

for name, func in strategies.items():
    alphas = []; sharpes = []; calmars = []; mdds = []
    for yr, s, e in years:
        df = fetch_stock_data('515880', s, e)
        try:
            r = func(df)
            alphas.append(r['alpha']); sharpes.append(r['sharpe']); calmars.append(r['calmar']); mdds.append(r['mdd'])
        except:
            pass
    if alphas:
        print(f"{name:<18} {np.mean(alphas):>+9.2%} {np.mean(sharpes):>9.2f} {np.mean(calmars):>9.2f} {np.mean(mdds):>9.2%}")

print(f"{'='*100}")
