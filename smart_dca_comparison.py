"""
智能定投策略对比 — 科创50 ETF
1. 普通定投（基准）— 每20个交易日固定投
2. 均线定投 — 价格<MA20加倍投，>MA20减半投
3. RSI定投 — RSI<30加倍投，RSI>70减半投
4. 价值平均策略 — 设定目标增长率，高抛低吸
5. 跌幅追加定投 — 每月定投+跌X%额外追加
对比: 马丁格尔（稳健版）
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

def calc_buy_cost(a): return max(a*COMMISSION_RATE, MIN_COMMISSION) + a*SLIPPAGE
def calc_sell_cost(a): return max(a*COMMISSION_RATE, MIN_COMMISSION) + a*STAMP_TAX + a*SLIPPAGE

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ─── 策略1: 普通定投 ───
def dca_normal(df, init_cash=100000, base_invest=5000, interval=20):
    """每interval个交易日投5000元"""
    df = compute_indicators(df)
    cash = init_cash; shares = 0; ti = 0; tc = 0; invest_count = 0
    
    for i in range(60, len(df)):
        price = df.iloc[i]['close']; date = df.iloc[i]['date']
        if cash >= base_invest and (i - 60) % interval == 0:
            cost = calc_buy_cost(base_invest); tc += cost
            cash -= base_invest + cost
            shares += base_invest / price
            ti += base_invest; invest_count += 1
    
    fe = cash + shares * df.iloc[-1]['close']
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    avg_cost = ti / shares if shares > 0 else 0
    return fe, ti, shares, avg_cost, tc, invest_count, bench

# ─── 策略2: 均线定投 ───
def dca_ma(df, init_cash=100000, base_invest=5000, interval=20, ma_period=20,
           boost=2.0, reduce=0.5):
    """价格<MA20投base*boost，>MA20投base*reduce"""
    df = compute_indicators(df)
    df['ma20'] = df['close'].rolling(ma_period).mean()
    cash = init_cash; shares = 0; ti = 0; tc = 0; invest_count = 0
    
    for i in range(60, len(df)):
        price = df.iloc[i]['close']; date = df.iloc[i]['date']
        ma = df.iloc[i]['ma20']
        if pd.isna(ma): continue
        
        if (i - 60) % interval == 0:
            if price < ma:
                amt = base_invest * boost
            else:
                amt = base_invest * reduce
            amt = min(amt, cash)
            if amt >= 100:
                cost = calc_buy_cost(amt); tc += cost
                cash -= amt + cost
                shares += amt / price
                ti += amt; invest_count += 1
    
    fe = cash + shares * df.iloc[-1]['close']
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    avg_cost = ti / shares if shares > 0 else 0
    return fe, ti, shares, avg_cost, tc, invest_count, bench

# ─── 策略3: RSI定投 ───
def dca_rsi(df, init_cash=100000, base_invest=5000, interval=20,
            rsi_low=30, rsi_high=70, boost=2.0, reduce=0.5):
    """RSI<30加倍投，RSI>70减半投，其他正常投"""
    df = compute_indicators(df)
    df['rsi'] = compute_rsi(df['close'], 14)
    cash = init_cash; shares = 0; ti = 0; tc = 0; invest_count = 0
    
    for i in range(60, len(df)):
        price = df.iloc[i]['close']; date = df.iloc[i]['date']
        rsi = df.iloc[i]['rsi']
        if pd.isna(rsi): continue
        
        if (i - 60) % interval == 0:
            if rsi < rsi_low:
                amt = base_invest * boost
            elif rsi > rsi_high:
                amt = base_invest * reduce
            else:
                amt = base_invest
            amt = min(amt, cash)
            if amt >= 100:
                cost = calc_buy_cost(amt); tc += cost
                cash -= amt + cost
                shares += amt / price
                ti += amt; invest_count += 1
    
    fe = cash + shares * df.iloc[-1]['close']
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    avg_cost = ti / shares if shares > 0 else 0
    return fe, ti, shares, avg_cost, tc, invest_count, bench

# ─── 策略4: 价值平均策略 ───
def value_averaging(df, init_cash=100000, target_growth=500, interval=20):
    """
    设定目标：每次定投后持仓市值 = 上次市值 + target_growth
    高抛低吸：涨多了就卖，跌多了就买更多
    """
    df = compute_indicators(df)
    cash = init_cash; shares = 0; ti = 0; tc = 0; invest_count = 0
    target_value = 0
    
    for i in range(60, len(df)):
        price = df.iloc[i]['close']; date = df.iloc[i]['date']
        
        if (i - 60) % interval == 0:
            target_value += target_growth
            current_value = shares * price
            
            diff = target_value - current_value  # 需要买入的金额
            
            if diff > 0:  # 需要买入
                amt = min(diff, cash)
                if amt >= 100:
                    cost = calc_buy_cost(amt); tc += cost
                    cash -= amt + cost
                    shares += amt / price
                    ti += amt; invest_count += 1
            elif diff < -100:  # 涨太多了，卖出
                sell_shares = abs(diff) / price
                sell_shares = min(sell_shares, shares)
                if sell_shares > 0:
                    sell_value = sell_shares * price
                    cost = calc_sell_cost(sell_value); tc += cost
                    cash += sell_value - cost
                    shares -= sell_shares
                    ti -= (sell_value - cost)  # 减少已投入
                    invest_count += 1
    
    fe = cash + shares * df.iloc[-1]['close']
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    avg_cost = ti / shares if shares > 0 else 0
    return fe, ti, shares, avg_cost, tc, invest_count, bench

# ─── 策略5: 跌幅追加定投 ───
def dca_dip_boost(df, init_cash=100000, base_invest=3000, interval=20,
                  dip_thresholds=None, boost_amounts=None):
    """
    每月定投3000元
    跌10%追加5000，跌20%追加15000，跌30%追加30000
    """
    if dip_thresholds is None: dip_thresholds = [0.10, 0.20, 0.30]
    if boost_amounts is None: boost_amounts = [5000, 15000, 30000]
    
    df = compute_indicators(df)
    cash = init_cash; shares = 0; ti = 0; tc = 0; invest_count = 0
    peak = df.iloc[60]['close']
    
    for i in range(60, len(df)):
        price = df.iloc[i]['close']; date = df.iloc[i]['date']
        peak = max(peak, price)
        drawdown = (peak - price) / peak
        
        # 定期定投
        if (i - 60) % interval == 0:
            amt = min(base_invest, cash)
            if amt >= 100:
                cost = calc_buy_cost(amt); tc += cost
                cash -= amt + cost
                shares += amt / price
                ti += amt; invest_count += 1
            
            # 跌幅追加（只触发一次，用标记避免重复）
            for j, (thresh, boost_amt) in enumerate(zip(dip_thresholds, boost_amounts)):
                if drawdown >= thresh:
                    amt = min(boost_amt, cash)
                    if amt >= 100:
                        cost = calc_buy_cost(amt); tc += cost
                        cash -= amt + cost
                        shares += amt / price
                        ti += amt; invest_count += 1
                    break  # 只触发最大一档
    
    fe = cash + shares * df.iloc[-1]['close']
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    avg_cost = ti / shares if shares > 0 else 0
    return fe, ti, shares, avg_cost, tc, invest_count, bench

# ─── 策略6: 马丁格尔（对比基准）───
def martin(df, init_cash=100000, base_amt=8000, dip_pct=0.07, tp_pct=0.08,
           mults=None, max_add=4):
    if mults is None: mults = [1,3,5,7]
    df = compute_indicators(df)
    cash = init_cash; shares = 0; ti = 0; ac = 0; avg = 0; tc = 0
    rounds = 0; profits = []
    
    for i in range(60, len(df)):
        price = df.iloc[i]['close']
        if i == 60 and shares == 0:
            c = calc_buy_cost(base_amt); tc += c; cash -= base_amt+c
            shares = base_amt/price; avg = price; ti = base_amt; ac = 0
        elif shares > 0:
            pnl = price/avg - 1
            if ac > 0 and pnl >= tp_pct:
                sv = shares*price; sc = calc_sell_cost(sv); tc += sc
                net = sv - sc; profit = net - ti
                profits.append(profit); rounds += 1
                cash += net; shares = 0; ti = 0; ac = 0; avg = 0
                if i < len(df)-5 and cash >= base_amt:
                    c = calc_buy_cost(base_amt); tc += c; cash -= base_amt+c
                    shares = base_amt/price; avg = price; ti = base_amt; ac = 0
            elif pnl <= -dip_pct*(ac+1) and ac < max_add:
                amt = base_amt*mults[ac]
                if cash >= amt:
                    c = calc_buy_cost(amt); tc += c; cash -= amt+c
                    ns = amt/price; ov = avg*shares; shares += ns
                    avg = (ov+amt)/shares; ti += amt; ac += 1
    
    fe = cash + shares*df.iloc[-1]['close']
    bench = df.iloc[-1]['close']/df.iloc[60]['close'] - 1
    return fe, ti, shares, 0, tc, rounds, bench, profits

# ─── 主程序 ───
df_all = fetch_stock_data('515880', '20210101', '20260708')
df_all['date'] = pd.to_datetime(df_all['date'])

segments = [
    ('2023H2', '2023-07-03', '2023-12-29'),
    ('2024H1', '2024-01-02', '2024-06-28'),
    ('2024H2', '2024-07-01', '2024-12-31'),
    ('2025H1', '2025-01-02', '2025-06-30'),
    ('2025H2', '2025-07-01', '2025-12-31'),
    ('2026H1', '2026-01-05', '2026-07-08'),
]

strategies = {
    '1.普通定投': lambda d: dca_normal(d, base_invest=5000, interval=20),
    '2.均线定投': lambda d: dca_ma(d, base_invest=5000, interval=20, boost=2.0, reduce=0.5),
    '3.RSI定投': lambda d: dca_rsi(d, base_invest=5000, interval=20, boost=2.0, reduce=0.5),
    '4.价值平均': lambda d: value_averaging(d, target_growth=2000, interval=20),
    '5.跌幅追加': lambda d: dca_dip_boost(d, base_invest=3000, interval=20),
}

print(f"科创50 ETF 6策略对比 — 每半年重置10万")
print(f"{'='*120}")

# 收集所有结果
results = {}

for name, func in strategies.items():
    results[name] = {'rets': [], 'pnls': [], 'benchs': [], 'alphas': [], 
                     'invests': [], 'costs': [], 'counts': []}

results['6.马丁格尔'] = {'rets': [], 'pnls': [], 'benchs': [], 'alphas': [],
                        'invests': [], 'costs': [], 'counts': [], 'profits': []}

for seg_name, start_s, end_s in segments:
    start_dt = pd.to_datetime(start_s)
    end_dt = pd.to_datetime(end_s)
    df_seg = df_all[(df_all['date'] >= start_dt) & (df_all['date'] <= end_dt)].copy()
    if len(df_seg) < 65:
        continue
    
    print(f"\n  {seg_name}（{str(df_seg.iloc[0]['date'])[:10]} → {str(df_seg.iloc[-1]['date'])[:10]}，{len(df_seg)}天）")
    print(f"  {'─'*100}")
    
    # 定投类策略
    for name, func in strategies.items():
        fe, ti, shares, avg_cost, tc, count, bench = func(df_seg)
        ret = (fe - 100000) / 100000
        pnl = fe - 100000
        alpha = ret - bench
        
        results[name]['rets'].append(ret)
        results[name]['pnls'].append(pnl)
        results[name]['benchs'].append(bench)
        results[name]['alphas'].append(alpha)
        results[name]['invests'].append(ti)
        results[name]['costs'].append(tc)
        results[name]['counts'].append(count)
        
        print(f"    {name:<12} | 投入{ti:>7,.0f} | 收益{ret:>+7.2%}（{pnl:>+7,.0f}）| Alpha{alpha:>+7.2%} | 操作{count:>2}次 | 成本{tc:>5.0f}")
    
    # 马丁格尔
    fe, ti, shares, avg_cost, tc, rounds, bench, profits = martin(df_seg)
    ret = (fe - 100000) / 100000
    pnl = fe - 100000
    alpha = ret - bench
    results['6.马丁格尔']['rets'].append(ret)
    results['6.马丁格尔']['pnls'].append(pnl)
    results['6.马丁格尔']['benchs'].append(bench)
    results['6.马丁格尔']['alphas'].append(alpha)
    results['6.马丁格尔']['invests'].append(ti)
    results['6.马丁格尔']['costs'].append(tc)
    results['6.马丁格尔']['counts'].append(rounds)
    results['6.马丁格尔']['profits'].append(sum(profits))
    
    print(f"    {'6.马丁格尔':<12} | 投入{ti:>7,.0f} | 收益{ret:>+7.2%}（{pnl:>+7,.0f}）| Alpha{alpha:>+7.2%} | {rounds:>2}轮 | 利润{sum(profits):>+6,.0f} | 成本{tc:>5.0f}")

# 汇总
print(f"\n{'='*120}")
print(f"\n汇总（{len(segments)}个半年平均）")
print(f"{'─'*120}")
print(f"  {'策略':<12} {'平均收益':>10} {'平均盈亏':>10} {'平均Alpha':>10} {'正Alpha率':>10} {'总盈亏':>10} {'平均投入':>10}")
print(f"  {'─'*80}")

all_names = list(strategies.keys()) + ['6.马丁格尔']
for name in all_names:
    r = results[name]
    n = len(r['rets'])
    avg_ret = np.mean(r['rets'])
    avg_pnl = np.mean(r['pnls'])
    avg_alpha = np.mean(r['alphas'])
    pos_alpha = sum(1 for a in r['alphas'] if a > 0)
    total_pnl = sum(r['pnls'])
    avg_invest = np.mean(r['invests'])
    
    print(f"  {name:<12} {avg_ret:>+9.2%} {avg_pnl:>+9,.0f} {avg_alpha:>+9.2%} {pos_alpha:>4}/{n:<6} {total_pnl:>+9,.0f} {avg_invest:>9,.0f}")

print(f"{'='*120}")
