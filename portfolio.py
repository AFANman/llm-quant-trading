"""
策略组合：按市场环境动态分配权重
- 下跌市 → RSI反转/布林带权重高
- 上涨市 → 动量/量价/趋势跟随权重高
- 震荡市 → 均衡分配

原理：对每个策略独立回测得到equity curve，计算每日收益率，
再按当日市场环境的权重加权平均，得到组合每日收益率。
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators, run_backtest
import json
import pandas as pd
import numpy as np

# 环境权重配置（基于Phase 3实验结论）
REGIME_WEIGHTS = {
    'downtrend': {
        '自适应RSI反转': 0.35,
        '自适应布林带均值回归': 0.30,
        '自适应量价策略': 0.15,
        '自适应动量策略': 0.10,
        '自适应趋势跟随': 0.05,
        '自适应综合策略': 0.05,
    },
    'uptrend': {
        '自适应动量策略': 0.30,
        '自适应量价策略': 0.25,
        '自适应趋势跟随': 0.20,
        '自适应RSI反转': 0.10,
        '自适应布林带均值回归': 0.10,
        '自适应综合策略': 0.05,
    },
    'sideways': {
        '自适应RSI反转': 0.20,
        '自适应布林带均值回归': 0.20,
        '自适应量价策略': 0.20,
        '自适应动量策略': 0.15,
        '自适应趋势跟随': 0.15,
        '自适应综合策略': 0.10,
    }
}

def run_portfolio(symbol, start_date, end_date, strategies_file, initial_capital=100000):
    df = fetch_stock_data(symbol, start_date, end_date)
    if df is None or len(df) < 60:
        return None
    df = compute_indicators(df)
    
    with open(strategies_file, 'r') as f:
        strategies = json.load(f)
    
    # 对每个策略独立回测，提取equity curve的每日收益率
    strategy_daily_ret = {}
    strategy_results = {}
    for strat in strategies:
        result = run_backtest(df, strat, initial_capital)
        eq = result.get('equity_curve', pd.DataFrame())
        name = strat['name']
        strategy_results[name] = result
        
        if len(eq) > 0:
            eq = eq.set_index('date')
            daily_r = eq['equity'].pct_change().fillna(0)
            strategy_daily_ret[name] = daily_r
    
    # 逐日：判断环境 → 按权重加权组合收益
    portfolio_daily = []
    regime_counts = {'uptrend': 0, 'downtrend': 0, 'sideways': 0}
    
    for i in range(60, len(df)):
        row = df.iloc[i]
        date = row['date']
        
        # 判断环境
        regime = 'sideways'
        if not pd.isna(row.get('ma20')) and not pd.isna(row.get('ma60')):
            if row['ma20'] > row['ma60'] and row['close'] > row['ma20']:
                regime = 'uptrend'
            elif row['ma20'] < row['ma60'] and row['close'] < row['ma20']:
                regime = 'downtrend'
        regime_counts[regime] += 1
        
        weights = REGIME_WEIGHTS[regime]
        daily_ret = 0.0
        for sname, w in weights.items():
            if sname in strategy_daily_ret:
                sr = strategy_daily_ret[sname]
                if date in sr.index:
                    daily_ret += sr[date] * w
        
        portfolio_daily.append({'date': date, 'ret': daily_ret, 'regime': regime})
    
    pdf = pd.DataFrame(portfolio_daily)
    cum = (1 + pdf['ret']).cumprod()
    portfolio_return = (cum.iloc[-1] - 1) * 100
    benchmark_return = (df.iloc[-1]['close'] - df.iloc[60]['close']) / df.iloc[60]['close'] * 100
    alpha = portfolio_return - benchmark_return
    
    # 夏普
    daily_std = pdf['ret'].std()
    sharpe = (pdf['ret'].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0
    
    # 最大回撤
    cum_max = cum.cummax()
    drawdown = (cum - cum_max) / cum_max * 100
    max_dd = drawdown.min()
    
    # 单策略最优
    best_name = ''
    best_ret = -999
    for name, res in strategy_results.items():
        eq = res.get('equity_curve', pd.DataFrame())
        if len(eq) > 0:
            r = (eq.iloc[-1]['equity'] - initial_capital) / initial_capital * 100
            if r > best_ret:
                best_ret = r
                best_name = name
    best_alpha = best_ret - benchmark_return
    
    print(f"\n{'='*60}")
    print(f"  {symbol} ({start_date}~{end_date})")
    print(f"{'='*60}")
    print(f"  基准: {benchmark_return:+.1f}%")
    print(f"  组合: {portfolio_return:+.1f}%  Alpha: {alpha:+.1f}%  夏普: {sharpe:+.2f}  回撤: {max_dd:.1f}%")
    print(f"  环境: 涨{regime_counts['uptrend']}天 震荡{regime_counts['sideways']}天 跌{regime_counts['downtrend']}天")
    print(f"  最优单策略: {best_name} ({best_ret:+.1f}%, Alpha {best_alpha:+.1f}%)")
    improved = '组合胜' if alpha > best_alpha else '单策略胜'
    print(f"  对比: {improved} (Alpha差 {alpha - best_alpha:+.1f}%)")
    
    return {
        'symbol': symbol,
        'benchmark': round(benchmark_return, 1),
        'portfolio': round(portfolio_return, 1),
        'alpha': round(alpha, 1),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 1),
        'best_single': best_name,
        'best_alpha': round(best_alpha, 1),
        'regimes': regime_counts
    }

if __name__ == '__main__':
    strategies_file = os.path.join(os.path.dirname(__file__), 'strategies', 'phase3_strategies.json')
    
    targets_in = [
        ('600519', '20220101', '20241231'),
        ('515880', '20220101', '20241231'),
        ('510050', '20220101', '20241231'),
        ('159915', '20220101', '20241231'),
        ('512660', '20220101', '20241231'),
        ('515030', '20220101', '20241231'),
        ('000858', '20220101', '20241231'),
        ('601318', '20220101', '20241231'),
        ('300750', '20220101', '20241231'),
    ]
    
    targets_oos = [
        ('600519', '20250101', '20251231'),
        ('515880', '20250101', '20251231'),
        ('159915', '20250101', '20251231'),
        ('510050', '20250101', '20251231'),
        ('300750', '20250101', '20251231'),
        ('000858', '20250101', '20251231'),
    ]
    
    all_results = []
    
    print("\n===== 样本内组合 (2022-2024) =====")
    for sym, s, e in targets_in:
        try:
            r = run_portfolio(sym, s, e, strategies_file)
            if r:
                all_results.append({'period': 'in-sample', **r})
        except Exception as ex:
            print(f"  {sym} 错误: {ex}")
    
    print("\n===== 样本外组合 (2025) =====")
    for sym, s, e in targets_oos:
        try:
            r = run_portfolio(sym, s, e, strategies_file)
            if r:
                all_results.append({'period': 'out-of-sample', **r})
        except Exception as ex:
            print(f"  {sym} 错误: {ex}")
    
    # 汇总表
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"{'标的':<10} {'时期':<12} {'基准':>7} {'组合':>7} {'Alpha':>7} {'夏普':>6} {'回撤':>7} {'单策略Alpha':>11} {'胜出':>6}")
    print(f"{'-'*80}")
    for r in all_results:
        winner = '组合' if r['alpha'] > r['best_alpha'] else '单策略'
        print(f"{r['symbol']:<10} {r['period']:<12} {r['benchmark']:>+6.1f}% {r['portfolio']:>+6.1f}% {r['alpha']:>+6.1f}% {r['sharpe']:>+5.2f} {r['max_dd']:>+6.1f}% {r['best_alpha']:>+10.1f}% {winner:>6}")
    
    for label, key in [('样本内', 'in-sample'), ('样本外', 'out-of-sample')]:
        items = [r for r in all_results if r['period'] == key]
        if items:
            avg_a = np.mean([r['alpha'] for r in items])
            pos = sum(1 for r in items if r['alpha'] > 0)
            combo_win = sum(1 for r in items if r['alpha'] > r['best_alpha'])
            print(f"\n{label}: 平均Alpha {avg_a:+.1f}%, 正Alpha {pos}/{len(items)}, 组合胜出 {combo_win}/{len(items)}")
