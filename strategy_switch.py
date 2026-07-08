"""
Phase 4: 全仓策略切换 (Full Position Strategy Switching)
根据市场环境100%切换到对应策略，不做组合分散

核心假设：
- 下跌市 → RSI反转（防守最强）
- 上涨市 → 动量/趋势跟随（进攻最强）  
- 震荡市 → 量价策略（均衡）
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators, run_backtest
import json
import pandas as pd
import numpy as np

# 全仓切换策略配置
SWITCH_CONFIG = {
    'downtrend': '自适应RSI反转',
    'uptrend': '自适应动量策略', 
    'sideways': '自适应量价策略'
}

def run_strategy_switch(symbol, start_date, end_date, strategies_file, initial_capital=100000):
    """
    全仓策略切换：每天根据环境100%运行对应策略
    """
    df = fetch_stock_data(symbol, start_date, end_date)
    if df is None or len(df) < 60:
        return None
    df = compute_indicators(df)
    
    with open(strategies_file, 'r') as f:
        strategies = json.load(f)
    
    # 对每个策略独立回测
    strategy_results = {}
    for strat in strategies:
        result = run_backtest(df, strat, initial_capital)
        strategy_results[strat['name']] = result
    
    # 逐日：判断环境 → 100%运行对应策略
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
        
        # 获取当日应运行的策略
        target_strategy = SWITCH_CONFIG[regime]
        
        # 从该策略的equity curve中提取当日收益
        if target_strategy in strategy_results:
            eq = strategy_results[target_strategy].get('equity_curve', pd.DataFrame())
            if len(eq) > 0:
                eq = eq.set_index('date')
                if date in eq.index and i > 60:
                    prev_date = df.iloc[i-1]['date']
                    if prev_date in eq.index:
                        daily_ret = (eq.loc[date, 'equity'] - eq.loc[prev_date, 'equity']) / eq.loc[prev_date, 'equity']
                    else:
                        daily_ret = 0
                else:
                    daily_ret = 0
            else:
                daily_ret = 0
        else:
            daily_ret = 0
        
        portfolio_daily.append({'date': date, 'ret': daily_ret, 'regime': regime, 'strategy': target_strategy})
    
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
    
    # 统计各策略运行天数
    strategy_days = pdf['strategy'].value_counts().to_dict()
    
    print(f"\n{'='*60}")
    print(f"  全仓切换: {symbol} ({start_date}~{end_date})")
    print(f"{'='*60}")
    print(f"  基准: {benchmark_return:+.1f}%")
    print(f"  切换: {portfolio_return:+.1f}%  Alpha: {alpha:+.1f}%  夏普: {sharpe:+.2f}  回撤: {max_dd:.1f}%")
    print(f"  环境: 涨{regime_counts['uptrend']}天 震荡{regime_counts['sideways']}天 跌{regime_counts['downtrend']}天")
    print(f"  策略运行: 涨→{strategy_days.get('自适应动量策略', 0)}天 跌→{strategy_days.get('自适应RSI反转', 0)}天 震→{strategy_days.get('自适应量价策略', 0)}天")
    print(f"  最优单策略: {best_name} ({best_ret:+.1f}%, Alpha {best_alpha:+.1f}%)")
    improved = '切换胜' if alpha > best_alpha else '单策略胜'
    print(f"  对比: {improved} (Alpha差 {alpha - best_alpha:+.1f}%)")
    
    return {
        'symbol': symbol,
        'benchmark': round(benchmark_return, 1),
        'switch': round(portfolio_return, 1),
        'alpha': round(alpha, 1),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 1),
        'best_single': best_name,
        'best_alpha': round(best_alpha, 1),
        'regimes': regime_counts,
        'strategy_days': strategy_days
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
    
    print("\n===== 样本内全仓切换 (2022-2024) =====")
    for sym, s, e in targets_in:
        try:
            r = run_strategy_switch(sym, s, e, strategies_file)
            if r:
                all_results.append({'period': 'in-sample', **r})
        except Exception as ex:
            print(f"  {sym} 错误: {ex}")
    
    print("\n===== 样本外全仓切换 (2025) =====")
    for sym, s, e in targets_oos:
        try:
            r = run_strategy_switch(sym, s, e, strategies_file)
            if r:
                all_results.append({'period': 'out-of-sample', **r})
        except Exception as ex:
            print(f"  {sym} 错误: {ex}")
    
    # 汇总表
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"{'标的':<10} {'时期':<12} {'基准':>7} {'切换':>7} {'Alpha':>7} {'夏普':>6} {'回撤':>7} {'单策略Alpha':>11} {'胜出':>6}")
    print(f"{'-'*80}")
    for r in all_results:
        winner = '切换' if r['alpha'] > r['best_alpha'] else '单策略'
        print(f"{r['symbol']:<10} {r['period']:<12} {r['benchmark']:>+6.1f}% {r['switch']:>+6.1f}% {r['alpha']:>+6.1f}% {r['sharpe']:>+5.2f} {r['max_dd']:>+6.1f}% {r['best_alpha']:>+10.1f}% {winner:>6}")
    
    for label, key in [('样本内', 'in-sample'), ('样本外', 'out-of-sample')]:
        items = [r for r in all_results if r['period'] == key]
        if items:
            avg_a = np.mean([r['alpha'] for r in items])
            pos = sum(1 for r in items if r['alpha'] > 0)
            switch_win = sum(1 for r in items if r['alpha'] > r['best_alpha'])
            print(f"\n{label}: 平均Alpha {avg_a:+.1f}%, 正Alpha {pos}/{len(items)}, 切换胜出 {switch_win}/{len(items)}")
