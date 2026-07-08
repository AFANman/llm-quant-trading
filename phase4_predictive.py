"""
Phase 4: 前瞻择时全仓切换
用3个前瞻择时器替代MA20/MA60滞后指标，提前5-10天切换策略

对比：
- Phase 4A: 滞后切换（MA20>MA60）→ 已测
- Phase 4B: 前瞻切换（动量加速度/量异动/RSI极值）→ 本次
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators, run_backtest, evaluate_condition
import json
import pandas as pd
import numpy as np

def determine_regime(row, predictor):
    """用前瞻择时器判断环境"""
    signals = predictor.get('regime_signals', {})
    
    # 检查上涨
    uptrend_conds = signals.get('uptrend', [])
    if uptrend_conds and all(evaluate_condition(row, c) for c in uptrend_conds):
        return 'uptrend'
    
    # 检查下跌
    downtrend_conds = signals.get('downtrend', [])
    if downtrend_conds and all(evaluate_condition(row, c) for c in downtrend_conds):
        return 'downtrend'
    
    # 检查震荡
    sideways_conds = signals.get('sideways', [])
    if sideways_conds and all(evaluate_condition(row, c) for c in sideways_conds):
        return 'sideways'
    
    return None  # 无信号

# 全仓切换配置：环境 → 最佳策略
SWITCH_MAP = {
    'downtrend': '自适应RSI反转',
    'uptrend': '自适应动量策略',
    'sideways': '自适应量价策略'
}

def run_phase4(symbol, start_date, end_date, strategies_file, predictors_file, initial_capital=100000):
    df = fetch_stock_data(symbol, start_date, end_date)
    if df is None or len(df) < 60:
        return None
    df = compute_indicators(df)
    
    with open(strategies_file, 'r') as f:
        strategies = json.load(f)
    with open(predictors_file, 'r') as f:
        predictors_raw = json.load(f)
    predictors = predictors_raw['strategies'] if 'strategies' in predictors_raw else predictors_raw
    
    # 独立回测每个策略
    strategy_equity = {}
    strategy_results = {}
    for strat in strategies:
        result = run_backtest(df, strat, initial_capital)
        strategy_results[strat['name']] = result
        eq = result.get('equity_curve', pd.DataFrame())
        if len(eq) > 0:
            eq = eq.set_index('date')
            strategy_equity[strat['name']] = eq
    
    # 对每个前瞻择时器分别测试
    predictor_results = []
    
    for predictor in predictors:
        pname = predictor['name']
        
        # 逐日判断环境 + 全仓切换
        portfolio_daily = []
        regime_counts = {'uptrend': 0, 'downtrend': 0, 'sideways': 0, 'unknown': 0}
        current_regime = 'sideways'
        
        for i in range(60, len(df)):
            row = df.iloc[i]
            date = row['date']
            
            # 用前瞻择时器判断
            new_regime = determine_regime(row, predictor)
            if new_regime:
                current_regime = new_regime
            
            regime_counts[current_regime] = regime_counts.get(current_regime, 0) + 1
            
            # 100%切换到对应策略
            target_strategy = SWITCH_MAP[current_regime]
            
            if target_strategy in strategy_equity:
                eq = strategy_equity[target_strategy]
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
            
            portfolio_daily.append({'date': date, 'ret': daily_ret, 'regime': current_regime})
        
        pdf = pd.DataFrame(portfolio_daily)
        cum = (1 + pdf['ret']).cumprod()
        portfolio_return = (cum.iloc[-1] - 1) * 100
        benchmark_return = (df.iloc[-1]['close'] - df.iloc[60]['close']) / df.iloc[60]['close'] * 100
        alpha = portfolio_return - benchmark_return
        
        daily_std = pdf['ret'].std()
        sharpe = (pdf['ret'].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        cum_max = cum.cummax()
        drawdown = (cum - cum_max) / cum_max * 100
        max_dd = drawdown.min()
        
        predictor_results.append({
            'predictor': pname,
            'return': round(portfolio_return, 1),
            'alpha': round(alpha, 1),
            'sharpe': round(sharpe, 2),
            'max_dd': round(max_dd, 1),
            'regimes': regime_counts
        })
    
    # 滞后基准（MA20>MA60）
    lag_portfolio = []
    for i in range(60, len(df)):
        row = df.iloc[i]
        date = row['date']
        regime = 'sideways'
        if not pd.isna(row.get('ma20')) and not pd.isna(row.get('ma60')):
            if row['ma20'] > row['ma60'] and row['close'] > row['ma20']:
                regime = 'uptrend'
            elif row['ma20'] < row['ma60'] and row['close'] < row['ma20']:
                regime = 'downtrend'
        target = SWITCH_MAP[regime]
        if target in strategy_equity:
            eq = strategy_equity[target]
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
        lag_portfolio.append(daily_ret)
    
    lag_cum = (1 + pd.Series(lag_portfolio)).cumprod()
    lag_return = (lag_cum.iloc[-1] - 1) * 100
    benchmark_return = (df.iloc[-1]['close'] - df.iloc[60]['close']) / df.iloc[60]['close'] * 100
    lag_alpha = lag_return - benchmark_return
    
    # 最优单策略
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
    
    # 打印
    print(f"\n{'='*70}")
    print(f"  Phase 4 前瞻择时: {symbol} ({start_date}~{end_date})")
    print(f"{'='*70}")
    print(f"  基准: {benchmark_return:+.1f}%")
    print(f"  滞后切换(MA): {lag_return:+.1f}%  Alpha: {lag_alpha:+.1f}%")
    print(f"  最优单策略: {best_name} Alpha: {best_alpha:+.1f}%")
    print(f"  {'-'*70}")
    
    best_pred_alpha = -999
    best_pred_name = ''
    for pr in predictor_results:
        winner = ''
        if pr['alpha'] > lag_alpha:
            winner = ' ★超前'
        if pr['alpha'] > best_alpha:
            winner = ' ★★全胜'
        print(f"  {pr['predictor']:<25} {pr['return']:+6.1f}% Alpha:{pr['alpha']:+6.1f}% 夏普:{pr['sharpe']:+.2f} 回撤:{pr['max_dd']:.1f}%{winner}")
        if pr['alpha'] > best_pred_alpha:
            best_pred_alpha = pr['alpha']
            best_pred_name = pr['predictor']
    
    return {
        'symbol': symbol,
        'benchmark': round(benchmark_return, 1),
        'lag_alpha': round(lag_alpha, 1),
        'best_single_alpha': round(best_alpha, 1),
        'best_predictor': best_pred_name,
        'best_predictor_alpha': round(best_pred_alpha, 1),
        'predictors': predictor_results
    }

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    strategies_file = os.path.join(base, 'strategies', 'phase3_strategies.json')
    predictors_file = os.path.join(base, 'strategies', 'phase4_regime_predictors.json')
    
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
    
    print("\n===== Phase 4 前瞻择时 样本内 (2022-2024) =====")
    for sym, s, e in targets_in:
        try:
            r = run_phase4(sym, s, e, strategies_file, predictors_file)
            if r:
                r['period'] = 'in-sample'
                all_results.append(r)
        except Exception as ex:
            print(f"  {sym} 错误: {ex}")
    
    print("\n===== Phase 4 前瞻择时 样本外 (2025) =====")
    for sym, s, e in targets_oos:
        try:
            r = run_phase4(sym, s, e, strategies_file, predictors_file)
            if r:
                r['period'] = 'out-of-sample'
                all_results.append(r)
        except Exception as ex:
            print(f"  {sym} 错误: {ex}")
    
    # 汇总
    print(f"\n{'='*80}")
    print(f"{'标的':<10} {'时期':<12} {'基准':>7} {'滞后Alpha':>9} {'最优预测Alpha':>13} {'单策略Alpha':>11} {'胜出':>8}")
    print(f"{'-'*80}")
    for r in all_results:
        best = max(r['lag_alpha'], r['best_predictor_alpha'])
        winner = '前瞻' if r['best_predictor_alpha'] > r['lag_alpha'] else '滞后'
        if best > r['best_single_alpha']:
            winner += '+胜单策略'
        print(f"{r['symbol']:<10} {r['period']:<12} {r['benchmark']:>+6.1f}% {r['lag_alpha']:>+8.1f}% {r['best_predictor_alpha']:>+12.1f}% {r['best_single_alpha']:>+10.1f}% {winner}")
    
    for label, key in [('样本内', 'in-sample'), ('样本外', 'out-of-sample')]:
        items = [r for r in all_results if r['period'] == key]
        if items:
            pred_win_lag = sum(1 for r in items if r['best_predictor_alpha'] > r['lag_alpha'])
            pred_win_single = sum(1 for r in items if r['best_predictor_alpha'] > r['best_single_alpha'])
            avg_pred = np.mean([r['best_predictor_alpha'] for r in items])
            avg_lag = np.mean([r['lag_alpha'] for r in items])
            avg_single = np.mean([r['best_single_alpha'] for r in items])
            print(f"\n{label}:")
            print(f"  前瞻平均Alpha: {avg_pred:+.1f}%")
            print(f"  滞后平均Alpha: {avg_lag:+.1f}%")
            print(f"  单策略平均Alpha: {avg_single:+.1f}%")
            print(f"  前瞻胜滞后: {pred_win_lag}/{len(items)}")
            print(f"  前瞻胜单策略: {pred_win_single}/{len(items)}")
