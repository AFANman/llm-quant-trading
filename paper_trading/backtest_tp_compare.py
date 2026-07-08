"""
马丁格尔回测：对比 +3% 止盈 vs +8% 止盈
多时段回测：2025年 + 2024年 + 2023-2024年（含震荡/回调）
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd
import numpy as np

def martingale_backtest(df, init_cash=100000, base_amt=8000,
                        dip_pct=0.07, tp_pct=0.08,
                        mults=[1, 3, 5, 7], max_add=4):
    df = compute_indicators(df)
    n = len(df)
    
    cash = init_cash - base_amt
    shares = base_amt / df.iloc[0]['close']
    cost_basis = df.iloc[0]['close']
    add_count = 0
    total_invested = base_amt
    peak_since_buy = cost_basis
    
    trades = []
    equity = []
    
    for i in range(1, n):
        row = df.iloc[i]
        price = row['close']
        
        position_value = shares * price
        
        if price > peak_since_buy:
            peak_since_buy = price
        
        pnl_pct = (price / cost_basis - 1) if cost_basis > 0 else 0
        total = cash + position_value
        
        equity.append({
            'date': row['date'], 'close': price,
            'equity': total, 'cash': cash,
            'position': position_value, 'pnl_pct': pnl_pct
        })
        
        # 止盈
        if add_count > 0 and pnl_pct >= tp_pct:
            base_shares = base_amt / price
            sell_shares = shares - base_shares
            if sell_shares > 0:
                sell_value = sell_shares * price
                cash += sell_value
                shares = base_shares
                add_count = 0
                total_invested = base_amt
                cost_basis = price
                peak_since_buy = price
                trades.append({
                    'date': row['date'], 'action': 'SELL_TP',
                    'price': price, 'amount': sell_value,
                    'add_count': 0,
                    'pnl_pct': pnl_pct, 'note': f'止盈+{pnl_pct:.1%}'
                })
        
        # 加仓
        elif add_count < max_add:
            dip = -pnl_pct
            threshold = dip_pct * (add_count + 1)
            if dip >= threshold:
                amt = base_amt * mults[add_count]
                if cash >= amt:
                    new_shares = amt / price
                    cash -= amt
                    shares += new_shares
                    total_invested += amt
                    add_count += 1
                    cost_basis = total_invested / shares
                    trades.append({
                        'date': row['date'], 'action': 'ADD',
                        'price': price, 'amount': amt,
                        'add_count': add_count,
                        'pnl_pct': pnl_pct, 'note': f'加仓第{add_count}次 跌{dip:.1%} 投{amt}'
                    })
    
    final_price = df.iloc[-1]['close']
    final_equity = cash + shares * final_price
    total_return = (final_equity - init_cash) / init_cash
    bench_return = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1)
    
    eq_df = pd.DataFrame(equity)
    peak = eq_df['equity'].cummax()
    max_dd = ((eq_df['equity'] - peak) / peak).min()
    
    # 计算每轮收益（ADD→SELL_TP）
    add_trades = [t for t in trades if t['action'] == 'ADD']
    sell_trades = [t for t in trades if t['action'] == 'SELL_TP']
    
    # 计算资金利用率（平均持仓占比）
    avg_pos_ratio = eq_df['position'] / eq_df['equity']
    
    return {
        'total_return': total_return,
        'benchmark_return': bench_return,
        'alpha': total_return - bench_return,
        'max_drawdown': max_dd,
        'final_equity': final_equity,
        'total_trades': len(trades),
        'add_count_total': len(add_trades),
        'sell_count_total': len(sell_trades),
        'avg_pos_ratio': avg_pos_ratio.mean(),
        'max_pos_ratio': avg_pos_ratio.max(),
        'trades': trades,
    }


def run_comparison(name, code, start, end):
    print(f"\n{'#'*70}")
    print(f"# {name} ({code}) | {start} ~ {end}")
    print(f"{'#'*70}")
    
    df = fetch_stock_data(code, start, end)
    print(f"数据: {len(df)} 条 | {df.iloc[0]['date'].strftime('%Y-%m-%d')} ~ {df.iloc[-1]['date'].strftime('%Y-%m-%d')}")
    print(f"价格: {df.iloc[0]['close']:.4f} → {df.iloc[-1]['close']:.4f} ({(df.iloc[-1]['close']/df.iloc[0]['close']-1):+.1%})")
    print(f"最低: {df['close'].min():.4f}  最高: {df['close'].max():.4f}")
    print(f"最大波幅: {(df['close'].max()/df['close'].min()-1):.1%}")
    
    results = {}
    for tp in [0.03, 0.08]:
        label = f"+{int(tp*100)}%"
        r = martingale_backtest(df, tp_pct=tp)
        results[label] = r
        
        print(f"\n  ── 止盈 {label} ──")
        print(f"  策略收益:    {r['total_return']:+.2%}")
        print(f"  Alpha:       {r['alpha']:+.2%}")
        print(f"  最大回撤:    {r['max_drawdown']:.2%}")
        print(f"  最终权益:    ¥{r['final_equity']:,.0f}")
        print(f"  加仓次数:    {r['add_count_total']}")
        print(f"  止盈次数:    {r['sell_count_total']}")
        print(f"  平均仓位比:  {r['avg_pos_ratio']:.1%}")
        print(f"  最大仓位比:  {r['max_pos_ratio']:.1%}")
        
        if r['trades']:
            print(f"  交易明细:")
            for t in r['trades']:
                d = pd.Timestamp(t['date']).strftime('%Y-%m-%d')
                print(f"    {d} | {t['action']:>8s} | 价:{t['price']:.4f} | 额:{t['amount']:>8.0f} | 盈亏:{t['pnl_pct']:+.2%} | {t['note']}")
        else:
            print(f"  （无加仓/止盈触发，仅底仓持有）")
    
    # 对比
    r3, r8 = results['+3%'], results['+8%']
    print(f"\n  ── 对比 ──")
    print(f"  {'指标':<16} {'止盈+3%':>12} {'止盈+8%':>12} {'差异':>12}")
    print(f"  {'─'*56}")
    print(f"  {'策略收益':<16} {r3['total_return']:>+11.2%} {r8['total_return']:>+11.2%} {r3['total_return']-r8['total_return']:>+11.2%}")
    print(f"  {'Alpha':<16} {r3['alpha']:>+11.2%} {r8['alpha']:>+11.2%} {r3['alpha']-r8['alpha']:>+11.2%}")
    print(f"  {'最大回撤':<16} {r3['max_drawdown']:>11.2%} {r8['max_drawdown']:>11.2%}")
    print(f"  {'加仓次数':<16} {r3['add_count_total']:>12} {r8['add_count_total']:>12}")
    print(f"  {'止盈次数':<16} {r3['sell_count_total']:>12} {r8['sell_count_total']:>12}")
    
    return results


# ── 多时段回测 ──
print("=" * 70)
print("马丁格尔止盈对比回测")
print("参数: 底仓8000 | 加仓间距7% | 翻倍1-3-5-7 | 总资金10万")
print("=" * 70)

# 1. 2025年（强牛市）
try:
    run_comparison("科创50 · 2025年", '588060', '20250101', '20251231')
except Exception as e:
    print(f"  错误: {e}")

# 2. 2024年（震荡+急跌）
try:
    run_comparison("科创50 · 2024年", '588060', '20240101', '20241231')
except Exception as e:
    print(f"  错误: {e}")

# 3. 2023-2024年（更长周期，含多轮回调）
try:
    run_comparison("科创50 · 2023-2024年", '588060', '20230101', '20241231')
except Exception as e:
    print(f"  错误: {e}")

# 4. 2024-2025年（跨年，含924行情）
try:
    run_comparison("科创50 · 2024-2025年", '588060', '20240101', '20251231')
except Exception as e:
    print(f"  错误: {e}")
