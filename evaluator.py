"""
评估模块 - 计算回测指标 + 打印报告
"""
import pandas as pd
import numpy as np

def compute_metrics(result: dict, benchmark_return: float = None) -> dict:
    trades = result["trades"]
    equity = result["equity_curve"]
    initial = result["initial_capital"]
    final = result["final_capital"]
    
    total_return = (final - initial) / initial
    days = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days
    annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1
    
    eq = equity["equity"]
    peak = eq.cummax()
    max_drawdown = ((eq - peak) / peak).min()
    
    daily_returns = eq.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() * 252 - 0.02) / (daily_returns.std() * np.sqrt(252))
    else:
        sharpe = 0
    
    num_trades = len(trades)
    if num_trades > 0:
        pnls = [t["pnl_pct"] for t in trades]
        win_rate = sum(1 for p in pnls if p > 0) / num_trades
        avg_pnl = np.mean(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    else:
        win_rate = avg_pnl = avg_win = avg_loss = profit_factor = 0
    
    m = {
        "strategy_name": result["strategy_name"],
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "num_trades": num_trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_pnl_pct": round(avg_pnl * 100, 2),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "final_capital": round(final, 2),
    }
    if benchmark_return is not None:
        m["benchmark_return_pct"] = round(benchmark_return * 100, 2)
        m["alpha_pct"] = round(total_return * 100 - benchmark_return * 100, 2)
    return m


def print_report(metrics: dict, trades: list):
    print(f"\n{'='*60}")
    print(f"  {metrics['strategy_name']}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if k == "strategy_name": continue
        print(f"  {k:<25} {v}")
    print(f"{'='*60}")
    if trades:
        print(f"\n  最近5笔交易:")
        for t in trades[-5:]:
            tag = "+" if t["pnl_pct"] > 0 else ""
            print(f"    {str(t['entry_date'])[:10]} -> {str(t['exit_date'])[:10]}  "
                  f"{t['entry_price']:.2f}->{t['exit_price']:.2f}  "
                  f"{tag}{t['pnl_pct']*100:.1f}%  ({t['exit_reason']})")
