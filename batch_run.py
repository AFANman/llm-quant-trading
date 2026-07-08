"""
批量实验脚本 - 接收策略JSON列表，对指定股票批量回测
用法: python3 batch_run.py <stock> <start> <end> <strategies.json>

strategies.json格式:
[
  {"name": "策略1", "entry_conditions": [...], ...},
  {"name": "策略2", ...}
]
"""
import sys
import json
import time
from data_fetcher import fetch_stock_data
from backtest_engine import run_backtest
from evaluator import compute_metrics, print_report

def main():
    if len(sys.argv) < 5:
        print("用法: python3 batch_run.py <stock> <start> <end> <strategies.json>")
        sys.exit(1)
    
    symbol = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    strategies_file = sys.argv[4]
    
    print(f"获取 {symbol} 数据...")
    df = fetch_stock_data(symbol, start_date, end_date)
    print(f"  {len(df)} 条数据\n")
    
    with open(strategies_file) as f:
        strategies = json.load(f)
    
    benchmark_return = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
    print(f"基准收益(买入持有): {benchmark_return*100:.1f}%\n")
    
    results = []
    for i, strategy in enumerate(strategies):
        print(f"[{i+1}/{len(strategies)}] {strategy.get('name', '未命名')}")
        result = run_backtest(df, strategy)
        metrics = compute_metrics(result, benchmark_return)
        print_report(metrics, result["trades"])
        results.append({"strategy": strategy, "metrics": metrics})
    
    # 汇总排名
    print(f"\n{'='*60}")
    print("  汇总排名 (按total_return排序)")
    print(f"{'='*60}")
    ranked = sorted(results, key=lambda x: x["metrics"]["total_return_pct"], reverse=True)
    for i, r in enumerate(ranked):
        m = r["metrics"]
        print(f"  #{i+1} {m['strategy_name']:<30} "
              f"收益:{m['total_return_pct']:>7.1f}%  "
              f"夏普:{m['sharpe_ratio']:>6.2f}  "
              f"胜率:{m['win_rate_pct']:>5.1f}%  "
              f"交易:{m['num_trades']:>3}笔  "
              f"alpha:{m.get('alpha_pct', 'N/A')}")
    
    # 保存
    output_file = f"results_{symbol}_{start_date}_{end_date}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {output_file}")

if __name__ == "__main__":
    main()
