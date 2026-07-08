"""
主运行脚本 - 串联 数据获取 → 回测 → 评估
用法:
  python3 run.py <stock_symbol> <start_date> <end_date> <strategy_json_file>
  python3 run.py 600519 20220101 20241231 strategies/rsi_oversold.json
"""
import sys
import json
from data_fetcher import fetch_stock_data
from backtest_engine import run_backtest
from evaluator import compute_metrics, print_report

def main():
    if len(sys.argv) < 5:
        print("用法: python3 run.py <stock> <start> <end> <strategy.json>")
        print("示例: python3 run.py 600519 20220101 20241231 strategies/rsi_oversold.json")
        sys.exit(1)
    
    symbol = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    strategy_file = sys.argv[4]
    
    # 1. 获取数据
    print(f"[1/3] 获取 {symbol} 数据 ({start_date} ~ {end_date})...")
    df = fetch_stock_data(symbol, start_date, end_date)
    print(f"  {len(df)} 条日线数据")
    
    # 2. 加载策略
    print(f"[2/3] 加载策略: {strategy_file}")
    with open(strategy_file) as f:
        strategy = json.load(f)
    print(f"  策略名: {strategy.get('name')}")
    print(f"  入场条件: {len(strategy.get('entry_conditions', []))} 个")
    print(f"  出场条件: {len(strategy.get('exit_conditions', []))} 个")
    
    # 3. 回测
    print(f"[3/3] 执行回测...")
    result = run_backtest(df, strategy)
    
    # 4. 计算基准收益（买入持有）
    benchmark_return = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
    
    # 5. 评估
    metrics = compute_metrics(result, benchmark_return)
    print_report(metrics, result["trades"])
    
    # 6. 输出JSON（方便后续批量分析）
    output = {
        "stock": symbol,
        "period": f"{start_date}~{end_date}",
        "strategy": strategy,
        "metrics": metrics
    }
    output_file = strategy_file.replace(".json", "_result.json")
    with open(output_file, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  结果已保存: {output_file}")

if __name__ == "__main__":
    main()
