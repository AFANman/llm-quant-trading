# 量化交易策略回测框架

基于LLM辅助生成的量化交易策略研究与回测系统。

## 架构

```mermaid
flowchart TB
    subgraph 数据层
        A[腾讯K线API] --> B[data_fetcher.py<br/>fetch_stock_data()<br/>fetch_realtime_quote()]
    end
    
    subgraph 指标层
        B --> C[backtest_engine.py<br/>compute_indicators()<br/>MA/RSI/ATR/布林]
        C --> D[evaluate_condition()]
    end
    
    subgraph 策略层
        E[Strategy ABC<br/>init_state()<br/>decide(row, state, ctx)] --> F[Buy / Sell / Hold]
        E --> G[MartinGrid]
        E --> H[NormalDCA]
        E --> I[MADCA]
        E --> J[RSIDCA]
        E --> K[ValueAvg]
        E --> L[DipBoost]
        M[注册中心<br/>register()<br/>clear_registry()] --> E
    end
    
    subgraph 回测层
        C --> N[backtest_framework.py<br/>backtest(strategy, df)<br/>run_all(segments)]
        O[成本模型<br/>佣金万三<br/>印花税千一<br/>滑点0.1%] --> N
        P[分时段回测<br/>每半年重置] --> N
    end
    
    subgraph 结果层
        N --> Q[性能指标<br/>Alpha / 夏普 / 卡尔马<br/>最大回撤 / 交易次数]
        N --> R[策略对比<br/>多策略排名]
        N --> S[交易记录<br/>买卖明细]
        N --> T[成本分析<br/>累计摩擦]
        N --> U[DataFrame]
    end
    
    subgraph 模拟盘
        V[paper_trading/<br/>daily_run.py<br/>state.json]
    end
    
    E --> N
    N -.-> V

    style A fill:#f5f5f5,stroke:#666
    style B fill:#fff2cc,stroke:#d6b656
    style C fill:#fff2cc,stroke:#d6b656
    style D fill:#fff2cc,stroke:#d6b656
    style E fill:#fff2cc,stroke:#d6b656
    style F fill:#f8cecc,stroke:#b85450
    style G fill:#fff2cc,stroke:#d6b656
    style H fill:#fff2cc,stroke:#d6b656
    style I fill:#fff2cc,stroke:#d6b656
    style J fill:#fff2cc,stroke:#d6b656
    style K fill:#fff2cc,stroke:#d6b656
    style L fill:#fff2cc,stroke:#d6b656
    style M fill:#fff2cc,stroke:#d6b656
    style N fill:#fff2cc,stroke:#d6b656
    style O fill:#f8cecc,stroke:#b85450
    style P fill:#fff2cc,stroke:#d6b656
    style Q fill:#fff2cc,stroke:#d6b656
    style R fill:#fff2cc,stroke:#d6b656
    style S fill:#fff2cc,stroke:#d6b656
    style T fill:#f8cecc,stroke:#b85450
    style U fill:#f5f5f5,stroke:#666
    style V fill:#ffe6cc,stroke:#d79b00
```

> 可编辑的drawio文件: [docs/architecture.drawio](docs/architecture.drawio)

## 核心特性

- **通用回测框架**: 策略注册制，支持多策略对比
- **真实交易成本**: 佣金万三 + 印花税千一 + 滑点0.1%
- **多种策略实现**:
  - 传统马丁格尔（优化版）
  - 智能定投（均线/RSI/价值平均/跌幅追加）
  - 混合策略
- **详细性能指标**: Alpha、夏普比率、最大回撤、交易次数、成本分析

## 项目结构

```
.
├── backtest_framework.py       # 通用回测框架（核心）
├── backtest_engine.py          # 回测引擎（技术指标计算）
├── data_fetcher.py             # 数据获取（腾讯K线API）
├── aggressive_strategies.py    # 8策略对比脚本
├── smart_dca_comparison.py     # 智能定投对比
├── realistic_backtest.py       # 实盘级别回测
├── detailed_backtest.py        # 详细交易记录分析
├── full_3year_backtest.py      # 3年全量回测
├── yearly_reset_backtest.py    # 每年重置回测
├── adaptive_strategy.py        # 自适应切换策略
├── martin_grid.py              # 马丁格尔参数网格搜索
├── martin_generalize.py        # 多市场泛化测试
├── trend_vs_martin.py          # 趋势 vs 马丁对比
├── paper_trading/              # 模拟盘
│   ├── daily_run.py            # 每日执行脚本
│   ├── month_replay.py         # 最近一个月回放
│   └── state.json              # 当前持仓状态
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy requests
```

### 2. 运行回测

```python
from backtest_framework import *

# 定义策略
class MyStrategy(Strategy):
    name = "我的策略"
    
    def init_state(self):
        return {'counter': 0}
    
    def decide(self, row, state, ctx):
        """
        每天被调用，返回交易决策
        
        Args:
            row: 当天数据 (close, ma20, ma60, rsi14...)
            state: 策略状态字典
            ctx: 上下文 (cash, shares, avg_cost, day_index...)
        
        Returns:
            list: [Buy(amount, reason)] 或 [Sell(shares, reason)] 或 [Hold()]
        """
        if row['close'] < row['ma20']:
            return [Buy(5000, "低于均线")]
        return [Hold()]

# 注册并运行
clear_registry()
register(MyStrategy())

segments = [
    ('2025H1', '2025-01-01', '2025-06-30'),
    ('2025H2', '2025-07-01', '2025-12-31'),
]

results = run_all(segments, init_cash=100000, code='515880')
print_summary(results)
```

### 3. 内置策略

```python
# 普通定投
NormalDCA(base_invest=5000, interval=20, label="定投")

# 均线定投（低于MA20加倍，高于减半）
MADCA(base_invest=5000, interval=20, boost=2.0, reduce=0.5, label="均线定投")

# RSI定投（RSI<30加倍，>70减半）
RSIDCA(base_invest=5000, interval=20, rsi_low=30, rsi_high=70, label="RSI定投")

# 价值平均（目标增长率）
ValueAveraging(target_growth=5000, interval=20, label="价值平均")

# 跌幅追加（跌5%加1万，跌10%加2万...）
DipBoostDCA(base_invest=5000, interval=20, 
            thresholds=[0.05, 0.10, 0.20],
            boosts=[10000, 20000, 30000], label="跌幅追加")

# 马丁格尔（底仓+加仓+止盈）
MartinGrid(base_amt=8000, dip_pct=0.07, tp_pct=0.08, 
           mults=[1, 3, 5, 7], max_add=4, label="马丁格尔")
```

## 策略对比结果（科创50，每半年重置10万）

| 策略 | 半年收益 | 总盈亏 | 平均回撤 | 特点 |
|------|---------|--------|---------|------|
| 马丁-激进3(30k) | +2.67% | +16,029 | -7.20% | 最高收益，最大回撤 |
| 均线定投(1万/10天) | +2.21% | +13,242 | -7.72% | 稳定收益 |
| 混合策略 | +1.47% | +8,846 | -2.66% | 最小回撤 |
| 马丁-稳健(8k) | +0.61% | +3,675 | -2.10% | 最保守 |
| 普通定投 | -0.37% | -2,225 | -8.59% | 基准参考 |

## 核心发现

1. **马丁格尔在暴跌中表现优异**: 2026H1科创50跌76%，马丁-稳健只亏7%，定投类亏20-30%
2. **资金利用率是关键**: 稳健版马丁（8k底仓）半年只赚600元，激进版（30k）赚2672元
3. **智能定投不如马丁格尔**: 定投在深跌时无法集中火力抄底
4. **混合策略回撤最小**: 平时定投积累，大跌切马丁加仓

## 数据来源

- 腾讯K线API: 515880（科创50ETF）
- 交易成本: 佣金万三 + 印花税千一 + 滑点0.1%

## 许可证

MIT License
