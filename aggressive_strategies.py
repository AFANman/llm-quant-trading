"""
激进版策略对比 — 提高资金利用率
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from backtest_framework import *

clear_registry()

# 马丁格尔系列 — 4个变体
register(MartinGrid(base_amt=8000, dip_pct=0.07, tp_pct=0.08, mults=[1,3,5,7], max_add=4, label="马丁-稳健(8k/7%/8%)"))
register(MartinGrid(base_amt=15000, dip_pct=0.07, tp_pct=0.08, mults=[1,3,5], max_add=3, label="马丁-中(15k/7%/8%)"))
register(MartinGrid(base_amt=20000, dip_pct=0.05, tp_pct=0.10, mults=[1,2,3], max_add=3, label="马丁-激进2(20k/5%/10%)"))
register(MartinGrid(base_amt=30000, dip_pct=0.10, tp_pct=0.15, mults=[1,2], max_add=2, label="马丁-激进3(30k/10%/15%)"))

# 定投系列
register(NormalDCA(base_invest=15000, interval=20, label="定投(1.5万/20天)"))
register(MADCA(base_invest=10000, interval=10, boost=2.0, reduce=0.5, label="均线定投(1万/10天)"))

# 跌幅追加
register(DipBoostDCA(base_invest=10000, interval=20, 
                     thresholds=[0.05, 0.10, 0.20, 0.30],
                     boosts=[10000, 20000, 30000, 40000], label="跌幅追加(4档)"))

# 混合型
class Hybrid(Strategy):
    """混合型：定投积累底仓，大跌切马丁加仓"""
    def __init__(self, dca_amt=5000, dca_interval=20, martin_base=15000, 
                 dip_threshold=0.10, martin_mults=None):
        self.name = "混合(定投+马丁)"
        self.dca_amt = dca_amt
        self.dca_interval = dca_interval
        self.martin_base = martin_base
        self.dip_threshold = dip_threshold
        self.martin_mults = martin_mults or [1, 3, 5]
    def init_state(self):
        return {'peak': 0, 'dca_count': 0, 'martin_count': 0, 'mode': 'dca'}
    def decide(self, row, state, ctx):
        price = ctx['price']
        state['peak'] = max(state['peak'], price)
        drawdown = (state['peak'] - price) / state['peak'] if state['peak'] > 0 else 0
        
        if drawdown >= self.dip_threshold and state['mode'] == 'dca':
            state['mode'] = 'martin'
        
        if state['mode'] == 'martin' and state['martin_count'] < len(self.martin_mults):
            expected_drawdown = self.dip_threshold * (state['martin_count'] + 1)
            if drawdown >= expected_drawdown:
                amt = self.martin_base * self.martin_mults[state['martin_count']]
                if ctx['cash'] >= amt:
                    state['martin_count'] += 1
                    return [Buy(amt, f"马丁加仓{state['martin_count']}")]
        
        if ctx['day_index'] % self.dca_interval == 0 and ctx['cash'] >= self.dca_amt:
            state['dca_count'] += 1
            return [Buy(self.dca_amt, "定投")]
        
        if ctx['shares'] > 0:
            pnl = price / ctx['avg_cost'] - 1
            if pnl >= 0.15:
                state['mode'] = 'dca'
                state['martin_count'] = 0
                return [Sell(-1, f"止盈{pnl:.1%}")]
        
        return [Hold()]

register(Hybrid())

# 跑对比
segments = [
    ('2023H2', '2023-07-03', '2023-12-29'),
    ('2024H1', '2024-01-02', '2024-06-28'),
    ('2024H2', '2024-07-01', '2024-12-31'),
    ('2025H1', '2025-01-02', '2025-06-30'),
    ('2025H2', '2025-07-01', '2025-12-31'),
    ('2026H1', '2026-01-05', '2026-07-08'),
]

df = run_all(segments, init_cash=100000)
