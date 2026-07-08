"""
通用回测框架 — 策略注册 + 统一评估 + 批量对比
用法：
    from backtest_framework import *
    
    # 定义策略
    class MyStrategy(Strategy):
        name = "我的策略"
        def init_state(self): return {}
        def decide(self, row, state, ctx): return [Buy(5000)]
    
    # 注册并跑
    register(MyStrategy)
    run_all(segments)
"""
import sys
sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional

# ─── 交易成本 ───
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.001
SLIPPAGE = 0.001

def calc_buy_cost(a): return max(a*COMMISSION_RATE, MIN_COMMISSION) + a*SLIPPAGE
def calc_sell_cost(a): return max(a*COMMISSION_RATE, MIN_COMMISSION) + a*STAMP_TAX + a*SLIPPAGE

# ─── 交易动作 ───
@dataclass
class Buy:
    amount: float  # 买入金额
    reason: str = ""

@dataclass
class Sell:
    shares: float  # 卖出股数（-1=全部卖出）
    reason: str = ""

@dataclass
class Hold:
    pass

Action = Buy | Sell | Hold

# ─── 策略基类 ───
class Strategy(ABC):
    name: str = "未命名"
    
    def __init__(self, params: dict = None):
        self.params = params or {}
    
    @abstractmethod
    def init_state(self) -> dict:
        """初始化策略状态"""
        pass
    
    @abstractmethod
    def decide(self, row: pd.Series, state: dict, ctx: dict) -> List[Action]:
        """
        每天决策：返回交易动作列表
        row: 当日数据（close, date, 以及所有指标）
        state: 策略自定义状态（可读写）
        ctx: 上下文（cash, shares, avg_cost, price, day_index, total_days）
        """
        pass

# ─── 回测引擎 ───
@dataclass
class BacktestResult:
    strategy_name: str
    init_cash: float
    final_equity: float
    total_invested: float
    ret: float          # 策略收益率
    bench: float        # 基准收益率
    alpha: float        # Alpha
    mdd: float          # 最大回撤
    sharpe: float       # 夏普比率
    calmar: float       # 卡尔马比率
    total_cost: float   # 总交易成本
    trade_count: int    # 交易次数
    realized_profit: float  # 已实现利润
    unrealized_profit: float  # 浮盈
    equity_curve: pd.Series  # 每日净值
    trade_log: list     # 交易日志
    start_date: str
    end_date: str

def backtest(strategy: Strategy, df: pd.DataFrame, init_cash: float = 100000) -> BacktestResult:
    """运行单个策略回测"""
    df = compute_indicators(df).reset_index(drop=True)
    
    cash = init_cash
    shares = 0.0
    total_invested = 0.0
    avg_cost = 0.0
    total_cost = 0.0
    trade_count = 0
    realized_profit = 0.0
    equity_list = []
    trade_log = []
    
    state = strategy.init_state()
    
    for i in range(60, len(df)):
        row = df.iloc[i]
        price = row['close']
        date = row['date']
        
        ctx = {
            'cash': cash,
            'shares': shares,
            'avg_cost': avg_cost,
            'price': price,
            'day_index': i - 60,
            'total_days': len(df) - 60,
            'row_index': i,
        }
        
        actions = strategy.decide(row, state, ctx)
        
        for action in actions:
            if isinstance(action, Buy):
                amt = min(action.amount, cash)
                if amt >= 100:
                    cost = calc_buy_cost(amt)
                    total_cost += cost
                    cash -= amt + cost
                    new_shares = amt / price
                    # 更新均价
                    old_value = avg_cost * shares
                    shares += new_shares
                    avg_cost = (old_value + amt) / shares if shares > 0 else 0
                    total_invested += amt
                    trade_count += 1
                    trade_log.append(f"{str(date)[:10]} 买入 {amt:,.0f}元@{price:.3f} {action.reason}")
                    
            elif isinstance(action, Sell):
                sell_shares = shares if action.shares == -1 else min(action.shares, shares)
                if sell_shares > 0:
                    sell_value = sell_shares * price
                    cost = calc_sell_cost(sell_value)
                    total_cost += cost
                    cash += sell_value - cost
                    # 计算利润
                    profit = (price - avg_cost) * sell_shares - cost
                    realized_profit += profit
                    total_invested -= avg_cost * sell_shares
                    trade_count += 1
                    trade_log.append(f"{str(date)[:10]} 卖出 {sell_shares:,.0f}股@{price:.3f} 利润{profit:+,.0f} {action.reason}")
                    shares -= sell_shares
                    if shares < 0.01:
                        shares = 0
                        avg_cost = 0
        
        # 记录每日净值
        position_value = shares * price
        equity = cash + position_value
        equity_list.append(equity)
    
    # 计算指标
    eq = pd.Series(equity_list)
    final_equity = eq.iloc[-1]
    ret = (final_equity - init_cash) / init_cash
    bench = df.iloc[-1]['close'] / df.iloc[60]['close'] - 1
    alpha = ret - bench
    
    peak = eq.cummax()
    drawdowns = (eq - peak) / peak
    mdd = drawdowns.min()
    
    daily_ret = eq.pct_change().dropna()
    rf = 0.02 / 252
    sharpe = (daily_ret.mean() - rf) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 and len(daily_ret) > 10 else 0
    
    n = len(eq)
    ann_ret = (1 + ret) ** (252 / max(n, 1)) - 1
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    
    unrealized = (df.iloc[-1]['close'] - avg_cost) * shares if shares > 0 else 0
    
    return BacktestResult(
        strategy_name=strategy.name,
        init_cash=init_cash,
        final_equity=final_equity,
        total_invested=total_invested,
        ret=ret,
        bench=bench,
        alpha=alpha,
        mdd=mdd,
        sharpe=sharpe,
        calmar=calmar,
        total_cost=total_cost,
        trade_count=trade_count,
        realized_profit=realized_profit,
        unrealized_profit=unrealized,
        equity_curve=eq,
        trade_log=trade_log,
        start_date=str(df.iloc[60]['date'])[:10],
        end_date=str(df.iloc[-1]['date'])[:10],
    )

# ─── 策略注册 ───
_registry: List[Strategy] = []
_id_counter = [0]

def register(strategy: Strategy):
    """注册策略（支持同名多实例，按注册顺序运行）"""
    _id_counter[0] += 1
    strategy._id = _id_counter[0]
    _registry.append(strategy)

def get_all_strategies() -> List[Strategy]:
    return list(_registry)

def clear_registry():
    _registry.clear()
    _id_counter[0] = 0

# ─── 批量运行 ───
def run_all(segments: list, strategies: list = None, init_cash: float = 100000,
            code: str = '515880', verbose: bool = True) -> pd.DataFrame:
    """
    批量跑所有策略 × 所有时段
    
    segments: [(name, start_date, end_date), ...]
    strategies: [Strategy, ...] 默认用注册表
    
    返回: DataFrame（每行=一个策略在一个时段的结果）
    """
    if strategies is None:
        strategies = list(_registry)
    
    # 拉数据
    all_starts = [s[1] for s in segments]
    all_ends = [s[2] for s in segments]
    earliest = min(all_starts)
    latest = max(all_ends)
    preheat_start = str(pd.to_datetime(earliest) - pd.Timedelta(days=180)).replace('-', '')[:8]
    
    if verbose:
        print(f"拉取数据: {code} {preheat_start} → {latest}")
    df_all = fetch_stock_data(code, preheat_start, latest.replace('-', '')[:8])
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    rows = []
    
    for seg_name, start_s, end_s in segments:
        start_dt = pd.to_datetime(start_s)
        end_dt = pd.to_datetime(end_s)
        df_seg = df_all[(df_all['date'] >= start_dt) & (df_all['date'] <= end_dt)].copy()
        
        if len(df_seg) < 65:
            if verbose:
                print(f"  {seg_name}: 数据不足，跳过")
            continue
        
        if verbose:
            print(f"\n  {seg_name}（{str(df_seg.iloc[0]['date'])[:10]} → {str(df_seg.iloc[-1]['date'])[:10]}）")
            print(f"  {'─'*110}")
        
        bench_ret = df_seg.iloc[-1]['close'] / df_seg.iloc[0]['close'] - 1
        
        for strategy in strategies:
            result = backtest(strategy, df_seg, init_cash)
            sid = getattr(strategy, '_id', 0)
            
            row = {
                'segment': seg_name,
                'strategy_id': sid,
                'strategy': result.strategy_name,
                'ret': result.ret,
                'pnl': result.ret * init_cash,
                'bench': result.bench,
                'alpha': result.alpha,
                'mdd': result.mdd,
                'sharpe': result.sharpe,
                'calmar': result.calmar,
                'cost': result.total_cost,
                'trades': result.trade_count,
                'invested': result.total_invested,
                'realized': result.realized_profit,
                'unrealized': result.unrealized_profit,
                'start': result.start_date,
                'end': result.end_date,
            }
            rows.append(row)
            
            if verbose:
                print(f"    {result.strategy_name:<24} | 投入{result.total_invested:>7,.0f} | "
                      f"收益{result.ret:>+7.2%}（{result.ret*init_cash:>+7,.0f}）| "
                      f"Alpha{result.alpha:>+7.2%} | 回撤{result.mdd:>6.2%} | "
                      f"夏普{result.sharpe:>5.2f} | {result.trade_count:>2}次 | 成本{result.total_cost:>5.0f}")
    
    df_results = pd.DataFrame(rows)
    
    # 汇总
    if verbose and len(rows) > 0:
        print(f"\n{'='*110}")
        print(f"\n汇总")
        print(f"{'─'*110}")
        print(f"  {'策略':<16} {'平均收益':>10} {'平均盈亏':>10} {'平均Alpha':>10} {'平均回撤':>10} {'正Alpha率':>10} {'总盈亏':>12}")
        print(f"  {'─'*85}")
        
        for strategy in strategies:
            sid = getattr(strategy, '_id', 0)
            s_rows = df_results[df_results['strategy_id'] == sid]
            if len(s_rows) == 0: continue
            n = len(s_rows)
            print(f"  {strategy.name:<16} "
                  f"{s_rows['ret'].mean():>+9.2%} "
                  f"{s_rows['pnl'].mean():>+9,.0f} "
                  f"{s_rows['alpha'].mean():>+9.2%} "
                  f"{s_rows['mdd'].mean():>9.2%} "
                  f"{(s_rows['alpha']>0).sum():>4}/{n:<6} "
                  f"{s_rows['pnl'].sum():>+11,.0f}")
        
        print(f"{'='*110}")
    
    return df_results

# ─── 内置策略 ───

class NormalDCA(Strategy):
    """普通定投：每N天固定买入"""
    def __init__(self, base_invest=5000, interval=20, label=None):
        self.base_invest = base_invest
        self.interval = interval
        self.name = label or f"定投({base_invest//1000}k/{interval}天)"
    def init_state(self):
        return {}
    def decide(self, row, state, ctx):
        if ctx['day_index'] % self.interval == 0 and ctx['cash'] >= self.base_invest:
            return [Buy(self.base_invest, "定投")]
        return [Hold()]

class MADCA(Strategy):
    """均线定投：价格<MA加倍，>MA减半"""
    def __init__(self, base_invest=5000, interval=20, boost=2.0, reduce=0.5, label=None):
        self.base_invest = base_invest
        self.interval = interval
        self.boost = boost
        self.reduce = reduce
        self.name = label or f"均线定投({base_invest//1000}k/{interval}天)"
    def init_state(self):
        return {}
    def decide(self, row, state, ctx):
        if ctx['day_index'] % self.interval == 0:
            ma20 = row.get('ma20', None)
            if ma20 is not None and not pd.isna(ma20):
                if row['close'] < ma20:
                    return [Buy(self.base_invest * self.boost, "低于MA20")]
                else:
                    return [Buy(self.base_invest * self.reduce, "高于MA20")]
            return [Buy(self.base_invest, "定投")]
        return [Hold()]

class RSIDCA(Strategy):
    """RSI定投：RSI低加倍，RSI高减半"""
    def __init__(self, base_invest=5000, interval=20, boost=2.0, reduce=0.5,
                 rsi_low=30, rsi_high=70, label=None):
        self.base_invest = base_invest
        self.interval = interval
        self.boost = boost
        self.reduce = reduce
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.name = label or f"RSI定投({base_invest//1000}k/{interval}天)"
    def init_state(self):
        return {}
    def decide(self, row, state, ctx):
        if ctx['day_index'] % self.interval == 0:
            # 计算RSI
            if ctx['row_index'] >= 74:
                closes = []
                for j in range(ctx['row_index']-14, ctx['row_index']+1):
                    closes.append(row['close'])  # 近似
                rsi = row.get('rsi14', 50)
                if rsi < self.rsi_low:
                    return [Buy(self.base_invest * self.boost, f"RSI={rsi:.0f}<30")]
                elif rsi > self.rsi_high:
                    return [Buy(self.base_invest * self.reduce, f"RSI={rsi:.0f}>70")]
            return [Buy(self.base_invest, "定投")]
        return [Hold()]

class ValueAveraging(Strategy):
    """价值平均：设定目标增长，高抛低吸"""
    def __init__(self, target_growth=2000, interval=20, label=None):
        self.target_growth = target_growth
        self.interval = interval
        self.name = label or f"价值平均({target_growth//1000}k/{interval}天)"
    def init_state(self):
        return {'target': 0}
    def decide(self, row, state, ctx):
        if ctx['day_index'] % self.interval == 0:
            state['target'] += self.target_growth
            current_value = ctx['shares'] * ctx['price']
            diff = state['target'] - current_value
            
            if diff > 100:
                return [Buy(min(diff, ctx['cash']), "补足目标")]
            elif diff < -100 and ctx['shares'] > 0:
                sell_value = abs(diff)
                sell_shares = sell_value / ctx['price']
                return [Sell(sell_shares, "超出目标")]
        return [Hold()]

class DipBoostDCA(Strategy):
    """跌幅追加定投：定期定投+跌X%追加"""
    def __init__(self, base_invest=3000, interval=20, thresholds=None, boosts=None, label=None):
        self.base_invest = base_invest
        self.interval = interval
        self.thresholds = thresholds or [0.10, 0.20, 0.30]
        self.boosts = boosts or [5000, 15000, 30000]
        th_str = '/'.join(f"{t:.0%}" for t in self.thresholds)
        self.name = label or f"跌幅追加({base_invest//1000}k,跌{th_str})"
    def init_state(self):
        return {'peak': 0}
    def decide(self, row, state, ctx):
        state['peak'] = max(state['peak'], ctx['price'])
        drawdown = (state['peak'] - ctx['price']) / state['peak'] if state['peak'] > 0 else 0
        
        actions = []
        if ctx['day_index'] % self.interval == 0 and ctx['cash'] >= self.base_invest:
            actions.append(Buy(self.base_invest, "定投"))
            
            # 跌幅追加
            for thresh, boost in zip(self.thresholds, self.boosts):
                if drawdown >= thresh and ctx['cash'] >= boost:
                    actions.append(Buy(boost, f"跌{drawdown:.1%}>{thresh:.0%}"))
                    break
        return actions if actions else [Hold()]

class MartinGrid(Strategy):
    """马丁格尔网格：跌加仓，涨止盈"""
    def __init__(self, base_amt=8000, dip_pct=0.07, tp_pct=0.08,
                 mults=None, max_add=4, label=None):
        self.base_amt = base_amt
        self.dip_pct = dip_pct
        self.tp_pct = tp_pct
        self.mults = mults or [1, 3, 5, 7]
        self.max_add = max_add
        mult_str = '-'.join(str(m) for m in self.mults)
        self.name = label or f"马丁({base_amt//1000}k,{dip_pct:.0%},{tp_pct:.0%},{mult_str})"
    def init_state(self):
        return {'add_count': 0, 'position_avg': 0, 'position_invested': 0}
    def decide(self, row, state, ctx):
        price = ctx['price']
        
        # 建仓
        if ctx['day_index'] == 0 and ctx['shares'] == 0 and ctx['cash'] >= self.base_amt:
            state['position_avg'] = price
            state['position_invested'] = self.base_amt
            state['add_count'] = 0
            return [Buy(self.base_amt, "建仓")]
        
        if ctx['shares'] > 0:
            pnl = price / ctx['avg_cost'] - 1
            
            # 止盈
            if state['add_count'] > 0 and pnl >= self.tp_pct:
                actions = [Sell(-1, f"止盈{pnl:.1%}")]
                state['add_count'] = 0
                state['position_invested'] = 0
                state['position_avg'] = 0
                # 重新建仓
                if ctx['day_index'] < ctx['total_days'] - 5:
                    # 下一天再建仓，这里先卖出
                    pass
                return actions
            
            # 加仓
            elif pnl <= -self.dip_pct * (state['add_count'] + 1) and state['add_count'] < self.max_add:
                amt = self.base_amt * self.mults[state['add_count']]
                if ctx['cash'] >= amt:
                    state['add_count'] += 1
                    return [Buy(amt, f"加仓{state['add_count']} 跌{pnl:.1%}")]
        
        return [Hold()]


# ─── RSI指标补充 ───
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 给compute_indicators补RSI
_orig_compute_indicators = compute_indicators
def compute_indicators(df):
    df = _orig_compute_indicators(df)
    if 'rsi14' not in df.columns:
        df['rsi14'] = compute_rsi(df['close'], 14)
    return df

import backtest_engine
backtest_engine.compute_indicators = compute_indicators

# ─── 快捷入口 ───
def quick_compare(segments=None, code='515880', init_cash=100000):
    """一键跑所有内置策略对比"""
    if segments is None:
        segments = [
            ('2023H2', '2023-07-03', '2023-12-29'),
            ('2024H1', '2024-01-02', '2024-06-28'),
            ('2024H2', '2024-07-01', '2024-12-31'),
            ('2025H1', '2025-01-02', '2025-06-30'),
            ('2025H2', '2025-07-01', '2025-12-31'),
            ('2026H1', '2026-01-05', '2026-07-08'),
        ]
    
    clear_registry()
    register(NormalDCA())
    register(MADCA())
    register(RSIDCA())
    register(ValueAveraging())
    register(DipBoostDCA())
    register(MartinGrid())
    
    return run_all(segments, init_cash=init_cash, code=code)

if __name__ == '__main__':
    # 默认跑一次全量对比
    df = quick_compare()
