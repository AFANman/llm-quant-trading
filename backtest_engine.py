"""
回测引擎 - 接收策略JSON + 行情数据，输出交易记录和收益曲线
"""
import pandas as pd
import numpy as np

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    df = df.copy()
    for w in [5, 10, 20, 60]:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    # 成交量
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
    # ATR
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - df["close"].shift()).abs(),
        "lc": (df["low"] - df["close"].shift()).abs()
    }).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_ma20"] = df["atr"].rolling(20).mean()
    # 布林带
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    # 涨跌幅
    df["pct_change"] = df["close"].pct_change()
    df["pct_5d"] = df["close"].pct_change(5)
    df["pct_20d"] = df["close"].pct_change(20)
    # 价格位置
    df["price_vs_ma60"] = (df["close"] - df["ma60"]) / df["ma60"]
    return df


def evaluate_condition(row: pd.Series, condition: dict) -> bool:
    """评估单个条件，支持value、value_indicator、derived、between"""
    indicator = condition.get("indicator")
    op = condition.get("operator")
    if not indicator or indicator not in row.index:
        return False
    actual = row[indicator]
    if pd.isna(actual):
        return False

    # between操作符: value是[low, high]
    if op == "between":
        bounds = condition.get("value", [])
        if len(bounds) != 2:
            return False
        return bounds[0] <= actual <= bounds[1]

    # 确定比较值: value_indicator > derived > value
    val_indicator = condition.get("value_indicator")
    derived = condition.get("derived")
    if val_indicator:
        if val_indicator not in row.index:
            return False
        value = row[val_indicator]
        if pd.isna(value):
            return False
    elif derived:
        try:
            ns = {str(k): float(v) for k, v in row.items()
                  if isinstance(v, (int, float, np.number)) and not pd.isna(v)}
            value = eval(derived, {"__builtins__": {}}, ns)
        except Exception:
            return False
    else:
        value = condition.get("value")
        if value is None:
            return False
        # 如果value是字符串且是有效指标名，视为value_indicator
        if isinstance(value, str) and value in row.index:
            value = row[value]
            if pd.isna(value):
                return False

    ops = {">": lambda a,b: a>b, "<": lambda a,b: a<b, ">=": lambda a,b: a>=b,
           "<=": lambda a,b: a<=b, "==": lambda a,b: a==b}
    return ops.get(op, lambda a,b: False)(actual, value)


def get_market_regime(row: pd.Series, regime_conditions: dict) -> str:
    """根据市场环境条件判断当前市场状态: uptrend/downtrend/sideways"""
    if not regime_conditions:
        return None  # 无环境判断，返回None表示用固定参数
    
    # 检查上升趋势
    uptrend = regime_conditions.get("uptrend", [])
    if uptrend and all(evaluate_condition(row, c) for c in uptrend):
        return "uptrend"
    
    # 检查下降趋势
    downtrend = regime_conditions.get("downtrend", [])
    if downtrend and all(evaluate_condition(row, c) for c in downtrend):
        return "downtrend"
    
    return "sideways"


def run_backtest(df: pd.DataFrame, strategy: dict, initial_capital: float = 100000) -> dict:
    """
    执行回测（支持市场环境自适应策略）
    strategy JSON格式:
    {
        "name": "策略名称",
        "entry_conditions": [{"indicator": "rsi", "operator": "<", "value": 30}],
        "exit_conditions": [{"indicator": "rsi", "operator": ">", "value": 70}],
        "stop_loss": 0.05,
        "take_profit": 0.15,
        "hold_days": 5,
        "position_size": 0.3,
        "market_regime_conditions": {  // 可选：市场环境判断
            "uptrend": [...],
            "downtrend": [...]
        },
        "params_by_regime": {  // 可选：不同环境下的参数
            "uptrend": {"position_size": 0.8, "take_profit": 0.30, "hold_days": 20},
            "downtrend": {"position_size": 0.2, "take_profit": 0.15, "hold_days": 10},
            "sideways": {"position_size": 0.4, "take_profit": 0.20, "hold_days": 15}
        }
    }
    可用指标: ma5/10/20/60, rsi, vol_ratio, atr, bb_upper/lower/mid,
              pct_change, pct_5d, pct_20d, price_vs_ma60, open, high, low, close, volume
    """
    df = compute_indicators(df)
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_idx = 0
    trades = []
    equity_curve = []
    
    entry_conds = strategy.get("entry_conditions", [])
    exit_conds = strategy.get("exit_conditions", [])
    regime_conditions = strategy.get("market_regime_conditions", {})
    params_by_regime = strategy.get("params_by_regime", {})
    
    # 默认参数
    default_stop_loss = strategy.get("stop_loss", 0.05)
    default_take_profit = strategy.get("take_profit", 0.15)
    default_hold_days = strategy.get("hold_days", 0)
    default_pos_size = strategy.get("position_size", 0.3)
    
    for i in range(len(df)):
        row = df.iloc[i]
        price = row["close"]
        
        # 判断市场环境
        regime = get_market_regime(row, regime_conditions)
        
        # 根据环境获取参数
        if regime and params_by_regime:
            rp = params_by_regime.get(regime, {})
            pos_size = rp.get("position_size", default_pos_size)
            take_profit = rp.get("take_profit", default_take_profit)
            hold_days_max = rp.get("hold_days", default_hold_days)
        else:
            pos_size = default_pos_size
            take_profit = default_take_profit
            hold_days_max = default_hold_days
        stop_loss = default_stop_loss
        
        equity = capital + position * price
        equity_curve.append({
            "date": row["date"], 
            "equity": equity,
            "regime": regime or "fixed"
        })
        
        # 卖出
        if position > 0:
            pnl_pct = (price - entry_price) / entry_price
            days_held = i - entry_idx
            exit_reason = None
            if pnl_pct <= -stop_loss:
                exit_reason = "stop_loss"
            elif pnl_pct >= take_profit:
                exit_reason = "take_profit"
            elif hold_days_max > 0 and days_held >= hold_days_max:
                exit_reason = "hold_days"
            elif exit_conds and all(evaluate_condition(row, c) for c in exit_conds):
                exit_reason = "signal"
            
            if exit_reason:
                capital += position * price
                trades.append({
                    "entry_date": df.iloc[entry_idx]["date"],
                    "exit_date": row["date"],
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "shares": position,
                    "regime": regime or "fixed"
                })
                position = 0
        
        # 买入
        if position == 0 and entry_conds:
            if all(evaluate_condition(row, c) for c in entry_conds):
                buy_value = capital * pos_size
                position = buy_value / price  # 碎股（模拟账户）
                if position > 0:
                    entry_price = price
                    entry_idx = i
                    capital -= position * price
    
    # 末尾强平
    if position > 0:
        price = df.iloc[-1]["close"]
        pnl_pct = (price - entry_price) / entry_price
        capital += position * price
        trades.append({
            "entry_date": df.iloc[entry_idx]["date"],
            "exit_date": df.iloc[-1]["date"],
            "entry_price": entry_price,
            "exit_price": price,
            "pnl_pct": pnl_pct,
            "exit_reason": "end",
            "shares": position,
            "regime": "end"
        })
    
    return {
        "strategy_name": strategy.get("name", "unknown"),
        "trades": trades,
        "equity_curve": pd.DataFrame(equity_curve),
        "final_capital": capital,
        "initial_capital": initial_capital
    }
