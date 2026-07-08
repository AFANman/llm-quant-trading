"""
数据获取 - 腾讯K线API (不依赖akshare，绕过Alilang代理)
"""
import requests
import pandas as pd
import time

def fetch_realtime_quote(symbol: str) -> dict:
    """
    获取实时行情（腾讯qt接口），返回当日OHLCV
    """
    # 上海: 6xx(股票), 5xx(ETF); 深圳: 0xx, 3xx, 1xx
    market = "sz" if symbol.startswith("0") or symbol.startswith("3") or symbol.startswith("1") else "sh"
    code = f"{market}{symbol}"
    url = f"https://qt.gtimg.cn/q={code}"
    resp = requests.get(url, timeout=10)
    text = resp.text
    # 解析: 字段用~分隔
    parts = text.split("~")
    if len(parts) < 50:
        return None
    return {
        "date": pd.to_datetime(parts[30][:8]),
        "open": float(parts[5]),
        "high": float(parts[33]),
        "low": float(parts[34]),
        "close": float(parts[3]),
        "volume": float(parts[6]),
    }

def fetch_stock_data(symbol: str, start_date: str = "20200101", end_date: str = "20241231") -> pd.DataFrame:
    """
    获取A股日线数据 (腾讯接口)
    symbol: 6位代码，如 600519
    """
    market = "sh" if symbol.startswith("6") else "sz"
    code = f"{market}{symbol}"
    
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    params = {"param": f"{code},day,{sd},{ed},800,qfq"}
    
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    klines = data.get("data", {}).get(code, {})
    
    rows = None
    for k in ["qfqday", "day"]:
        if k in klines and klines[k]:
            rows = klines[k]
            break
    
    if not rows:
        raise ValueError(f"无法获取 {symbol} 数据, keys={list(klines.keys())}")
    
    # 统一列数（有些行可能有额外字段）
    ncols = len(rows[0])
    cols6 = ["date", "open", "close", "high", "low", "volume"]
    df = pd.DataFrame([r[:6] for r in rows], columns=cols6)
    
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    
    # 如果K线数据最新日期不是今天，尝试用实时行情补充当日数据
    from datetime import datetime as _dt
    today = _dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
    last_date = df["date"].iloc[-1].replace(hour=0, minute=0, second=0, microsecond=0)
    if last_date < today:
        try:
            rt = fetch_realtime_quote(symbol)
            if rt and rt["date"].replace(hour=0, minute=0, second=0, microsecond=0) >= today and rt["close"] > 0:
                new_row = pd.DataFrame([rt])
                df = pd.concat([df, new_row], ignore_index=True)
        except Exception:
            pass
    
    return df


def fetch_multiple_stocks(symbols: list, start_date: str = "20200101", end_date: str = "20241231") -> dict:
    """批量获取多只股票"""
    result = {}
    for sym in symbols:
        try:
            df = fetch_stock_data(sym, start_date, end_date)
            result[sym] = df
            print(f"  {sym}: {len(df)} 条")
            time.sleep(0.3)
        except Exception as e:
            print(f"  {sym}: 失败 - {e}")
    return result


if __name__ == "__main__":
    print("测试获取贵州茅台数据...")
    df = fetch_stock_data("600519", "20230101", "20241231")
    print(f"获取到 {len(df)} 条数据")
    print(df.head(3).to_string())
    print("...")
    print(df.tail(3).to_string())
