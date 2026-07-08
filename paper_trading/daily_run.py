"""
马丁格尔模拟盘 — 科创50
每天盘末执行一次，本地文件保存状态
参数：底仓8000, 间距7%, 止盈8%, 翻倍1-3-5-7（稳健版）
"""
import json, os, sys
from datetime import datetime

sys.path.insert(0, '/Users/fanzhangmu/llm_trading_lab')
from data_fetcher import fetch_stock_data
from backtest_engine import compute_indicators

# ── 配置 ──
STATE_FILE = '/Users/fanzhangmu/llm_trading_lab/paper_trading/state.json'
LOG_FILE = '/Users/fanzhangmu/llm_trading_lab/paper_trading/trade_log.csv'
CODE = '588060'
INIT_CASH = 100000
BASE_AMT = 8000
DIP_PCT = 0.07
TP_PCT = 0.08
MULTS = [1, 3, 5, 7]
MAX_ADD = 4

def init_state():
    """初始化状态"""
    return {
        'cash': INIT_CASH - BASE_AMT,
        'position_value': BASE_AMT,
        'shares': BASE_AMT,  # 用金额模拟（简化）
        'cost_basis': 0,     # 建仓时填入
        'add_count': 0,
        'total_invested': BASE_AMT,
        'init_cash': INIT_CASH,
        'started_at': datetime.now().isoformat(),
        'last_price': 0,
        'last_date': '',
        'trades': [],
        'peak_since_buy': 0,
    }

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def log_trade(state, action, amount, price, note=''):
    """记录交易到CSV"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a') as f:
        if not file_exists:
            f.write('date,action,amount,price,cash,position,total,pnl_pct,add_count,note\n')
        total = state['cash'] + state['position_value']
        pnl_pct = (total - state['init_cash']) / state['init_cash'] * 100
        f.write(f"{datetime.now().strftime('%Y-%m-%d')},{action},{amount:.2f},{price:.4f},"
                f"{state['cash']:.2f},{state['position_value']:.2f},{total:.2f},{pnl_pct:.2f},"
                f"{state['add_count']},{note}\n")

def run_daily():
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 拉取数据（最近120天够用）
    from datetime import timedelta
    start = (datetime.now() - timedelta(days=150)).strftime('%Y%m%d')
    end = datetime.now().strftime('%Y%m%d')
    df = fetch_stock_data(CODE, start, end)
    df = compute_indicators(df)
    
    if len(df) < 5:
        print(f"[{today}] 数据不足，跳过")
        return
    
    today_row = df.iloc[-1]
    price = today_row['close']
    open_price = today_row['open']
    high_price = today_row['high']
    low_price = today_row['low']
    today_date = str(today_row.get('date', today))
    
    # 加载或初始化状态
    state = load_state()
    if state is None:
        state = init_state()
        state['cost_basis'] = price
        state['last_price'] = price
        state['last_date'] = today_date
        state['peak_since_buy'] = price
        # 记录初始买入
        log_trade(state, 'BUY_INIT', BASE_AMT, price, '初始建仓')
        save_state(state)
        total = state['cash'] + state['position_value']
        pos_ratio = state['position_value'] / total * 100
        print(f"[{today}] 操作: BUY_INIT 初始建仓")
        print(f"  开盘价: {open_price:.4f}  收盘价: {price:.4f}")
        print(f"  持仓成本: {state['cost_basis']:.4f}")
        print(f"  收益率: 0.00%")
        print(f"  持仓比例: {pos_ratio:.1f}%  (持仓{state['position_value']:.0f} / 总值{total:.0f})")
        print(f"  现金: {state['cash']:.0f}  账户总值: {total:.0f}")
        return
    
    # 已经执行过今天了
    if state.get('last_date') == today_date:
        total = state['cash'] + state['position_value']
        pnl_pct = (price / state['cost_basis'] - 1) if state['cost_basis'] > 0 else 0
        total_pnl = (total - state['init_cash']) / state['init_cash'] * 100
        pos_ratio = state['position_value'] / total * 100 if total > 0 else 0
        print(f"[{today}] 今天已执行过，跳过")
        print(f"  开盘价: {open_price:.4f}  收盘价: {price:.4f}")
        print(f"  持仓成本: {state['cost_basis']:.4f}")
        print(f"  收益率: {total_pnl:+.2f}%")
        print(f"  持仓比例: {pos_ratio:.1f}%  (持仓{state['position_value']:.0f} / 总值{total:.0f})")
        return
    
    prev_price = state['last_price']
    
    # 更新持仓市值
    if prev_price > 0 and state['position_value'] > 0:
        state['position_value'] = state['position_value'] * (price / prev_price)
    
    state['last_price'] = price
    state['last_date'] = today_date
    
    # 更新峰值
    if price > state.get('peak_since_buy', 0):
        state['peak_since_buy'] = price
    
    # 判断盈亏
    pnl_pct = (price / state['cost_basis'] - 1) if state['cost_basis'] > 0 else 0
    total = state['cash'] + state['position_value']
    total_pnl = (total - state['init_cash']) / state['init_cash'] * 100
    
    action_taken = 'HOLD'
    note = ''
    
    # ── 止盈判断 ──
    if state['add_count'] > 0 and pnl_pct >= TP_PCT:
        # 卖出所有加仓部分，保留底仓
        sell_value = state['position_value'] - BASE_AMT
        if sell_value > 100:
            state['cash'] += sell_value
            state['position_value'] = BASE_AMT
            state['add_count'] = 0
            state['total_invested'] = BASE_AMT
            state['cost_basis'] = price
            state['peak_since_buy'] = price
            action_taken = 'SELL_TP'
            note = f'止盈+{pnl_pct:.1%}'
            log_trade(state, action_taken, sell_value, price, note)
            print(f"[{today}] 止盈! 卖出{sell_value:.0f}元 | 盈利{pnl_pct:.1%} | 价格:{price:.4f}")
    
    # ── 加仓判断 ──
    elif state['add_count'] < MAX_ADD:
        # 计算当前跌幅（相对成本价）
        dip = -pnl_pct  # 正数=亏损幅度
        
        if dip >= DIP_PCT * (state['add_count'] + 1):
            amt = BASE_AMT * MULTS[state['add_count']]
            if state['cash'] >= amt:
                state['cash'] -= amt
                state['position_value'] += amt
                state['total_invested'] += amt
                state['add_count'] += 1
                action_taken = 'ADD'
                note = f'加仓第{state["add_count"]}次 跌{dip:.1%}'
                log_trade(state, action_taken, amt, price, note)
                print(f"[{today}] 加仓! 投入{amt:.0f}元(第{state['add_count']}次) | 跌幅{dip:.1%} | 价格:{price:.4f}")
    
    if action_taken == 'HOLD':
        print(f"[{today}] 操作: HOLD 持有")
    else:
        print(f"  操作: {action_taken} {note}")

    print(f"  开盘价: {open_price:.4f}  收盘价: {price:.4f}")
    print(f"  持仓成本: {state['cost_basis']:.4f}")
    print(f"  收益率: {total_pnl:+.2f}%")
    print(f"  持仓比例: {state['position_value'] / total * 100:.1f}%  (持仓{state['position_value']:.0f} / 总值{total:.0f})")
    print(f"  现金: {state['cash']:.0f}  账户总值: {total:.0f}")
    
    # 保存状态
    state['trades'].append({
        'date': today_date,
        'action': action_taken,
        'price': price,
        'pnl_pct': pnl_pct,
        'note': note,
    })
    # 只保留最近100条
    state['trades'] = state['trades'][-100:]
    save_state(state)

if __name__ == '__main__':
    run_daily()
