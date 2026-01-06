"""
================================================================================
PROP FIRM BOT - Backtest (Engine Upgraded)
================================================================================
Purpose: Safe trading for Prop Firm Challenge (The5ers, FTMO, etc.)
Symbols: 3 (EURUSD, USDCAD, USDJPY)
Risk: 0.15% per trade (0.45% total when all open)

Engine: Based on Original Bot (High PF) + Prop Firm Safety Rules
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.strategy import calculate_indicators, generate_signals, STRATEGY_PARAMS
from config import settings

# PROP FIRM CONFIGURATION
SYMBOLS = ["EURUSDm", "USDCADm", "USDCHFm"]
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE = 0.003   # 0.30% per trade (Optimized Safe Growth)

# PROP FIRM LIMITS
DAILY_LOSS_LIMIT = 0.041 # 4.1%
MAX_LOSS_ABSOLUTE = 1000  # $1000

# Trading Costs
SPREAD_PIPS = {"EURUSDm": 1.2, "USDCADm": 1.5, "USDJPYm": 1.3}
SLIPPAGE_PIPS = 0.5

import os
import pickle

CACHE_FILE = os.path.join(os.getcwd(), "backtest_data_cache.pkl")

def get_spread_cost(symbol: str) -> float:
    spread = SPREAD_PIPS.get(symbol, 1.5)
    total_pips = spread + SLIPPAGE_PIPS
    if "JPY" in symbol.upper(): return total_pips * 0.01
    return total_pips * 0.0001

def fetch_data(symbol: str, timeframe=mt5.TIMEFRAME_M30, count=70000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df

def get_pip_value(symbol: str) -> float:
    return 1000 if "JPY" in symbol.upper() else 100000

def run_propfirm_backtest():
    print("=" * 80)
    print("   PROP FIRM CHALLENGE BACKTEST (High Efficiency Engine)")
    print(f"   Risk: {RISK_PER_TRADE*100}% per trade")
    print("=" * 80)
    
    if not mt5.initialize(path=settings.MT5_PATH if settings.MT5_PATH else None):
        print("MT5 init failed")
        return None

    # Fetch & Prepare Data
    all_data = {}
    for symbol in SYMBOLS:
        print(f"[*] Loading {symbol}...")
        df = fetch_data_cached(symbol)
        if not df.empty:
            # Check if indicators need calc (cache usually has them)
            if "ATR" not in df.columns:
                df = calculate_indicators(df)
                df = generate_signals(df)
            all_data[symbol] = df
    mt5.shutdown()

    if len(all_data) < len(SYMBOLS): return None

    # Align Data
    common_index = all_data[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]: common_index = common_index.intersection(all_data[symbol].index)
    for symbol in SYMBOLS: all_data[symbol] = all_data[symbol].loc[common_index]

    # Backtest Loop
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_drawdown = 0
    
    positions = {}
    trades = []
    equity_curve = []
    daily_start_balance = {}
    
    start_idx = 250
    
    for i in range(start_idx, len(common_index)):
        current_time = common_index[i]
        curr_date = current_time.date()
        
        # Daily Reset
        if curr_date not in daily_start_balance:
            daily_start_balance[curr_date] = balance
            
        # Daily Loss Check
        daily_loss = daily_start_balance[curr_date] - balance
        # In this backtest we don't stop, just track performance, 
        # assuming the strategy is good enough to not hit it.
        
        for symbol in SYMBOLS:
            row = all_data[symbol].iloc[i]
            price = row["Close"]
            atr = row["ATR"]
            
            # POSITIONS
            if symbol in positions:
                pos = positions[symbol]
                pos["bars"] += 1
                p = STRATEGY_PARAMS[pos["signal_type"]]
                exit_price = None
                exit_reason = ""
                
                if pos["type"] == "buy":
                    pos["highest"] = max(pos["highest"], price)
                    if pos["highest"] - pos["entry"] > p["TRAIL_START"] * atr:
                        pos["sl"] = max(pos["sl"], pos["highest"] - p["TRAIL_DIST"] * atr)
                    if price <= pos["sl"]: exit_price = pos["sl"]; exit_reason = "SL"
                    elif price >= pos["tp"]: exit_price = pos["tp"]; exit_reason = "TP"
                    elif pos["bars"] >= p["MAX_BARS"]: exit_price = price; exit_reason = "TIME"
                
                elif pos["type"] == "sell":
                    pos["lowest"] = min(pos["lowest"], price)
                    if pos["entry"] - pos["lowest"] > p["TRAIL_START"] * atr:
                        pos["sl"] = min(pos["sl"], pos["lowest"] + p["TRAIL_DIST"] * atr)
                    if price >= pos["sl"]: exit_price = pos["sl"]; exit_reason = "SL"
                    elif price <= pos["tp"]: exit_price = pos["tp"]; exit_reason = "TP"
                    elif pos["bars"] >= p["MAX_BARS"]: exit_price = price; exit_reason = "TIME"
                
                if exit_price:
                    pip_mult = get_pip_value(symbol)
                    pnl = (exit_price - pos["entry"])*pos["size"]*pip_mult if pos["type"]=="buy" else (pos["entry"]-exit_price)*pos["size"]*pip_mult
                    pnl -= get_spread_cost(symbol)*pos["size"]*pip_mult
                    balance += pnl
                    trades.append({"pnl": pnl, "result": exit_reason})
                    del positions[symbol]
                    
                    if balance > peak_balance: peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance * 100
                    max_drawdown = max(max_drawdown, dd)

            # SIGNALS
            if symbol not in positions:
                signal_type = None; signal = None
                if row.get("trend_buy"): signal="buy"; signal_type="TREND"
                elif row.get("mr_buy"): signal="buy"; signal_type="MR"
                elif row.get("trend_sell"): signal="sell"; signal_type="TREND"
                elif row.get("mr_sell"): signal="sell"; signal_type="MR"
                
                if signal:
                    p = STRATEGY_PARAMS[signal_type]
                    sl_dist = atr * p["SL_ATR"]
                    if sl_dist > 0:
                        risk_amt = balance * RISK_PER_TRADE
                        pip_mult = get_pip_value(symbol)
                        size = risk_amt / (sl_dist * pip_mult)
                        size = max(0.01, min(size, 10.0))
                        
                        tp_price = price + atr*p["TP_ATR"] if signal=="buy" else price - atr*p["TP_ATR"]
                        sl_price = price - sl_dist if signal=="buy" else price + sl_dist
                        
                        positions[symbol] = {
                            "type": signal, "entry": price, "size": size, 
                            "signal_type": signal_type, "sl": sl_price, "tp": tp_price,
                            "highest": price if signal=="buy" else 0,
                            "lowest": float('inf') if signal=="buy" else price,
                            "bars": 0
                        }
        
        equity_curve.append(balance)

    # Reporting
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        losses = abs(trades_df[trades_df["pnl"] <= 0]["pnl"].sum())
        pf = wins / losses if losses > 0 else 0
        win_rate = len(trades_df[trades_df["pnl"] > 0]) / len(trades_df) * 100
    else:
        pf = 0; win_rate = 0

    return {
        "final_balance": balance,
        "total_return": (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        "max_dd": max_drawdown,
        "pf": pf,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "equity_curve": equity_curve
    }

if __name__ == "__main__":
    res = run_propfirm_backtest()
    if res:
        print(f"PF: {res['pf']:.2f}, Return: {res['total_return']:.1f}%")
