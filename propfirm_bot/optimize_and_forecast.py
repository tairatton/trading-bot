import sys
import os
# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from services.strategy import calculate_indicators, generate_signals, STRATEGY_PARAMS
from config import settings

# CONFIGURATION
SYMBOLS = ["EURUSDm", "USDCADm", "USDCHFm"]
INITIAL_BALANCE = 10000.0
CACHE_FILE = os.path.join(os.getcwd(), "backtest", "backtest_data_cache.pkl")

# TRADING COSTS
SPREAD_PIPS = {"EURUSDm": 1.2, "USDCADm": 1.5, "USDJPYm": 1.3}
SLIPPAGE_PIPS = 0.5

def get_spread_cost(symbol: str) -> float:
    spread = SPREAD_PIPS.get(symbol, 1.5)
    total_pips = spread + SLIPPAGE_PIPS
    if "JPY" in symbol.upper(): return total_pips * 0.01
    return total_pips * 0.0001

def get_pip_value(symbol: str) -> float:
    return 1000 if "JPY" in symbol.upper() else 100000

def fetch_data(symbol: str, timeframe=mt5.TIMEFRAME_M30, count=70000):
    if not mt5.initialize():
        print("MT5 Init Failed")
        return pd.DataFrame()
        
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df

def fetch_data_cached(symbol):
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            if symbol in cache:
                return cache[symbol]
        except Exception as e:
            # print(f"Cache read error: {e}")
            pass

    print(f"[*] Fetching fresh data for {symbol}...")
    df = fetch_data(symbol)
    
    if not df.empty:
        try:
            cache = {}
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "rb") as f:
                    cache = pickle.load(f)
            cache[symbol] = df
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Cache write error: {e}")
            
    return df

def run_backtest_engine(risk_percent, sl_atr_trend, sl_atr_mr):
    # LOCAL PARAMS
    import copy
    PARAMS = copy.deepcopy(STRATEGY_PARAMS)
    PARAMS["TREND"]["SL_ATR"] = sl_atr_trend
    PARAMS["MR"]["SL_ATR"] = sl_atr_mr

    # LOAD DATA
    all_data = {}
    for symbol in SYMBOLS:
        df = fetch_data_cached(symbol)
        if df.empty: continue
        
        if "ATR" not in df.columns:
            df = calculate_indicators(df)
            df = generate_signals(df)
        all_data[symbol] = df
        
    if len(all_data) < len(SYMBOLS):
        print("Not enough data.")
        return None

    # ALIGN
    common_index = all_data[SYMBOLS[0]].index
    for s in SYMBOLS[1:]: common_index = common_index.intersection(all_data[s].index)
    for s in SYMBOLS: all_data[s] = all_data[s].loc[common_index]

    # SIMULATION
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    max_dd = 0
    trades = []
    positions = {}
    equity_curve = [INITIAL_BALANCE]
    
    start_idx = 250
    
    for i in range(start_idx, len(common_index) - 1):
        # Current Bar [i]: Signal Generation
        # Next Bar [i+1]: Execution & Price Action
        
        row_curr = {s: all_data[s].iloc[i] for s in SYMBOLS}
        row_next = {s: all_data[s].iloc[i+1] for s in SYMBOLS}
        
        for symbol in SYMBOLS:
            # MANAGE POSITIONS
            if symbol in positions:
                pos = positions[symbol]
                pos["bars"] += 1
                p = PARAMS[pos["signal_type"]]
                
                # Next Bar Price Action
                high = row_next[symbol]["High"]
                low = row_next[symbol]["Low"]
                close = row_next[symbol]["Close"]
                atr = row_curr[symbol]["ATR"]
                
                exit_price = None
                
                if pos["type"] == "buy":
                    pos["highest"] = max(pos["highest"], high)
                    if pos["highest"] - pos["entry"] > p["TRAIL_START"] * atr:
                        pos["sl"] = max(pos["sl"], pos["highest"] - p["TRAIL_DIST"] * atr)
                    
                    if low <= pos["sl"]: exit_price = pos["sl"]
                    elif high >= pos["tp"]: exit_price = pos["tp"]
                    elif pos["bars"] >= p["MAX_BARS"]: exit_price = close
                
                elif pos["type"] == "sell":
                    pos["lowest"] = min(pos["lowest"], low)
                    if pos["entry"] - pos["lowest"] > p["TRAIL_START"] * atr:
                        pos["sl"] = min(pos["sl"], pos["lowest"] + p["TRAIL_DIST"] * atr)
                        
                    if high >= pos["sl"]: exit_price = pos["sl"]
                    elif low <= pos["tp"]: exit_price = pos["tp"]
                    elif pos["bars"] >= p["MAX_BARS"]: exit_price = close
                
                if exit_price:
                    pip_val = get_pip_value(symbol)
                    direction = 1 if pos["type"] == "buy" else -1
                    gross_pnl = (exit_price - pos["entry"]) * direction * pos["size"] * pip_val
                    cost = get_spread_cost(symbol) * pos["size"] * pip_val
                    net_pnl = gross_pnl - cost
                    
                    balance += net_pnl
                    trades.append(net_pnl)
                    del positions[symbol]
                    
                    if balance > peak: peak = balance
                    dd = (peak - balance) / peak * 100
                    max_dd = max(max_dd, dd)

            # OPEN POSITIONS
            if symbol not in positions:
                sig = row_curr[symbol]
                s_type = None; signal = None
                
                if sig.get("trend_buy"): signal="buy"; s_type="TREND"
                elif sig.get("mr_buy"): signal="buy"; s_type="MR"
                elif sig.get("trend_sell"): signal="sell"; s_type="TREND"
                elif sig.get("mr_sell"): signal="sell"; s_type="MR"
                
                if signal:
                    p = PARAMS[s_type]
                    atr = sig["ATR"]
                    entry = row_next[symbol]["Open"]
                    
                    sl_dist = atr * p["SL_ATR"]
                    if sl_dist > 0:
                        risk_amt = balance * risk_percent
                        pip_val = get_pip_value(symbol)
                        size = risk_amt / (sl_dist * pip_val)
                        size = max(0.01, min(size, 10.0))
                        
                        tp = entry + atr * p["TP_ATR"] if signal == "buy" else entry - atr * p["TP_ATR"]
                        sl = entry - sl_dist if signal == "buy" else entry + sl_dist
                        
                        positions[symbol] = {
                            "type": signal, "entry": entry, "size": size,
                            "signal_type": s_type, "sl": sl, "tp": tp,
                            "highest": entry if signal == "buy" else 0,
                            "lowest": float('inf') if signal == "buy" else entry,
                            "bars": 0
                        }
        
        equity_curve.append(balance)

    # METRICS
    total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    start_date = common_index[start_idx]
    end_date = common_index[-1]
    days = (end_date - start_date).days
    months = days / 30.44
    avg_monthly = total_return / months if months > 0 else 0
    
    pf = 0
    if trades:
        wins = sum([t for t in trades if t > 0])
        losses = abs(sum([t for t in trades if t <= 0]))
        pf = wins / losses if losses > 0 else 0

    return {
        "risk": risk_percent,
        "trend_sl": sl_atr_trend,
        "mr_sl": sl_atr_mr,
        "return": total_return,
        "max_dd": max_dd,
        "pf": pf,
        "months": months,
        "avg_monthly": avg_monthly,
        "equity_curve": equity_curve
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   PROP FIRM FORECAST (High/Low Physics Engine)")
    print("   Goal: Pass Challenge (10% Target) | Safe DD (<10%)")
    print("="*60 + "\n")
    
    # Optimized Params
    RISK = 0.0009
    TREND_SL = 1.5
    MR_SL = 0.6
    
    res = run_backtest_engine(RISK, TREND_SL, MR_SL)
    
    if res:
        target = 10.0 # 10% Profit Target
        est_months = target / res["avg_monthly"] if res["avg_monthly"] > 0 else 999
        
        print(f"Risk: {res['risk']*100:.2f}% | Trend SL: {res['trend_sl']} | MR SL: {res['mr_sl']}\n")
        print(f"Total Return:     {res['return']:.2f}%")
        print(f"Max Drawdown:     {res['max_dd']:.2f}%  {'[PASSED]' if res['max_dd'] < 10 else '[FAILED]'}")
        print(f"Profit Factor:    {res['pf']:.2f}")
        print(f"Duration Data:    {res['months']:.1f} Months")
        print(f"Avg Monthly Ret:  {res['avg_monthly']:.2f}%")
        print("-" * 40)
        print(f"ESTIMATED TIME TO PASS (10%):  {est_months:.1f} Months")
        print("=" * 60)
        
        # PLOTTING
        try:
            import matplotlib.pyplot as plt
            
            # Prepare data
            curve = res.get("equity_curve", [])
            if not curve:
                print("No equity curve data returned.")
            else:
                plt.figure(figsize=(12, 6))
                plt.plot(curve, label=f"Equity (Risk={res['risk']*100}%)", color='blue', linewidth=1)
                
                # Highlight Peak
                peak_val = max(curve)
                plt.axhline(y=peak_val, color='green', linestyle='--', alpha=0.5, label='Peak Equity')
                
                # Labels
                plt.title(f"Prop Firm Forecast: +{res['return']:.1f}% | DD: {res['max_dd']:.2f}%")
                plt.xlabel("Trade Bars (Time)")
                plt.ylabel("Balance ($)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save
                filename = "equity_curve_forecast.png"
                plt.savefig(filename)
                print(f"Graph saved to: {os.path.abspath(filename)}")
                # plt.show() # Non-blocking in script
            
        except ImportError:
            print("Matplotlib not installed. Skipping plot.")
        except Exception as e:
            print(f"Plotting error: {e}")
