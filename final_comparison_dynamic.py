import sys
import os
import copy
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

# ... Imports ...
sys.path.append(os.path.join(os.getcwd(), "propfirm_bot"))
sys.path.append(os.path.join(os.getcwd(), "original_bot"))

try:
    from services.strategy import calculate_indicators, generate_signals
except ImportError:
    import ta
    def calculate_indicators(df):
        df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
        return df
    def generate_signals(df): return df

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
SYMBOLS = ["EURUSDm", "USDCADm", "USDCHFm"]
SPREAD_PIPS = {"EURUSDm": 1.2, "USDCADm": 1.5, "USDJPYm": 1.3}
SLIPPAGE_PIPS = 0.5
INITIAL_BALANCE = 2500.0

# 1. ORIGINAL BOT (5-Tier)
# Target: Maximize Return, DD < 20%
ORIGINAL_PROFILE = {
    "NAME": "Original Bot (5-Tier Dynamic)",
    "BASE_RISK": 0.0050,  # 0.50%
    "TIERS": [
        (3.0,  0.8), (6.0,  0.6), (9.0,  0.4), (12.0, 0.2), (15.0, 0.1)
    ],
    "PARAMS": { "TREND_SL": 1.5, "MR_SL": 0.6 }
}

# 2. PROP FIRM BOT (5-Tier Small Acc)
# Target: Strict DD < 9%
PROP_PROFILE = {
    "NAME": "Prop Firm Bot (5-Tier Dynamic)",
    "BASE_RISK": 0.0028,  # 0.28%
    "TIERS": [
        (1.5, 0.8), (3.0, 0.6), (4.0, 0.4), (5.0, 0.2), (6.0, 0.1)
    ],
    "PARAMS": { "TREND_SL": 1.5, "MR_SL": 0.6 }
}

BASE_PARAMS = {
    "TREND": { "SL_ATR": 1.5, "TP_ATR": 5.0, "TRAIL_START": 1.0, "TRAIL_DIST": 0.6, "MAX_BARS": 50 },
    "MR":    { "SL_ATR": 0.6, "TP_ATR": 3.0, "TRAIL_START": 0.53, "TRAIL_DIST": 0.4, "MAX_BARS": 25 }
}

def get_pip_val(s): return 1000 if "JPY" in s else 100000
def get_cost(s): return (SPREAD_PIPS.get(s,1.5)+SLIPPAGE_PIPS) * (0.01 if "JPY" in s else 0.0001)

def fetch_data(symbol):
    if os.path.exists(os.path.join(os.getcwd(), "propfirm_bot", "backtest", "backtest_data_cache.pkl")):
         path = os.path.join(os.getcwd(), "propfirm_bot", "backtest", "backtest_data_cache.pkl")
    else: path = "backtest_data_cache.pkl"
    if os.path.exists(path):
        try:
            with open(path, "rb") as f: cache = pickle.load(f)
            if symbol in cache: return cache[symbol]
        except: pass
    if not mt5.initialize(): return pd.DataFrame()
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, 70000)
    mt5.shutdown()
    if rates is None: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df

def get_dynamic_risk(current_dd, profile):
    base = profile["BASE_RISK"]
    multiplier = 1.0
    for threshold, mult in profile["TIERS"]:
        if current_dd > threshold:
            multiplier = mult 
    return base * multiplier

def run_simulation(profile, all_data, common_index):
    balance = INITIAL_BALANCE; peak = INITIAL_BALANCE; max_dd = 0
    positions = {}; equity_curve = []
    
    start_idx = 250
    daily_start_balance = {}
    max_daily_dd = 0
    
    for i in range(start_idx, len(common_index) - 1):
        # Daily DD Check
        curr_time = common_index[i]
        curr_date = curr_time.date()
        if curr_date not in daily_start_balance:
            daily_start_balance[curr_date] = balance
        
        day_start = daily_start_balance[curr_date]
        if day_start > 0:
            current_daily_dd = (day_start - balance) / day_start * 100
            max_daily_dd = max(max_daily_dd, current_daily_dd)
        
        current_dd = (peak - balance) / peak * 100
        max_dd = max(max_dd, current_dd)
        
        # NOTE: Using Total DD for Risk Sizing (as per Anti-Martingale logic)
        risk_percent = get_dynamic_risk(current_dd, profile)
        
        row_curr = {s: all_data[s].iloc[i] for s in SYMBOLS}
        row_next = {s: all_data[s].iloc[i+1] for s in SYMBOLS}
        
        for symbol in SYMBOLS:
            if symbol in positions:
                pos = positions[symbol]; pos["bars"]+=1
                p = BASE_PARAMS[pos["signal_type"]]
                high=row_next[symbol]["High"]; low=row_next[symbol]["Low"]
                close=row_next[symbol]["Close"]; atr=row_curr[symbol]["ATR"]
                exit_price = None
                
                if pos["type"] == "buy":
                    pos["highest"] = max(pos["highest"], high)
                    if pos["highest"]-pos["entry"] > p["TRAIL_START"]*atr:
                        pos["sl"] = max(pos["sl"], pos["highest"]-p["TRAIL_DIST"]*atr)
                    if low <= pos["sl"]: exit_price = pos["sl"]
                    elif high >= pos["tp"]: exit_price = pos["tp"]
                    elif pos["bars"] >= p["MAX_BARS"]: exit_price = close
                elif pos["type"] == "sell":
                    pos["lowest"] = min(pos["lowest"], low)
                    if pos["entry"]-pos["lowest"] > p["TRAIL_START"]*atr:
                        pos["sl"] = min(pos["sl"], pos["lowest"]+p["TRAIL_DIST"]*atr)
                    if high >= pos["sl"]: exit_price = pos["sl"]
                    elif low <= pos["tp"]: exit_price = pos["tp"]
                    elif pos["bars"] >= p["MAX_BARS"]: exit_price = close
                
                if exit_price:
                    pip = get_pip_val(symbol)
                    direction = 1 if pos["type"]=="buy" else -1
                    net = ((exit_price-pos["entry"])*direction*pos["size"]*pip) - (get_cost(symbol)*pos["size"]*pip)
                    balance += net
                    del positions[symbol]
                    if balance > peak: peak = balance
                    # Update daily min balance? No, balance updates normally.
            
            if symbol not in positions:
                sig = row_curr[symbol]
                if sig.get("trend_buy"): signal="buy"; s_type="TREND"
                elif sig.get("mr_buy"): signal="buy"; s_type="MR"
                elif sig.get("trend_sell"): signal="sell"; s_type="TREND"
                elif sig.get("mr_sell"): signal="sell"; s_type="MR"
                else: signal=None
                
                if signal:
                    p = BASE_PARAMS[s_type]
                    atr = sig["ATR"]
                    entry = row_next[symbol]["Open"]
                    sl_dist = atr * p["SL_ATR"]
                    if sl_dist > 0:
                        raw_size = (balance * risk_percent) / (sl_dist * get_pip_val(symbol))
                        size = max(0.01, min(raw_size, 10.0)) # 0.01 Floor
                        tp = entry + atr*p["TP_ATR"] if signal=="buy" else entry - atr*p["TP_ATR"]
                        sl = entry - sl_dist if signal=="buy" else entry + sl_dist
                        positions[symbol] = {
                            "type": signal, "entry": entry, "size": size,
                            "signal_type": s_type, "sl": sl, "tp": tp,
                            "highest": entry if signal=="buy" else 0,
                            "lowest": float('inf') if signal=="buy" else entry,
                            "bars": 0
                        }
        equity_curve.append(balance)
    return { "return": (balance-INITIAL_BALANCE)/INITIAL_BALANCE*100, "max_dd": max_dd, "max_daily_dd": max_daily_dd, "equity": equity_curve }

def main():
    print("FINAL COMPARISON: Original (5-Tier) vs Prop Firm (5-Tier Small Acc)")
    all_data = {}
    for s in SYMBOLS:
        df = fetch_data(s)
        if df.empty: continue
        if "ATR" not in df.columns:
            df = calculate_indicators(df); df = generate_signals(df)
        all_data[s] = df
        
    keys = list(all_data.keys()); idx = all_data[keys[0]].index
    for k in keys[1:]: idx = idx.intersection(all_data[k].index)
    
    res_orig = run_simulation(ORIGINAL_PROFILE, all_data, idx)
    res_prop = run_simulation(PROP_PROFILE, all_data, idx)
    
    # Report
    print("="*80)
    print(f"{'METRIC':<20} | {'ORIGINAL (5-Tier)':<25} | {'PROP FIRM (5-Tier)':<25}")
    print("-" * 80)
    print(f"{'Start Risk':<20} | {ORIGINAL_PROFILE['BASE_RISK']*100:>23.2f}% | {PROP_PROFILE['BASE_RISK']*100:>23.2f}%")
    print(f"{'Total Return':<20} | {res_orig['return']:>24.1f}% | {res_prop['return']:>24.1f}%")
    print(f"{'Max Drawdown (Tot)':<20} | {res_orig['max_dd']:>24.2f}% | {res_prop['max_dd']:>24.2f}%")
    print(f"{'Max Daily DD':<20} | {res_orig['max_daily_dd']:>24.2f}% | {res_prop['max_daily_dd']:>24.2f}%")
    print("="*80)
    
    plt.figure(figsize=(12, 7))
    plt.plot(res_orig['equity'], label=f"Original (Ret {res_orig['return']:.0f}%, DD {res_orig['max_dd']:.2f}%)", color='orange')
    plt.plot(res_prop['equity'], label=f"Prop Firm (Ret {res_prop['return']:.0f}%, DD {res_prop['max_dd']:.2f}%)", color='teal')
    plt.legend()
    plt.title("Final Comparison (Dynamic 5-Tier Strategy)")
    plt.grid(True, alpha=0.3)
    plt.savefig("final_comparison_dynamic.png")

if __name__ == "__main__":
    main()
