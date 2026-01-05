"""
Optimize Prop Firm Bot to achieve DD < 9%
Test different risk levels and SL/TP parameters
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_bot"))

import pandas as pd
import pickle
from services.strategy import calculate_indicators, generate_signals, STRATEGY_PARAMS

SYMBOLS = ["EURUSDm", "USDCADm", "USDJPYm"]
INITIAL_BALANCE = 10000.0
SPREAD_PIPS = {"EURUSDm": 1.2, "USDCADm": 1.5, "USDJPYm": 1.3}
SLIPPAGE_PIPS = 0.5
CACHE_FILE = "backtest_data_cache.pkl"

def get_spread_cost(symbol):
    spread = SPREAD_PIPS.get(symbol, 1.5)
    total_pips = spread + SLIPPAGE_PIPS
    if "JPY" in symbol.upper():
        return total_pips * 0.01
    return total_pips * 0.0001

def get_pip_value(symbol):
    if "JPY" in symbol.upper():
        return 1000
    return 100000

def run_backtest_with_params(risk, params):
    """Run backtest with given risk and parameters"""
    # Load cached data
    with open(CACHE_FILE, 'rb') as f:
        cached_data = pickle.load(f)
    
    all_data = {}
    for symbol in SYMBOLS:
        if symbol in cached_data:
            all_data[symbol] = cached_data[symbol]
    
    # Align indices
    common_index = all_data[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common_index = common_index.intersection(all_data[symbol].index)
    
    for symbol in SYMBOLS:
        all_data[symbol] = all_data[symbol].loc[common_index]
    
    # Backtest
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_drawdown = 0
    positions = {}
    trades = []
    
    start_idx = 250
    
    for i in range(start_idx, len(common_index)):
        for symbol in SYMBOLS:
            row = all_data[symbol].iloc[i]
            price = row["Close"]
            atr = row["ATR"]
            
            if symbol in positions:
                pos = positions[symbol]
                pos["bars"] += 1
                p = params[pos["signal_type"]]
                
                exit_price = None
                exit_reason = ""
                
                if pos["type"] == "buy":
                    pos["highest"] = max(pos["highest"], price)
                    if pos["highest"] - pos["entry"] > p["TRAIL_START"] * atr:
                        pos["sl"] = max(pos["sl"], pos["highest"] - p["TRAIL_DIST"] * atr)
                    
                    if price <= pos["sl"]:
                        exit_price, exit_reason = pos["sl"], "SL"
                    elif price >= pos["tp"]:
                        exit_price, exit_reason = pos["tp"], "TP"
                    elif pos["bars"] >= p["MAX_BARS"]:
                        exit_price, exit_reason = price, "TIME"
                        
                elif pos["type"] == "sell":
                    pos["lowest"] = min(pos["lowest"], price)
                    if pos["entry"] - pos["lowest"] > p["TRAIL_START"] * atr:
                        pos["sl"] = min(pos["sl"], pos["lowest"] + p["TRAIL_DIST"] * atr)
                    
                    if price >= pos["sl"]:
                        exit_price, exit_reason = pos["sl"], "SL"
                    elif price <= pos["tp"]:
                        exit_price, exit_reason = pos["tp"], "TP"
                    elif pos["bars"] >= p["MAX_BARS"]:
                        exit_price, exit_reason = price, "TIME"
                
                if exit_price:
                    pip_mult = get_pip_value(symbol)
                    spread_cost = get_spread_cost(symbol) * pos["size"] * pip_mult
                    
                    if pos["type"] == "buy":
                        pnl = (exit_price - pos["entry"]) * pos["size"] * pip_mult
                    else:
                        pnl = (pos["entry"] - exit_price) * pos["size"] * pip_mult
                    
                    pnl -= spread_cost
                    balance += pnl
                    trades.append({"pnl": pnl})
                    del positions[symbol]
            
            if symbol not in positions:
                signal_type = None
                signal = None
                
                if row.get("trend_buy", False):
                    signal, signal_type = "buy", "TREND"
                elif row.get("mr_buy", False):
                    signal, signal_type = "buy", "MR"
                elif row.get("trend_sell", False):
                    signal, signal_type = "sell", "TREND"
                elif row.get("mr_sell", False):
                    signal, signal_type = "sell", "MR"
                
                if signal:
                    p = params[signal_type]
                    sl_dist = atr * p["SL_ATR"]
                    pip_mult = get_pip_value(symbol)
                    
                    if sl_dist > 0:
                        risk_amount = balance * risk
                        size = risk_amount / (sl_dist * pip_mult)
                        size = max(0.01, min(size, 10.0))
                        
                        if signal == "buy":
                            positions[symbol] = {
                                "type": "buy", "entry": price,
                                "sl": price - sl_dist, "tp": price + atr * p["TP_ATR"],
                                "size": size, "signal_type": signal_type,
                                "bars": 0, "highest": price, "lowest": float('inf')
                            }
                        else:
                            positions[symbol] = {
                                "type": "sell", "entry": price,
                                "sl": price + sl_dist, "tp": price - atr * p["TP_ATR"],
                                "size": size, "signal_type": signal_type,
                                "bars": 0, "highest": 0, "lowest": price
                            }
        
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100
        max_drawdown = max(max_drawdown, dd)
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    return {
        "final_balance": balance,
        "total_return": total_return,
        "max_dd": max_drawdown,
        "total_trades": len(trades)
    }

def main():
    print("=" * 70)
    print("   PROP FIRM BOT OPTIMIZATION - Target: DD < 9%")
    print("=" * 70)
    
    # Test different risk levels
    risk_levels = [0.0018, 0.0016, 0.0015, 0.0014, 0.0013, 0.0012]
    
    # Base parameters from propfirm_bot
    base_params = {
        "TREND": {"SL_ATR": 1.2, "TP_ATR": 3.0, "TRAIL_START": 1.0, "TRAIL_DIST": 0.7, "MAX_BARS": 50},
        "MR": {"SL_ATR": 0.8, "TP_ATR": 2.0, "TRAIL_START": 0.7, "TRAIL_DIST": 0.5, "MAX_BARS": 25}
    }
    
    print("\nTesting different risk levels...")
    print(f"{'Risk %':<10} {'Return %':<12} {'DD %':<10} {'Balance':<15} {'Trades':<10}")
    print("-" * 70)
    
    best_config = None
    
    for risk in risk_levels:
        result = run_backtest_with_params(risk, base_params)
        risk_pct = risk * 100
        
        status = "[OK]" if result["max_dd"] < 9.0 else "[!]"
        print(f"{risk_pct:<10.2f} {result['total_return']:<12.1f} {result['max_dd']:<10.1f} ${result['final_balance']:<14,.0f} {result['total_trades']:<10,} {status}")
        
        if result["max_dd"] < 9.0:
            if best_config is None or result["total_return"] > best_config["return"]:
                best_config = {
                    "risk": risk,
                    "return": result["total_return"],
                    "dd": result["max_dd"],
                    "balance": result["final_balance"],
                    "trades": result["total_trades"]
                }
    
    print("=" * 70)
    
    if best_config:
        print("\n[SUCCESS] Found optimal configuration:")
        print(f"  Risk per trade: {best_config['risk']*100:.2f}%")
        print(f"  Total return: +{best_config['return']:.1f}%")
        print(f"  Max drawdown: {best_config['dd']:.1f}%")
        print(f"  Final balance: ${best_config['balance']:,.0f}")
        print(f"  Total trades: {best_config['trades']:,}")
        print(f"\nUpdate propfirm_bot/.env with: RISK_PERCENT={best_config['risk']*100:.2f}")
    else:
        print("\n[WARNING] Could not find config with DD < 9%")
        print("Consider adjusting SL/TP parameters or reducing symbols")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
