"""
================================================================================
ORIGINAL TRADING BOT - Backtest
================================================================================
Purpose: High return trading for personal account
Symbols: 3 (EURUSD, USDCAD, USDJPY)
Risk: 0.5% per trade (1.5% total when all open)
Expected: ~9% monthly return, ~12% max DD

Use this for: เทรดเอง (Personal Trading)
NOT for: Prop Firm Challenge (use backtest_propfirm.py instead)
================================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from services.strategy import calculate_indicators, generate_signals, STRATEGY_PARAMS

SYMBOLS = ["EURUSDm", "USDCADm", "USDCHFm"]
INITIAL_BALANCE = 10000.0  # 10,000 THB for Exness Cent
RISK_PER_TRADE = 0.01   # 1.0% per symbol (Target < 20% DD)

# MOONSHOT Parameters (matches live bot and verified backtest)
CUSTOM_PARAMS = {
    "TREND": {"SL_ATR": 1.2, "TP_ATR": 5.0, "TRAIL_START": 1.2, "TRAIL_DIST": 0.6, "MAX_BARS": 50},
    "MR": {"SL_ATR": 0.8, "TP_ATR": 3.0, "TRAIL_START": 0.64, "TRAIL_DIST": 0.4, "MAX_BARS": 25}
}
USE_CUSTOM_PARAMS = False  # Uses STRATEGY_PARAMS from strategy.py (also Moonshot)

# Trading Costs (Exness Cent Account)
SPREAD_PIPS = {
    "EURUSDm": 1.2,   # ~1.2 pips average
    "USDCADm": 1.5,
    "USDCHFm": 1.5,
    "USDJPYm": 1.3,   # Raw spread in pips for JPY
}
SLIPPAGE_PIPS = 0.5  # Average slippage per trade

def get_spread_cost(symbol: str) -> float:
    """Get spread + slippage in price units."""
    spread = SPREAD_PIPS.get(symbol, 1.5)
    total_pips = spread + SLIPPAGE_PIPS
    # Convert pips to price
    if "JPY" in symbol.upper():
        return total_pips * 0.01  # JPY: 1 pip = 0.01
    return total_pips * 0.0001  # Standard: 1 pip = 0.0001

def fetch_data(symbol: str, timeframe=mt5.TIMEFRAME_M30, count=70000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df


def get_pip_value(symbol: str) -> float:
    """Get pip value multiplier based on symbol type.
    
    For USD pairs (EURUSD, etc): 1 pip = 0.0001, multiplier = 100000
    For JPY pairs (USDJPY, etc): 1 pip = 0.01, multiplier = 1000 (100x smaller)
    """
    if "JPY" in symbol.upper():
        return 1000  # JPY pairs have 2 decimal pips
    return 100000  # Standard pairs have 4 decimal pips


def run_combined_backtest():
    """Run backtest with all symbols sharing one account."""
    print("=" * 70)
    print("   COMBINED MULTI-SYMBOL BACKTEST")
    print(f"   Symbols: {SYMBOLS}")
    print(f"   Risk per trade: {RISK_PER_TRADE*100}%")
    print("=" * 70)
    
    if not mt5.initialize():
        print("MT5 init failed")
        return
    
    # Fetch and prepare data for all symbols
    all_data = {}
    for symbol in SYMBOLS:
        print(f"[*] Loading {symbol}...")
        df = fetch_data(symbol)
        if df.empty:
            print(f"    No data for {symbol}")
            continue
        df = calculate_indicators(df)
        df = generate_signals(df)
        all_data[symbol] = df
        print(f"    {len(df)} bars loaded")
    
    mt5.shutdown()
    
    # Save to cache for unified_comparison.py
    try:
        import pickle
        with open("backtest_data_cache.pkl", "wb") as f:
            pickle.dump(all_data, f)
        print(f"[*] Data cached to backtest_data_cache.pkl")
    except Exception as e:
        print(f"[!] Failed to save cache: {e}")
    
    if len(all_data) < len(SYMBOLS):
        print("Not all symbols loaded")
        return
    
    # Find common time range
    common_index = all_data[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common_index = common_index.intersection(all_data[symbol].index)
    
    print(f"\n[*] Common bars: {len(common_index)}")
    print(f"    From: {common_index[0]}")
    print(f"    To:   {common_index[-1]}")
    
    # Align all data to common index
    for symbol in SYMBOLS:
        all_data[symbol] = all_data[symbol].loc[common_index]
    
    # Backtest state
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_drawdown = 0
    
    # Daily tracking
    daily_start_balance = {}
    daily_pnl = {}
    
    positions = {}  # {symbol: {type, entry, sl, tp, size, signal_type, entry_time, bars, highest, lowest}}
    trades = []
    equity_curve = []
    
    start_idx = 250
    
    for i in range(start_idx, len(common_index)):
        current_time = common_index[i]
        current_date = current_time.date()
        
        # Track daily start balance
        if current_date not in daily_start_balance:
            daily_start_balance[current_date] = balance
            daily_pnl[current_date] = 0
        
        # Process each symbol
        for symbol in SYMBOLS:
            row = all_data[symbol].iloc[i]
            price = row["Close"]
            atr = row["ATR"]
            
            # Check exits for existing position
            if symbol in positions:
                pos = positions[symbol]
                pos["bars"] += 1
                params = (CUSTOM_PARAMS if USE_CUSTOM_PARAMS else STRATEGY_PARAMS)[pos["signal_type"]]
                
                exit_price = None
                exit_reason = ""
                
                if pos["type"] == "buy":
                    pos["highest"] = max(pos["highest"], price)
                    # Trailing stop
                    if pos["highest"] - pos["entry"] > params["TRAIL_START"] * atr:
                        new_sl = pos["highest"] - params["TRAIL_DIST"] * atr
                        pos["sl"] = max(pos["sl"], new_sl)
                    
                    if price <= pos["sl"]:
                        exit_price, exit_reason = pos["sl"], "SL"
                    elif price >= pos["tp"]:
                        exit_price, exit_reason = pos["tp"], "TP"
                    elif pos["bars"] >= params["MAX_BARS"]:
                        exit_price, exit_reason = price, "TIME"
                        
                elif pos["type"] == "sell":
                    pos["lowest"] = min(pos["lowest"], price)
                    # Trailing stop
                    if pos["entry"] - pos["lowest"] > params["TRAIL_START"] * atr:
                        new_sl = pos["lowest"] + params["TRAIL_DIST"] * atr
                        pos["sl"] = min(pos["sl"], new_sl)
                    
                    if price >= pos["sl"]:
                        exit_price, exit_reason = pos["sl"], "SL"
                    elif price <= pos["tp"]:
                        exit_price, exit_reason = pos["tp"], "TP"
                    elif pos["bars"] >= params["MAX_BARS"]:
                        exit_price, exit_reason = price, "TIME"
                
                # Close position
                if exit_price:
                    pip_mult = get_pip_value(symbol)
                    spread_cost = get_spread_cost(symbol) * pos["size"] * pip_mult
                    
                    if pos["type"] == "buy":
                        pnl = (exit_price - pos["entry"]) * pos["size"] * pip_mult
                    else:
                        pnl = (pos["entry"] - exit_price) * pos["size"] * pip_mult
                    
                    # Deduct spread + slippage cost
                    pnl -= spread_cost
                    
                    balance += pnl
                    daily_pnl[current_date] += pnl
                    trades.append({
                        "symbol": symbol, "type": pos["type"], "entry": pos["entry"],
                        "exit": exit_price, "pnl": pnl, "reason": exit_reason,
                        "entry_time": pos["entry_time"], "exit_time": current_time
                    })
                    del positions[symbol]
            
            # Check entry (only if no position for this symbol)
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
                    params = (CUSTOM_PARAMS if USE_CUSTOM_PARAMS else STRATEGY_PARAMS)[signal_type]
                    sl_dist = atr * params["SL_ATR"]
                    pip_mult = get_pip_value(symbol)
                    if sl_dist > 0:
                        risk_amount = balance * RISK_PER_TRADE
                        size = risk_amount / (sl_dist * pip_mult)
                        size = max(0.01, min(size, 10.0))
                        
                        if signal == "buy":
                            sl = price - sl_dist
                            tp = price + atr * params["TP_ATR"]
                            positions[symbol] = {
                                "type": "buy", "entry": price, "sl": sl, "tp": tp,
                                "size": size, "signal_type": signal_type,
                                "entry_time": current_time, "bars": 0,
                                "highest": price, "lowest": float('inf')
                            }
                        else:
                            sl = price + sl_dist
                            tp = price - atr * params["TP_ATR"]
                            positions[symbol] = {
                                "type": "sell", "entry": price, "sl": sl, "tp": tp,
                                "size": size, "signal_type": signal_type,
                                "entry_time": current_time, "bars": 0,
                                "highest": 0, "lowest": price
                            }
        
        # Track equity and drawdown
        equity_curve.append({"time": current_time, "balance": balance, "open_positions": len(positions)})
        
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100
        max_drawdown = max(max_drawdown, dd)
    
    # Results
    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    
    total_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
    profit_factor = total_profit / total_loss
    
    print("\n" + "=" * 70)
    print("   COMBINED BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Symbols:          {', '.join(SYMBOLS)}")
    print(f"  Risk per trade:   {RISK_PER_TRADE*100}%")
    print(f"  Total Risk:       {RISK_PER_TRADE*100*len(SYMBOLS)}% (when all open)")
    print("-" * 70)
    print(f"  Total Trades:     {len(trades)}")
    print(f"  Wins/Losses:      {len(wins)} / {len(losses)}")
    print(f"  Win Rate:         {len(wins)/len(trades)*100:.1f}%")
    print("-" * 70)
    print(f"  Initial Balance:  ${INITIAL_BALANCE:,.0f}")
    print(f"  Final Balance:    ${balance:,.0f}")
    print(f"  Total Return:     {(balance-INITIAL_BALANCE)/INITIAL_BALANCE*100:.1f}%")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  MAX DRAWDOWN:     {max_drawdown:.1f}%")
    
    # Calculate max daily loss
    max_daily_loss_pct = 0
    max_daily_loss_usd = 0
    for date, pnl in daily_pnl.items():
        if pnl < 0:
            day_start = daily_start_balance.get(date, INITIAL_BALANCE)
            loss_pct = abs(pnl / day_start * 100)
            if loss_pct > max_daily_loss_pct:
                max_daily_loss_pct = loss_pct
                max_daily_loss_usd = abs(pnl)
    
    print(f"  Max Daily Loss:   ${max_daily_loss_usd:,.0f} ({max_daily_loss_pct:.1f}%)")
    print("=" * 70)
    
    # Per-symbol breakdown
    print("\n  Per-Symbol Breakdown:")
    for symbol in SYMBOLS:
        sym_trades = trades_df[trades_df["symbol"] == symbol]
        sym_pnl = sym_trades["pnl"].sum()
        sym_wins = len(sym_trades[sym_trades["pnl"] > 0])
        print(f"    {symbol}: {len(sym_trades)} trades, ${sym_pnl:,.0f} PnL, {sym_wins} wins")
    
    # Results displayed above (no CSV output)


if __name__ == "__main__":
    run_combined_backtest()
