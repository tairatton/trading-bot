"""
================================================================================
PROP FIRM BOT - Backtest
================================================================================
Purpose: Safe trading for Prop Firm Challenge (The5ers, FTMO, etc.)
Symbols: 3 (EURUSD, USDCAD, USDJPY)
Risk: 0.15% per trade (0.45% total when all open)

Prop Firm Limits:
- Daily Loss: 5% max (we use 4% buffer)
- Max Loss: $1,000 (for $10K account)
- Profit Target: 8% Phase 1, 5% Phase 2

Use this for: Prop Firm Challenge
NOT for: Personal Trading (use backtest_original.py for higher returns)
================================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.strategy import calculate_indicators, generate_signals

# PROP FIRM CONFIGURATION
SYMBOLS = ["EURUSDm", "USDCADm", "USDJPYm"]  # 3 symbols like original
INITIAL_BALANCE = 10000.0  # $10,000 account
RISK_PER_TRADE = 0.0018  # 0.18% per trade (0.54% total when all open)

# PROP FIRM LIMITS
DAILY_LOSS_LIMIT = 0.041  # 4.1% daily stop (buffer below 5% limit)
MAX_LOSS_ABSOLUTE = 1000  # $1000 max loss limit

# Aggressive trailing for faster profits
CUSTOM_PARAMS = {
    "TREND": {"SL_ATR": 1.2, "TP_ATR": 3.0, "TRAIL_START": 1.0, "TRAIL_DIST": 0.7, "MAX_BARS": 50},
    "MR": {"SL_ATR": 0.8, "TP_ATR": 2.0, "TRAIL_START": 0.7, "TRAIL_DIST": 0.5, "MAX_BARS": 25}
}
USE_CUSTOM_PARAMS = True

# Trading Costs
SPREAD_PIPS = {
    "EURUSDm": 1.2,
    "USDCADm": 1.5,
    "USDJPYm": 1.3
}
SLIPPAGE_PIPS = 0.5

def get_spread_cost(symbol: str) -> float:
    """Get spread + slippage in price units."""
    spread = SPREAD_PIPS.get(symbol, 1.5)
    total_pips = spread + SLIPPAGE_PIPS
    if "JPY" in symbol.upper():
        return total_pips * 0.01
    return total_pips * 0.0001

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
    """Get pip value multiplier based on symbol type."""
    if "JPY" in symbol.upper():
        return 1000
    return 100000

def run_propfirm_backtest():
    """Run backtest optimized for prop firm challenge."""
    print("=" * 80)
    print("   PROP FIRM CHALLENGE BACKTEST")
    print(f"   Target: 8% profit in 1 month")
    print(f"   Risk: {RISK_PER_TRADE*100}% per trade (AGGRESSIVE)")
    print(f"   Daily Loss Limit: {DAILY_LOSS_LIMIT*100}%")
    print("=" * 80)
    
    if not mt5.initialize():
        print("MT5 init failed")
        return
    
    # Fetch data
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
    
    if len(all_data) < len(SYMBOLS):
        print("Not all symbols loaded")
        return
    
    # Common time range
    common_index = all_data[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common_index = common_index.intersection(all_data[symbol].index)
    
    print(f"\n[*] Common bars: {len(common_index)}")
    print(f"    From: {common_index[0]}")
    print(f"    To:   {common_index[-1]}")
    
    for symbol in SYMBOLS:
        all_data[symbol] = all_data[symbol].loc[common_index]
    
    # Backtest state
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_drawdown = 0
    
    # Daily tracking
    daily_start_balance = {}
    daily_pnl = {}
    daily_stopped = set()  # Days where trading stopped due to daily loss limit
    
    positions = {}
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
        
        # Check daily loss limit
        day_pnl = balance - daily_start_balance[current_date]
        day_loss_pct = abs(day_pnl / daily_start_balance[current_date]) if day_pnl < 0 else 0
        
        # Stop trading for the day if hit daily limit
        if day_loss_pct >= DAILY_LOSS_LIMIT:
            if current_date not in daily_stopped:
                daily_stopped.add(current_date)
                print(f"[!] DAILY STOP: {current_date} - Loss {day_loss_pct*100:.2f}%")
            continue  # Skip trading for rest of the day
        
        # Process each symbol
        for symbol in SYMBOLS:
            row = all_data[symbol].iloc[i]
            price = row["Close"]
            atr = row["ATR"]
            
            # Check exits for existing position
            if symbol in positions:
                pos = positions[symbol]
                pos["bars"] += 1
                params = CUSTOM_PARAMS[pos["signal_type"]]
                
                exit_price = None
                exit_reason = ""
                
                if pos["type"] == "buy":
                    pos["highest"] = max(pos["highest"], price)
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
                    
                    pnl -= spread_cost
                    balance += pnl
                    daily_pnl[current_date] += pnl
                    
                    trades.append({
                        "symbol": symbol, "type": pos["type"], "entry": pos["entry"],
                        "exit": exit_price, "pnl": pnl, "reason": exit_reason,
                        "entry_time": pos["entry_time"], "exit_time": current_time
                    })
                    del positions[symbol]
            
            # Check entry (only if no position and not daily stopped)
            if symbol not in positions and current_date not in daily_stopped:
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
                    params = CUSTOM_PARAMS[signal_type]
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
        
        # Check max loss absolute
        total_loss = INITIAL_BALANCE - balance
        if total_loss >= MAX_LOSS_ABSOLUTE:
            print(f"\n[!] MAX LOSS HIT: ${total_loss:.0f}")
            break
    
    # Results
    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    
    total_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
    profit_factor = total_profit / total_loss
    
    # Calculate daily stats - find worst daily loss percentage
    max_daily_loss_pct = 0
    max_daily_loss_usd = 0
    for date, pnl in daily_pnl.items():
        if pnl < 0:
            day_start = daily_start_balance.get(date, INITIAL_BALANCE)
            loss_pct = abs(pnl / day_start * 100)
            if loss_pct > max_daily_loss_pct:
                max_daily_loss_pct = loss_pct
                max_daily_loss_usd = abs(pnl)
    
    # Monthly return estimate
    total_days = (common_index[-1] - common_index[start_idx]).days
    total_months = total_days / 30
    monthly_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE / total_months * 100) if total_months > 0 else 0
    
    print("\n" + "=" * 80)
    print("   PROP FIRM BACKTEST RESULTS")
    print("=" * 80)
    print(f"  Risk per trade:   {RISK_PER_TRADE*100}%")
    print(f"  Daily Loss Limit: {DAILY_LOSS_LIMIT*100}%")
    print("-" * 80)
    print(f"  Total Trades:     {len(trades)}")
    print(f"  Wins/Losses:      {len(wins)} / {len(losses)}")
    print(f"  Win Rate:         {len(wins)/len(trades)*100:.1f}%")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print("-" * 80)
    print(f"  Initial Balance:  ${INITIAL_BALANCE:,.0f}")
    print(f"  Final Balance:    ${balance:,.0f}")
    print(f"  Total Return:     {(balance-INITIAL_BALANCE)/INITIAL_BALANCE*100:.1f}%")
    print(f"  Monthly Return:   {monthly_return:.1f}%")
    print("-" * 80)
    print(f"  MAX DRAWDOWN:     {max_drawdown:.1f}%")
    print(f"  Max Daily Loss:   ${max_daily_loss_usd:.0f} ({max_daily_loss_pct:.1f}%)")
    print(f"  Days Stopped:     {len(daily_stopped)}")
    print("-" * 80)
    
    # Check if meets prop firm requirements
    phase1_pass = monthly_return >= 8 and max_daily_loss_pct < 5 and max_drawdown < 15
    if phase1_pass:
        print(f"  STATUS:           [PASS] Phase 1 Requirements Met!")
        print(f"                    - Monthly return >= 8% [OK]")
        print(f"                    - Daily loss < 5% [OK]")
        print(f"                    - Overall DD < 15% [OK]")
    else:
        print(f"  STATUS:           [WARN] Phase 1 Requirements NOT MET")
        if monthly_return < 8:
            print(f"                    - Monthly return {monthly_return:.1f}% < 8% [FAIL]")
        if max_daily_loss_pct >= 5:
            print(f"                    - Daily loss {max_daily_loss_pct:.1f}% >= 5% [FAIL]")
        if max_drawdown >= 15:
            print(f"                    - DD {max_drawdown:.1f}% >= 15% [FAIL]")
    
    print("=" * 80)
    
    # Save equity curve for plotting
    equity_df = pd.DataFrame(equity_curve)
    csv_path = os.path.join(os.path.dirname(__file__), "propfirm_equity_curve.csv")
    equity_df.to_csv(csv_path, index=False)
    print(f"\n[*] Equity curve saved: {csv_path}")
    
    # Generate plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'Prop Firm Backtest - Risk {RISK_PER_TRADE*100}% | Monthly Return: {monthly_return:.1f}%', fontsize=14, fontweight='bold')
        
        # Equity Curve
        ax1.plot(equity_df['time'], equity_df['balance'], linewidth=1.5, color='#2ecc71')
        ax1.axhline(y=INITIAL_BALANCE, color='blue', linestyle='--', alpha=0.5, label=f'Initial: ${INITIAL_BALANCE:,.0f}')
        ax1.set_ylabel('Balance ($)')
        ax1.set_title('Equity Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown
        equity_df['peak'] = equity_df['balance'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['balance']) / equity_df['peak'] * 100
        ax2.fill_between(equity_df['time'], 0, equity_df['drawdown'], alpha=0.4, color='red')
        ax2.axhline(y=max_drawdown, color='orange', linestyle=':', label=f'Max DD: {max_drawdown:.1f}%')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.invert_yaxis()
        
        plt.tight_layout()
        chart_path = os.path.join(os.path.dirname(__file__), "propfirm_backtest_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"[*] Chart saved: {chart_path}")
        plt.show()
    except Exception as e:
        print(f"[!] Plot error: {e}")


if __name__ == "__main__":
    run_propfirm_backtest()
