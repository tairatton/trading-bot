"""
Comprehensive Bot Comparison
Shows all metrics: Return, DD, Daily Loss, Win Rate, Profit Factor
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_bot"))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from services.strategy import calculate_indicators, generate_signals, STRATEGY_PARAMS

# Configuration
CONFIGS = {
    "Original Bot": {
        "symbols": ["EURUSDm", "USDCADm", "USDJPYm"],
        "risk": 0.005,
        "params": STRATEGY_PARAMS,
        "color": "#2ecc71"
    },
    "Prop Firm Bot": {
        "symbols": ["EURUSDm", "USDCADm", "USDJPYm"],
        "risk": 0.0015,
        "params": {
            "TREND": {"SL_ATR": 1.2, "TP_ATR": 3.0, "TRAIL_START": 1.0, "TRAIL_DIST": 0.7, "MAX_BARS": 50},
            "MR": {"SL_ATR": 0.8, "TP_ATR": 2.0, "TRAIL_START": 0.7, "TRAIL_DIST": 0.5, "MAX_BARS": 25}
        },
        "color": "#3498db"
    }
}

INITIAL_BALANCE = 10000.0
SPREAD_PIPS = {"EURUSDm": 1.2, "USDCADm": 1.5, "USDJPYm": 1.3}
SLIPPAGE_PIPS = 0.5

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

def run_backtest(config_name, config):
    """Run full backtest with all metrics"""
    symbols = config["symbols"]
    risk = config["risk"]
    params = config["params"]
    
    all_data = {}
    for symbol in symbols:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, 70000)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        df = calculate_indicators(df)
        df = generate_signals(df)
        all_data[symbol] = df
    
    common_index = all_data[symbols[0]].index
    for symbol in symbols[1:]:
        common_index = common_index.intersection(all_data[symbol].index)
    
    for symbol in symbols:
        all_data[symbol] = all_data[symbol].loc[common_index]
    
    # Backtest
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_drawdown = 0
    
    daily_start_balance = {}
    daily_pnl = {}
    
    positions = {}
    trades = []
    equity_curve = []
    drawdown_curve = []
    
    start_idx = 250
    
    for i in range(start_idx, len(common_index)):
        current_time = common_index[i]
        current_date = current_time.date()
        
        if current_date not in daily_start_balance:
            daily_start_balance[current_date] = balance
            daily_pnl[current_date] = 0
        
        for symbol in symbols:
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
                    daily_pnl[current_date] += pnl
                    trades.append({"pnl": pnl, "reason": exit_reason})
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
        
        equity_curve.append({"time": current_time, "balance": balance})
        
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100
        max_drawdown = max(max_drawdown, dd)
        drawdown_curve.append({"time": current_time, "dd": dd})
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    
    total_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    total_loss_amt = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
    
    max_daily_loss_pct = 0
    for date, pnl in daily_pnl.items():
        if pnl < 0:
            day_start = daily_start_balance.get(date, INITIAL_BALANCE)
            loss_pct = abs(pnl / day_start * 100)
            max_daily_loss_pct = max(max_daily_loss_pct, loss_pct)
    
    # Calculate monthly return
    total_days = (common_index[-1] - common_index[start_idx]).days
    total_months = total_days / 30
    monthly_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE / total_months * 100) if total_months > 0 else 0
    
    return {
        "equity_df": pd.DataFrame(equity_curve),
        "dd_df": pd.DataFrame(drawdown_curve),
        "final_balance": balance,
        "total_return": (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        "monthly_return": monthly_return,
        "max_dd": max_drawdown,
        "max_daily_loss": max_daily_loss_pct,
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if len(trades) > 0 else 0,
        "profit_factor": total_profit / total_loss_amt if total_loss_amt > 0 else 0,
        "color": config["color"]
    }

def main():
    print("=" * 70)
    print("   COMPREHENSIVE BOT COMPARISON")
    print("=" * 70)
    
    if not mt5.initialize():
        print("MT5 init failed")
        return
    
    results = {}
    
    for name, config in CONFIGS.items():
        print(f"\n[*] Running {name}...")
        result = run_backtest(name, config)
        if result:
            results[name] = result
            print(f"    Done: ${result['final_balance']:,.0f}")
    
    mt5.shutdown()
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Bot Comparison: Original vs Prop Firm', fontsize=16, fontweight='bold')
    
    # Layout: 2x3 grid
    ax1 = plt.subplot(2, 3, 1)  # Equity curve
    ax2 = plt.subplot(2, 3, 2)  # Drawdown curve
    ax3 = plt.subplot(2, 3, 3)  # Return comparison
    ax4 = plt.subplot(2, 3, 4)  # DD comparison
    ax5 = plt.subplot(2, 3, 5)  # Win rate comparison
    ax6 = plt.subplot(2, 3, 6)  # Profit Factor comparison
    
    names = list(results.keys())
    colors = [results[n]["color"] for n in names]
    
    # Plot 1: Equity Curves
    # Plot 1: Equity Curves
    for name, data in results.items():
        total_ret = data["total_return"]
        total_profit = data["final_balance"] - INITIAL_BALANCE
        label_text = f"{name} (+{total_ret:.0f}% | +${total_profit:,.0f})"
        
        ax1.plot(data["equity_df"]["time"], data["equity_df"]["balance"], 
                 label=label_text, linewidth=1.5, color=data["color"])
    ax1.set_title('Equity Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Drawdown Curves
    for name, data in results.items():
        ax2.fill_between(data["dd_df"]["time"], 0, data["dd_df"]["dd"], alpha=0.3, color=data["color"])
        ax2.plot(data["dd_df"]["time"], data["dd_df"]["dd"], label=name, linewidth=1, color=data["color"])
    ax2.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # Plot 3: Total Return
    returns = [results[n]["total_return"] for n in names]
    bars = ax3.bar(names, returns, color=colors, alpha=0.8)
    ax3.set_title('Total Return (%)', fontsize=12, fontweight='bold')
    for bar, ret in zip(bars, returns):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{ret:.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Max DD comparison
    dds = [results[n]["max_dd"] for n in names]
    bars = ax4.bar(names, dds, color=colors, alpha=0.8)
    ax4.set_title('Max Drawdown (%)', fontsize=12, fontweight='bold')
    for bar, dd in zip(bars, dds):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{dd:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax4.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='15% limit')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Win Rate
    win_rates = [results[n]["win_rate"] for n in names]
    bars = ax5.bar(names, win_rates, color=colors, alpha=0.8)
    ax5.set_title('Win Rate (%)', fontsize=12, fontweight='bold')
    for bar, wr in zip(bars, win_rates):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{wr:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Profit Factor
    pfs = [results[n]["profit_factor"] for n in names]
    bars = ax6.bar(names, pfs, color=colors, alpha=0.8)
    ax6.set_title('Profit Factor', fontsize=12, fontweight='bold')
    for bar, pf in zip(bars, pfs):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{pf:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax6.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_full.png', dpi=150, bbox_inches='tight')
    print("\n[*] Chart saved: comparison_full.png")
    
    # Print full summary
    print("\n" + "=" * 70)
    print("   FULL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Original Bot':>15} {'Prop Firm Bot':>15}")
    print("-" * 70)
    print(f"{'Risk/Trade':<20} {'0.5%':>15} {'0.15%':>15}")
    print(f"{'Total Risk':<20} {'1.5%':>15} {'0.45%':>15}")
    print(f"{'Trades':<20} {results['Original Bot']['total_trades']:>15,} {results['Prop Firm Bot']['total_trades']:>15,}")
    print(f"{'Win Rate':<20} {results['Original Bot']['win_rate']:>14.1f}% {results['Prop Firm Bot']['win_rate']:>14.1f}%")
    print(f"{'Profit Factor':<20} {results['Original Bot']['profit_factor']:>15.2f} {results['Prop Firm Bot']['profit_factor']:>15.2f}")
    print(f"{'Final Balance':<20} ${results['Original Bot']['final_balance']:>13,.0f} ${results['Prop Firm Bot']['final_balance']:>13,.0f}")
    print(f"{'Total Return':<20} {results['Original Bot']['total_return']:>14.1f}% {results['Prop Firm Bot']['total_return']:>14.1f}%")
    print(f"{'Monthly Return':<20} {results['Original Bot']['monthly_return']:>14.1f}% {results['Prop Firm Bot']['monthly_return']:>14.1f}%")
    print(f"{'Max DD':<20} {results['Original Bot']['max_dd']:>14.1f}% {results['Prop Firm Bot']['max_dd']:>14.1f}%")
    print(f"{'Max Daily Loss':<20} {results['Original Bot']['max_daily_loss']:>14.1f}% {results['Prop Firm Bot']['max_daily_loss']:>14.1f}%")
    print("=" * 70)
    
    plt.show()

if __name__ == "__main__":
    main()
