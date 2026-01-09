"""
Comprehensive Financial Performance Analysis
Generates all charts needed for Prop Firm evaluation
"""
import argparse
import sys
import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import importlib.util

# Allow imports from bot dirs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine import calculate_dynamic_risk

def get_strategy_module(bot_name):
    """Load strategy module directly."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if bot_name == "propfirm":
        strategy_path = os.path.join(base_dir, "propfirm_bot", "services", "strategy.py")
    else:
        strategy_path = os.path.join(base_dir, "original_bot", "services", "strategy.py")
    
    spec = importlib.util.spec_from_file_location("strategy", strategy_path)
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    return strategy

def fetch_data(symbol: str, count=48000):
    """Fetch historical data from MT5."""
    if not mt5.initialize():
        print(f"MT5 Init Failed for {symbol}")
        return pd.DataFrame()
    
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, count)
    if rates is None or len(rates) == 0:
        print(f"No data for {symbol}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
    return df

def run_detailed_backtest(df, strategy_params, initial_balance, risk_per_trade, risk_tiers, daily_loss_limit=4.1):
    """Run backtest and return detailed equity curve and trade data."""
    balance = initial_balance
    peak_balance = initial_balance
    equity_curve = []
    trades = []
    daily_pnl = {}
    
    # Position state
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    position_size = 0
    bars_in_trade = 0
    highest = 0
    lowest = float('inf')
    signal_type = ""
    entry_time = None
    
    # Daily tracking
    daily_start_balance = initial_balance
    current_day = None
    
    start_idx = min(250, len(df) // 10)
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        price = row["Close"]
        high = row["High"]
        low = row["Low"]
        atr = row["ATR"]
        
        # Daily reset
        bar_day = current_time.date() if hasattr(current_time, 'date') else current_time
        if current_day != bar_day:
            if current_day is not None:
                daily_pnl[current_day] = balance - daily_start_balance
            current_day = bar_day
            daily_start_balance = balance
        
        # Update peak
        if balance > peak_balance:
            peak_balance = balance
        
        dd_pct = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        
        # Record equity
        equity_curve.append({
            "time": current_time,
            "balance": balance,
            "peak": peak_balance,
            "drawdown_pct": dd_pct
        })
        
        # Exit logic
        if position is not None:
            bars_in_trade += 1
            exit_price = None
            exit_reason = ""
            params = strategy_params.get(signal_type, {})
            if not params:
                continue
            
            if position == "buy":
                highest = max(highest, high)
                if highest - entry_price > params["TRAIL_START"] * atr:
                    new_sl = highest - params["TRAIL_DIST"] * atr
                    stop_loss = max(stop_loss, new_sl)
                
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                elif high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                elif bars_in_trade >= params["MAX_BARS"]:
                    exit_price = price
                    exit_reason = "TIME"
            else:
                lowest = min(lowest, low)
                if entry_price - lowest > params["TRAIL_START"] * atr:
                    new_sl = lowest + params["TRAIL_DIST"] * atr
                    stop_loss = min(stop_loss, new_sl)
                
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                elif low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                elif bars_in_trade >= params["MAX_BARS"]:
                    exit_price = price
                    exit_reason = "TIME"
            
            if exit_price is not None:
                if position == "buy":
                    pnl = (exit_price - entry_price) * position_size * 100000
                else:
                    pnl = (entry_price - exit_price) * position_size * 100000
                
                balance += pnl
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "type": position,
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl / daily_start_balance * 100 if daily_start_balance > 0 else 0,
                    "reason": exit_reason,
                    "bars_held": bars_in_trade
                })
                position = None
        
        # Entry logic
        if position is None:
            current_risk_pct = calculate_dynamic_risk(balance, peak_balance, risk_per_trade, risk_tiers)
            risk_amount = balance * current_risk_pct
            
            if row.get("trend_buy", False):
                signal_type = "TREND"
                params = strategy_params["TREND"]
            elif row.get("mr_buy", False):
                signal_type = "MR"
                params = strategy_params["MR"]
            elif row.get("trend_sell", False):
                signal_type = "TREND"
                params = strategy_params["TREND"]
            elif row.get("mr_sell", False):
                signal_type = "MR"
                params = strategy_params["MR"]
            else:
                continue
            
            sl_dist = atr * params["SL_ATR"]
            if sl_dist <= 0:
                continue
            
            if row.get("trend_buy", False) or row.get("mr_buy", False):
                position = "buy"
                entry_price = price + 0.00015  # spread
                stop_loss = entry_price - sl_dist
                take_profit = entry_price + atr * params["TP_ATR"]
                highest = entry_price
            else:
                position = "sell"
                entry_price = price
                stop_loss = entry_price + sl_dist
                take_profit = entry_price - atr * params["TP_ATR"]
                lowest = entry_price
            
            position_size = risk_amount / (sl_dist * 100000)
            position_size = max(0.01, min(position_size, 10.0))
            bars_in_trade = 0
            entry_time = current_time
    
    # Final day PnL
    if current_day is not None:
        daily_pnl[current_day] = balance - daily_start_balance
    
    return {
        "equity_curve": pd.DataFrame(equity_curve),
        "trades": pd.DataFrame(trades),
        "daily_pnl": daily_pnl,
        "final_balance": balance,
        "initial_balance": initial_balance
    }

def plot_comprehensive_analysis(results, bot_name, save_path):
    """Generate comprehensive financial charts."""
    equity_df = results["equity_curve"]
    trades_df = results["trades"]
    daily_pnl = results["daily_pnl"]
    initial_balance = results["initial_balance"]
    final_balance = results["final_balance"]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'{bot_name.upper()} BOT - Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Equity Curve with Drawdown
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(equity_df["time"], equity_df["balance"], 'b-', linewidth=1.5, label='Balance')
    ax1.plot(equity_df["time"], equity_df["peak"], 'g--', linewidth=1, alpha=0.7, label='Peak Balance')
    ax1.axhline(y=initial_balance, color='gray', linestyle=':', alpha=0.5, label='Initial Balance')
    ax1.axhline(y=initial_balance * 1.08, color='orange', linestyle='--', alpha=0.7, label='Phase 1 Target (+8%)')
    ax1.axhline(y=initial_balance * 1.13, color='green', linestyle='--', alpha=0.7, label='Phase 2 Target (+13%)')
    ax1.set_title('Equity Curve', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Balance ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Drawdown Chart
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.fill_between(equity_df["time"], 0, -equity_df["drawdown_pct"], color='red', alpha=0.5)
    ax2.axhline(y=-9, color='darkred', linestyle='--', linewidth=2, label='Max DD Limit (-9%)')
    ax2.axhline(y=-5, color='orange', linestyle='--', linewidth=1, label='Daily DD Limit (-5%)')
    max_dd = equity_df["drawdown_pct"].max()
    ax2.set_title(f'Drawdown (Max: {max_dd:.2f}%)', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-15, 1)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Daily P&L Distribution
    ax3 = fig.add_subplot(4, 2, 3)
    daily_returns = list(daily_pnl.values())
    colors = ['green' if x >= 0 else 'red' for x in daily_returns]
    ax3.hist(daily_returns, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    mean_daily = np.mean(daily_returns)
    ax3.axvline(x=mean_daily, color='orange', linestyle='--', linewidth=2, label=f'Mean: ${mean_daily:.2f}')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Daily P&L Distribution', fontweight='bold')
    ax3.set_xlabel('Daily P&L ($)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Monthly Returns
    ax4 = fig.add_subplot(4, 2, 4)
    if not trades_df.empty and len(trades_df) > 0:
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        monthly_pnl = trades_df.groupby('month')['pnl'].sum()
        months = [str(m) for m in monthly_pnl.index]
        values = monthly_pnl.values
        colors = ['green' if v >= 0 else 'red' for v in values]
        bars = ax4.bar(range(len(months)), values, color=colors, edgecolor='black', alpha=0.7)
        ax4.set_title('Monthly P&L', fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('P&L ($)')
        ax4.set_xticks(range(0, len(months), max(1, len(months)//12)))
        ax4.set_xticklabels([months[i] for i in range(0, len(months), max(1, len(months)//12))], rotation=45)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Win/Loss Analysis
    ax5 = fig.add_subplot(4, 2, 5)
    if not trades_df.empty:
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] <= 0])
        ax5.pie([wins, losses], labels=[f'Wins ({wins})', f'Losses ({losses})'], 
                colors=['green', 'red'], autopct='%1.1f%%', startangle=90,
                explode=(0.05, 0))
        ax5.set_title(f'Win Rate: {wins/(wins+losses)*100:.1f}%', fontweight='bold')
    
    # 6. Trade Duration Analysis
    ax6 = fig.add_subplot(4, 2, 6)
    if not trades_df.empty and 'bars_held' in trades_df.columns:
        ax6.hist(trades_df['bars_held'], bins=30, color='purple', edgecolor='black', alpha=0.7)
        avg_bars = trades_df['bars_held'].mean()
        ax6.axvline(x=avg_bars, color='orange', linestyle='--', linewidth=2, label=f'Avg: {avg_bars:.1f} bars')
        ax6.set_title('Trade Duration (Bars)', fontweight='bold')
        ax6.set_xlabel('Bars Held')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Cumulative Returns %
    ax7 = fig.add_subplot(4, 2, 7)
    cumulative_return = (equity_df["balance"] / initial_balance - 1) * 100
    ax7.plot(equity_df["time"], cumulative_return, 'b-', linewidth=1.5)
    ax7.axhline(y=8, color='orange', linestyle='--', label='Phase 1 (8%)')
    ax7.axhline(y=13, color='green', linestyle='--', label='Phase 2 (13%)')
    ax7.fill_between(equity_df["time"], 0, cumulative_return, 
                     where=(cumulative_return >= 0), color='green', alpha=0.3)
    ax7.fill_between(equity_df["time"], 0, cumulative_return, 
                     where=(cumulative_return < 0), color='red', alpha=0.3)
    ax7.set_title(f'Cumulative Return ({cumulative_return.iloc[-1]:.1f}%)', fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Return (%)')
    ax7.legend(loc='upper left')
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax7.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
    
    # 8. Key Metrics Summary
    ax8 = fig.add_subplot(4, 2, 8)
    ax8.axis('off')
    
    # Calculate metrics
    total_return = (final_balance / initial_balance - 1) * 100
    max_dd = equity_df["drawdown_pct"].max()
    total_trades = len(trades_df)
    win_rate = len(trades_df[trades_df['pnl'] > 0]) / total_trades * 100 if total_trades > 0 else 0
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if not trades_df.empty else 0
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if not trades_df.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
    
    # Find phase completion days
    phase1_idx = equity_df[equity_df["balance"] >= initial_balance * 1.08].index
    phase1_days = (equity_df.loc[phase1_idx[0], "time"] - equity_df.iloc[0]["time"]).days if len(phase1_idx) > 0 else "N/A"
    
    phase2_idx = equity_df[equity_df["balance"] >= initial_balance * 1.13].index
    phase2_days = (equity_df.loc[phase2_idx[0], "time"] - equity_df.iloc[0]["time"]).days if len(phase2_idx) > 0 else "N/A"
    
    metrics_text = f"""
╔══════════════════════════════════════════════════════════╗
║              KEY PERFORMANCE METRICS                      ║
╠══════════════════════════════════════════════════════════╣
║  Initial Balance:     ${initial_balance:,.2f}                         
║  Final Balance:       ${final_balance:,.2f}                         
║  Total Return:        {total_return:+.2f}%                           
║  Max Drawdown:        {max_dd:.2f}%  {'✓ PASS' if max_dd < 9 else '✗ FAIL'}               
╠══════════════════════════════════════════════════════════╣
║  Total Trades:        {total_trades:,}                              
║  Win Rate:            {win_rate:.1f}%                                
║  Profit Factor:       {profit_factor:.2f}                            
║  Avg Win:             ${avg_win:.2f}                              
║  Avg Loss:            ${avg_loss:.2f}                              
╠══════════════════════════════════════════════════════════╣
║  Phase 1 (8%):        {phase1_days} days                             
║  Phase 2 (13%):       {phase2_days} days                             
╚══════════════════════════════════════════════════════════╝
"""
    ax8.text(0.1, 0.5, metrics_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Chart saved: {save_path}")
    
    return {
        "total_return": total_return,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "phase1_days": phase1_days,
        "phase2_days": phase2_days
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", type=str, default="propfirm", choices=["propfirm", "original"])
    parser.add_argument("--days", type=int, default=1000)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"COMPREHENSIVE ANALYSIS: {args.bot.upper()} BOT")
    print("=" * 60)
    
    # Load strategy
    strategy = get_strategy_module(args.bot)
    
    # Fetch data
    bars_count = args.days * 48
    print(f"Fetching {bars_count} bars...")
    df = fetch_data("EURUSDm", bars_count)
    
    if df.empty:
        print("No data!")
        return
    
    # Calculate indicators
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)
    
    print(f"Data loaded: {len(df)} bars")
    print("Running detailed backtest...")
    
    # Set parameters based on bot type
    if args.bot == "propfirm":
        initial_balance = 2500.0
        risk = 0.0075  # 0.75%
        risk_tiers = [
            (1.5, 0.8), (3.0, 0.6), (4.5, 0.4),
            (6.0, 0.2), (7.5, 0.1), (8.5, 0.05)
        ]
        daily_loss = 4.1
    else:
        initial_balance = 10000.0
        risk = 0.018  # 1.8%
        risk_tiers = [
            (5.0, 0.8), (10.0, 0.6), (15.0, 0.4),
            (18.0, 0.2), (19.0, 0.1)
        ]
        daily_loss = 10.0
    
    # Run backtest
    results = run_detailed_backtest(
        df,
        strategy.STRATEGY_PARAMS,
        initial_balance=initial_balance,
        risk_per_trade=risk,
        risk_tiers=risk_tiers,
        daily_loss_limit=daily_loss
    )
    
    # Generate charts
    save_path = os.path.join(os.path.dirname(__file__), f"{args.bot}_performance_analysis.png")
    metrics = plot_comprehensive_analysis(results, args.bot, save_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Return:    {metrics['total_return']:+.2f}%")
    print(f"Max Drawdown:    {metrics['max_dd']:.2f}%")
    print(f"Total Trades:    {metrics['total_trades']}")
    print(f"Win Rate:        {metrics['win_rate']:.1f}%")
    print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    print(f"Phase 1 (8%):    {metrics['phase1_days']} days")
    print(f"Phase 2 (13%):   {metrics['phase2_days']} days")
    print("=" * 60)

if __name__ == "__main__":
    main()
