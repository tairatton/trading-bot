"""
Comprehensive Portfolio Financial Analysis
Generates all charts needed for Prop Firm evaluation (Multi-Pair)
"""
import argparse
import sys
import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import importlib.util
from datetime import datetime

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

def run_portfolio_backtest_detailed(pairs_data, strategy_params, initial_balance, risk_per_trade, risk_tiers, daily_loss_limit):
    """Run portfolio backtest and return comprehensive data for plotting."""
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0
    equity_curve = []
    all_trades = []
    
    # Position state per symbol
    positions = {}
    
    # Daily tracking
    daily_start_balance = initial_balance
    current_day = None
    daily_pnl = {}
    
    # Merge all dataframes by time
    all_times = set()
    for symbol, df in pairs_data.items():
        all_times.update(df.index.tolist())
    all_times = sorted(all_times)
    
    start_idx = min(250, len(all_times) // 10)
    
    for i, current_time in enumerate(all_times[start_idx:], start=start_idx):
        # Daily reset
        bar_day = current_time.date() if hasattr(current_time, 'date') else current_time
        if current_day != bar_day:
            if current_day is not None:
                daily_pnl[current_day] = balance - daily_start_balance
            current_day = bar_day
            daily_start_balance = balance
        
        # Check daily loss
        daily_loss_pct = ((daily_start_balance - balance) / daily_start_balance * 100) if daily_start_balance > 0 else 0
        daily_blocked = daily_loss_pct >= daily_loss_limit
        
        # Update peak and DD
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        
        equity_curve.append({
            "time": current_time,
            "balance": balance,
            "peak": peak_balance,
            "drawdown_pct": dd
        })
        
        # Process each symbol
        for symbol, df in pairs_data.items():
            if current_time not in df.index:
                continue
                
            row = df.loc[current_time]
            price = row["Close"]
            high = row["High"]
            low = row["Low"]
            atr = row.get("ATR", 0.001)
            
            pip_value = 0.0068 if "JPY" in symbol.upper() else 0.0001
            spread_pips = 1.5
            spread_price = spread_pips * pip_value
            
            if symbol not in positions:
                positions[symbol] = {"position": None}
            pos = positions[symbol]
            
            # Exit logic
            if pos["position"] is not None:
                pos["bars_in_trade"] = pos.get("bars_in_trade", 0) + 1
                exit_price = None
                exit_reason = ""
                params = strategy_params.get(pos["signal_type"], {})
                
                if pos["position"] == "buy":
                    pos["highest"] = max(pos.get("highest", price), high)
                    if pos["highest"] - pos["entry_price"] > params.get("TRAIL_START", 1.0) * atr:
                        new_sl = pos["highest"] - params.get("TRAIL_DIST", 0.6) * atr
                        pos["stop_loss"] = max(pos["stop_loss"], new_sl)
                    
                    if low <= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        exit_reason = "SL"
                    elif high >= pos["take_profit"]:
                        exit_price = pos["take_profit"]
                        exit_reason = "TP"
                    elif pos["bars_in_trade"] >= params.get("MAX_BARS", 50):
                        exit_price = price
                        exit_reason = "TIME"
                else: 
                    pos["lowest"] = min(pos.get("lowest", price), low)
                    if pos["entry_price"] - pos["lowest"] > params.get("TRAIL_START", 1.0) * atr:
                        new_sl = pos["lowest"] + params.get("TRAIL_DIST", 0.6) * atr
                        pos["stop_loss"] = min(pos["stop_loss"], new_sl)
                    
                    if high >= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        exit_reason = "SL"
                    elif low <= pos["take_profit"]:
                        exit_price = pos["take_profit"]
                        exit_reason = "TP"
                    elif pos["bars_in_trade"] >= params.get("MAX_BARS", 50):
                        exit_price = price
                        exit_reason = "TIME"
                
                if exit_price is not None:
                    if pos["position"] == "buy":
                        pnl = (exit_price - pos["entry_price"]) * pos["position_size"] * 100000
                    else:
                        pnl = (pos["entry_price"] - exit_price) * pos["position_size"] * 100000
                    
                    balance += pnl
                    all_trades.append({
                        "entry_time": pos["entry_time"],
                        "exit_time": current_time,
                        "symbol": symbol,
                        "type": pos["position"],
                        "pnl": pnl,
                        "reason": exit_reason,
                        "bars_held": pos["bars_in_trade"]
                    })
                    pos["position"] = None
            
            # Entry logic
            if pos["position"] is None and not daily_blocked:
                current_risk_pct = calculate_dynamic_risk(balance, peak_balance, risk_per_trade, risk_tiers)
                risk_amount = balance * current_risk_pct
                
                signal_type = None
                direction = None
                
                if row.get("trend_buy", False):
                    signal_type = "TREND"; direction = "buy"
                elif row.get("mr_buy", False):
                    signal_type = "MR"; direction = "buy"
                elif row.get("trend_sell", False):
                    signal_type = "TREND"; direction = "sell"
                elif row.get("mr_sell", False):
                    signal_type = "MR"; direction = "sell"
                
                if signal_type and direction:
                    params = strategy_params[signal_type]
                    sl_dist = atr * params["SL_ATR"]
                    
                    if sl_dist > 0:
                        pos["position"] = direction
                        pos["signal_type"] = signal_type
                        pos["bars_in_trade"] = 0
                        pos["entry_time"] = current_time
                        
                        if direction == "buy":
                            pos["entry_price"] = price + spread_price
                            pos["stop_loss"] = pos["entry_price"] - sl_dist
                            pos["take_profit"] = pos["entry_price"] + atr * params["TP_ATR"]
                            pos["highest"] = pos["entry_price"]
                        else:
                            pos["entry_price"] = price
                            pos["stop_loss"] = pos["entry_price"] + sl_dist
                            pos["take_profit"] = pos["entry_price"] - atr * params["TP_ATR"]
                            pos["lowest"] = pos["entry_price"]
                        
                        pos["position_size"] = risk_amount / (sl_dist * 100000)
                        pos["position_size"] = max(0.01, min(pos["position_size"], 10.0))

    if current_day is not None:
        daily_pnl[current_day] = balance - daily_start_balance

    return {
        "equity_curve": pd.DataFrame(equity_curve),
        "trades": pd.DataFrame(all_trades),
        "daily_pnl": daily_pnl,
        "final_balance": balance,
        "initial_balance": initial_balance
    }

def plot_portfolio_analysis(results, bot_name, save_path):
    equity_df = results["equity_curve"]
    trades_df = results["trades"]
    daily_pnl = results["daily_pnl"]
    initial_balance = results["initial_balance"]
    final_balance = results["final_balance"]
    
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'{bot_name.upper()} BOT - Portfolio Performance (3 Pairs)', fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(equity_df["time"], equity_df["balance"], 'b-', linewidth=1.5, label='Balance')
    ax1.plot(equity_df["time"], equity_df["peak"], 'g--', linewidth=1, alpha=0.7, label='Peak Balance')
    ax1.axhline(y=initial_balance, color='gray', linestyle=':', label='Initial')
    ax1.set_title('Equity Curve', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Balance ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Drawdown
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.fill_between(equity_df["time"], 0, -equity_df["drawdown_pct"], color='red', alpha=0.5)
    ax2.axhline(y=-9, color='darkred', linestyle='--', label='Prop Limit (-9%)')
    ax2.axhline(y=-20, color='purple', linestyle='--', label='Original Limit (-20%)')
    ax2.set_title(f'Drawdown (Max: {equity_df["drawdown_pct"].max():.2f}%)', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('DD (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Daily P&L
    ax3 = fig.add_subplot(4, 2, 3)
    daily_returns = list(daily_pnl.values())
    ax3.hist(daily_returns, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.set_title('Daily P&L Distribution', fontweight='bold')
    ax3.axvline(x=0, color='black')
    
    # 4. Monthly P&L
    ax4 = fig.add_subplot(4, 2, 4)
    if not trades_df.empty:
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        monthly_pnl = trades_df.groupby('month')['pnl'].sum()
        months = [str(m) for m in monthly_pnl.index]
        values = monthly_pnl.values
        colors = ['green' if v >= 0 else 'red' for v in values]
        ax4.bar(range(len(months)), values, color=colors, edgecolor='black')
        ax4.set_title('Monthly P&L', fontweight='bold')
        ax4.set_xticks(range(0, len(months), max(1, len(months)//12)))
        ax4.set_xticklabels([months[i] for i in range(0, len(months), max(1, len(months)//12))], rotation=45)
        ax4.grid(True, alpha=0.3)

    # 5. Trades by Symbol
    ax5 = fig.add_subplot(4, 2, 5)
    if not trades_df.empty:
        symbol_counts = trades_df['symbol'].value_counts()
        ax5.pie(symbol_counts.values, labels=symbol_counts.index, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Trades by Symbol', fontweight='bold')

    # 6. Win Rate
    ax6 = fig.add_subplot(4, 2, 6)
    if not trades_df.empty:
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] <= 0])
        ax6.pie([wins, losses], labels=['Wins', 'Losses'], colors=['green', 'red'], autopct='%1.1f%%')
        ax6.set_title(f'Win Rate: {wins/(wins+losses)*100:.1f}%', fontweight='bold')

    # 7. Cumulative Return
    ax7 = fig.add_subplot(4, 2, 7)
    cum_ret = (equity_df["balance"] / initial_balance - 1) * 100
    ax7.plot(equity_df["time"], cum_ret, 'b-')
    ax7.fill_between(equity_df["time"], 0, cum_ret, color='green', alpha=0.1)
    ax7.set_title(f'Cumulative Return ({cum_ret.iloc[-1]:.0f}%)', fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Return (%)')
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)

    # 8. Metrics
    ax8 = fig.add_subplot(4, 2, 8)
    ax8.axis('off')
    
    total_return = (final_balance / initial_balance - 1) * 100
    max_dd = equity_df["drawdown_pct"].max()
    net_profit = final_balance - initial_balance
    
    metrics_text = f"""
╔════════════════════════════════════════════════╗
║         PORTFOLIO PERFORMANCE METRICS          ║
╠════════════════════════════════════════════════╣
║  Bot Type:            {bot_name.upper()} 
║  Pairs:               EURUSD, USDCAD, USDCHF            
║  Initial Balance:     ${initial_balance:,.2f}
║  Final Balance:       ${final_balance:,.2f}
║  Net Profit:          ${net_profit:,.2f}
║  Total Return:        {total_return:+.2f}%
║  Max Drawdown:        {max_dd:.2f}%
╠════════════════════════════════════════════════╣
║  Total Trades:        {len(trades_df):,}
║  Win Rate:            {len(trades_df[trades_df['pnl']>0])/len(trades_df)*100:.1f}%
╚════════════════════════════════════════════════╝
"""
    ax8.text(0.1, 0.5, metrics_text, transform=ax8.transAxes, fontsize=12,
             fontfamily='monospace', bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", type=str, required=True)
    parser.add_argument("--days", type=int, default=1500)
    args = parser.parse_args()
    
    strategy = get_strategy_module(args.bot)
    pairs = ["EURUSDm", "USDCADm", "USDCHFm"]
    bars_count = args.days * 48
    
    pairs_data = {}
    print(f"Fetching data for {pairs}...")
    for sym in pairs:
        df = fetch_data(sym, bars_count)
        if not df.empty:
            df = strategy.calculate_indicators(df)
            df = strategy.generate_signals(df)
            pairs_data[sym] = df
            print(f"Loaded {sym}: {len(df)} bars")

    if not pairs_data:
        print("No data!")
        return

    # Configuration based on optimization results
    if args.bot == "propfirm":
        risk = 0.0040  # 0.40%
        daily_loss = 4.1
        risk_tiers = [
            (1.5, 0.8), (3.0, 0.6), (4.5, 0.4),
            (6.0, 0.2), (7.5, 0.1), (8.5, 0.05)
        ]
        initial = 2500.0
    else:
        risk = 0.0070  # 0.70%
        daily_loss = 10.0
        risk_tiers = [
            (5.0, 0.8), (10.0, 0.6), (15.0, 0.4),
            (18.0, 0.2), (19.0, 0.1)
        ]
        initial = 10000.0

    print(f"Running Portfolio Backtest ({args.bot.upper()})... Risk: {risk*100:.2f}%")
    results = run_portfolio_backtest_detailed(
        pairs_data, 
        strategy.STRATEGY_PARAMS, 
        initial, risk, risk_tiers, daily_loss
    )
    
    save_path = f"{args.bot}_portfolio_analysis.png"
    plot_portfolio_analysis(results, args.bot, save_path)

if __name__ == "__main__":
    main()
