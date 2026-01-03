"""
Backtest Script - Trading Bot Strategy
=======================================
Uses shared strategy module for consistency with live trading bot.

Usage:
    py -3.11 backtest.py
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import shared strategy module
from services.strategy import (
    STRATEGY_PARAMS,
    calculate_indicators,
    generate_signals,
    run_backtest
)


def fetch_data(symbol: str = "EURUSDm", timeframe=mt5.TIMEFRAME_M30) -> pd.DataFrame:
    """Fetch historical data from MT5."""
    print(f"[*] Fetching M30 data from MT5...")
    
    # Connect to MT5
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return pd.DataFrame()
    
    
    
    # Fetch data: Verified that ~42k bars are available, so 90k is a safe upper limit request
    count = 90000
    print(f"[*] Requesting {count} bars (approx 5 years M30)...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    # Check for error details before shutdown
    if rates is None:
        error_code = mt5.last_error()
        print(f"[!] Failed to fetch data. MT5 Error Code: {error_code}")
        
    mt5.shutdown()
    
    if rates is None:
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'tick_volume': 'Volume'
    }, inplace=True)
    
    return df


def plot_performance_charts(results: dict):
    """Generate performance visualization charts."""
    if not results.get('equity_curve') or not results.get('trades'):
        print("[!] Not enough data to generate charts")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Backtest Performance Analysis - EURUSD M30', fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    equity_df = pd.DataFrame(results['equity_curve'])
    equity_df['time'] = pd.to_datetime(equity_df['time'])
    
    axes[0].plot(equity_df['time'], equity_df['balance'], linewidth=2, color='#2E86AB', label='Equity')
    axes[0].axhline(y=results['initial_balance'], color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
    axes[0].set_title('Equity Curve', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Balance ($)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 2. Drawdown Chart
    equity_df['peak'] = equity_df['balance'].cummax()
    equity_df['drawdown'] = ((equity_df['balance'] - equity_df['peak']) / equity_df['peak']) * 100
    
    axes[1].fill_between(equity_df['time'], equity_df['drawdown'], 0, color='#A23B72', alpha=0.6)
    axes[1].set_title('Drawdown %', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 3. Trade P&L Distribution
    trades_df = pd.DataFrame(results['trades'])
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] < 0]['pnl']
    
    axes[2].hist([wins, losses], bins=30, label=['Wins', 'Losses'], 
                 color=['#06A77D', '#D62246'], alpha=0.7, edgecolor='black')
    axes[2].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[2].set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Profit/Loss ($)', fontsize=10)
    axes[2].set_ylabel('Frequency', fontsize=10)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_performance.png', dpi=150, bbox_inches='tight')
    print("\n[*] Performance charts saved to: backtest_performance.png")
    plt.show()


def print_results(results: dict):
    """Print backtest results."""
    print("\n" + "=" * 60)
    print("     BACKTEST RESULTS - EURUSD (Shared Strategy)")
    print("=" * 60)
    print(f"  Total Trades:      {results['total_trades']}")
    print(f"    - Trend Trades:  {results.get('trend_trades', 0)}")
    print(f"    - MR Trades:     {results.get('mr_trades', 0)}")
    print(f"  Winning Trades:    {results.get('winning_trades', 0)}")
    print(f"  Losing Trades:     {results.get('losing_trades', 0)}")
    print(f"  Win Rate:          {results['win_rate']:.2f}%")
    print("-" * 60)
    print(f"  Total Return:      {results['total_return']:+.2f}%")
    print(f"  Profit Factor:     {results['profit_factor']:.2f}")
    print(f"  Max Drawdown:      {results['max_drawdown']:.2f}%")
    print("-" * 60)
    print(f"  Avg Win:           ${results.get('avg_win', 0):.2f}")
    print(f"  Avg Loss:          ${results.get('avg_loss', 0):.2f}")
    print(f"  Final Balance:     ${results['final_balance']:.2f}")
    print("=" * 60)
    
    # Assessment
    pf = results['profit_factor']
    if pf > 1.5:
        status = "[EXCELLENT] Strategy is profitable!"
    elif pf > 1.2:
        status = "[GOOD] Strategy shows potential"
    elif pf > 1.0:
        status = "[OK] Strategy is marginally profitable"
    else:
        status = "[POOR] Strategy needs optimization"
    print(f"\n  {status}")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("     EURUSD BACKTEST - Using Shared Strategy Module")
    print("=" * 60)
    print(f"  Strategy Parameters: {STRATEGY_PARAMS}")
    
    # Fetch data
    print("\n[*] Fetching data from MT5...")
    df = fetch_data(symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M30)
    
    if df.empty:
        print("[X] No data available. Make sure MT5 is running.")
        return
    
    print(f"    Loaded {len(df)} bars")
    print(f"    From: {df.index[0]}")
    print(f"    To:   {df.index[-1]}")
    
    # Calculate indicators (using shared module)
    print("[*] Calculating indicators...")
    df = calculate_indicators(df)
    
    # Generate signals (using shared module)
    print("[*] Generating signals...")
    df = generate_signals(df)
    
    buy_signals = df['buy_signal'].sum()
    sell_signals = df['sell_signal'].sum()
    print(f"    Buy signals:  {buy_signals}")
    print(f"    Sell signals: {sell_signals}")
    
    # Run backtest (using shared module)
    print("[*] Running backtest...")
    results = run_backtest(
        df,
        initial_balance=10000.0,
        risk_per_trade=0.02
    )
    
    # Print results
    print_results(results)
    
    # Save trades to CSV
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv('backtest_trades.csv', index=False)
        print("\n[*] Trades saved to: backtest_trades.csv")
    
    # Generate performance charts
    print("\n[*] Generating performance charts...")
    plot_performance_charts(results)


if __name__ == "__main__":
    main()
