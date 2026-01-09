import argparse
import sys
import os
import pandas as pd
import MetaTrader5 as mt5

# Allow imports from bot dirs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine import run_backtest

def get_bot_module(bot_name):
    if bot_name == "propfirm":
        sys.path.insert(0, os.path.join(os.getcwd(), "propfirm_bot"))
        import propfirm_bot.services.strategy as strategy
        return strategy
    else:
        sys.path.insert(0, os.path.join(os.getcwd(), "original_bot"))
        import original_bot.services.strategy as strategy
        return strategy

def fetch_data(symbol: str, count=70000):
    if not mt5.initialize():
        print("MT5 Init Failed")
        return pd.DataFrame()
        
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, count)
    mt5.shutdown()
    
    if rates is None: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser(description='Unified Trading Bot Backtester')
    parser.add_argument('--bot', choices=['original', 'propfirm'], default='propfirm', help='Which bot strategy to use')
    parser.add_argument('--symbol', default='EURUSDm', help='Symbol to backtest')
    parser.add_argument('--days', type=int, default=1000, help='Days of data to fetch (approx)')
    parser.add_argument('--dynamic-risk', action='store_true', help='Enable dynamic risk (anti-martingale)')
    parser.add_argument('--risk', type=float, help='Override risk per trade (e.g. 0.005 for 0.5%)')
    args = parser.parse_args()
    
    print(f"========== BACKTEST: {args.bot.upper()} BOT ==========")
    print(f"Symbol: {args.symbol}")
    
    # Load Strategy Module
    strategy = get_bot_module(args.bot)
    print(f"Loaded strategy settings from: {args.bot}_bot")
    
    # Fetch Data
    # 30min bars * 48 bars/day * days
    count = args.days * 48
    print(f"Fetching {count} bars...")
    df = fetch_data(args.symbol, count=count)
    
    if df.empty:
        print("No data found!")
        return
        
    print(f"Data loaded: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    
    # Indicators & Signals
    print("Calculating indicators...")
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)
    
    # Run Simulation
    print("Running simulation...")
    
    # Determine risk settings based on bot type
    if args.bot == "propfirm":
        initial_balance = 2500.0
        risk = 0.0028 # 0.28%
    else:
        initial_balance = 10000.0
        risk = 0.01   # 1.0%
        
    if args.risk:
        risk = args.risk

    # Define Dynamic Risk Tiers
    # Prop Firm: Conservative (protect 5% daily / 10% max)
    prop_tiers = [
        (1.5, 0.8),    # > 1.5% DD -> Risk 80%
        (3.0, 0.6),    # > 3.0% DD -> Risk 60% 
        (4.0, 0.4),    # > 4.0% DD -> Risk 40%
        (5.0, 0.2),    # > 5.0% DD -> Risk 20%
        (6.0, 0.1)     # > 6.0% DD -> Risk 10%
    ]
    
    # Original: Aggressive (High growth, tolerate 20% DD)
    original_tiers = [
        (5.0, 0.8),    # > 5% DD -> Risk 80%
        (10.0, 0.6),   # > 10% DD -> Risk 60%
        (15.0, 0.4),   # > 15% DD -> Risk 40%
        (18.0, 0.2),   # > 18% DD -> Risk 20%
        (20.0, 0.1)    # > 20% DD -> Risk 10%
    ]
    
    risk_tiers = prop_tiers if args.bot == "propfirm" else original_tiers

    results = run_backtest(
        df, 
        strategy.STRATEGY_PARAMS,
        initial_balance=initial_balance,
        risk_per_trade=risk,
        use_dynamic_risk=args.dynamic_risk,
        risk_tiers=risk_tiers
    )
    
    # Report
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Final Balance:   ${results['final_balance']:,.2f}")
    print(f"Total Return:    {results['total_return']:.2f}%")
    print(f"Max Drawdown:    {results['max_drawdown']:.2f}%")
    print(f"Max Daily DD:    {results.get('max_daily_drawdown', 0):.2f}%")
    print(f"Phase 1 (8%):    {results.get('days_to_phase1', 'N/A')} days")
    print(f"Phase 2 (5%):    {results.get('days_to_phase2', 'N/A')} days")
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Win Rate:        {results['win_rate']:.1f}%")
    print(f"Profit Factor:   {results['profit_factor']:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
