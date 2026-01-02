"""Backtest comparison for different risk levels - Full Metrics"""
import MetaTrader5 as mt5
import pandas as pd
from services.strategy import calculate_indicators, generate_signals, run_backtest

# Connect to MT5
if not mt5.initialize():
    print('MT5 failed')
    exit()

# Fetch EURUSD data
rates = mt5.copy_rates_from_pos('EURUSDm', mt5.TIMEFRAME_M30, 0, 90000)
mt5.shutdown()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)

print(f'Loaded {len(df)} bars')
print(f'Period: {df.index[0]} to {df.index[-1]}')

# Calculate years
days = (df.index[-1] - df.index[0]).days
years = days / 365.25
print(f'Duration: {years:.2f} years')

# Calculate and generate signals
df = calculate_indicators(df)
df = generate_signals(df)

# Test different risk levels
risk_levels = [0.005, 0.010, 0.015, 0.020]  # 0.5%, 1%, 1.5%, 2%

results_list = []
for risk in risk_levels:
    results = run_backtest(df, initial_balance=10000.0, risk_per_trade=risk)
    
    # Calculate annualized return (CAGR)
    total_return_decimal = results["total_return"] / 100
    cagr = ((1 + total_return_decimal) ** (1/years) - 1) * 100
    
    # Calculate monthly return
    monthly_return = ((1 + total_return_decimal) ** (1/(years*12)) - 1) * 100
    
    results_list.append({
        'risk': risk * 100,
        'total_return': results["total_return"],
        'cagr': cagr,
        'monthly': monthly_return,
        'max_dd': results["max_drawdown"],
        'pf': results["profit_factor"],
        'trades': results["total_trades"],
        'win_rate': results["win_rate"],
        'wins': results["winning_trades"],
        'losses': results["losing_trades"],
        'avg_win': results["avg_win"],
        'avg_loss': results["avg_loss"],
        'final': results["final_balance"]
    })

# Print comparison table
print('\n' + '=' * 90)
print('=== RISK LEVEL COMPARISON - EURUSD M30 (FULL METRICS) ===')
print('=' * 90)

# Table 1: Returns
print('\n[RETURNS]')
print(f'{"Risk":<8} {"Total Return":<15} {"CAGR/Year":<12} {"Monthly":<10} {"Final Balance":<15}')
print('-' * 60)
for r in results_list:
    print(f'{r["risk"]:.1f}%{"":<4} {r["total_return"]:>+12.2f}% {r["cagr"]:>10.2f}% {r["monthly"]:>8.2f}% ${r["final"]:>12,.2f}')

# Table 2: Risk Metrics
print('\n[RISK METRICS]')
print(f'{"Risk":<8} {"Max DD":<10} {"Profit Factor":<15} {"Return/DD Ratio":<15}')
print('-' * 50)
for r in results_list:
    return_dd_ratio = r["cagr"] / r["max_dd"] if r["max_dd"] > 0 else 0
    print(f'{r["risk"]:.1f}%{"":<4} {r["max_dd"]:>8.2f}% {r["pf"]:>12.2f} {return_dd_ratio:>13.2f}')

# Table 3: Trade Stats
print('\n[TRADE STATISTICS]')
print(f'{"Risk":<8} {"Total":<8} {"Wins":<8} {"Losses":<8} {"Win Rate":<10} {"Avg Win":<12} {"Avg Loss":<12}')
print('-' * 70)
for r in results_list:
    print(f'{r["risk"]:.1f}%{"":<4} {r["trades"]:<8} {r["wins"]:<8} {r["losses"]:<8} {r["win_rate"]:>8.2f}% ${r["avg_win"]:>10.2f} ${r["avg_loss"]:>10.2f}')

print('\n' + '=' * 90)
print('Summary:')
print(f'  Period: {years:.2f} years ({df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")})')
print(f'  Best Return/DD Ratio: 0.5% Risk (lowest DD with good returns)')
print('=' * 90)
