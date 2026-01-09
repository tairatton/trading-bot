"""
Generate Financial Report
=========================
Reads backtest results (CSV) and generates a comprehensive financial analysis report.

Charts:
1. Growth: Equity Curve & Monthly Returns
2. Risk: Drawdown & Underwater Plot
3. Efficiency: MAE/MFE Analysis
4. Stats: Win/Loss Distribution & Duration
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

# Set style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def load_data(results_dir):
    """Load trades and equity data."""
    trades_path = os.path.join(results_dir, "trades.csv")
    equity_path = os.path.join(results_dir, "equity.csv")
    
    if not os.path.exists(trades_path) or not os.path.exists(equity_path):
        print("Error: Results files not found. Run backtest first.")
        return None, None
    
    trades = pd.read_csv(trades_path)
    equity = pd.read_csv(equity_path)
    
    # Convert dates
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    equity['time'] = pd.to_datetime(equity['time'])
    
    return trades, equity

def plot_growth_analysis(equity, trades, output_dir):
    """1. Growth & Equity Analysis"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Equity Curve
    ax1.plot(equity['time'], equity['balance'], label='Balance', linewidth=1.5, alpha=0.7)
    ax1.plot(equity['time'], equity['equity'], label='Equity', linewidth=1.0, alpha=0.9)
    ax1.set_title('Balance vs Equity Curve')
    ax1.legend()
    ax1.set_ylabel('Account Value ($)')
    
    # Monthly Heatmap
    if not trades.empty:
        trades['month_year'] = trades['exit_time'].dt.to_period('M')
        monthly_pnl = trades.groupby('month_year')['pnl'].sum().reset_index()
        monthly_pnl['period'] = monthly_pnl['month_year'].astype(str)
        # Parse into year and month
        monthly_pnl['year'] = monthly_pnl['month_year'].dt.year
        monthly_pnl['month'] = monthly_pnl['month_year'].dt.month
        
        pivot_table = monthly_pnl.pivot(index='year', columns='month', values='pnl')
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="RdYlGn", center=0, ax=ax2, cbar_kws={'label': 'PnL ($)'})
        ax2.set_title('Monthly PnL Heatmap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_growth_analysis.png"))
    plt.close()

def plot_risk_analysis(equity, output_dir):
    """2. Risk & Drawdown Analysis"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Calculate Drawdown
    peak = equity['equity'].cummax()
    drawdown_abs = equity['equity'] - peak
    drawdown_pct = (drawdown_abs / peak) * 100
    
    ax1.fill_between(equity['time'], drawdown_pct, 0, color='red', alpha=0.3)
    ax1.plot(equity['time'], drawdown_pct, color='red', linewidth=1)
    ax1.set_title('Underwater Plot (Drawdown %)')
    ax1.set_ylabel('Drawdown (%)')
    ax1.set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_risk_analysis.png"))
    plt.close()

def plot_efficiency_analysis(trades, output_dir):
    """3. Entry/Exit Efficiency (MAE/MFE)"""
    if trades.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE vs MFE Scatter
    # MAE is negative PnL potential (adverse), MFE is positive PnL potential (favorable)
    # We plot magnitude
    mae_abs = trades['mae'].abs()
    mfe_abs = trades['mfe'].abs()
    
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] <= 0]
    
    ax1.scatter(winning_trades['mae'].abs(), winning_trades['mfe'].abs(), color='green', alpha=0.5, label='Wins')
    ax1.scatter(losing_trades['mae'].abs(), losing_trades['mfe'].abs(), color='red', alpha=0.5, label='Losses')
    
    # Add diagonal line (1:1 Ratio)
    max_val = max(mae_abs.max(), mfe_abs.max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    ax1.set_title('MAE vs MFE (Trade Efficiency)')
    ax1.set_xlabel('MAE (Max Adverse Excursion) $')
    ax1.set_ylabel('MFE (Max Favorable Excursion) $')
    ax1.legend()
    
    # Holding Time Histogram
    sns.histplot(data=trades, x='duration_bars', bins=30, kde=True, ax=ax2)
    ax2.set_title('Trade Duration Distribution (Bars)')
    ax2.set_xlabel('Duration (30m Bars)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_efficiency_analysis.png"))
    plt.close()

def plot_stats_analysis(trades, output_dir):
    """4. Advanced Statistics"""
    if trades.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Win/Loss Distribution
    wins = len(trades[trades['pnl'] > 0])
    losses = len(trades[trades['pnl'] <= 0])
    
    labels = [f'Wins ({wins})', f'Losses ({losses})']
    sizes = [wins, losses]
    colors = ['#66b3ff', '#ff9999']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax1.set_title('Win/Loss Ratio')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_stats_analysis.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="backtest_results", help="Directory containing trades.csv and equity.csv")
    args = parser.parse_args()
    
    print(f"Generating charts from {args.dir}...")
    trades, equity = load_data(args.dir)
    
    if trades is not None:
        plot_growth_analysis(equity, trades, args.dir)
        plot_risk_analysis(equity, args.dir)
        plot_efficiency_analysis(trades, args.dir)
        plot_stats_analysis(trades, args.dir)
        print("Charts generated successfully!")
        print(f"- {os.path.join(args.dir, '1_growth_analysis.png')}")
        print(f"- {os.path.join(args.dir, '2_risk_analysis.png')}")
        print(f"- {os.path.join(args.dir, '3_efficiency_analysis.png')}")
        print(f"- {os.path.join(args.dir, '4_stats_analysis.png')}")

if __name__ == "__main__":
    main()
