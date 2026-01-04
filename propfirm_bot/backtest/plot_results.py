"""
Plot Prop Firm Backtest Results
Visualize equity curve and drawdown
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
equity_df = pd.read_csv("propfirm_equity_curve.csv")
equity_df['time'] = pd.to_datetime(equity_df['time'])

# Calculate values
initial_balance = 10000
max_loss_threshold = 9000  # $1000 max loss

# Calculate peak and drawdown
equity_df['peak'] = equity_df['balance'].cummax()
equity_df['drawdown'] = (equity_df['peak'] - equity_df['balance']) / equity_df['peak'] * 100
equity_df['drawdown_usd'] = equity_df['peak'] - equity_df['balance']

# Find minimum balance
min_balance = equity_df['balance'].min()
max_dd_pct = equity_df['drawdown'].max()
max_dd_usd = equity_df['drawdown_usd'].max()

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Prop Firm Challenge Backtest - Risk 0.45%', fontsize=16, fontweight='bold')

# Plot 1: Equity Curve
ax1.plot(equity_df['time'], equity_df['balance'], linewidth=1.5, color='#2ecc71', label='Balance')
ax1.axhline(y=initial_balance, color='blue', linestyle='--', alpha=0.5, label='Initial: $10,000')
ax1.axhline(y=max_loss_threshold, color='red', linestyle='--', linewidth=2, label='Max Loss Limit: $9,000')
ax1.axhline(y=min_balance, color='orange', linestyle=':', alpha=0.7, label=f'Min Balance: ${min_balance:,.0f}')

# Highlight area below max loss
ax1.fill_between(equity_df['time'], 0, max_loss_threshold, alpha=0.1, color='red', label='Danger Zone')

ax1.set_ylabel('Balance ($)', fontsize=12, fontweight='bold')
ax1.set_title('Equity Curve', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add text box with key metrics
textstr = f'Final Balance: ${equity_df["balance"].iloc[-1]:,.0f}\n'
textstr += f'Total Return: {(equity_df["balance"].iloc[-1] - initial_balance) / initial_balance * 100:.1f}%\n'
textstr += f'Min Balance: ${min_balance:,.0f}\n'
textstr += f'Loss from Initial: ${initial_balance - min_balance:,.0f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Plot 2: Drawdown Percentage
ax2.fill_between(equity_df['time'], 0, equity_df['drawdown'], alpha=0.4, color='red')
ax2.plot(equity_df['time'], equity_df['drawdown'], linewidth=1, color='darkred')
ax2.axhline(y=15, color='orange', linestyle='--', linewidth=2, label='15% Target Limit')
ax2.axhline(y=max_dd_pct, color='red', linestyle=':', alpha=0.7, label=f'Max DD: {max_dd_pct:.1f}%')

ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_title('Drawdown Percentage', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')
ax2.invert_yaxis()  # Invert so drawdown goes down

# Plot 3: Drawdown in USD
ax3.fill_between(equity_df['time'], 0, equity_df['drawdown_usd'], alpha=0.4, color='purple')
ax3.plot(equity_df['time'], equity_df['drawdown_usd'], linewidth=1, color='darkviolet')
ax3.axhline(y=1000, color='red', linestyle='--', linewidth=2, label='$1,000 Max Loss')
ax3.axhline(y=max_dd_usd, color='orange', linestyle=':', alpha=0.7, label=f'Max DD: ${max_dd_usd:,.0f}')

ax3.set_ylabel('Drawdown ($)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
ax3.set_title('Drawdown in USD', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax3.invert_yaxis()

# Format x-axis for all plots
for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('propfirm_backtest_chart.png', dpi=150, bbox_inches='tight')
print("\n[*] Chart saved: propfirm_backtest_chart.png")

# Print summary
print("\n" + "=" * 70)
print("BACKTEST SUMMARY")
print("=" * 70)
print(f"Period: {equity_df['time'].iloc[0].strftime('%Y-%m-%d')} to {equity_df['time'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"Initial Balance: ${initial_balance:,.0f}")
print(f"Final Balance: ${equity_df['balance'].iloc[-1]:,.0f}")
print(f"Total Return: {(equity_df['balance'].iloc[-1] - initial_balance) / initial_balance * 100:.1f}%")
print("-" * 70)
print(f"Min Balance: ${min_balance:,.0f}")
print(f"Max Drawdown: ${max_dd_usd:,.0f} ({max_dd_pct:.1f}%)")
print(f"Loss from Initial: ${initial_balance - min_balance:,.0f}")
print("-" * 70)

# Check if passed
passed_max_loss = (initial_balance - min_balance) < 1000
print(f"\nMax Loss Check ($1,000 limit):")
if passed_max_loss:
    print(f"  [PASS] Loss from initial: ${initial_balance - min_balance:,.0f} < $1,000")
else:
    print(f"  [FAIL] Loss from initial: ${initial_balance - min_balance:,.0f} >= $1,000")

print("=" * 70)
print("\nView the chart: propfirm_backtest_chart.png")

plt.show()
