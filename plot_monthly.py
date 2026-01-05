
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
try:
    df = pd.read_csv("monthly_returns.csv")
except FileNotFoundError:
    print("Error: monthly_returns.csv not found. Please run compare_bots.py first.")
    exit(1)

# Convert Month to datetime
df['Month_Dt'] = pd.to_datetime(df['Month'])

# Create figure
fig, ax = plt.subplots(figsize=(15, 8))
fig.suptitle('Monthly Returns Comparison: Original vs Prop Firm', fontsize=16, fontweight='bold')

# Plot bars
width = 20  # Width of bars in days relative to plot scale (approx)
# Actually better to just use index for bar plot and set labels
indices = range(len(df))
width = 0.35

# Conditional colors
orig_colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df['Original_Ret']]
prop_colors = ['#3498db' if x >= 0 else '#e74c3c' for x in df['Prop_Ret']]
# For prop firm, maybe use a different shade of red/blue? Or keep consistent red for loss.
# Let's use: Orig (Green/Red), Prop (Blue/Orange) to distinguish? 
# Or just standard Green/Red for both but maybe different alpha or hatch?
# Let's stick to User request: "Negative use Red tone".
# To distinguish bots, we can use edge color or different shades.
# Orig: Green (#2ecc71) / Red (#e74c3c)
# Prop: Blue (#3498db) / Dark Red (#c0392b)

# Conditional colors
# Original: Green (Pos), Red (Neg)
# Prop: Blue (Pos), Orange (Neg - to distinguish from Original's Red? Or use same Red but different hatch?)
# Let's use:
# Original: Green (#2ecc71) / Red (#e74c3c)
# Prop: Blue (#3498db) / Dark Red (#c0392b)

orig_colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df['Original_Ret']]
prop_colors = ['#3498db' if x >= 0 else '#c0392b' for x in df['Prop_Ret']]

rects1 = ax.bar([i - width/2 for i in indices], df['Original_Ret'], width=width, label='Original Bot', color=orig_colors, alpha=0.85)
rects2 = ax.bar([i + width/2 for i in indices], df['Prop_Ret'], width=width, label='Prop Firm Bot', color=prop_colors, alpha=0.85)

# Formatting
ax.set_ylabel('Monthly Return (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Custom Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Original (Profit)'),
    Patch(facecolor='#e74c3c', label='Original (Loss)'),
    Patch(facecolor='#3498db', label='Prop Firm (Profit)'),
    Patch(facecolor='#c0392b', label='Prop Firm (Loss)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

# Set x-ticks
# Show every 6th month label (semiannual)
tick_indices = [i for i in indices if i % 6 == 0]
tick_labels = [df['Month'].iloc[i] for i in tick_indices]
ax.set_xticks(tick_indices)
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)

# Add values for significant bars
def autolabel(rects, is_prop=False):
    for rect in rects:
        height = rect.get_height()
        if abs(height) > 5:  # Only label items > 5% or < -5%
            val_text = f'{height:.0f}%'
            xy_pos = (rect.get_x() + rect.get_width() / 2, height)
            
            # Offset text
            va = 'bottom' if height > 0 else 'top'
            offset = 3 if height > 0 else -3
            
            # Color
            txt_color = 'darkgreen' if height > 0 else 'darkred'
            if is_prop and height > 0: txt_color = 'darkblue'
            
            ax.annotate(val_text,
                        xy=xy_pos,
                        xytext=(0, offset),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va=va, fontsize=9, fontweight='bold', color=txt_color)

autolabel(rects1)
autolabel(rects2, is_prop=True)


plt.tight_layout()
plt.savefig('monthly_returns_chart.png', dpi=150, bbox_inches='tight')
print("Chart saved: monthly_returns_chart.png")

# Show summary stats
print("\nStats Summary:")
print(f"Original Avg: {df['Original_Ret'].mean():.2f}% | Max: {df['Original_Ret'].max():.2f}% | Min: {df['Original_Ret'].min():.2f}%")
print(f"Prop Firm Avg: {df['Prop_Ret'].mean():.2f}% | Max: {df['Prop_Ret'].max():.2f}% | Min: {df['Prop_Ret'].min():.2f}%")

plt.show()
