"""
================================================================================
COMPREHENSIVE BOT PERFORMANCE ANALYSIS
================================================================================
This script performs a complete evaluation of both trading bots:
- Original Bot (0.7% risk - Personal Trading)
- Prop Firm Bot (0.2% risk - Prop Firm Challenge)

Metrics calculated:
- Total Return, Monthly Returns, Profit Factor
- Win Rate, Max Drawdown, Max Daily Loss
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Bell Curve distribution of returns
- Monthly P/L breakdown
================================================================================
"""

import sys
import os
import pickle
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / "original_bot"))
sys.path.insert(0, str(Path(__file__).parent / "propfirm_bot"))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from datetime import datetime, timedelta

# Import strategy from original_bot (both have same strategy)
from services.strategy import calculate_indicators, generate_signals, STRATEGY_PARAMS

# ============================================================
# CONFIGURATION
# ============================================================
SYMBOLS = ["EURUSDm", "USDCADm", "USDCHFm"]
INITIAL_BALANCE = 10000.0
CACHE_FILE = Path(__file__).parent / "backtest_data_cache.pkl"

# Bot configurations (OPTIMIZED 2026-01-07)
BOT_CONFIGS = {
    "Original Bot": {
        "risk_per_trade": 0.015,  # 1.5% per trade (was 1.0%)
        "color": "#2ecc71",       # Green
        "label": "Original (1.5%)"
    },
    "Prop Firm Bot": {
        "risk_per_trade": 0.004,  # 0.4% per trade (was 0.3%)
        "color": "#3498db",       # Blue
        "label": "PropFirm (0.4%)"
    }
}

# Trading Costs (Exness Cent Account)
SPREAD_PIPS = {
    "EURUSDm": 1.2,
    "USDCADm": 1.5,
    "USDCHFm": 1.5,
}
SLIPPAGE_PIPS = 0.5

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_spread_cost(symbol: str) -> float:
    """Get spread + slippage in price units."""
    spread = SPREAD_PIPS.get(symbol, 1.5)
    total_pips = spread + SLIPPAGE_PIPS
    if "JPY" in symbol.upper():
        return total_pips * 0.01
    return total_pips * 0.0001

def get_pip_value(symbol: str) -> float:
    """Get pip value multiplier."""
    return 1000 if "JPY" in symbol.upper() else 100000

def fetch_data(symbol: str, timeframe=None, count=70000):
    """Fetch OHLC data from MT5."""
    if timeframe is None and MT5_AVAILABLE:
        timeframe = mt5.TIMEFRAME_M30
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df

def load_or_fetch_data():
    """Load cached data or fetch from MT5."""
    if CACHE_FILE.exists():
        print("[*] Loading cached data...")
        with open(CACHE_FILE, "rb") as f:
            all_data = pickle.load(f)
        
        # Check if data has all symbols with indicators
        if all(symbol in all_data and "ATR" in all_data[symbol].columns for symbol in SYMBOLS):
            print(f"    Loaded {len(all_data)} symbols from cache")
            return all_data
    
    # Check if MT5 is available
    if not MT5_AVAILABLE:
        print("[!] MT5 not available and no cache found")
        return None
    
    # Fetch from MT5
    print("[*] Fetching data from MT5...")
    if not mt5.initialize():
        print("[!] MT5 initialization failed")
        return None
    
    all_data = {}
    for symbol in SYMBOLS:
        print(f"    Loading {symbol}...")
        df = fetch_data(symbol)
        if not df.empty:
            df = calculate_indicators(df)
            df = generate_signals(df)
            all_data[symbol] = df
            print(f"    {len(df)} bars loaded")
    
    mt5.shutdown()
    
    # Save to cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(all_data, f)
    print(f"[*] Data cached to {CACHE_FILE}")
    
    return all_data

# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(all_data: dict, risk_per_trade: float) -> dict:
    """Run backtest with given risk per trade."""
    
    # Align all data to common index
    common_index = all_data[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        common_index = common_index.intersection(all_data[symbol].index)
    
    aligned_data = {symbol: all_data[symbol].loc[common_index] for symbol in SYMBOLS}
    
    # Backtest state
    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_drawdown = 0
    
    positions = {}  # {symbol: {type, entry, sl, tp, size, signal_type, entry_time, bars, highest, lowest}}
    trades = []
    equity_curve = []
    daily_start_balance = {}
    daily_pnl = {}
    
    start_idx = 250
    
    for i in range(start_idx, len(common_index)):
        current_time = common_index[i]
        current_date = current_time.date()
        
        # Track daily start balance
        if current_date not in daily_start_balance:
            daily_start_balance[current_date] = balance
            daily_pnl[current_date] = 0
        
        for symbol in SYMBOLS:
            row = aligned_data[symbol].iloc[i]
            price = row["Close"]
            high = row["High"]
            low = row["Low"]
            atr = row["ATR"]
            
            # Check exits - using High/Low for realistic SL/TP simulation
            if symbol in positions:
                pos = positions[symbol]
                pos["bars"] += 1
                params = STRATEGY_PARAMS[pos["signal_type"]]
                
                exit_price = None
                exit_reason = ""
                
                if pos["type"] == "buy":
                    # Use High for trailing (best price during bar)
                    pos["highest"] = max(pos["highest"], high)
                    if pos["highest"] - pos["entry"] > params["TRAIL_START"] * atr:
                        new_sl = pos["highest"] - params["TRAIL_DIST"] * atr
                        pos["sl"] = max(pos["sl"], new_sl)
                    
                    # Check exits using Low (worst case for buy)
                    if low <= pos["sl"]:
                        exit_price, exit_reason = pos["sl"], "SL"
                    elif high >= pos["tp"]:
                        exit_price, exit_reason = pos["tp"], "TP"
                    elif pos["bars"] >= params["MAX_BARS"]:
                        exit_price, exit_reason = price, "TIME"
                
                elif pos["type"] == "sell":
                    # Use Low for trailing (best price during bar)
                    pos["lowest"] = min(pos["lowest"], low)
                    if pos["entry"] - pos["lowest"] > params["TRAIL_START"] * atr:
                        new_sl = pos["lowest"] + params["TRAIL_DIST"] * atr
                        pos["sl"] = min(pos["sl"], new_sl)
                    
                    # Check exits using High (worst case for sell)
                    if high >= pos["sl"]:
                        exit_price, exit_reason = pos["sl"], "SL"
                    elif low <= pos["tp"]:
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
                        "symbol": symbol,
                        "type": pos["type"],
                        "signal_type": pos["signal_type"],
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "entry_time": pos["entry_time"],
                        "exit_time": current_time,
                        "bars": pos["bars"]
                    })
                    del positions[symbol]
            
            # Check entry
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
                    params = STRATEGY_PARAMS[signal_type]
                    sl_dist = atr * params["SL_ATR"]
                    pip_mult = get_pip_value(symbol)
                    
                    if sl_dist > 0:
                        risk_amount = balance * risk_per_trade
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
        equity_curve.append({"time": current_time, "balance": balance})
        
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100
        max_drawdown = max(max_drawdown, dd)
    
    return {
        "trades": pd.DataFrame(trades),
        "equity_curve": pd.DataFrame(equity_curve),
        "daily_pnl": daily_pnl,
        "daily_start_balance": daily_start_balance,
        "final_balance": balance,
        "max_drawdown": max_drawdown
    }

# ============================================================
# RISK METRICS CALCULATION
# ============================================================
def calculate_metrics(result: dict, bot_name: str) -> dict:
    """Calculate comprehensive risk metrics."""
    trades_df = result["trades"]
    equity_df = result["equity_curve"]
    daily_pnl = result["daily_pnl"]
    daily_start = result["daily_start_balance"]
    
    if trades_df.empty:
        return {"error": "No trades"}
    
    # Basic metrics
    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins) / total_trades * 100
    
    total_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
    profit_factor = total_profit / total_loss
    
    # Monthly returns
    equity_df["month"] = equity_df["time"].dt.to_period("M")
    monthly_balance = equity_df.groupby("month")["balance"].last()
    monthly_returns = monthly_balance.pct_change().dropna() * 100
    
    # Daily returns
    daily_returns = []
    for date, pnl in daily_pnl.items():
        start_bal = daily_start.get(date, INITIAL_BALANCE)
        daily_returns.append(pnl / start_bal * 100)
    daily_returns = pd.Series(daily_returns)
    
    # Risk-adjusted metrics
    risk_free_rate = 0.02 / 252  # Annual 2% / trading days
    
    # Sharpe Ratio (annualized)
    avg_daily_return = daily_returns.mean()
    daily_std = daily_returns.std()
    sharpe_ratio = (avg_daily_return - risk_free_rate * 100) / daily_std * np.sqrt(252) if daily_std > 0 else 0
    
    # Sortino Ratio (only downside volatility)
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std() if len(negative_returns) > 0 else 1
    sortino_ratio = (avg_daily_return - risk_free_rate * 100) / downside_std * np.sqrt(252) if downside_std > 0 else 0
    
    # Calmar Ratio (return / max drawdown)
    total_return = (result["final_balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    calmar_ratio = total_return / result["max_drawdown"] if result["max_drawdown"] > 0 else 0
    
    # Max daily loss
    max_daily_loss = min(daily_returns) if len(daily_returns) > 0 else 0
    
    # Consecutive losses
    is_loss = trades_df["pnl"] < 0
    max_consecutive_losses = 0
    current_streak = 0
    for loss in is_loss:
        if loss:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0
    
    # Trade duration
    avg_bars = trades_df["bars"].mean()
    
    # By exit reason
    sl_trades = len(trades_df[trades_df["reason"] == "SL"])
    tp_trades = len(trades_df[trades_df["reason"] == "TP"])
    time_trades = len(trades_df[trades_df["reason"] == "TIME"])
    
    # By signal type
    trend_trades = trades_df[trades_df["signal_type"] == "TREND"]
    mr_trades = trades_df[trades_df["signal_type"] == "MR"]
    
    return {
        "bot_name": bot_name,
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_return": total_return,
        "final_balance": result["final_balance"],
        "max_drawdown": result["max_drawdown"],
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_daily_loss": max_daily_loss,
        "max_consecutive_losses": max_consecutive_losses,
        "avg_trade_duration_bars": avg_bars,
        "sl_exits": sl_trades,
        "tp_exits": tp_trades,
        "time_exits": time_trades,
        "trend_trades": len(trend_trades),
        "mr_trades": len(mr_trades),
        "trend_pnl": trend_trades["pnl"].sum() if len(trend_trades) > 0 else 0,
        "mr_pnl": mr_trades["pnl"].sum() if len(mr_trades) > 0 else 0,
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
        "monthly_returns": monthly_returns,
        "daily_returns": daily_returns
    }

# ============================================================
# VISUALIZATION
# ============================================================
def create_visualizations(results: dict, all_metrics: dict):
    """Create comprehensive visualizations."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Trading Bot Comprehensive Analysis", fontsize=20, fontweight="bold", y=0.98)
    
    # ========== 1. Equity Curves Comparison ==========
    ax1 = fig.add_subplot(4, 2, 1)
    for bot_name, result in results.items():
        config = BOT_CONFIGS[bot_name]
        equity_df = result["equity_curve"]
        ax1.plot(equity_df["time"], equity_df["balance"], 
                 color=config["color"], linewidth=1.5, label=config["label"])
    
    ax1.axhline(y=INITIAL_BALANCE, color="gray", linestyle="--", alpha=0.5, label="Initial $10,000")
    ax1.set_title("Equity Curve Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Balance ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    # ========== 2. Drawdown Comparison ==========
    ax2 = fig.add_subplot(4, 2, 2)
    for bot_name, result in results.items():
        config = BOT_CONFIGS[bot_name]
        equity_df = result["equity_curve"].copy()
        equity_df["peak"] = equity_df["balance"].cummax()
        equity_df["drawdown"] = (equity_df["peak"] - equity_df["balance"]) / equity_df["peak"] * 100
        ax2.fill_between(equity_df["time"], 0, equity_df["drawdown"], 
                         alpha=0.3, color=config["color"])
        ax2.plot(equity_df["time"], equity_df["drawdown"], 
                 color=config["color"], linewidth=1, label=config["label"])
    
    ax2.set_title("Drawdown Comparison", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # ========== 3. Monthly Returns - Original Bot ==========
    ax3 = fig.add_subplot(4, 2, 3)
    metrics_orig = all_metrics["Original Bot"]
    monthly_orig = metrics_orig["monthly_returns"]
    colors_orig = ["#2ecc71" if x > 0 else "#e74c3c" for x in monthly_orig.values]
    x_pos = range(len(monthly_orig))
    ax3.bar(x_pos, monthly_orig.values, color=colors_orig, alpha=0.8)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.axhline(y=monthly_orig.mean(), color="blue", linestyle="--", 
                label=f"Avg: {monthly_orig.mean():.2f}%")
    ax3.set_title("Original Bot - Monthly Returns", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Return (%)")
    ax3.set_xlabel("Month")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    
    # Set x-ticks every 6 months
    tick_positions = list(range(0, len(monthly_orig), 6))
    tick_labels = [str(monthly_orig.index[i]) for i in tick_positions if i < len(monthly_orig)]
    ax3.set_xticks(tick_positions[:len(tick_labels)])
    ax3.set_xticklabels(tick_labels, rotation=45, ha="right")
    
    # ========== 4. Monthly Returns - PropFirm Bot ==========
    ax4 = fig.add_subplot(4, 2, 4)
    metrics_prop = all_metrics["Prop Firm Bot"]
    monthly_prop = metrics_prop["monthly_returns"]
    colors_prop = ["#3498db" if x > 0 else "#e74c3c" for x in monthly_prop.values]
    x_pos = range(len(monthly_prop))
    ax4.bar(x_pos, monthly_prop.values, color=colors_prop, alpha=0.8)
    ax4.axhline(y=0, color="black", linewidth=0.5)
    ax4.axhline(y=monthly_prop.mean(), color="blue", linestyle="--",
                label=f"Avg: {monthly_prop.mean():.2f}%")
    ax4.set_title("Prop Firm Bot - Monthly Returns", fontsize=14, fontweight="bold")
    ax4.set_ylabel("Return (%)")
    ax4.set_xlabel("Month")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    
    tick_positions = list(range(0, len(monthly_prop), 6))
    tick_labels = [str(monthly_prop.index[i]) for i in tick_positions if i < len(monthly_prop)]
    ax4.set_xticks(tick_positions[:len(tick_labels)])
    ax4.set_xticklabels(tick_labels, rotation=45, ha="right")
    
    # ========== 5. Bell Curve - Daily Returns Distribution (Original) ==========
    ax5 = fig.add_subplot(4, 2, 5)
    daily_orig = metrics_orig["daily_returns"]
    
    # Histogram
    n, bins, patches = ax5.hist(daily_orig, bins=50, density=True, alpha=0.7, 
                                 color="#2ecc71", edgecolor="black")
    
    # Fit normal distribution
    mu, std = daily_orig.mean(), daily_orig.std()
    x = np.linspace(daily_orig.min(), daily_orig.max(), 100)
    pdf = stats.norm.pdf(x, mu, std)
    ax5.plot(x, pdf, 'r-', linewidth=2, label=f"Normal Fit\nμ={mu:.3f}%, σ={std:.3f}%")
    
    # Add skewness and kurtosis
    skew = stats.skew(daily_orig)
    kurt = stats.kurtosis(daily_orig)
    ax5.axvline(x=mu, color="red", linestyle="--", alpha=0.7)
    textstr = f"Skewness: {skew:.2f}\nKurtosis: {kurt:.2f}"
    ax5.text(0.02, 0.98, textstr, transform=ax5.transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    ax5.set_title("Original Bot - Daily Returns Distribution (Bell Curve)", fontsize=14, fontweight="bold")
    ax5.set_xlabel("Daily Return (%)")
    ax5.set_ylabel("Density")
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)
    
    # ========== 6. Bell Curve - Daily Returns Distribution (PropFirm) ==========
    ax6 = fig.add_subplot(4, 2, 6)
    daily_prop = metrics_prop["daily_returns"]
    
    n, bins, patches = ax6.hist(daily_prop, bins=50, density=True, alpha=0.7,
                                 color="#3498db", edgecolor="black")
    
    mu, std = daily_prop.mean(), daily_prop.std()
    x = np.linspace(daily_prop.min(), daily_prop.max(), 100)
    pdf = stats.norm.pdf(x, mu, std)
    ax6.plot(x, pdf, 'r-', linewidth=2, label=f"Normal Fit\nμ={mu:.3f}%, σ={std:.3f}%")
    
    skew = stats.skew(daily_prop)
    kurt = stats.kurtosis(daily_prop)
    ax6.axvline(x=mu, color="red", linestyle="--", alpha=0.7)
    textstr = f"Skewness: {skew:.2f}\nKurtosis: {kurt:.2f}"
    ax6.text(0.02, 0.98, textstr, transform=ax6.transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    ax6.set_title("Prop Firm Bot - Daily Returns Distribution (Bell Curve)", fontsize=14, fontweight="bold")
    ax6.set_xlabel("Daily Return (%)")
    ax6.set_ylabel("Density")
    ax6.legend(loc="upper right")
    ax6.grid(True, alpha=0.3)
    
    # ========== 7. Metrics Comparison Bar Chart ==========
    ax7 = fig.add_subplot(4, 2, 7)
    
    comparison_metrics = ["win_rate", "profit_factor", "sharpe_ratio", "sortino_ratio", "calmar_ratio"]
    metric_labels = ["Win Rate (%)", "Profit Factor", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]
    
    x = np.arange(len(comparison_metrics))
    width = 0.35
    
    orig_values = [metrics_orig[m] for m in comparison_metrics]
    prop_values = [metrics_prop[m] for m in comparison_metrics]
    
    bars1 = ax7.bar(x - width/2, orig_values, width, label="Original Bot", color="#2ecc71", alpha=0.8)
    bars2 = ax7.bar(x + width/2, prop_values, width, label="PropFirm Bot", color="#3498db", alpha=0.8)
    
    ax7.set_title("Key Metrics Comparison", fontsize=14, fontweight="bold")
    ax7.set_xticks(x)
    ax7.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax7.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    # ========== 8. Trade Distribution by Exit Type ==========
    ax8 = fig.add_subplot(4, 2, 8)
    
    exit_types = ["SL", "TP", "TIME"]
    orig_exits = [metrics_orig["sl_exits"], metrics_orig["tp_exits"], metrics_orig["time_exits"]]
    prop_exits = [metrics_prop["sl_exits"], metrics_prop["tp_exits"], metrics_prop["time_exits"]]
    
    x = np.arange(len(exit_types))
    width = 0.35
    
    ax8.bar(x - width/2, orig_exits, width, label="Original Bot", color="#2ecc71", alpha=0.8)
    ax8.bar(x + width/2, prop_exits, width, label="PropFirm Bot", color="#3498db", alpha=0.8)
    
    ax8.set_title("Trade Exits by Type", fontsize=14, fontweight="bold")
    ax8.set_xticks(x)
    ax8.set_xticklabels(["Stop Loss", "Take Profit", "Time Exit"])
    ax8.set_ylabel("Number of Trades")
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save chart
    chart_path = Path(__file__).parent / "comprehensive_analysis_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n[*] Chart saved: {chart_path}")
    
    return chart_path

# ============================================================
# MAIN REPORT
# ============================================================
def print_report(all_metrics: dict):
    """Print comprehensive text report."""
    
    print("\n" + "=" * 100)
    print("                     COMPREHENSIVE TRADING BOT ANALYSIS REPORT")
    print("=" * 100)
    
    for bot_name, metrics in all_metrics.items():
        config = BOT_CONFIGS[bot_name]
        print(f"\n{'-' * 50}")
        print(f"  {bot_name.upper()} (Risk: {config['risk_per_trade']*100}% per trade)")
        print(f"{'-' * 50}")
        
        print(f"\n  [PERFORMANCE SUMMARY]")
        print(f"  {'-' * 40}")
        print(f"  Initial Balance:      ${INITIAL_BALANCE:,.0f}")
        print(f"  Final Balance:        ${metrics['final_balance']:,.0f}")
        print(f"  Total Return:         {metrics['total_return']:.1f}%")
        print(f"  Max Drawdown:         {metrics['max_drawdown']:.1f}%")
        
        print(f"\n  [TRADE STATISTICS]")
        print(f"  {'-' * 40}")
        print(f"  Total Trades:         {metrics['total_trades']}")
        print(f"  Wins / Losses:        {metrics['wins']} / {metrics['losses']}")
        print(f"  Win Rate:             {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor:        {metrics['profit_factor']:.2f}")
        print(f"  Avg Win:              ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss:             ${metrics['avg_loss']:.2f}")
        print(f"  Avg Trade Duration:   {metrics['avg_trade_duration_bars']:.1f} bars")
        
        print(f"\n  [RISK METRICS]")
        print(f"  {'-' * 40}")
        print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:        {metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:         {metrics['calmar_ratio']:.2f}")
        print(f"  Max Daily Loss:       {metrics['max_daily_loss']:.2f}%")
        print(f"  Max Consec. Losses:   {metrics['max_consecutive_losses']}")
        
        print(f"\n  [EXIT BREAKDOWN]")
        print(f"  {'-' * 40}")
        print(f"  Stop Loss:            {metrics['sl_exits']} ({metrics['sl_exits']/metrics['total_trades']*100:.1f}%)")
        print(f"  Take Profit:          {metrics['tp_exits']} ({metrics['tp_exits']/metrics['total_trades']*100:.1f}%)")
        print(f"  Time Exit:            {metrics['time_exits']} ({metrics['time_exits']/metrics['total_trades']*100:.1f}%)")
        
        print(f"\n  [STRATEGY BREAKDOWN]")
        print(f"  {'-' * 40}")
        print(f"  TREND Trades:         {metrics['trend_trades']} (PnL: ${metrics['trend_pnl']:,.0f})")
        print(f"  MR Trades:            {metrics['mr_trades']} (PnL: ${metrics['mr_pnl']:,.0f})")
        
        print(f"\n  [MONTHLY RETURNS]")
        print(f"  {'-' * 40}")
        monthly = metrics["monthly_returns"]
        print(f"  Avg Monthly Return:   {monthly.mean():.2f}%")
        print(f"  Best Month:           {monthly.max():.2f}%")
        print(f"  Worst Month:          {monthly.min():.2f}%")
        print(f"  Positive Months:      {len(monthly[monthly > 0])} / {len(monthly)}")
        print(f"  Negative Months:      {len(monthly[monthly <= 0])}")
    
    # Comparison
    print("\n" + "=" * 100)
    print("                                  BOT COMPARISON")
    print("=" * 100)
    
    orig = all_metrics["Original Bot"]
    prop = all_metrics["Prop Firm Bot"]
    
    print(f"\n  {'Metric':<30} {'Original Bot':>15} {'PropFirm Bot':>15} {'Winner':>15}")
    print(f"  {'-' * 75}")
    
    comparisons = [
        ("Total Return", f"{orig['total_return']:.1f}%", f"{prop['total_return']:.1f}%", 
         "Original" if orig['total_return'] > prop['total_return'] else "PropFirm"),
        ("Max Drawdown", f"{orig['max_drawdown']:.1f}%", f"{prop['max_drawdown']:.1f}%",
         "PropFirm" if prop['max_drawdown'] < orig['max_drawdown'] else "Original"),
        ("Profit Factor", f"{orig['profit_factor']:.2f}", f"{prop['profit_factor']:.2f}",
         "Original" if orig['profit_factor'] > prop['profit_factor'] else "PropFirm"),
        ("Sharpe Ratio", f"{orig['sharpe_ratio']:.2f}", f"{prop['sharpe_ratio']:.2f}",
         "Original" if orig['sharpe_ratio'] > prop['sharpe_ratio'] else "PropFirm"),
        ("Sortino Ratio", f"{orig['sortino_ratio']:.2f}", f"{prop['sortino_ratio']:.2f}",
         "Original" if orig['sortino_ratio'] > prop['sortino_ratio'] else "PropFirm"),
        ("Calmar Ratio", f"{orig['calmar_ratio']:.2f}", f"{prop['calmar_ratio']:.2f}",
         "Original" if orig['calmar_ratio'] > prop['calmar_ratio'] else "PropFirm"),
        ("Win Rate", f"{orig['win_rate']:.1f}%", f"{prop['win_rate']:.1f}%",
         "Tie" if abs(orig['win_rate'] - prop['win_rate']) < 0.5 else ("Original" if orig['win_rate'] > prop['win_rate'] else "PropFirm")),
    ]
    
    for metric, orig_val, prop_val, winner in comparisons:
        print(f"  {metric:<30} {orig_val:>15} {prop_val:>15} {winner:>15}")
    
    # Recommendations
    print("\n" + "=" * 100)
    print("                                  RECOMMENDATIONS")
    print("=" * 100)
    
    print("""
  [ORIGINAL BOT] (0.7% Risk)
     + Best for: Personal trading accounts with higher risk tolerance
     + Pros: Higher returns, better capital growth
     - Cons: Higher drawdown, not suitable for prop firm challenges
     > Recommended for: Accounts where >10% monthly DD is acceptable
  
  [PROP FIRM BOT] (0.2% Risk)  
     + Best for: Prop firm challenges (5%ers, FTMO, etc.)
     + Pros: Lower drawdown, safer for passing challenges
     - Cons: Lower returns, slower capital growth
     > Recommended for: Prop firm rules with 5-10% max loss limits
  
  [KEY INSIGHTS]
     * Both bots use the same strategy (MOONSHOT) with identical signals
     * The only difference is position sizing (0.7% vs 0.2% risk)
     * PropFirm bot should comfortably pass 5%ers challenge (<10% max loss)
     * Original bot is optimized for personal account growth
    """)
    
    print("=" * 100)

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("   COMPREHENSIVE BOT PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Load data
    all_data = load_or_fetch_data()
    if all_data is None:
        print("[!] Failed to load data")
        sys.exit(1)
    
    # Run backtests for both bots
    results = {}
    all_metrics = {}
    
    for bot_name, config in BOT_CONFIGS.items():
        print(f"\n[*] Running backtest for {bot_name}...")
        result = run_backtest(all_data, config["risk_per_trade"])
        results[bot_name] = result
        
        print(f"    Calculating metrics...")
        metrics = calculate_metrics(result, bot_name)
        all_metrics[bot_name] = metrics
        
        print(f"    [OK] {metrics['total_trades']} trades, ${metrics['final_balance']:,.0f} final balance")
    
    # Print report
    print_report(all_metrics)
    
    # Create visualizations
    print("\n[*] Creating visualizations...")
    chart_path = create_visualizations(results, all_metrics)
    
    print("\n[*] Analysis complete!")
    print(f"    View chart: {chart_path}")
