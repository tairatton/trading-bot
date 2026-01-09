"""
Multi-Pair Portfolio Backtest
Run multiple pairs on the SAME account balance to get realistic portfolio performance.
"""
import argparse
import sys
import os
import pandas as pd
import MetaTrader5 as mt5
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

def run_portfolio_backtest(
    pairs_data: dict,  # {symbol: df}
    strategy_params: dict,
    initial_balance: float = 2500.0,
    risk_per_trade: float = 0.0015,  # 0.15% per pair
    use_dynamic_risk: bool = True,
    risk_tiers: list = None,
    daily_loss_limit: float = 4.1,
    spread_pips: float = 1.5,
    output_dir: str = "backtest_results"
):
    """
    Run portfolio backtest with multiple pairs on SAME balance.
    """
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0
    equity_curve = []
    all_trades = []
    
    # Position state per symbol
    positions = {}  # {symbol: {position, entry_price, stop_loss, take_profit, ...}}
    
    # Daily tracking
    daily_start_balance = initial_balance
    current_day = None
    max_daily_dd_pct = 0.0
    
    # Merge all dataframes by time
    all_times = set()
    for symbol, df in pairs_data.items():
        all_times.update(df.index.tolist())
    all_times = sorted(all_times)
    
    # Start after warmup
    start_idx = 250
    
    
    # OUTPUT DIRECTORY
    os.makedirs(output_dir, exist_ok=True)
    
    for i, current_time in enumerate(all_times[start_idx:], start=start_idx):
        # Daily reset
        bar_day = current_time.date() if hasattr(current_time, 'date') else current_time
        if current_day != bar_day:
            current_day = bar_day
            daily_start_balance = balance
        
        # Check daily loss
        unrealized_pnl = 0.0
        
        # Calculate floating PnL for Equity
        for symbol, pos in positions.items():
            if pos["position"] is not None:
                # Get current price
                df = pairs_data[symbol]
                if current_time in df.index:
                    curr_price = df.loc[current_time, "Close"]
                    
                    if pos["position"] == "buy":
                        floating = (curr_price - pos["entry_price"]) * pos["position_size"] * 100000
                    else:
                        floating = (pos["entry_price"] - curr_price) * pos["position_size"] * 100000
                    
                    unrealized_pnl += floating
        
        equity = balance + unrealized_pnl
        equity_curve.append({
            "time": current_time, 
            "balance": balance, 
            "equity": equity
        })
        
        # Daily Loss Calculation (Equity based)
        daily_loss_pct = ((daily_start_balance - equity) / daily_start_balance * 100) if daily_start_balance > 0 else 0
        if daily_loss_pct > max_daily_dd_pct:
            max_daily_dd_pct = daily_loss_pct
        daily_blocked = daily_loss_pct >= daily_loss_limit
        
        # Update peak and DD
        if equity > peak_balance:
            peak_balance = equity
        dd = (peak_balance - equity) / peak_balance * 100 if peak_balance > 0 else 0
        max_drawdown = max(max_drawdown, dd)
        
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
            spread_price = spread_pips * pip_value
            
            # Initialize position state
            if symbol not in positions:
                positions[symbol] = {"position": None}
            
            pos = positions[symbol]
            
            # Exit logic
            if pos["position"] is not None:
                pos["bars_in_trade"] = pos.get("bars_in_trade", 0) + 1
                exit_price = None
                exit_reason = ""
                params = strategy_params.get(pos["signal_type"], {})
                
                # Update Highest/Lowest for MAE/MFE
                if pos["position"] == "buy":
                    pos["highest"] = max(pos.get("highest", price), high)
                    pos["lowest"] = min(pos.get("lowest", price), low) # Track lowest for MAE
                    
                    # Trailing Stop
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
                        
                else:  # sell
                    pos["lowest"] = min(pos.get("lowest", price), low)
                    pos["highest"] = max(pos.get("highest", price), high) # Track highest for MAE
                    
                    # Trailing Stop
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
                    # Calculate MFE/MAE
                    mae = 0.0
                    mfe = 0.0
                    if pos["position"] == "buy":
                        pnl = (exit_price - pos["entry_price"]) * pos["position_size"] * 100000
                        # MAE: Max loss potential (Entry - Lowest)
                        mae = (pos["entry_price"] - pos["lowest"]) * pos["position_size"] * 100000
                        # MFE: Max profit potential (Highest - Entry)
                        mfe = (pos["highest"] - pos["entry_price"]) * pos["position_size"] * 100000
                        mae = min(mae, 0) # MAE is usually negative or zero relative to entry? No, usually positive distance. 
                        # Let's align with common definition: MAE is max adverse excursion (distance).
                        # But user wants scatter plot. Let's keep PnL units for easier plotting.
                        # Revise: MAE as negative PnL value, MFE as positive PnL value.
                        mae = (pos["lowest"] - pos["entry_price"]) * pos["position_size"] * 100000
                        mfe = (pos["highest"] - pos["entry_price"]) * pos["position_size"] * 100000
                    else:
                        pnl = (pos["entry_price"] - exit_price) * pos["position_size"] * 100000
                        # MAE: Max loss (Highest - Entry)
                        mae = (pos["entry_price"] - pos["highest"]) * pos["position_size"] * 100000
                        # MFE: Max profit (Entry - Lowest)
                        mfe = (pos["entry_price"] - pos["lowest"]) * pos["position_size"] * 100000
                    
                    balance += pnl
                    all_trades.append({
                        "ticket": len(all_trades) + 1,
                        "symbol": symbol,
                        "type": pos["position"],
                        "entry_time": current_time, # Approximation (exit time is current)
                        "exit_time": current_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "lots": pos["position_size"],
                        "pnl": pnl,
                        "mae": mae,
                        "mfe": mfe,
                        "reason": exit_reason,
                        "duration_bars": pos["bars_in_trade"]
                    })
                    pos["position"] = None
            
            # Entry logic (skip if daily blocked or already in position)
            if pos["position"] is None and not daily_blocked:
                current_risk_pct = calculate_dynamic_risk(balance, peak_balance, risk_per_trade, risk_tiers) if use_dynamic_risk else risk_per_trade
                risk_amount = balance * current_risk_pct
                
                signal_type = None
                direction = None
                
                if row.get("trend_buy", False):
                    signal_type = "TREND"
                    direction = "buy"
                elif row.get("mr_buy", False):
                    signal_type = "MR"
                    direction = "buy"
                elif row.get("trend_sell", False):
                    signal_type = "TREND"
                    direction = "sell"
                elif row.get("mr_sell", False):
                    signal_type = "MR"
                    direction = "sell"
                
                if signal_type and direction:
                    params = strategy_params[signal_type]
                    sl_dist = atr * params["SL_ATR"]
                    
                    if sl_dist > 0:
                        pos["position"] = direction
                        pos["signal_type"] = signal_type
                        pos["bars_in_trade"] = 0
                        
                        if direction == "buy":
                            pos["entry_price"] = price + spread_price
                            pos["stop_loss"] = pos["entry_price"] - sl_dist
                            pos["take_profit"] = pos["entry_price"] + atr * params["TP_ATR"]
                            pos["highest"] = pos["entry_price"]
                            pos["lowest"] = pos["entry_price"]
                        else:
                            pos["entry_price"] = price
                            pos["stop_loss"] = pos["entry_price"] + sl_dist
                            pos["take_profit"] = pos["entry_price"] - atr * params["TP_ATR"]
                            pos["lowest"] = pos["entry_price"]
                            pos["highest"] = pos["entry_price"]
                        
                        pos["position_size"] = risk_amount / (sl_dist * 100000)
                        pos["position_size"] = max(0.01, min(pos["position_size"], 10.0))
        
    
    # Calculate stats
    total_trades = len(all_trades)
    wins = len([t for t in all_trades if t["pnl"] > 0])
    losses = len([t for t in all_trades if t["pnl"] <= 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    gross_profit = sum([t["pnl"] for t in all_trades if t["pnl"] > 0])
    gross_loss = abs(sum([t["pnl"] for t in all_trades if t["pnl"] < 0]))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999
    
    # Export CSVs
    pd.DataFrame(all_trades).to_csv(os.path.join(output_dir, "trades.csv"), index=False)
    pd.DataFrame(equity_curve).to_csv(os.path.join(output_dir, "equity.csv"), index=False)
    
    print(f"Results saved to {output_dir}")
    
    return {
        "final_balance": balance,
        "total_return": (balance - initial_balance) / initial_balance * 100,
        "max_drawdown": max_drawdown,
        "max_daily_drawdown": max_daily_dd_pct,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trades_by_symbol": {s: len([t for t in all_trades if t["symbol"] == s]) for s in pairs_data.keys()}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["EURUSDm", "GBPUSDm", "AUDUSDm", "USDCADm", "NZDUSDm"])
    parser.add_argument("--days", type=int, default=1000)
    parser.add_argument("--risk", type=float, default=0.0015)  # 0.15% per pair
    parser.add_argument("--output", type=str, default="backtest_results")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PORTFOLIO BACKTEST: 5 PAIRS (SHARED BALANCE)")
    print("=" * 60)
    print(f"Pairs: {', '.join(args.pairs)}")
    print(f"Risk per pair: {args.risk * 100:.2f}%")
    print(f"Output Directory: {args.output}")
    print()
    
    # Load strategy
    strategy = get_strategy_module("propfirm")
    
    # Fetch data for all pairs
    pairs_data = {}
    bars_count = args.days * 48  # 30min bars
    
    for symbol in args.pairs:
        print(f"Fetching {symbol}...")
        df = fetch_data(symbol, bars_count)
        if not df.empty:
            df = strategy.calculate_indicators(df)
            df = strategy.generate_signals(df)
            pairs_data[symbol] = df
            print(f"  -> {len(df)} bars loaded")
    
    if not pairs_data:
        print("No data loaded!")
        return
    
    print("\nRunning portfolio simulation...")
    
    # Risk tiers
    if args.risk >= 0.005: # Assume aggressive/original bot if risk >= 0.5%
        print("Using ORIGINAL BOT Risk Tiers (Aggressive)")
        risk_tiers = [
            (5.0, 0.8),
            (10.0, 0.6),
            (15.0, 0.4),
            (18.0, 0.2),
            (19.0, 0.1)
        ]
        daily_loss = 10.0
    else:
        print("Using PROPFIRM BOT Risk Tiers (Safe)")
        risk_tiers = [
            (1.5, 0.8),
            (3.0, 0.6),
            (4.5, 0.4),
            (6.0, 0.2),
            (7.5, 0.1),
            (8.5, 0.05)
        ]
        daily_loss = 4.1
    
    results = run_portfolio_backtest(
        pairs_data,
        strategy.STRATEGY_PARAMS,
        initial_balance=2500.0,
        risk_per_trade=args.risk,
        use_dynamic_risk=True,
        risk_tiers=risk_tiers,
        daily_loss_limit=daily_loss,
        spread_pips=1.5,
        output_dir=args.output
    )
    
    # Report
    print("\n" + "=" * 60)
    print("PORTFOLIO RESULTS (5 PAIRS)")
    print("=" * 60)
    print(f"Final Balance:   ${results['final_balance']:,.2f}")
    print(f"Total Return:    {results['total_return']:.2f}%")
    print(f"Max Drawdown:    {results['max_drawdown']:.2f}%")
    print(f"Max Daily DD:    {results['max_daily_drawdown']:.2f}%")
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Win Rate:        {results['win_rate']:.1f}%")
    print(f"Profit Factor:   {results['profit_factor']:.2f}")
    print()
    print("Trades by Symbol:")
    for symbol, count in results['trades_by_symbol'].items():
        print(f"  {symbol}: {count} trades")
    print("=" * 60)

if __name__ == "__main__":
    main()
