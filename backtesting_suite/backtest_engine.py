import pandas as pd
import numpy as np

def calculate_dynamic_risk(balance: float, peak_balance: float, base_risk: float, risk_tiers: list = None) -> float:
    """Calculate risk percentage based on drawdown (Prop Firm Logic)."""
    if peak_balance <= 0 or balance >= peak_balance:
        return base_risk
        
    dd_abs = peak_balance - balance
    current_dd = (dd_abs / peak_balance) * 100
    
    # Tiered Risk Logic
    # Default Tiers (Prop Firm): (Threshold, Multiplier)
    if risk_tiers is None:
        risk_tiers = [
            (1.5, 0.8),    # > 1.5% DD -> 80% risk
            (3.0, 0.6),    # > 3.0% DD -> 60% risk
            (4.0, 0.4),    # > 4.0% DD -> 40% risk
            (5.0, 0.2),    # > 5.0% DD -> 20% risk
            (6.0, 0.1)     # > 6.0% DD -> 10% risk
        ]
    
    multiplier = 1.0
    for threshold, mult in risk_tiers:
        if current_dd > threshold:
            multiplier = mult
            
    return base_risk * multiplier

def run_backtest(
    df: pd.DataFrame,
    strategy_params: dict,
    initial_balance: float = 10000.0,
    risk_per_trade: float = 0.02,
    pip_value: float = 0.0001,
    use_dynamic_risk: bool = False,
    risk_tiers: list = None
) -> dict:
    """
    Run backtest with trailing stop.
    
    Args:
        df: DataFrame with OHLC data and signals (must have indicators calculated)
        strategy_params: Dictionary of strategy parameters (TREND, MR configs)
        initial_balance: Starting balance in USD
        risk_per_trade: Risk per trade as decimal (0.02 = 2%)
        pip_value: Pip value for the instrument (0.0001 for EUR/USD)
        use_dynamic_risk: If True, adjust risk based on drawdown
    
    Returns:
        Dictionary with backtest results
    """
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0
    equity_curve = []
    trades = []
    
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
    
    # Start after warmup period for indicators
    start_idx = min(250, len(df) // 10)
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        price = row["Close"]
        high = row["High"]
        low = row["Low"]
        atr = row["ATR"]
        
        # Update peak balance for dynamic risk
        if balance > peak_balance:
            peak_balance = balance
            
        # Exit logic - using High/Low for realistic SL/TP simulation
        if position is not None:
            bars_in_trade += 1
            exit_price = None
            exit_reason = ""
            params = strategy_params.get(signal_type, {})
            if not params:
                 # Fallback if signal type missing
                 continue

            if position == "buy":
                # Use High for trailing (best price during bar)
                highest = max(highest, high)
                
                # Trailing stop - update based on highest price
                if highest - entry_price > params["TRAIL_START"] * atr:
                    new_sl = highest - params["TRAIL_DIST"] * atr
                    stop_loss = max(stop_loss, new_sl)
                
                # Check exits using Low (worst case for buy)
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                elif high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                elif bars_in_trade >= params["MAX_BARS"]:
                    exit_price = price
                    exit_reason = "TIME"
                    
            elif position == "sell":
                # Use Low for trailing (best price during bar)
                lowest = min(lowest, low)
                
                # Trailing stop - update based on lowest price
                if entry_price - lowest > params["TRAIL_START"] * atr:
                    new_sl = lowest + params["TRAIL_DIST"] * atr
                    stop_loss = min(stop_loss, new_sl)
                
                # Check exits using High (worst case for sell)
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                elif low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                elif bars_in_trade >= params["MAX_BARS"]:
                    exit_price = price
                    exit_reason = "TIME"
            
            # Close position
            if exit_price is not None:
                if position == "buy":
                    pnl = (exit_price - entry_price) * position_size * 100000
                else:
                    pnl = (entry_price - exit_price) * position_size * 100000
                
                balance += pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'type': position,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'result': exit_reason,
                    'bars_held': bars_in_trade
                })
                
                position = None
                
                # Update max drawdown
                if balance > peak_balance:
                    peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100
                max_drawdown = max(max_drawdown, dd)
        
        # SIGNAL REVERSAL: Check if opposite signal appears while in position
        if position is not None:
            signal_reversal = False
            new_signal_type = None
            
            # Check for opposite signal
            if position == "buy" and (row.get("trend_sell", False) or row.get("mr_sell", False)):
                signal_reversal = True
                new_signal_type = "TREND" if row.get("trend_sell", False) else "MR"
            elif position == "sell" and (row.get("trend_buy", False) or row.get("mr_buy", False)):
                signal_reversal = True  
                new_signal_type = "TREND" if row.get("trend_buy", False) else "MR"
            
            # If reversal signal detected, close current and open opposite
            if signal_reversal:
                # Close current position
                exit_price = price
                if position == "buy":
                    pnl = (exit_price - entry_price) * position_size * 100000
                else:
                    pnl = (entry_price - exit_price) * position_size * 100000
                
                balance += pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'type': position,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position_size,
                    'pnl': pnl,
                    'balance': balance,
                    'exit_reason': 'REVERSAL',
                    'bars_held': bars_in_trade
                })
                
                # Update max drawdown
                if balance > peak_balance:
                    peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100
                max_drawdown = max(max_drawdown, dd)
                
                # Open new position in opposite direction
                current_risk_pct = calculate_dynamic_risk(balance, peak_balance, risk_per_trade, risk_tiers) if use_dynamic_risk else risk_per_trade
                risk_amount = balance * current_risk_pct
                
                params = strategy_params.get(new_signal_type, {})
                if not params: continue

                sl_dist = atr * params["SL_ATR"]
                
                if sl_dist > 0:
                    # Determine new position direction
                    new_position = "sell" if position == "buy" else "buy"
                    position = new_position
                    signal_type = new_signal_type
                    entry_price = price
                    position_size = risk_amount / (sl_dist * 100000)
                    position_size = max(0.01, min(position_size, 10.0))
                    bars_in_trade = 0
                    entry_time = current_time
                    
                    if position == "buy":
                        stop_loss = price - sl_dist
                        take_profit = price + atr * params["TP_ATR"]
                        highest = price
                        lowest = float('inf')
                    else:
                        stop_loss = price + sl_dist
                        take_profit = price - atr * params["TP_ATR"]
                        lowest = price
                        highest = 0
                else:
                    position = None
                    
                continue  # Skip normal exit/entry logic for this bar
        
        # Entry logic
        if position is None:
            current_risk_pct = calculate_dynamic_risk(balance, peak_balance, risk_per_trade, risk_tiers) if use_dynamic_risk else risk_per_trade
            risk_amount = balance * current_risk_pct

            if row.get("trend_buy", False):
                signal_type = "TREND"
                params = strategy_params["TREND"]
                sl_dist = atr * params["SL_ATR"]
                
                if sl_dist > 0:
                    position = "buy"
                    entry_price = price
                    stop_loss = price - sl_dist
                    take_profit = price + atr * params["TP_ATR"]
                    position_size = risk_amount / (sl_dist * 100000)
                    position_size = max(0.01, min(position_size, 10.0))
                    highest = price
                    bars_in_trade = 0
                    entry_time = current_time
                    
            elif row.get("mr_buy", False):
                signal_type = "MR"
                params = strategy_params["MR"]
                sl_dist = atr * params["SL_ATR"]
                
                if sl_dist > 0:
                    position = "buy"
                    entry_price = price
                    stop_loss = price - sl_dist
                    take_profit = price + atr * params["TP_ATR"]
                    position_size = risk_amount / (sl_dist * 100000)
                    position_size = max(0.01, min(position_size, 10.0))
                    highest = price
                    bars_in_trade = 0
                    entry_time = current_time
                    
            elif row.get("trend_sell", False):
                signal_type = "TREND"
                params = strategy_params["TREND"]
                sl_dist = atr * params["SL_ATR"]
                
                if sl_dist > 0:
                    position = "sell"
                    entry_price = price
                    stop_loss = price + sl_dist
                    take_profit = price - atr * params["TP_ATR"]
                    position_size = risk_amount / (sl_dist * 100000)
                    position_size = max(0.01, min(position_size, 10.0))
                    lowest = price
                    bars_in_trade = 0
                    entry_time = current_time
                    
            elif row.get("mr_sell", False):
                signal_type = "MR"
                params = strategy_params["MR"]
                sl_dist = atr * params["SL_ATR"]
                
                if sl_dist > 0:
                    position = "sell"
                    entry_price = price
                    stop_loss = price + sl_dist
                    take_profit = price - atr * params["TP_ATR"]
                    position_size = risk_amount / (sl_dist * 100000)
                    position_size = max(0.01, min(position_size, 10.0))
                    lowest = price
                    bars_in_trade = 0
                    entry_time = current_time
        
        # Track equity at each timestamp
        equity_curve.append({'time': current_time, 'balance': balance})
    
    # Calculate statistics
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'final_balance': balance,
            'trades': [],
            'equity_curve': equity_curve
        }
    
    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    
    total_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate Max Daily Drawdown
    equity_df = pd.DataFrame(equity_curve)
    equity_df['date'] = equity_df['time'].dt.date
    equity_df['balance'] = equity_df['balance']
    
    # Calculate daily start (open) and min (low) balance
    daily_stats = equity_df.groupby('date')['balance'].agg(['first', 'min'])
    
    # Daily Drawdown = (Daily Open - Daily Low) / Daily Open
    # Only counts if Daily Low < Daily Open (loss for the day)
    daily_stats['drawdown_pct'] = ((daily_stats['first'] - daily_stats['min']) / daily_stats['first']) * 100
    max_daily_drawdown = daily_stats['drawdown_pct'].max() if not daily_stats.empty else 0
    
    # Count by signal type
    trend_trades = trades_df[trades_df['signal_type'] == 'TREND']
    mr_trades = trades_df[trades_df['signal_type'] == 'MR']
    
    # Calculate days to pass
    phase1_target = initial_balance * 1.08
    phase2_target = initial_balance * 1.05
    
    days_to_phase1 = None
    days_to_phase2 = None
    
    start_date = equity_df['date'].iloc[0]
    
    # Check Phase 1 (8%)
    pass_p1 = equity_df[equity_df['balance'] >= phase1_target]
    if not pass_p1.empty:
        pass_date = pass_p1.iloc[0]['date']
        days_to_phase1 = (pass_date - start_date).days
        
    # Check Phase 2 (5%)
    pass_p2 = equity_df[equity_df['balance'] >= phase2_target]
    if not pass_p2.empty:
        pass_date = pass_p2.iloc[0]['date']
        days_to_phase2 = (pass_date - start_date).days

    return {
        'initial_balance': initial_balance,
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'total_return': (balance - initial_balance) / initial_balance * 100,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'max_daily_drawdown': max_daily_drawdown,
        'days_to_phase1': days_to_phase1,
        'days_to_phase2': days_to_phase2,
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'final_balance': balance,
        'trend_trades': len(trend_trades),
        'mr_trades': len(mr_trades),
        'trades': trades,
        'equity_curve': equity_curve
    }
