"""
Shared Strategy Module
======================
This module contains the core strategy logic used by both:
- Live Trading Bot (trading_service.py)
- Backtest Script (backtest.py)

Modify this file to update the strategy for both systems.
"""
import pandas as pd
import numpy as np
import ta


# ============================================================
# STRATEGY PARAMETERS (OPTIMIZED 2026-01-07)
# ============================================================
STRATEGY_PARAMS = {
    "TREND": {
        "SL_ATR": 1.5,       # Stop Loss = 1.5 x ATR (Optimized for <10% DD)
        "TP_ATR": 5.0,       # Take Profit = 5.0 x ATR (Moonshot)
        "TRAIL_START": 1.0,  # Start trailing after 1.0 x ATR profit (was 1.2)
        "TRAIL_DIST": 0.6,   # Trail distance = 0.6 x ATR
        "MAX_BARS": 50       # Max bars in trade
    },
    "MR": {
        "SL_ATR": 0.6,       # Stop Loss = 0.6 x ATR (Optimized for <10% DD)
        "TP_ATR": 3.0,       # Take Profit = 3.0 x ATR (Moonshot)
        "TRAIL_START": 0.53, # Start trailing after 0.53 x ATR (was 0.64)
        "TRAIL_DIST": 0.4,   # Trail distance = 0.4 x ATR
        "MAX_BARS": 25       # Max bars in trade
    }
}


# ============================================================
# INDICATOR CALCULATION
# ============================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for the strategy.
    
    Indicators:
    - SMA 20, 50, 200
    - EMA 10
    - RSI 14
    - Bollinger Bands (20, 2)
    - ATR 14
    - ADX 14 with DI+/DI-
    - Stochastic K (14, 3)
    """
    df = df.copy()
    
    # Moving Averages
    df["SMA_20"] = ta.trend.SMAIndicator(df["Close"], 20).sma_indicator()
    df["SMA_50"] = ta.trend.SMAIndicator(df["Close"], 50).sma_indicator()
    df["SMA_200"] = ta.trend.SMAIndicator(df["Close"], 200).sma_indicator()
    df["EMA_10"] = ta.trend.EMAIndicator(df["Close"], 10).ema_indicator()
    
    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Middle"] = bb.bollinger_mavg()
    
    # ATR
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
    
    # ADX
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], 14)
    df["ADX"] = adx.adx()
    df["DI_Plus"] = adx.adx_pos()
    df["DI_Minus"] = adx.adx_neg()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], 14, 3)
    df["Stoch_K"] = stoch.stoch()
    # Drop NaN rows except keep the last row (current forming candle)
    if len(df) > 1:
        df_clean = df.iloc[:-1].dropna()
        df_last = df.iloc[-1:].ffill()
        df = pd.concat([df_clean, df_last])
    return df


# ============================================================
# SIGNAL GENERATION
# ============================================================
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy/sell signals based on the hybrid strategy.
    
    Strategy 1: TREND FOLLOWING
    - Buy: Uptrend + pullback to SMA20 + RSI 40-65
    - Sell: Downtrend + rally to SMA20 + RSI 35-60
    
    Strategy 2: MEAN REVERSION
    - Buy: Price <= BB Lower + RSI < 35 + Stoch < 25 + ADX < 30
    - Sell: Price >= BB Upper + RSI > 65 + Stoch > 75 + ADX < 30
    """
    df = df.copy()
    
    # TREND FOLLOWING
    df["uptrend"] = (df["Close"] > df["SMA_50"]) & (df["SMA_50"] > df["SMA_200"])
    df["downtrend"] = (df["Close"] < df["SMA_50"]) & (df["SMA_50"] < df["SMA_200"])
    
    df["trend_buy"] = (
        df["uptrend"] &
        (df["Low"] <= df["SMA_20"]) &
        (df["Close"] > df["SMA_20"]) &
        (df["RSI"] > 40) & (df["RSI"] < 65) &
        (df["ADX"] > 30)  # Optimization: Filter out weak trends (PF 1.48 -> 1.57)
    )
    
    df["trend_sell"] = (
        df["downtrend"] &
        (df["High"] >= df["SMA_20"]) &
        (df["Close"] < df["SMA_20"]) &
        (df["RSI"] > 35) & (df["RSI"] < 60) &
        (df["ADX"] > 30)  # Optimization: Filter out weak trends
    )
    
    # MEAN REVERSION
    df["mr_buy"] = (
        (df["Close"] <= df["BB_Lower"]) &
        (df["RSI"] < 35) &
        (df["Stoch_K"] < 25) &
        (df["ADX"] < 30)
    )
    
    df["mr_sell"] = (
        (df["Close"] >= df["BB_Upper"]) &
        (df["RSI"] > 65) &
        (df["Stoch_K"] > 75) &
        (df["ADX"] < 30)
    )
    
    # Combined signals
    df["buy_signal"] = df["trend_buy"] | df["mr_buy"]
    df["sell_signal"] = df["trend_sell"] | df["mr_sell"]
    
    return df


def calculate_signal_strength(row: pd.Series, signal: str, signal_type: str) -> dict:
    """
    Calculate signal strength based on multiple indicators.
    
    Factors:
    1. ADX (trend strength): Higher = stronger for TREND, lower = better for MR
    2. RSI extremity: More extreme = stronger signal
    3. Trend alignment: Price vs SMA20/50/200
    4. Stochastic confirmation
    5. Bollinger Band position
    
    Returns:
        dict with 'strength' (WEAK/MEDIUM/STRONG), 'score' (0-100), 'factors' list
    """
    score = 0
    factors = []
    
    # Get indicator values
    adx = row.get("ADX", 25)
    rsi = row.get("RSI", 50)
    stoch = row.get("Stoch_K", 50)
    price = row.get("Close", 0)
    sma20 = row.get("SMA_20", price)
    sma50 = row.get("SMA_50", price)
    sma200 = row.get("SMA_200", price)
    bb_upper = row.get("BB_Upper", price)
    bb_lower = row.get("BB_Lower", price)
    di_plus = row.get("DI_Plus", 20)
    di_minus = row.get("DI_Minus", 20)
    
    if signal == "none":
        return {"strength": "NONE", "score": 0, "factors": []}
    
    # ========== TREND FOLLOWING SIGNALS ==========
    if signal_type == "TREND":
        # 1. ADX (Trend strength) - Higher is better (max 25 points)
        if adx >= 30:
            score += 25
            factors.append("🔥 Strong trend (ADX≥30)")
        elif adx >= 25:
            score += 20
            factors.append("📈 Good trend (ADX≥25)")
        elif adx >= 20:
            score += 15
            factors.append("📊 Moderate trend")
        else:
            score += 5
            factors.append("⚠️ Weak trend")
        
        # 2. DI+ vs DI- alignment (max 20 points)
        if signal == "buy":
            if di_plus > di_minus + 10:
                score += 20
                factors.append("✅ DI+ >> DI-")
            elif di_plus > di_minus:
                score += 10
                factors.append("👍 DI+ > DI-")
        else:  # sell
            if di_minus > di_plus + 10:
                score += 20
                factors.append("✅ DI- >> DI+")
            elif di_minus > di_plus:
                score += 10
                factors.append("👍 DI- > DI+")
        
        # 3. RSI confirmation (max 20 points)
        if signal == "buy":
            if 45 <= rsi <= 55:  # RSI in neutral, good for pullback entry
                score += 20
                factors.append("🎯 RSI optimal zone")
            elif 40 <= rsi <= 60:
                score += 15
                factors.append("✓ RSI in range")
        else:  # sell
            if 45 <= rsi <= 55:
                score += 20
                factors.append("🎯 RSI optimal zone")
            elif 40 <= rsi <= 60:
                score += 15
                factors.append("✓ RSI in range")
        
        # 4. Moving Average alignment (max 20 points)
        if signal == "buy":
            ma_score = sum([
                price > sma20,
                sma20 > sma50,
                sma50 > sma200,
                price > sma200
            ])
            score += ma_score * 5
            if ma_score >= 3:
                factors.append("📊 MA alignment strong")
        else:  # sell
            ma_score = sum([
                price < sma20,
                sma20 < sma50,
                sma50 < sma200,
                price < sma200
            ])
            score += ma_score * 5
            if ma_score >= 3:
                factors.append("📊 MA alignment strong")
        
        # 5. Stochastic confirmation (max 15 points)
        if signal == "buy" and stoch < 50:
            score += 15
            factors.append("✓ Stoch not overbought")
        elif signal == "sell" and stoch > 50:
            score += 15
            factors.append("✓ Stoch not oversold")
    
    # ========== MEAN REVERSION SIGNALS ==========
    elif signal_type == "MR":
        # 1. ADX (Low = better for MR) (max 25 points)
        if adx < 20:
            score += 25
            factors.append("🎯 Ranging market (ADX<20)")
        elif adx < 25:
            score += 20
            factors.append("👍 Low trend (ADX<25)")
        elif adx < 30:
            score += 10
            factors.append("⚠️ Moderate ADX")
        
        # 2. RSI extremity (max 25 points)
        if signal == "buy":
            if rsi < 25:
                score += 25
                factors.append("🔥 RSI extremely oversold")
            elif rsi < 30:
                score += 20
                factors.append("📉 RSI oversold")
            elif rsi < 35:
                score += 15
                factors.append("✓ RSI low")
        else:  # sell
            if rsi > 75:
                score += 25
                factors.append("🔥 RSI extremely overbought")
            elif rsi > 70:
                score += 20
                factors.append("📈 RSI overbought")
            elif rsi > 65:
                score += 15
                factors.append("✓ RSI high")
        
        # 3. Stochastic extremity (max 20 points)
        if signal == "buy":
            if stoch < 15:
                score += 20
                factors.append("🎯 Stoch extremely low")
            elif stoch < 25:
                score += 15
                factors.append("✓ Stoch low")
        else:  # sell
            if stoch > 85:
                score += 20
                factors.append("🎯 Stoch extremely high")
            elif stoch > 75:
                score += 15
                factors.append("✓ Stoch high")
        
        # 4. Bollinger Band touch (max 20 points)
        bb_range = bb_upper - bb_lower if bb_upper > bb_lower else 1
        if signal == "buy":
            distance_from_lower = (price - bb_lower) / bb_range
            if distance_from_lower < 0:  # Below BB lower
                score += 20
                factors.append("🔥 Price below BB lower")
            elif distance_from_lower < 0.1:
                score += 15
                factors.append("✓ Price near BB lower")
        else:  # sell
            distance_from_upper = (bb_upper - price) / bb_range
            if distance_from_upper < 0:  # Above BB upper
                score += 20
                factors.append("🔥 Price above BB upper")
            elif distance_from_upper < 0.1:
                score += 15
                factors.append("✓ Price near BB upper")
        
        # 5. Reversal candle pattern hint (max 10 points)
        # Simple check: close vs open direction
        if signal == "buy" and price > row.get("Open", price):
            score += 10
            factors.append("🕯️ Bullish candle")
        elif signal == "sell" and price < row.get("Open", price):
            score += 10
            factors.append("🕯️ Bearish candle")
    
    # Determine strength level
    if score >= 70:
        strength = "STRONG 💪"
    elif score >= 50:
        strength = "MEDIUM ⚡"
    else:
        strength = "WEAK ⚠️"
    
    return {
        "strength": strength,
        "score": min(score, 100),
        "factors": factors[:4]  # Top 4 factors
    }


def get_latest_signal(df: pd.DataFrame) -> dict:
    """Get the latest signal from the dataframe."""
    if df is None or df.empty:
        return {"signal": "none", "price": 0, "atr": 0, "strength": "NONE", "strength_score": 0}
    
    row = df.iloc[-1]
    signal = "none"
    signal_type = ""
    
    if row.get("trend_buy", False):
        signal = "buy"
        signal_type = "TREND"
    elif row.get("mr_buy", False):
        signal = "buy"
        signal_type = "MR"
    elif row.get("trend_sell", False):
        signal = "sell"
        signal_type = "TREND"
    elif row.get("mr_sell", False):
        signal = "sell"
        signal_type = "MR"
    
    # Calculate signal strength
    strength_info = calculate_signal_strength(row, signal, signal_type)
    
    return {
        "signal": signal,
        "signal_type": signal_type,
        "price": row["Close"],
        "atr": row.get("ATR", 0),
        "rsi": row.get("RSI", 0),
        "adx": row.get("ADX", 0),
        "stoch": row.get("Stoch_K", 0),
        "time": df.index[-1],
        "strength": strength_info["strength"],
        "strength_score": strength_info["score"],
        "strength_factors": strength_info["factors"]
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 10000.0,
    risk_per_trade: float = 0.02,
    pip_value: float = 0.0001,
) -> dict:
    """
    Run backtest with trailing stop and optimized parameters.
    
    Args:
        df: DataFrame with OHLC data and signals (must have indicators calculated)
        initial_balance: Starting balance in USD
        risk_per_trade: Risk per trade as decimal (0.02 = 2%)
        pip_value: Pip value for the instrument (0.0001 for EUR/USD)
    
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
        
        # Exit logic - using High/Low for realistic SL/TP simulation
        # This catches intra-bar spikes that Close price misses
        if position is not None:
            bars_in_trade += 1
            exit_price = None
            exit_reason = ""
            params = STRATEGY_PARAMS[signal_type]
            
            if position == "buy":
                # Use High for trailing (best price during bar)
                highest = max(highest, high)
                
                # Trailing stop - update based on highest price
                if highest - entry_price > params["TRAIL_START"] * atr:
                    new_sl = highest - params["TRAIL_DIST"] * atr
                    stop_loss = max(stop_loss, new_sl)
                
                # Check exits using Low (worst case for buy)
                # Simulate intra-bar spike that hits SL
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
                # Simulate intra-bar spike that hits SL
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
                risk_amount = balance * risk_per_trade
                params = STRATEGY_PARAMS[new_signal_type]
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
                    # If can't open new position, just close
                    position = None
                    
                continue  # Skip normal exit/entry logic for this bar
        
        # Entry logic
        if position is None:
            risk_amount = balance * risk_per_trade
            
            if row.get("trend_buy", False):
                signal_type = "TREND"
                params = STRATEGY_PARAMS["TREND"]
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
                params = STRATEGY_PARAMS["MR"]
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
                params = STRATEGY_PARAMS["TREND"]
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
                params = STRATEGY_PARAMS["MR"]
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
    
    # Count by signal type
    trend_trades = trades_df[trades_df['signal_type'] == 'TREND']
    mr_trades = trades_df[trades_df['signal_type'] == 'MR']
    
    return {
        'initial_balance': initial_balance,
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'total_return': (balance - initial_balance) / initial_balance * 100,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'final_balance': balance,
        'trend_trades': len(trend_trades),
        'mr_trades': len(mr_trades),
        'trades': trades,
        'equity_curve': equity_curve
    }
