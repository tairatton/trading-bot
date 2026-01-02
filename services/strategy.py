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
# STRATEGY PARAMETERS
# ============================================================
STRATEGY_PARAMS = {
    "TREND": {
        "SL_ATR": 1.2,       # Stop Loss = 1.2 x ATR
        "TP_ATR": 3.5,       # Take Profit = 3.5 x ATR
        "TRAIL_START": 1.8,  # Start trailing after 1.8 x ATR profit
        "TRAIL_DIST": 1.2,   # Trail distance = 1.2 x ATR
        "MAX_BARS": 60       # Max bars in trade
    },
    "MR": {
        "SL_ATR": 1.0,
        "TP_ATR": 2.5,
        "TRAIL_START": 1.2,
        "TRAIL_DIST": 0.8,
        "MAX_BARS": 30
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
        (df["RSI"] > 40) & (df["RSI"] < 65)
    )
    
    df["trend_sell"] = (
        df["downtrend"] &
        (df["High"] >= df["SMA_20"]) &
        (df["Close"] < df["SMA_20"]) &
        (df["RSI"] > 35) & (df["RSI"] < 60)
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


def get_latest_signal(df: pd.DataFrame) -> dict:
    """Get the latest signal from the dataframe."""
    if df is None or df.empty:
        return {"signal": "none", "price": 0, "atr": 0}
    
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
    
    return {
        "signal": signal,
        "signal_type": signal_type,
        "price": row["Close"],
        "atr": row.get("ATR", 0),
        "rsi": row.get("RSI", 0),
        "adx": row.get("ADX", 0),
        "time": df.index[-1]
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
        atr = row["ATR"]
        
        # Exit logic
        if position is not None:
            bars_in_trade += 1
            exit_price = None
            exit_reason = ""
            params = STRATEGY_PARAMS[signal_type]
            
            if position == "buy":
                highest = max(highest, price)
                
                # Trailing stop
                if highest - entry_price > params["TRAIL_START"] * atr:
                    new_sl = highest - params["TRAIL_DIST"] * atr
                    stop_loss = max(stop_loss, new_sl)
                
                # Check exits
                if price <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                elif price >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                elif bars_in_trade >= params["MAX_BARS"]:
                    exit_price = price
                    exit_reason = "TIME"
                    
            elif position == "sell":
                lowest = min(lowest, price)
                
                # Trailing stop
                if entry_price - lowest > params["TRAIL_START"] * atr:
                    new_sl = lowest + params["TRAIL_DIST"] * atr
                    stop_loss = min(stop_loss, new_sl)
                
                # Check exits
                if price >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                elif price <= take_profit:
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
