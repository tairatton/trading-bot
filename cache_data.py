"""
Cache MT5 Data for Faster Backtests
Loads and saves historical data to avoid slow MT5 queries
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_bot"))

import MetaTrader5 as mt5
import pandas as pd
import pickle
from services.strategy import calculate_indicators, generate_signals

SYMBOLS = ["EURUSDm", "USDCADm", "USDJPYm"]
CACHE_FILE = "backtest_data_cache.pkl"

def load_and_cache_data():
    """Load data from MT5 and cache to file"""
    print("=" * 60)
    print("   MT5 DATA CACHING TOOL")
    print("=" * 60)
    
    if not mt5.initialize():
        print("[X] MT5 initialization failed")
        return
    
    print(f"\n[*] Loading {len(SYMBOLS)} symbols from MT5...")
    all_data = {}
    
    for symbol in SYMBOLS:
        print(f"  - Loading {symbol}...", end=" ")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, 70000)
        
        if rates is None or len(rates) == 0:
            print(f"[FAILED]")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        
        # Calculate indicators
        df = calculate_indicators(df)
        df = generate_signals(df)
        
        all_data[symbol] = df
        print(f"[OK] {len(df):,} bars")
    
    mt5.shutdown()
    
    # Save to file
    print(f"\n[*] Saving to {CACHE_FILE}...", end=" ")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(all_data, f)
    
    file_size = os.path.getsize(CACHE_FILE) / (1024 * 1024)
    print(f"[OK] ({file_size:.1f} MB)")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Cache created successfully!")
    print(f"   Use this cache in compare_bots.py to skip MT5 loading")
    print("=" * 60)

if __name__ == "__main__":
    load_and_cache_data()
