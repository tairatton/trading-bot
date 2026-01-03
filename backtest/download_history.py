import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import time

SYMBOLS = ["EURUSDm", "USDCADm", "USDJPYm"]
TIMEFRAME = mt5.TIMEFRAME_M30
YEARS = 10

def download_data():
    if not mt5.initialize():
        print("MT5 init failed")
        return

    print(f"[*] Downloading {YEARS} years of M30 data...")
    
    # Calculate start date (10 years ago)
    end_date = datetime.now()
    start_year = end_date.year - YEARS
    
    for symbol in SYMBOLS:
        print(f"\nProcessing {symbol}...")
        
        # Select symbol to ensure it's available
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select {symbol}")
            continue

        total_bars = 0
        current_date = datetime(start_year, 1, 1)
        
        # Download year by year to avoid timeouts
        for year in range(start_year, end_date.year + 1):
            date_from = datetime(year, 1, 1)
            date_to = datetime(year + 1, 1, 1)
            if date_to > end_date: date_to = end_date
            
            print(f"  > Fetching {year}...", end="\r")
            
            rates = mt5.copy_rates_range(symbol, TIMEFRAME, date_from, date_to)
            
            if rates is not None and len(rates) > 0:
                total_bars += len(rates)
            else:
                # Force synchronization (dummy call)
                mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 1)
                time.sleep(0.5) # Wait for server sync
        
        print(f"  > Done. Total bars found: {total_bars:,}")

    mt5.shutdown()
    print("\n[*] Download process completed!")
    print("    You can now run the backtest script.")

if __name__ == "__main__":
    download_data()
