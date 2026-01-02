"""
Test Script - Check M30 Data Availability in MT5
================================================
Check how far back we can fetch M30 data by fetching month by month.
"""
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

def check_data_depth():
    """Check how far back M30 data is available."""
    print("=" * 60)
    print("  Checking M30 Data Availability in MT5")
    print("=" * 60)
    
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return
    
    symbol = "EURUSDm"
    timeframe = mt5.TIMEFRAME_M30
    
    # Try to fetch oldest available data
    print(f"\n[*] Symbol: {symbol}")
    print(f"[*] Timeframe: M30")
    
    # Method 1: Try fetching maximum bars at once
    print("\n--- Method 1: Single fetch ---")
    for bars in [10000, 50000, 100000, 200000]:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is not None and len(rates) > 0:
            oldest = datetime.fromtimestamp(rates[0]['time'])
            newest = datetime.fromtimestamp(rates[-1]['time'])
            print(f"  {bars:,} bars: {oldest.date()} to {newest.date()} ({len(rates):,} bars received)")
        else:
            print(f"  {bars:,} bars: FAILED")
    
    # Method 2: Fetch by year going backwards
    print("\n--- Method 2: Fetch by year ---")
    all_data = []
    current_year = datetime.now().year
    
    for year in range(current_year, current_year - 15, -1):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59)
        
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is not None and len(rates) > 0:
            print(f"  {year}: {len(rates):,} bars available")
            all_data.append(pd.DataFrame(rates))
        else:
            print(f"  {year}: No data available - LIMIT REACHED")
            break
    
    # Summary
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined['time'] = pd.to_datetime(combined['time'], unit='s')
        oldest = combined['time'].min()
        newest = combined['time'].max()
        
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"  Total bars available: {len(combined):,}")
        print(f"  Oldest data: {oldest}")
        print(f"  Newest data: {newest}")
        print(f"  Time span: {(newest - oldest).days / 365:.1f} years")
        print("=" * 60)
    
    mt5.shutdown()

if __name__ == "__main__":
    check_data_depth()
