"""Data Service for market data and indicators."""
import pandas as pd
from datetime import datetime
from typing import Optional

# Import shared strategy module
from services.strategy import (
    STRATEGY_PARAMS,
    calculate_indicators as calc_indicators,
    generate_signals as gen_signals,
    get_latest_signal as get_signal
)


class DataService:
    """Service to fetch and process market data."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
    
    def fetch_data_from_mt5(self, mt5_service, symbol: str = None, count: int = 500) -> pd.DataFrame:
        """Fetch OHLC data from MT5 for a specific symbol."""
        data = mt5_service.get_ohlc_data(symbol=symbol, count=count)
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'tick_volume': 'Volume'
        }, inplace=True)
        
        self.df = df
        self.last_update = datetime.now()
        return df
    
    def calculate_indicators(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate all technical indicators using shared strategy module."""
        if df is None:
            df = self.df
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Use shared module
        df = calc_indicators(df)
        self.df = df
        return df
    
    def generate_signals(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate buy/sell signals using shared strategy module."""
        if df is None:
            df = self.df
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Use shared module
        df = gen_signals(df)
        self.df = df
        return df
    
    def get_latest_signal(self) -> dict:
        """Get the latest signal using shared strategy module."""
        return get_signal(self.df)


# Singleton instance
data_service = DataService()
