"""Configuration settings loaded from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Settings:
    """Application settings from environment."""
    
    # MT5 Connection
    MT5_LOGIN: int = int(os.getenv('MT5_LOGIN', '0'))
    MT5_PASSWORD: str = os.getenv('MT5_PASSWORD', '')
    MT5_SERVER: str = os.getenv('MT5_SERVER', 'Exness-MT5Real')
    MT5_PATH: str = os.getenv('MT5_PATH', '')
    
    # Trading Parameters - Symbol Selection
    AVAILABLE_SYMBOLS = [
        "EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "USDCHFm",
        "USDCADm", "NZDUSDm", "EURGBPm", "EURJPYm", "GBPJPYm",
        "XAUUSDm", "XAGUSDm",  # Gold, Silver
        "BTCUSDm", "ETHUSDm",  # Crypto
    ]
    SYMBOL: str = os.getenv('SYMBOL', 'EURUSDm')  # Active trading symbol
    TIMEFRAME: str = os.getenv('TIMEFRAME', 'M30')
    RISK_PERCENT: float = float(os.getenv('RISK_PERCENT', '1.5'))
    SESSION_START_UTC: int = int(os.getenv('SESSION_START_UTC', '8'))
    SESSION_END_UTC: int = int(os.getenv('SESSION_END_UTC', '20'))
    
    # Strategy Parameters (Optimized - matching backtest.py)
    PARAMS = {
        "TREND": {"SL_ATR": 1.2, "TP_ATR": 3.5, "TRAIL_START": 1.8, "TRAIL_DIST": 1.2, "MAX_BARS": 60},
        "MR": {"SL_ATR": 1.0, "TP_ATR": 2.5, "TRAIL_START": 1.2, "TRAIL_DIST": 0.8, "MAX_BARS": 30}
    }
    
    # Spread Filter
    MAX_SPREAD_PIPS: float = float(os.getenv('MAX_SPREAD_PIPS', '3.0'))  # Skip order if spread > this
    
    # News Filter (High Impact Events)
    NEWS_BLACKOUT_MINUTES: int = 30
    
    # Web Dashboard
    WEB_HOST: str = os.getenv('WEB_HOST', '0.0.0.0')
    WEB_PORT: int = int(os.getenv('WEB_PORT', '8000'))
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'change-this-secret')
    
    # Telegram (Optional)
    TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    @classmethod
    def is_mt5_configured(cls) -> bool:
        """Check if MT5 credentials are set."""
        return cls.MT5_LOGIN > 0 and cls.MT5_PASSWORD != ''


settings = Settings()
