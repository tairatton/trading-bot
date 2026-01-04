"""Configuration settings loaded from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Settings:
    """Application settings from environment."""
    
    # MT5 Connection (Loaded from accounts.py)
    try:
        import sys
        # Add parent directory to path to find accounts.py
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from accounts import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    except ImportError:
        # Fallback to env or empty
        MT5_LOGIN = int(os.getenv('MT5_LOGIN', '0'))
        MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
        MT5_SERVER = os.getenv('MT5_SERVER', 'Exness-MT5Real')
        MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\MetaTrader 5\terminal64.exe")
        TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Trading Parameters - Symbol Selection
    AVAILABLE_SYMBOLS = [
        "EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "USDCHFm",
        "USDCADm", "NZDUSDm", "EURGBPm", "EURJPYm", "GBPJPYm",
        "XAUUSDm", "XAGUSDm",  # Gold, Silver
        "BTCUSDm", "ETHUSDm",  # Crypto
    ]
    
    # Active symbols for trading (Top 3 by Profit Factor)
    # Supports trading multiple symbols with 1 order per symbol max
    ACTIVE_SYMBOLS_STR: str = os.getenv('ACTIVE_SYMBOLS', 'EURUSDm,USDCADm,USDCHFm')
    ACTIVE_SYMBOLS: list = [s.strip() for s in ACTIVE_SYMBOLS_STR.split(',') if s.strip()]
    
    SYMBOL: str = os.getenv('SYMBOL', 'EURUSDm')  # Legacy: Default single symbol
    TIMEFRAME: str = os.getenv('TIMEFRAME', 'M30')
    RISK_PERCENT: float = float(os.getenv('RISK_PERCENT', '1.5'))
    SESSION_START_UTC: int = int(os.getenv('SESSION_START_UTC', '8'))
    SESSION_END_UTC: int = int(os.getenv('SESSION_END_UTC', '20'))
    
    # Weekend Safety: Close all trades on Friday to avoid gap risk
    CLOSE_ON_FRIDAY: bool = os.getenv('CLOSE_ON_FRIDAY', 'true').lower() == 'true'
    FRIDAY_CLOSE_HOUR_UTC: int = int(os.getenv('FRIDAY_CLOSE_HOUR_UTC', '21'))  # 21:00 UTC = 04:00 TH (Sat)
    
    # Strategy Parameters (Configurable via .env)
    PARAMS = {
        "TREND": {
            "SL_ATR": float(os.getenv('TREND_SL_ATR', '1.2')),
            "TP_ATR": float(os.getenv('TREND_TP_ATR', '3.5')),
            "TRAIL_START": float(os.getenv('TREND_TRAIL_START', '1.8')),
            "TRAIL_DIST": float(os.getenv('TREND_TRAIL_DIST', '1.2')),
            "MAX_BARS": int(os.getenv('TREND_MAX_BARS', '60'))
        },
        "MR": {
            "SL_ATR": float(os.getenv('MR_SL_ATR', '1.0')),
            "TP_ATR": float(os.getenv('MR_TP_ATR', '2.5')),
            "TRAIL_START": float(os.getenv('MR_TRAIL_START', '1.2')),
            "TRAIL_DIST": float(os.getenv('MR_TRAIL_DIST', '0.8')),
            "MAX_BARS": int(os.getenv('MR_MAX_BARS', '30'))
        }
    }
    
    # Spread Filter - Per Asset Class
    MAX_SPREAD_FOREX: float = float(os.getenv('MAX_SPREAD_FOREX', '3.0'))   # Forex pairs
    MAX_SPREAD_METALS: float = float(os.getenv('MAX_SPREAD_METALS', '50.0'))  # Gold/Silver
    MAX_SPREAD_CRYPTO: float = float(os.getenv('MAX_SPREAD_CRYPTO', '5000.0'))  # BTC/ETH
    MAX_SPREAD_PIPS: float = MAX_SPREAD_FOREX  # Legacy fallback
    
    @classmethod
    def get_max_spread(cls, symbol: str) -> float:
        """Get max spread limit for a symbol based on asset class."""
        symbol_upper = symbol.upper()
        if "BTC" in symbol_upper or "ETH" in symbol_upper:
            return cls.MAX_SPREAD_CRYPTO
        elif "XAU" in symbol_upper or "XAG" in symbol_upper:
            return cls.MAX_SPREAD_METALS
        return cls.MAX_SPREAD_FOREX
    
    # News Filter (High Impact Events)
    NEWS_BLACKOUT_MINUTES: int = int(os.getenv('NEWS_BLACKOUT_MINUTES', '30'))
    
    # Web Dashboard
    WEB_HOST: str = os.getenv('WEB_HOST', '0.0.0.0')
    WEB_PORT: int = int(os.getenv('WEB_PORT', '8000'))
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'change-this-secret')
    
    # Telegram (Optional)
    # Token loaded above from accounts.py
    ENABLE_SIGNAL_ALERTS: bool = os.getenv('ENABLE_SIGNAL_ALERTS', 'false').lower() == 'true'
    
    @classmethod
    def is_mt5_configured(cls) -> bool:
        """Check if MT5 credentials are set."""
        return cls.MT5_LOGIN > 0 and cls.MT5_PASSWORD != ''


settings = Settings()
