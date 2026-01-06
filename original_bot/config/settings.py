"""Configuration settings loaded from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Settings:
    """Application settings from environment."""
    
    # Bot Identification
    BOT_NAME: str = "Original Bot"
    BOT_TYPE: str = "ORIGINAL"  # PROP_FIRM or ORIGINAL
    
    # MT5 Connection (Single Account from .env)
    @staticmethod
    def _find_mt5_path() -> str:
        """Auto-detect MT5 terminal path."""
        import os
        from pathlib import Path
        
        # Common MT5 installation paths
        possible_paths = [
            # Default Roaming install (most common for Exness)
            Path(os.getenv('APPDATA', '')) / "MetaTrader 5" / "terminal64.exe",
            # Program Files installs
            Path("C:/Program Files/MetaTrader 5/terminal64.exe"),
            Path("C:/Program Files (x86)/MetaTrader 5/terminal64.exe"),
            # Exness specific
            Path(os.getenv('APPDATA', '')) / "Exness-MT5" / "terminal64.exe",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return ""  # Let MT5 use default if not found
    
    MT5_LOGIN: int = int(os.getenv('MT5_LOGIN', '0'))
    MT5_PASSWORD: str = os.getenv('MT5_PASSWORD', '')
    MT5_SERVER: str = os.getenv('MT5_SERVER', 'Exness-MT5Real')
    MT5_PATH: str = os.getenv('MT5_PATH', '') or _find_mt5_path.__func__()
    
    # Telegram Settings (from .env)
    TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
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
    RISK_PERCENT: float = float(os.getenv('RISK_PERCENT', '1.0'))
    SESSION_START_UTC: int = int(os.getenv('SESSION_START_UTC', '8'))
    SESSION_END_UTC: int = int(os.getenv('SESSION_END_UTC', '20'))
    
    # Weekend Safety: Close all trades on Friday to avoid gap risk
    CLOSE_ON_FRIDAY: bool = os.getenv('CLOSE_ON_FRIDAY', 'true').lower() == 'true'
    FRIDAY_CLOSE_HOUR_UTC: int = int(os.getenv('FRIDAY_CLOSE_HOUR_UTC', '21'))  # 21:00 UTC = 04:00 TH (Sat)
    
    # Note: Strategy PARAMS moved to services/strategy.py for simplicity
    # PARAMS is no longer loaded from .env
    
    
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
    

    
    # Web Dashboard
    WEB_HOST: str = os.getenv('WEB_HOST', '0.0.0.0')
    WEB_PORT: int = int(os.getenv('WEB_PORT', '8000'))
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'change-this-secret')
    
    # Telegram (Optional)
    # Telegram (Optional) - Loaded from .env
    ENABLE_SIGNAL_ALERTS: bool = os.getenv('ENABLE_SIGNAL_ALERTS', 'false').lower() == 'true'
    
    @classmethod
    def is_mt5_configured(cls) -> bool:
        """Check if MT5 credentials are set."""
        return cls.MT5_LOGIN > 0 and cls.MT5_PASSWORD != ''


settings = Settings()
