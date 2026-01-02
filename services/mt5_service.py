"""MetaTrader 5 Service for Exness broker connection."""
from datetime import datetime
from typing import Optional, Dict, Any, List
from config import settings

# Try to import MT5 (only works on Windows)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[MT5] WARNING: MetaTrader5 not available")



class MT5Service:
    """Service to interact with MetaTrader 5 (Exness)."""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
    
    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        if not MT5_AVAILABLE:
            print("[MT5] Cannot connect - MetaTrader5 package not available")
            return False
        
        if not settings.is_mt5_configured():
            print("[MT5] Credentials not configured in .env")
            return False
        
        # Initialize MT5
        if not mt5.initialize(path=settings.MT5_PATH if settings.MT5_PATH else None):
            print(f"[MT5] Failed to initialize: {mt5.last_error()}")
            return False
        
        # Login to account
        authorized = mt5.login(
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER
        )
        
        if authorized:
            self.connected = True
            self.account_info = mt5.account_info()._asdict()
            print(f"[MT5] Connected to Exness - Account: {settings.MT5_LOGIN}")
            print(f"[MT5] Balance: ${self.account_info['balance']:.2f}")
            return True
        else:
            print(f"[MT5] Login failed: {mt5.last_error()}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5."""
        mt5.shutdown()
        self.connected = False
        print("[MT5] Disconnected")
    
    def is_connected(self) -> bool:
        """Check if MT5 connection is alive."""
        if not self.connected:
            return False
        
        # Try to get account info to verify connection
        try:
            info = mt5.account_info()
            if info is None:
                self.connected = False
                return False
            return True
        except:
            self.connected = False
            return False
    
    def ensure_connected(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """Ensure MT5 is connected, reconnect if needed."""
        import time
        
        # Check if already connected
        if self.is_connected():
            return True
        
        print("[MT5] Connection lost, attempting to reconnect...")
        
        # Try to reconnect
        for attempt in range(1, max_retries + 1):
            print(f"[MT5] Reconnect attempt {attempt}/{max_retries}...")
            
            # Shutdown first to clean up
            try:
                mt5.shutdown()
            except:
                pass
            
            # Wait before retry
            if attempt > 1:
                time.sleep(retry_delay)
            
            # Try to connect
            if self.connect():
                print(f"[MT5] Reconnected successfully on attempt {attempt}")
                return True
            else:
                print(f"[MT5] Reconnect attempt {attempt} failed")
        
        print(f"[MT5] Failed to reconnect after {max_retries} attempts")
        return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.connected:
            return {}
        info = mt5.account_info()
        return info._asdict() if info else {}
    
    def get_symbol_info(self, symbol: str = None) -> Dict[str, Any]:
        """Get symbol information."""
        symbol = symbol or settings.SYMBOL
        info = mt5.symbol_info(symbol)
        if info:
            return info._asdict()
        return {}
    
    def get_current_price(self, symbol: str = None) -> Dict[str, float]:
        """Get current bid/ask prices."""
        symbol = symbol or settings.SYMBOL
        
        # Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol: {symbol}")
            return {}
            
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return {"bid": tick.bid, "ask": tick.ask, "time": datetime.fromtimestamp(tick.time)}
        return {}
    
    def get_current_spread(self, symbol: str = None) -> float:
        """Get current spread in pips."""
        symbol = symbol or settings.SYMBOL
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            spread = tick.ask - tick.bid
            # Convert to pips (for EURUSD: 1 pip = 0.0001)
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                point = symbol_info.point
                # Pip size is usually 10 points for 5-digit brokers
                pip_size = point * 10 if symbol_info.digits == 5 or symbol_info.digits == 3 else point
                return spread / pip_size
        return 0.0
    
    def get_ohlc_data(self, symbol: str = None, timeframe: str = None, count: int = 500) -> List[Dict]:
        """Get OHLC candle data."""
        symbol = symbol or settings.SYMBOL
        
        # Map timeframe string to MT5 constant
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1
        }
        tf = tf_map.get(timeframe or settings.TIMEFRAME, mt5.TIMEFRAME_H1)
        
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is not None:
            return [dict(zip(['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'], r)) for r in rates]
        return []
    
    def place_order(self, order_type: str, symbol: str = None, volume: float = 0.01, 
                    sl: float = 0, tp: float = 0, comment: str = "") -> Dict[str, Any]:
        """Place a market order."""
        symbol = symbol or settings.SYMBOL
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"success": False, "error": "Cannot get price"}
        
        # Determine order type and price
        if order_type.lower() == "buy":
            trade_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            trade_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        # Create order request (without SL/TP if they are 0)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Only add SL/TP if they are valid (non-zero)
        if sl > 0:
            request["sl"] = sl
        if tp > 0:
            request["tp"] = tp
        
        result = mt5.order_send(request)
        
        # Check if result is None (order_send failed completely)
        if result is None:
            last_error = mt5.last_error()
            return {
                "success": False,
                "error": f"Order send failed: {last_error}"
            }
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {
                "success": True,
                "order_id": result.order,
                "price": result.price,
                "volume": result.volume
            }
        else:
            return {
                "success": False,
                "error": f"Order failed: {result.retcode} - {result.comment}"
            }
    
    def close_position(self, position_id: int) -> Dict[str, Any]:
        """Close an open position."""
        positions = mt5.positions_get(ticket=position_id)
        if not positions:
            return {"success": False, "error": "Position not found"}
        
        pos = positions[0]
        symbol = pos.symbol
        volume = pos.volume
        
        # Opposite order to close
        if pos.type == mt5.POSITION_TYPE_BUY:
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "position": position_id,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "order_id": result.order}
        else:
            return {"success": False, "error": f"Close failed: {result.retcode}"}
    
    def modify_position_sl_tp(self, position_id: int, sl: float = None, tp: float = None) -> Dict[str, Any]:
        """Modify SL/TP of an existing position."""
        positions = mt5.positions_get(ticket=position_id)
        if not positions:
            return {"success": False, "error": "Position not found"}
        
        pos = positions[0]
        symbol = pos.symbol
        
        # Use current SL/TP if not specified
        new_sl = sl if sl is not None else pos.sl
        new_tp = tp if tp is not None else pos.tp
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position_id,
            "sl": new_sl,
            "tp": new_tp,
            "magic": 123456,
            "comment": "Modify SL/TP",
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            last_error = mt5.last_error()
            return {
                "success": False,
                "error": f"Modify SL/TP failed: {last_error}"
            }
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {
                "success": True,
                "order_id": result.order,
                "sl": new_sl,
                "tp": new_tp
            }
        else:
            return {
                "success": False,
                "error": f"Modify failed: {result.retcode} - {result.comment}"
            }

    
    def get_open_positions(self, symbol: str = None) -> List[Dict]:
        """Get open positions. If symbol=None, gets all positions."""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()  # Get all positions
        if positions:
            return [pos._asdict() for pos in positions]
        return []
    
    def get_history_deals(self, days: int = 7) -> List[Dict]:
        """Get recent closed deal history from MT5."""
        from datetime import timedelta
        
        # Get deals from the last N days
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        deals = mt5.history_deals_get(from_date, to_date)
        if deals:
            result = []
            for deal in deals:
                d = deal._asdict()
                # Only include exit deals (closed positions)
                # entry: 0=IN, 1=OUT, 2=INOUT, 3=OUT_BY
                # type: 0=buy, 1=sell
                if d.get('type') in [0, 1] and d.get('entry') == 1:  # entry=1 means OUT (close)
                    result.append({
                        "ticket": d.get('ticket'),
                        "time": datetime.fromtimestamp(d.get('time', 0)).strftime("%Y-%m-%d %H:%M"),
                        "type": "SELL" if d.get('type') == 0 else "BUY",  # Flip: closing BUY means original was SELL
                        "symbol": d.get('symbol'),
                        "volume": d.get('volume'),
                        "price": d.get('price'),
                        "profit": d.get('profit'),
                        "comment": d.get('comment', '')
                    })
            return result[-20:]  # Return last 20 closed deals
        return []


# Singleton instance
mt5_service = MT5Service()
