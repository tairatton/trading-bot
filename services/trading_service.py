"""Trading Service - main bot logic."""
import threading
import time
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
from config import settings
from services.telegram_service import telegram_service

# Setup file logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger("TradingBot")


class TradingService:
    """Main trading bot service."""
    
    def __init__(self):
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        self.last_signal_time: Optional[datetime] = None
        self.current_position: Optional[Dict] = None
        self.trade_history: List[Dict] = []
        self.stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "last_update": None,
            "last_action": "Waiting..."
        }
        
        # Trailing stop tracking
        self.position_entry_price: float = 0
        self.position_highest: float = 0
        self.position_lowest: float = float('inf')
        self.position_entry_time: Optional[datetime] = None  # Track entry time for time exit
        self.position_signal_type: str = ""
        self.position_original_sl: float = 0
        
        # Time exit setting (hours) - different from backtest for faster exits
        self.max_hours_in_trade: float = 10.0  # 10 hours timeout
        
        # Multi-symbol position tracking {symbol: {...}}
        self.positions_tracking: Dict[str, Dict] = {}
        
        # Error tracking for browser notification
        self.last_error: Optional[str] = None
        
        # News events (UTC times)
        self.high_impact_news = [
            "2025-01-10 13:30", "2025-02-07 13:30", "2025-03-07 13:30",
            "2025-04-04 13:30", "2025-05-02 13:30", "2025-06-06 13:30",
            "2025-07-03 13:30", "2025-08-01 13:30", "2025-09-05 13:30",
            "2025-10-03 13:30", "2025-11-07 13:30", "2025-12-05 13:30",
            "2025-01-29 19:00", "2025-03-19 18:00", "2025-05-07 18:00",
            "2025-06-18 18:00", "2025-07-30 18:00", "2025-09-17 18:00",
        ]
    
    def is_news_time(self, current_time: datetime) -> bool:
        """Check if during news blackout."""
        for news_str in self.high_impact_news:
            try:
                news_time = pd.to_datetime(news_str)
                time_diff = abs((current_time - news_time).total_seconds() / 60)
                if time_diff <= settings.NEWS_BLACKOUT_MINUTES:
                    return True
            except:
                pass
        return False
    
    def is_session_time(self, current_time: datetime) -> bool:
        """Check if within trading session."""
        hour_utc = current_time.hour
        if current_time.weekday() >= 5:  # Weekend
            return False
        return settings.SESSION_START_UTC <= hour_utc < settings.SESSION_END_UTC
    
    def can_trade(self, current_time: datetime) -> tuple:
        """Check if trading is allowed.
        
        Note: Session filter disabled for 24-hour trading.
        Backtest showed 24h trading has higher returns (+3125% vs +2747%).
        """
        if self.is_news_time(current_time):
            return False, "News blackout"
        # Session filter disabled - trade 24 hours
        return True, "OK"
    
    def get_last_error(self) -> Optional[str]:
        """Get and clear last error."""
        error = self.last_error
        self.last_error = None  # Clear after reading
        return error
    
    def calculate_position_size(self, balance: float, atr: float, signal_type: str) -> float:
        """Calculate position size based on risk."""
        risk_amount = balance * (settings.RISK_PERCENT / 100)
        sl_atr = settings.PARAMS[signal_type]["SL_ATR"]
        sl_distance = atr * sl_atr
        
        if sl_distance <= 0:
            return 0.01  # Minimum lot
        
        # For EURUSD: 1 lot = 100,000 units, 1 pip = $10
        # position_size in terms of price movement
        position_units = risk_amount / sl_distance
        
        # Convert to lots (approximate for EURUSD)
        # 1 mini lot (0.1) = 10,000 units = $1/pip
        lots = position_units / 10000 * 0.1
        
        # Round to 2 decimals, minimum 0.01
        lots = max(0.01, round(lots, 2))
        return lots
    
    def process_signal(self, mt5_service, data_service, signal_info: Dict, symbol: str) -> Dict:
        """Process a trading signal and place order if conditions met."""
        signal = signal_info.get("signal")
        signal_type = signal_info.get("signal_type")
        price = signal_info.get("price")
        atr = signal_info.get("atr")
        
        if signal == "none":
            return {"action": "none", "reason": "No signal"}
        
        # Get account info
        account = mt5_service.get_account_info()
        balance = account.get("balance", 500)
        
        # Check if can trade
        can, reason = self.can_trade(datetime.utcnow())
        if not can:
            self.stats["last_action"] = f"Skipped: {reason}"
            return {"action": "skip", "reason": reason}
        
        # Check if already have position for THIS symbol
        positions = mt5_service.get_open_positions(symbol=symbol)
        if positions:
            # SIGNAL REVERSAL: Check if signal is opposite to current position
            current_pos = positions[0]
            current_type = "buy" if current_pos["type"] == 0 else "sell"
            
            # If signal is opposite, close current and continue to open new
            if (current_type == "buy" and signal == "sell") or (current_type == "sell" and signal == "buy"):
                logger.info(f"[{symbol}] SIGNAL REVERSAL: Closing {current_type.upper()} to open {signal.upper()}")
                close_result = mt5_service.close_position(current_pos["ticket"])
                
                if close_result.get("success"):
                    self.stats["last_action"] = f"REVERSAL: Closed {current_type.upper()}"
                    logger.info(f"[{symbol}] Position closed for reversal")
                    # Continue to open new position below
                else:
                    self.stats["last_action"] = f"REVERSAL FAILED: {close_result.get('error')}"
                    return {"action": "failed", "error": f"Failed to close for reversal: {close_result.get('error')}"}
            else:
                # Same direction signal, skip
                self.stats["last_action"] = f"Skipped: Position open ({symbol})"
                return {"action": "skip", "reason": f"Position open for {symbol}"}
        
        # Check spread filter
        # Check spread filter
        current_spread = mt5_service.get_current_spread(symbol=symbol)
        if current_spread > settings.MAX_SPREAD_PIPS:
            logger.warning(f"[{symbol}] SPREAD FILTER: {current_spread:.1f} pips > Max {settings.MAX_SPREAD_PIPS}")
            self.stats["last_action"] = f"Skipped: Spread {current_spread:.1f} > {settings.MAX_SPREAD_PIPS}"
            return {"action": "skip", "reason": f"Spread too high ({current_spread:.1f} pips)"}
        
        # Calculate position size
        lots = self.calculate_position_size(balance, atr, signal_type)
        
        # Calculate SL/TP
        params = settings.PARAMS[signal_type]
        sl_distance = atr * params["SL_ATR"]
        tp_distance = atr * params["TP_ATR"]
        
        if signal == "buy":
            sl = price - sl_distance
            tp = price + tp_distance
        else:
            sl = price + sl_distance
            tp = price - tp_distance
        
        # Place order
        logger.info(f"[{symbol}] PLACING ORDER: {signal.upper()} {lots} lots @ {price:.5f}")
        result = mt5_service.place_order(
            order_type=signal,
            symbol=symbol,
            volume=lots,
            sl=sl,
            tp=tp,
            comment=f"Bot_{signal_type}"
        )
        
        if result.get("success"):
            self.stats["total_trades"] += 1
            self.last_signal_time = datetime.now()
            self.trade_history.append({
                "time": datetime.now(),
                "signal": signal,
                "type": signal_type,
                "price": result.get("price"),
                "lots": lots,
                "sl": sl,
                "tp": tp
            })
            logger.info(f"ORDER SUCCESS: {signal.upper()} @ {result.get('price'):.5f}, ID={result.get('order_id')}")
            self.stats["last_action"] = f"OPENED {signal.upper()} @ {result.get('price'):.5f}"
            
            # Send Telegram notification
            telegram_service.notify_trade_opened(
                symbol=symbol,
                trade_type=signal,
                price=result.get("price", price),
                lots=lots,
                sl=sl,
                tp=tp,
                signal_type=signal_type
            )
            
            return {"action": "opened", "result": result}
        else:
            error_msg = result.get("error", "Unknown error")
            self.last_error = error_msg  # Store for browser notification
            logger.error(f"ORDER FAILED: {error_msg}")
            self.stats["last_action"] = f"FAILED: {error_msg}"
            
            # Send Telegram error notification
            telegram_service.notify_error(f"Order failed for {symbol}: {error_msg}")
            
            return {"action": "failed", "error": error_msg}
    
    def monitor_positions(self, mt5_service, data_service) -> None:
        """Monitor open positions for trailing stop and time-based exit."""
        from services.strategy import STRATEGY_PARAMS
        
        positions = mt5_service.get_open_positions()
        if not positions:
            # No positions, reset tracking
            if self.current_position:
                self.current_position = None
                self.position_entry_price = 0
                self.position_highest = 0
                self.position_lowest = float('inf')
                self.position_entry_time = None
                self.position_signal_type = ""
                self.position_original_sl = 0
            return
        
        # Get current price and ATR
        current_price_info = mt5_service.get_current_price()
        if not current_price_info:
            return
        
        # Get latest data for ATR
        df = data_service.fetch_data_from_mt5(mt5_service, count=50)
        if df.empty:
            return
        df = data_service.calculate_indicators(df)
        current_atr = df.iloc[-1]["ATR"]
        
        for pos in positions:
            ticket = pos["ticket"]
            pos_type = "buy" if pos["type"] == 0 else "sell"
            entry_price = pos["price_open"]
            current_sl = pos["sl"]
            current_tp = pos["tp"]
            
            # Initialize tracking if this is a new position
            if self.current_position is None or self.current_position.get("ticket") != ticket:
                # Extract signal type from comment (e.g., "Bot_TREND" or "Bot_MR")
                comment = pos.get("comment", "")
                if "TREND" in comment:
                    signal_type = "TREND"
                elif "MR" in comment:
                    signal_type = "MR"
                else:
                    signal_type = "TREND"  # Default
                
                self.current_position = {"ticket": ticket, "type": pos_type}
                self.position_entry_price = entry_price
                self.position_signal_type = signal_type
                self.position_original_sl = current_sl
                self.position_entry_time = datetime.now()  # Record entry time
                
                if pos_type == "buy":
                    self.position_highest = entry_price
                    self.position_lowest = float('inf')
                else:
                    self.position_lowest = entry_price
                    self.position_highest = 0
            
            # Calculate hours in trade (real time based)
            hours_in_trade = 0.0
            if self.position_entry_time:
                hours_in_trade = (datetime.now() - self.position_entry_time).total_seconds() / 3600
            
            # Get parameters for this signal type
            params = STRATEGY_PARAMS.get(self.position_signal_type, STRATEGY_PARAMS["TREND"])
            
            # Get current price
            if pos_type == "buy":
                current_price = current_price_info.get("bid", 0)
            else:
                current_price = current_price_info.get("ask", 0)
            
            if current_price == 0:
                continue
            
            # Update highest/lowest
            if pos_type == "buy":
                self.position_highest = max(self.position_highest, current_price)
            else:
                self.position_lowest = min(self.position_lowest, current_price)
            
            # Check for TIME-BASED EXIT (using real hours converted from MAX_BARS)
            max_bars = params["MAX_BARS"]
            
            # Convert bars to hours based on timeframe
            tf_multiplier = 1.0 # Default H1
            if settings.TIMEFRAME == "M30":
                tf_multiplier = 0.5
            elif settings.TIMEFRAME == "M15":
                tf_multiplier = 0.25
            
            max_hours = max_bars * tf_multiplier
            
            if hours_in_trade >= max_hours:
                logger.info(f"TIME EXIT: Position held {hours_in_trade:.1f}/{max_hours} hours ({max_bars} bars)")
                result = mt5_service.close_position(ticket)
                if result.get("success"):
                    logger.info(f"POSITION CLOSED: Time-based exit")
                    self.stats["total_trades"] += 1
                else:
                    logger.error(f"CLOSE FAILED: {result.get('error')}")
                continue
            
            # Check for TRAILING STOP
            new_sl = None
            
            if pos_type == "buy":
                profit = self.position_highest - self.position_entry_price
                # Check if profit exceeds TRAIL_START threshold
                if profit >= params["TRAIL_START"] * current_atr:
                    # Calculate new trailing SL
                    new_sl = self.position_highest - params["TRAIL_DIST"] * current_atr
                    # Only update if new SL is higher than current SL
                    if new_sl > current_sl:
                        logger.info(f"TRAILING STOP: SL {current_sl:.5f} -> {new_sl:.5f} (+{profit*10000:.1f} pips)")
                        result = mt5_service.modify_position_sl_tp(ticket, sl=new_sl)
                        if result.get("success"):
                            logger.info(f"SL UPDATED: {new_sl:.5f}")
                        else:
                            logger.error(f"SL UPDATE FAILED: {result.get('error')}")
            
            elif pos_type == "sell":
                profit = self.position_entry_price - self.position_lowest
                # Check if profit exceeds TRAIL_START threshold
                if profit >= params["TRAIL_START"] * current_atr:
                    # Calculate new trailing SL
                    new_sl = self.position_lowest + params["TRAIL_DIST"] * current_atr
                    # Only update if new SL is lower than current SL
                    if new_sl < current_sl or current_sl == 0:
                        logger.info(f"TRAILING STOP: SL {current_sl:.5f} -> {new_sl:.5f} (+{profit*10000:.1f} pips)")
                        result = mt5_service.modify_position_sl_tp(ticket, sl=new_sl)
                        if result.get("success"):
                            logger.info(f"SL UPDATED: {new_sl:.5f}")
                        else:
                            logger.error(f"SL UPDATE FAILED: {result.get('error')}")

    
    def run_loop(self, mt5_service, data_service):
        """Main trading loop."""
        logger.info("========== TRADING BOT STARTED ==========")
        logger.info(f"Active symbol: {settings.SYMBOL}")
        logger.info(f"Scanning all symbols: {settings.AVAILABLE_SYMBOLS}")
        
        # Track last signal per symbol to avoid duplicate alerts
        last_signals = {}
        
        while self.running:
            if self.paused:
                time.sleep(5)
                continue
            
            try:
                # Ensure MT5 connection is alive (auto reconnect if needed)
                if not mt5_service.ensure_connected():
                    print("[BOT] MT5 connection failed, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                
                # Monitor existing positions (trailing stop & time-based exit)
                self.monitor_positions(mt5_service, data_service)
                
                # Scan ALL symbols for signals
                for symbol in settings.AVAILABLE_SYMBOLS:
                    try:
                        # Fetch data for this symbol
                        df = data_service.fetch_data_from_mt5(mt5_service, symbol=symbol, count=300)
                        if df.empty:
                            continue
                        
                        # Calculate indicators and signals
                        df = data_service.calculate_indicators(df)
                        df = data_service.generate_signals(df)
                        
                        # Get latest signal
                        signal_info = data_service.get_latest_signal()
                        
                        if signal_info["signal"] != "none":
                            # Check if this is a new signal (avoid duplicate alerts)
                            signal_key = f"{symbol}_{signal_info['signal']}_{signal_info['signal_type']}"
                            
                            if signal_key != last_signals.get(symbol):
                                # New signal detected - send Telegram alert
                                is_active = (symbol == settings.SYMBOL)
                                
                                telegram_service.notify_signal_detected(
                                    symbol=symbol,
                                    signal=signal_info["signal"],
                                    signal_type=signal_info["signal_type"],
                                    price=signal_info["price"],
                                    rsi=signal_info.get("rsi", 0),
                                    adx=signal_info.get("adx", 0),
                                    is_active_symbol=is_active,
                                    strength=signal_info.get("strength", ""),
                                    strength_score=signal_info.get("strength_score", 0),
                                    strength_factors=signal_info.get("strength_factors", [])
                                )
                                
                                last_signals[symbol] = signal_key
                                logger.info(f"[{symbol}] Signal: {signal_info['signal'].upper()} ({signal_info['signal_type']}) - Alert sent")
                            
                            # Only trade if this is the active symbol
                            if symbol == settings.SYMBOL:
                                result = self.process_signal(mt5_service, data_service, signal_info, symbol)
                                print(f"[{symbol}] ACTIVE: {signal_info['signal'].upper()} ({signal_info['signal_type']}) - {result['action']}")
                        else:
                            # Clear last signal if no signal
                            last_signals[symbol] = None
                            
                    except Exception as e:
                        logger.warning(f"[{symbol}] Error scanning: {e}")
                        continue
                
                # Update stats
                self.stats["last_update"] = datetime.now().isoformat()
                
                # Wait for next check (check every 5 minutes for M30 candle updates)
                time.sleep(300)
                
            except Exception as e:
                print(f"[BOT] ERROR: {e}")
                time.sleep(60)
        
        print("[BOT] Trading bot stopped")
    
    def start(self, mt5_service, data_service):
        """Start the trading bot."""
        if self.running:
            return {"success": False, "message": "Already running"}
        
        self.running = True
        self.paused = False
        self.thread = threading.Thread(
            target=self.run_loop,
            args=(mt5_service, data_service),
            daemon=True
        )
        self.thread.start()
        
        # Send Telegram notification
        telegram_service.notify_bot_started(settings.SYMBOL, settings.RISK_PERCENT)
        
        return {"success": True, "message": "Bot started"}
    
    def stop(self):
        """Stop the trading bot."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        # Send Telegram notification
        telegram_service.notify_bot_stopped()
        
        return {"success": True, "message": "Bot stopped"}
    
    def pause(self):
        """Pause trading."""
        self.paused = True
        return {"success": True, "message": "Bot paused"}
    
    def resume(self):
        """Resume trading."""
        self.paused = False
        return {"success": True, "message": "Bot resumed"}
    
    def get_status(self) -> Dict:
        """Get bot status."""
        return {
            "running": self.running,
            "paused": self.paused,
            "stats": self.stats,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "trade_history": self.trade_history[-10:]  # Last 10 trades
        }


# Singleton instance
trading_service = TradingService()
