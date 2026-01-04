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
        
        # Check spread filter - use per-symbol limit
        current_spread = mt5_service.get_current_spread(symbol=symbol)
        max_spread = settings.get_max_spread(symbol)
        if current_spread > max_spread:
            logger.warning(f"[{symbol}] SPREAD FILTER: {current_spread:.1f} pips > Max {max_spread}")
            self.stats["last_action"] = f"Skipped: Spread {current_spread:.1f} > {max_spread}"
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
    
    def monitor_all_accounts(self, mt5_service, data_service) -> None:
        """Monitor open positions for ALL accounts."""
        from services.strategy import STRATEGY_PARAMS
        
        # Get list of accounts
        accounts = mt5_service.accounts
        if not accounts:
            logger.warning("No accounts configured for monitoring.")
            return

        for acc in accounts:
            try:
                # Login to account
                if not mt5_service.login_account(acc):
                    logger.error(f"[MONITOR] Failed to login to {acc['name']}")
                    continue
                
                # Get Account ID
                acc_id = str(acc['login'])
                
                # Initialize tracking for this account if needed
                if acc_id not in self.positions_tracking:
                    self.positions_tracking[acc_id] = {}
                
                # Get Open Positions
                positions = mt5_service.get_open_positions()
                
                # If no positions, clear tracking for this account
                if not positions:
                    self.positions_tracking[acc_id] = {}
                    continue
                
                # Check each position
                for pos in positions:
                    ticket = pos["ticket"]
                    pos_symbol = pos.get("symbol", settings.SYMBOL)
                    pos_type = "buy" if pos["type"] == 0 else "sell"
                    entry_price = pos["price_open"]
                    current_sl = pos["sl"]
                    
                    # Track Position Details
                    if ticket not in self.positions_tracking[acc_id]:
                        # Infer signal type from comment
                        comment = pos.get("comment", "")
                        signal_type = "MR" if "MR" in comment else "TREND"
                        
                        self.positions_tracking[acc_id][ticket] = {
                            "entry_price": entry_price,
                            "highest": entry_price if pos_type == "buy" else 0,
                            "lowest": entry_price if pos_type == "sell" else float('inf'),
                            "entry_time": datetime.now(), # Approximate if just started
                            "signal_type": signal_type,
                            "type": pos_type
                        }
                    
                    track_info = self.positions_tracking[acc_id][ticket]
                    
                    # Logic: Trailing Stop & Time Exit
                    # Get Current Price
                    price_info = mt5_service.get_current_price(pos_symbol)
                    if not price_info: continue
                    
                    current_price = price_info["bid"] if pos_type == "buy" else price_info["ask"]
                    
                    # Update High/Low
                    if pos_type == "buy":
                        track_info["highest"] = max(track_info["highest"], current_price)
                    else:
                        track_info["lowest"] = min(track_info["lowest"], current_price)
                    
                    # Get ATR
                    df = data_service.fetch_data_from_mt5(mt5_service, symbol=pos_symbol, count=50)
                    if df.empty: continue
                    df = data_service.calculate_indicators(df)
                    current_atr = df.iloc[-1]["ATR"]
                    
                    # Check Trailing
                    params = STRATEGY_PARAMS.get(track_info["signal_type"], STRATEGY_PARAMS["TREND"])
                    
                    new_sl = None
                    if pos_type == "buy":
                        profit = track_info["highest"] - track_info["entry_price"]
                        if profit >= params["TRAIL_START"] * current_atr:
                            new_sl = track_info["highest"] - params["TRAIL_DIST"] * current_atr
                            if new_sl > current_sl:
                                mt5_service.modify_position_sl_tp(ticket, sl=new_sl)
                                logger.info(f"[{acc['name']}] TRAILING BUY: {current_sl} -> {new_sl}")
                    
                    elif pos_type == "sell":
                        profit = track_info["entry_price"] - track_info["lowest"]
                        if profit >= params["TRAIL_START"] * current_atr:
                            new_sl = track_info["lowest"] + params["TRAIL_DIST"] * current_atr
                            if new_sl < current_sl or current_sl == 0:
                                mt5_service.modify_position_sl_tp(ticket, sl=new_sl)
                                logger.info(f"[{acc['name']}] TRAILING SELL: {current_sl} -> {new_sl}")
                                
                    # Time Exit Logic (Simplified for brevity)
                    hours_held = (datetime.now() - track_info["entry_time"]).total_seconds() / 3600
                    if hours_held > 10.0: # 10 Hours Max
                        mt5_service.close_position(ticket)
                        logger.info(f"[{acc['name']}] TIME EXIT: Held {hours_held:.1f} hours")

            except Exception as e:
                logger.error(f"[MONITOR] Error on {acc.get('name')}: {e}")


    
    def run_loop(self, mt5_service, data_service):
        """Main trading loop."""
        logger.info("========== TRADING BOT STARTED ==========")
        logger.info(f"Active trading symbols: {settings.ACTIVE_SYMBOLS}")
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

                # =========================================================
                # WEEKEND SAFETY MODE (Friday Force Close)
                # =========================================================
                if settings.CLOSE_ON_FRIDAY and datetime.utcnow().weekday() == 4:
                    if datetime.utcnow().hour >= settings.FRIDAY_CLOSE_HOUR_UTC:
                        logger.warning("[WEEKEND] Cut-off time reached! Closing all positions to sleep well 😴")
                        
                        # Close all open positions
                        positions = mt5_service.get_open_positions()
                        if positions:
                            res = mt5_service.close_all_positions()
                            if res["closed"] > 0:
                                msg = f"🛑 <b>WEEKEND FORCE CLOSE</b>\n\nClosed {res['closed']} positions to avoid gap risk.\nSee you next week! 👋"
                                telegram_service.send_message(msg)
                                logger.info(f"[WEEKEND] Closed {res['closed']} positions")
                        
                        # Wait and skip trading scan
                        logger.info("[WEEKEND] Standing by until Monday...")
                        time.sleep(300) # Sleep 5 minutes
                        continue
                # =========================================================
                
                # Monitor ALL accounts (Loop Login -> Check -> Logout)
                self.monitor_all_accounts(mt5_service, data_service)
                
                # Scan ALL symbols for signals (Just need to login to one account to fetch data)
                # Ensure we are connected to at least one account for data fetching
                if not mt5_service.is_connected():
                    mt5_service.login_account(mt5_service.accounts[0])
                
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
                                # New signal detected
                                is_active = (symbol in settings.ACTIVE_SYMBOLS)
                                
                                # Send Notification
                                if settings.ENABLE_SIGNAL_ALERTS:
                                    telegram_service.notify_signal_detected(
                                        symbol=symbol,
                                        signal=signal_info["signal"],
                                        signal_type=signal_info["signal_type"],
                                        price=signal_info["price"],
                                        rsi=signal_info.get("rsi", 0),
                                        adx=signal_info.get("adx", 0),
                                        is_active_symbol=is_active
                                    )
                                    logger.info(f"[{symbol}] Signal Alert Sent: {signal_info['signal']}")
                                
                                last_signals[symbol] = signal_key
                            
                            # Trade if ACTIVE (Multi-Account Execution)
                            if symbol in settings.ACTIVE_SYMBOLS:
                                print(f"[{symbol}] TRADING SIGNAL: {signal_info['signal'].upper()}")
                                
                                # Loop through all accounts to place trades
                                for acc in mt5_service.accounts:
                                    try:
                                        # Login
                                        if not mt5_service.login_account(acc):
                                            logger.error(f"Failed to login {acc['name']} for trading")
                                            continue
                                            
                                        # Get fresh balance for THIS account
                                        acc_info = mt5_service.get_account_info()
                                        balance = acc_info.get("balance", 0)
                                        
                                        # Calculate Position Size based on THIS account's balance
                                        atr = signal_info["atr"]
                                        signal_type = signal_info["signal_type"]
                                        lots = self.calculate_position_size(balance, atr, signal_type)
                                        
                                        # Calculate SL/TP
                                        price = signal_info["price"]
                                        params = settings.PARAMS[signal_type]
                                        sl_dist = atr * params["SL_ATR"]
                                        tp_dist = atr * params["TP_ATR"]
                                        
                                        if signal_info["signal"] == "buy":
                                            sl = price - sl_dist
                                            tp = price + tp_dist
                                        else:
                                            sl = price + sl_dist
                                            tp = price - tp_dist
                                            
                                        # Place Order for THIS account
                                        logger.info(f"[{acc['name']}] Placing {signal_info['signal']} {lots} lots")
                                        result = mt5_service.place_order(
                                            order_type=signal_info["signal"],
                                            symbol=symbol,
                                            volume=lots,
                                            sl=sl,
                                            tp=tp,
                                            comment=f"Bot_{signal_type}"
                                        )
                                        
                                        if result.get("success"):
                                            logger.info(f"[{acc['name']}] Order SUCCESS: #{result.get('order_id')}")
                                            # Optional: Verify trade with specific account checking logic if needed
                                        else:
                                            logger.error(f"[{acc['name']}] Order FAILED: {result.get('error')}")
                                            
                                    except Exception as e:
                                        logger.error(f"[{acc['name']}] Trade Error: {e}")
                                
                                self.stats["last_action"] = f"Executed {signal_info['signal']} on {len(mt5_service.accounts)} accs"
                                
                        else:
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
        symbols_str = ', '.join(settings.ACTIVE_SYMBOLS)
        telegram_service.notify_bot_started(symbols_str, settings.RISK_PERCENT)
        
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
