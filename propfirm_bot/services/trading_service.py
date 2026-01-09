"""Trading Service - main bot logic."""
import threading
import time
import logging
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import math
from config import settings
from services.strategy import STRATEGY_PARAMS
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
        
        # Position tracking for trailing stop/time exit (single account)
        self.positions_tracking: Dict[int, Dict[str, Any]] = {}
        
        # Error tracking for browser notification
        self.last_error: Optional[str] = None
        
        # ========== DYNAMIC RISK (5-Tier Anti-Martingale) ==========
        self.peak_balance: float = 0  # Track peak for DD calculation
        self.dynamic_risk_config = {
            "BASE_RISK": 1.0,  # 1.0% base (Optimized: Safe via Dynamic Risk)
            "TIERS": [
                (1.5, 0.8),    # > 1.5% DD -> 0.224%
                (3.0, 0.6),    # > 3.0% DD -> 0.168%
                (4.0, 0.4),    # > 4.0% DD -> 0.112%
                (5.0, 0.2),    # > 5.0% DD -> 0.056% (Floor hit)
                (6.0, 0.1)     # > 6.0% DD -> 0.028% (Floor hit)
            ]
        }
        
        # Max Drawdown Tracking
        self.max_dd_abs: float = 0.0
        self.max_dd_pct: float = 0.0
        
        # Daily Loss Protection for Prop Firm (The 5%ers: 5% daily limit)
        self.DAILY_LOSS_LIMIT: float = 4.1  # Stop trading if daily loss exceeds 4.1%
        self.daily_starting_balance: float = 0  # Track starting balance of the day
        self.last_daily_reset: Optional[datetime] = None
        
        # Persistence file for daily state (survives restarts)
        self.DAILY_STATE_FILE: str = os.path.join(log_dir, "daily_state.json")
        self._load_daily_state()
        
        # News filter removed - using spread filter instead
    

    

    
    def can_trade(self, current_time: datetime) -> tuple:
        """Check if trading is allowed. Using spread filter instead of news filter."""
        # News filter removed - spread filter handles volatility
        # Session filter disabled - trade 24 hours (higher returns in backtest)
        return True, "OK"
    
    def calculate_dynamic_risk(self, balance: float) -> float:
        """Calculate risk percentage based on current drawdown (5-Tier Anti-Martingale).
        
        Returns risk as percentage (e.g., 0.28 for 0.28%).
        """
        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Calculate current drawdown
        if self.peak_balance <= 0:
            return self.dynamic_risk_config["BASE_RISK"]
        
        dd_abs = self.peak_balance - balance
        current_dd = (dd_abs / self.peak_balance) * 100
        
        # Update Max Drawdown Stats
        metrics_changed = False
        
        if balance > self.peak_balance:
            self.peak_balance = balance
            metrics_changed = True

        if current_dd > self.max_dd_pct:
            self.max_dd_pct = current_dd
            metrics_changed = True
            
        if dd_abs > self.max_dd_abs:
            self.max_dd_abs = dd_abs
            metrics_changed = True
            
        if metrics_changed:
            self._save_daily_state()
        
        # Apply tier logic
        base_risk = self.dynamic_risk_config["BASE_RISK"]
        multiplier = 1.0
        
        for threshold, mult in self.dynamic_risk_config["TIERS"]:
            if current_dd > threshold:
                multiplier = mult
        
        risk_percent = base_risk * multiplier
        
        logger.debug(f"[DYNAMIC RISK] DD: {current_dd:.2f}%, MaxDD: {self.max_dd_pct:.2f}%, Risk: {risk_percent:.2f}%")
        return risk_percent
        
    def get_dashboard_metrics(self, mt5_service) -> Dict:
        """Get detailed dashboard metrics including MaxDD and Fees."""
        # Get metrics
        account = mt5_service.get_account_info()
        fees = mt5_service.get_daily_fees()
        
        balance = account.get("balance", 0.0)
        
        # Calculate current DD stats
        if self.peak_balance > 0:
            current_dd_abs = self.peak_balance - balance
            current_dd_pct = (current_dd_abs / self.peak_balance) * 100
        else:
            current_dd_abs = 0.0
            current_dd_pct = 0.0
            
        return {
            "balance": balance,
            "equity": account.get("equity", 0.0),
            "profit": account.get("profit", 0.0),
            "peak_balance": self.peak_balance,
            "current_dd_abs": current_dd_abs,
            "current_dd_pct": current_dd_pct,
            "max_dd_abs": self.max_dd_abs,
            "max_dd_pct": self.max_dd_pct,
            "fees_commission": fees.get("commission", 0.0),
            "fees_swap": fees.get("swap", 0.0),
            "fees_total": fees.get("total", 0.0),
            "current_risk": self.calculate_dynamic_risk(balance)
        }
    
    def get_last_error(self) -> Optional[str]:
        """Get and clear last error."""
        error = self.last_error
        self.last_error = None  # Clear after reading
        return error
    
    def _load_daily_state(self) -> None:
        """Load daily state from JSON file (for restart persistence)."""
        try:
            if os.path.exists(self.DAILY_STATE_FILE):
                with open(self.DAILY_STATE_FILE, 'r') as f:
                    state = json.load(f)
                    saved_date = state.get("date", "")
                    today = datetime.utcnow().strftime("%Y-%m-%d")
                    
                    if saved_date == today:
                        self.daily_starting_balance = state.get("starting_balance", 0)
                        self.last_daily_reset = datetime.fromisoformat(state.get("reset_time", datetime.utcnow().isoformat()))
                        logger.info(f"[DAILY] Restored state from file - Balance: ${self.daily_starting_balance:.2f}")
                    else:
                        logger.info(f"[DAILY] State file is from {saved_date}, ignoring (new day)")
                    
                    # Load MaxDD stats (persist regardless of day)
                    self.peak_balance = state.get("peak_balance", 0.0)
                    self.max_dd_abs = state.get("max_dd_abs", 0.0)
                    self.max_dd_pct = state.get("max_dd_pct", 0.0)
                    logger.info(f"[STATE] Loaded MaxDD stats: Peak=${self.peak_balance:.2f}, MaxDD=${self.max_dd_abs:.2f} ({self.max_dd_pct:.2f}%)")

        except Exception as e:
            logger.warning(f"[DAILY] Failed to load state: {e}")
    
    def _save_daily_state(self) -> None:
        """Save daily state to JSON file."""
        try:
            state = {
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "starting_balance": self.daily_starting_balance,
                "reset_time": self.last_daily_reset.isoformat() if self.last_daily_reset else datetime.utcnow().isoformat(),
                "peak_balance": self.peak_balance,
                "max_dd_abs": self.max_dd_abs,
                "max_dd_pct": self.max_dd_pct
            }
            with open(self.DAILY_STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning(f"[DAILY] Failed to save state: {e}")
    
    def is_weekend_close_time(self) -> bool:
        """Check if it's time to close all positions before weekend."""
        if not settings.CLOSE_ON_FRIDAY:
            return False
        
        now = datetime.utcnow()
        # Friday = weekday 4
        if now.weekday() == 4 and now.hour >= settings.FRIDAY_CLOSE_HOUR_UTC:
            return True
        # Saturday or Sunday should not have positions
        if now.weekday() >= 5:
            return True
        return False
    
    def check_daily_loss(self, mt5_service) -> tuple:
        """Check if daily loss limit has been reached.
        
        Returns:
            (can_trade: bool, daily_loss_pct: float, message: str)
        """
        account = mt5_service.get_account_info()
        current_balance = account.get("balance", 0)
        current_equity = account.get("equity", current_balance)
        
        # Reset daily tracking at start of new TRADING day (05:00 Thai Time = 22:00 UTC)
        # We define "Trading Date" as changing at 22:00 UTC.
        # UTC 21:59 -> +2hr = 23:59 (Day X)
        # UTC 22:00 -> +2hr = 00:00 (Day X+1)
        # So we use (UTC + 2 hours).date() to track the trading day.
        
        now_utc = datetime.utcnow()
        current_trading_date = (now_utc + timedelta(hours=2)).date()
        
        # Init last_daily_reset data if not present (load from file/init)
        if not hasattr(self, 'last_reset_trading_date') or self.last_reset_trading_date is None:
             # Try to infer from last_daily_reset or just use current
             if self.last_daily_reset:
                 self.last_reset_trading_date = (self.last_daily_reset + timedelta(hours=2)).date()
             else:
                 self.last_reset_trading_date = current_trading_date
                 # Initial setup, assume current balance is start balance
                 self.daily_starting_balance = current_balance

        # Check if we need to reset (Date changed)
        if self.last_reset_trading_date != current_trading_date:
            self.daily_starting_balance = current_balance
            self.last_reset_trading_date = current_trading_date
            self.last_daily_reset = now_utc
            self._save_daily_state()  # Persist to file
            logger.info(f"[DAILY] New Trading Day Started (05:00 Thai) - Balance Reset: ${current_balance:.2f}")
        
        if self.daily_starting_balance <= 0:
            return True, 0, "OK"
        
        # Calculate daily loss (use equity to include unrealized P&L)
        daily_loss = self.daily_starting_balance - current_equity
        daily_loss_pct = (daily_loss / self.daily_starting_balance) * 100
        
        if daily_loss_pct >= self.DAILY_LOSS_LIMIT:
            return False, daily_loss_pct, f"Daily loss {daily_loss_pct:.2f}% >= {self.DAILY_LOSS_LIMIT}%"
        
        return True, daily_loss_pct, "OK"
    
    def calculate_position_size(self, balance: float, atr: float, signal_type: str, symbol: str = None, mt5_service = None) -> float:
        """Calculate position size based on risk.
        
        Properly handles all symbol types:
        - Standard Forex (EURUSD, GBPUSD): pip = 0.0001, pip value = 10 USD/lot
        - JPY pairs (USDJPY, EURJPY): pip = 0.01, pip value varies with price
        - Metals (XAUUSD): pip = 0.1, pip value = 10 USD/lot
        - Crypto (BTCUSD): pip = 1, pip value = 1 USD/lot
        """
        from config import settings
        symbol = symbol or settings.SYMBOL
        
        # Calculate DYNAMIC risk based on current DD
        risk_percent = self.calculate_dynamic_risk(balance)
        risk_amount = balance * (risk_percent / 100)
        
        sl_atr = STRATEGY_PARAMS[signal_type]["SL_ATR"]
        sl_distance = atr * sl_atr
        
        if sl_distance <= 0:
            return 0.01  # Minimum lot
        
        # Get symbol info for accurate lot calculation
        pip_value_per_lot = 10.0  # Default: $10 per pip for standard forex (1 lot = 100,000 units)
        
        if mt5_service and mt5_service.connected:
            try:
                symbol_info = mt5_service.get_symbol_info(symbol)
                if symbol_info:
                    contract_size = symbol_info.get('trade_contract_size', 100000)
                    digits = symbol_info.get('digits', 5)
                    point = symbol_info.get('point', 0.00001)
                    
                    # Calculate pip size (usually 10 points for 5-digit/3-digit brokers)
                    if digits == 5 or digits == 3:
                        pip_size = point * 10
                    elif digits == 2:  # JPY pairs
                        pip_size = point * 10
                    else:
                        pip_size = point
                    
                    # Determine pip value based on symbol type
                    symbol_upper = symbol.upper()
                    
                    if 'JPY' in symbol_upper:
                        # JPY pairs: pip value = (pip_size / current_price) * contract_size
                        # For USDJPY at ~156: pip_value = 0.01 / 156 * 100000 = ~6.41 USD
                        price_info = mt5_service.get_current_price(symbol)
                        if price_info:
                            current_price = (price_info.get('bid', 156) + price_info.get('ask', 156)) / 2
                            pip_value_per_lot = (pip_size / current_price) * contract_size
                    elif 'XAU' in symbol_upper or 'XAG' in symbol_upper:
                        # Gold/Silver: typically $10/pip for 1 lot (100oz), pip = 0.1
                        pip_value_per_lot = 10.0
                    elif 'BTC' in symbol_upper or 'ETH' in symbol_upper:
                        # Crypto: varies by broker, usually pip = 1, pip_value = $1/lot
                        pip_value_per_lot = 1.0
                    else:
                        # Standard forex (EURUSD, GBPUSD, etc.): $10/pip for 1 lot
                        pip_value_per_lot = 10.0
                        
                    logger.debug(f"[{symbol}] pip_value_per_lot: ${pip_value_per_lot:.2f}")
            except Exception as e:
                logger.warning(f"[{symbol}] Error getting symbol info: {e}, using default pip value")
        
        # Calculate lot size: lots = risk_amount / (SL in pips * pip_value_per_lot)
        # SL_distance is in price units, need to convert to pips
        symbol_upper = symbol.upper() if symbol else ""
        
        if 'JPY' in symbol_upper:
            sl_pips = sl_distance * 100  # JPY: 0.01 = 1 pip, so multiply by 100
        elif 'XAU' in symbol_upper:
            sl_pips = sl_distance * 10   # Gold: 0.1 = 1 pip
        elif 'BTC' in symbol_upper or 'ETH' in symbol_upper:
            sl_pips = sl_distance        # Crypto: 1 = 1 pip
        else:
            sl_pips = sl_distance * 10000  # Standard forex: 0.0001 = 1 pip
        
        if sl_pips <= 0:
            return 0.01
            
        lots = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Round to 2 decimals, minimum 0.01, maximum 10.0
        lots = max(0.01, min(10.0, math.floor(lots * 100) / 100))
        
        logger.info(f"[{symbol}] Lot calc: Risk ${risk_amount:.2f}, SL {sl_pips:.1f} pips, PipVal ${pip_value_per_lot:.2f} => {lots} lots")
        
        return lots
    
    def process_signal(self, mt5_service, data_service, signal_info: Dict, symbol: str, account_name: str = "") -> Dict:
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
        
        # Check Daily Loss Protection
        can_trade_daily, daily_loss_pct, daily_msg = self.check_daily_loss(mt5_service)
        if not can_trade_daily:
            logger.warning(f"[DAILY LOSS] LIMIT REACHED: {daily_msg}")
            self.stats["last_action"] = f"STOPPED: Daily Loss {daily_loss_pct:.1f}%"
            telegram_service.notify_error(f"⚠️ DAILY LOSS LIMIT: {daily_loss_pct:.2f}% - Trading paused")
            return {"action": "skip", "reason": daily_msg}
        
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
        
        # PRE-TRADE RISK CHECK: Ensure total risk (existing + new) doesn't exceed 4.1%
        # This prevents opening positions that would risk more than daily loss limit
        account_info = mt5_service.get_account_info()
        current_equity = account_info.get("equity", balance)
        
        # Calculate risk from existing positions (max potential loss if all SL hit)
        existing_positions = mt5_service.get_open_positions()
        existing_risk = 0.0
        
        for pos in existing_positions:
            pos_symbol = pos.get("symbol", "")
            pos_volume = pos.get("volume", 0)
            pos_price_open = pos.get("price_open", 0)
            pos_sl = pos.get("sl", 0)
            pos_type = pos.get("type", 0)  # 0=buy, 1=sell
            
            if pos_sl > 0:  # Only count if SL is set
                # Calculate pip value for this position's symbol
                if "JPY" in pos_symbol.upper():
                    pip_mult = 1000
                else:
                    pip_mult = 100000
                
                # Calculate potential loss if SL hits
                if pos_type == 0:  # Buy position
                    sl_loss = (pos_price_open - pos_sl) * pos_volume * pip_mult
                else:  # Sell position
                    sl_loss = (pos_sl - pos_price_open) * pos_volume * pip_mult
                
                existing_risk += abs(sl_loss)
        
        # Calculate risk from new position
        params = STRATEGY_PARAMS[signal_type]
        sl_distance = atr * params["SL_ATR"]
        
        # Calculate DYNAMIC risk for pre-trade check
        risk_percent = self.calculate_dynamic_risk(balance)
        new_position_risk = balance * (risk_percent / 100)
        
        # Total risk if we open this position
        total_risk = existing_risk + new_position_risk
        total_risk_pct = (total_risk / current_equity) * 100
        
        # Check if total risk exceeds limit (5% target for Prop Firm)
        MAX_TOTAL_RISK_PCT = 4.1
        if total_risk_pct > MAX_TOTAL_RISK_PCT:
            logger.warning(f"[{symbol}] PRE-TRADE RISK BLOCK: Total risk {total_risk_pct:.2f}% > {MAX_TOTAL_RISK_PCT}%")
            logger.warning(f"  Existing risk: ${existing_risk:.2f} ({existing_risk/current_equity*100:.2f}%)")
            logger.warning(f"  New position risk (Risk {risk_percent:.2f}%): ${new_position_risk:.2f} ({new_position_risk/current_equity*100:.2f}%)")
            self.stats["last_action"] = f"BLOCKED: Total risk {total_risk_pct:.1f}% > {MAX_TOTAL_RISK_PCT}%"
            
            # Send Telegram warning
            telegram_service.notify_error(
                f"⚠️ Trade Blocked - Risk Protection\n\n"
                f"Symbol: {symbol}\n"
                f"Existing risk: ${existing_risk:.0f} ({existing_risk/current_equity*100:.1f}%)\n"
                f"New position (Risk {risk_percent:.2f}%): ${new_position_risk:.0f} ({new_position_risk/current_equity*100:.1f}%)\n"
                f"Total: {total_risk_pct:.1f}% > {MAX_TOTAL_RISK_PCT}% limit"
            )
            
            return {"action": "skip", "reason": f"Total risk {total_risk_pct:.1f}% > {MAX_TOTAL_RISK_PCT}%"}
        
        logger.info(f"[{symbol}] Pre-trade risk check: {total_risk_pct:.2f}% \u003c {MAX_TOTAL_RISK_PCT}% [OK]")
        
        # Calculate position size
        lots = self.calculate_position_size(balance, atr, signal_type, symbol=symbol, mt5_service=mt5_service)
        
        # Calculate SL/TP

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
            
            # Send Telegram notification (includes dynamic risk %)
            telegram_service.notify_trade_opened(
                symbol=symbol,
                trade_type=signal,
                price=result.get("price", price),
                lots=lots,
                sl=sl,
                tp=tp,
                signal_type=signal_type,
                account_name=account_name,
                risk_percent=risk_percent
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
        """Monitor open positions for current account (single account)."""
        from services.strategy import STRATEGY_PARAMS

        positions = mt5_service.get_open_positions()
        if not positions:
            self.positions_tracking = {}
            return

        live_tickets = set()
        for pos in positions:
            try:
                ticket = pos["ticket"]
                live_tickets.add(ticket)
                pos_symbol = pos.get("symbol", settings.SYMBOL)
                pos_type = "buy" if pos["type"] == 0 else "sell"
                entry_price = pos["price_open"]
                current_sl = pos.get("sl", 0)

                if ticket not in self.positions_tracking:
                    comment = pos.get("comment", "")
                    signal_type = "MR" if "MR" in comment else "TREND"
                    
                    # Use MT5 position time instead of datetime.now()
                    pos_time = pos.get("time", 0)
                    if pos_time > 0:
                        entry_time = datetime.fromtimestamp(pos_time)
                    else:
                        entry_time = datetime.now()  # Fallback
                    
                    self.positions_tracking[ticket] = {
                        "entry_price": entry_price,
                        "highest": entry_price if pos_type == "buy" else 0,
                        "lowest": entry_price if pos_type == "sell" else float('inf'),
                        "entry_time": entry_time,
                        "signal_type": signal_type,
                        "type": pos_type,
                    }

                track_info = self.positions_tracking[ticket]

                price_info = mt5_service.get_current_price(pos_symbol)
                if not price_info:
                    continue

                current_price = price_info["bid"] if pos_type == "buy" else price_info["ask"]

                if pos_type == "buy":
                    track_info["highest"] = max(track_info["highest"], current_price)
                else:
                    track_info["lowest"] = min(track_info["lowest"], current_price)

                df = data_service.fetch_data_from_mt5(mt5_service, symbol=pos_symbol, count=50)
                if df.empty:
                    continue
                df = data_service.calculate_indicators(df)
                current_atr = df.iloc[-1]["ATR"]

                params = STRATEGY_PARAMS.get(track_info["signal_type"], STRATEGY_PARAMS["TREND"])

                if pos_type == "buy":
                    profit = track_info["highest"] - track_info["entry_price"]
                    if profit >= params["TRAIL_START"] * current_atr:
                        new_sl = track_info["highest"] - params["TRAIL_DIST"] * current_atr
                        if new_sl > current_sl:
                            mt5_service.modify_position_sl_tp(ticket, sl=new_sl)
                            logger.info(f"TRAILING BUY: {current_sl} -> {new_sl}")
                else:
                    profit = track_info["entry_price"] - track_info["lowest"]
                    if profit >= params["TRAIL_START"] * current_atr:
                        new_sl = track_info["lowest"] + params["TRAIL_DIST"] * current_atr
                        if new_sl < current_sl or current_sl == 0:
                            mt5_service.modify_position_sl_tp(ticket, sl=new_sl)
                            logger.info(f"TRAILING SELL: {current_sl} -> {new_sl}")

                hours_held = (datetime.now() - track_info["entry_time"]).total_seconds() / 3600
                # Dynamic time exit based on MAX_BARS (matches backtest)
                max_hours = params["MAX_BARS"] * 0.5  # 30min bars = 0.5hr each
                if hours_held > max_hours:
                    mt5_service.close_position(ticket)
                    logger.info(f"TIME EXIT: Held {hours_held:.1f}hrs (Max: {max_hours}hrs)")
            except Exception as e:
                logger.error(f"[MONITOR] Error: {e}")

        # Keep only live tickets
        self.positions_tracking = {t: info for t, info in self.positions_tracking.items() if t in live_tickets}


    
    def run_loop(self, mt5_service, data_service):
        """Main trading loop."""
        logger.info("========== TRADING BOT STARTED ==========")
        logger.info(f"Active trading symbols: {settings.ACTIVE_SYMBOLS}")
        logger.info(f"Scanning all symbols: {settings.AVAILABLE_SYMBOLS}")
        
        # Track last signal per symbol to avoid duplicate alerts
        last_signals = {}
        last_dashboard_time = datetime.min
        
        while self.running:
            if self.paused:
                time.sleep(5)
                continue
            
            # ---------------------------------------------------------
            # DASHBOARD PERIODIC UPDATE (Every 1 Hour)
            # ---------------------------------------------------------

            
            try:
                # Ensure MT5 connection is alive (auto reconnect if needed)
                if not mt5_service.ensure_connected():
                    print("[BOT] MT5 connection failed, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                
                # Check metrics & Save state if balance changed (Order Closed)
                current_balance = mt5_service.get_account_info().get("balance", 0)
                self.calculate_dynamic_risk(current_balance)

                # =========================================================
                # WEEKEND SAFETY MODE (Close positions before weekend)
                # =========================================================
                if self.is_weekend_close_time():
                    logger.warning("[WEEKEND] Cut-off time reached! Closing ALL positions (silent mode - no alerts)")

                    res = mt5_service.close_all_positions()
                    total_closed = int(res.get("closed", 0))
                            
                    if total_closed > 0:
                        logger.info(f"[WEEKEND] Closed {total_closed} positions. Weekend mode active.")
                        # Note: No Telegram notification to avoid disturbing on weekends
                    
                    # Wait and skip trading scan
                    logger.info("[WEEKEND] Standing by until Monday...")
                    time.sleep(300)  # Sleep 5 minutes
                    continue
                # =========================================================
                
                # =========================================================
                
                # Check Daily Loss Limit (Trigger Reset if needed)
                can_trade, daily_loss_pct, reason = self.check_daily_loss(mt5_service)
                if not can_trade:
                    if self.stats["last_action"] != f"DAILY STOP: {reason}":
                         logger.warning(f"[DAILY] TRADING STOPPED: {reason}")
                         self.stats["last_action"] = f"DAILY STOP: {reason}"
                         telegram_service.notify_error(f"⛔ Daily Loss Limit Hit: {daily_loss_pct:.2f}% (Stop Trading)")
                    time.sleep(60) # Wait a bit before retry (wait for next day reset)
                    continue

                # Monitor positions (single account)
                self.monitor_positions(mt5_service, data_service)
                
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
                                
                                # Calculate SL/TP for signal notification
                                from services.strategy import STRATEGY_PARAMS
                                sig_price = signal_info["price"]
                                sig_atr = signal_info.get("atr", 0)
                                sig_type = signal_info["signal_type"]
                                sig_params = STRATEGY_PARAMS.get(sig_type, STRATEGY_PARAMS["TREND"])
                                
                                if signal_info["signal"] == "buy":
                                    sig_sl = sig_price - sig_atr * sig_params["SL_ATR"]
                                    sig_tp = sig_price + sig_atr * sig_params["TP_ATR"]
                                else:  # sell
                                    sig_sl = sig_price + sig_atr * sig_params["SL_ATR"]
                                    sig_tp = sig_price - sig_atr * sig_params["TP_ATR"]
                                
                                # Send Notification
                                if settings.ENABLE_SIGNAL_ALERTS:
                                    telegram_service.notify_signal_detected(
                                        symbol=symbol,
                                        signal=signal_info["signal"],
                                        signal_type=signal_info["signal_type"],
                                        price=sig_price,
                                        rsi=signal_info.get("rsi", 0),
                                        adx=signal_info.get("adx", 0),
                                        is_active_symbol=is_active,
                                        strength=signal_info.get("strength", ""),
                                        strength_score=signal_info.get("strength_score", 0),
                                        strength_factors=signal_info.get("strength_factors", []),
                                        sl=sig_sl,
                                        tp=sig_tp
                                    )
                                    logger.info(f"[{symbol}] Signal Alert Sent: {signal_info['signal']}")
                                
                                last_signals[symbol] = signal_key
                            
                            # Trade if ACTIVE (Multi-Account Execution)
                            if symbol in settings.ACTIVE_SYMBOLS:
                                print(f"[{symbol}] TRADING SIGNAL: {signal_info['signal'].upper()}")
                                trade_result = self.process_signal(mt5_service, data_service, signal_info, symbol)

                                if trade_result.get("action") == "opened":
                                    self.stats["last_action"] = f"OPENED {signal_info['signal'].upper()} ({symbol})"
                                elif trade_result.get("action") == "failed":
                                    self.stats["last_action"] = f"FAILED: {trade_result.get('error', 'Unknown error')}"
                                else:
                                    self.stats["last_action"] = f"{trade_result.get('action', 'none')}: {trade_result.get('reason', '')}".strip(": ")
                                
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
        print(f"[TELEGRAM] Sending Bot Started notification...")
        telegram_service.notify_bot_started(symbols_str, str(settings.MT5_LOGIN))
        
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
