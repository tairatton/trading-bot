"""Telegram Notification Service for Trading Bot."""
import requests
import logging
from typing import Optional
from config import settings

logger = logging.getLogger("TradingBot")


class TelegramService:
    """Send notifications via Telegram Bot."""
    
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if self.enabled:
            logger.info("[Telegram] Notifications enabled")
        else:
            logger.info("[Telegram] Not configured - notifications disabled")
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram."""
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                logger.info("[Telegram] Message sent successfully")
                return True
            else:
                logger.error(f"[Telegram] Failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"[Telegram] Error: {e}")
            return False
    
    
    def _clean_symbol(self, symbol: str) -> str:
        """Remove broker suffixes from symbol (e.g. EURUSDm -> EURUSD)."""
        import re
        # Remove lowercase suffixes at the end
        clean = re.sub(r'[a-z]+$', '', symbol)
        return clean

    # ===== Trade Notifications =====
    
    def notify_trade_opened(self, symbol: str, trade_type: str, price: float, 
                           lots: float, sl: float, tp: float, signal_type: str = "",
                           account_name: str = ""):
        """Notify when a trade is opened."""
        clean_symbol = self._clean_symbol(symbol)
        emoji = "🟢" if trade_type.upper() == "BUY" else "🔴"
        acc_line = f"<b>Account:</b> {account_name}\n" if account_name else ""
        msg = f"""
{emoji} <b>TRADE OPENED</b>

{acc_line}<b>Symbol:</b> {clean_symbol}
<b>Type:</b> {trade_type.upper()} ({signal_type})
<b>Price:</b> {price:.5f}
<b>Lots:</b> {lots}
<b>SL:</b> {sl:.5f}
<b>TP:</b> {tp:.5f}
"""
        self.send_message(msg.strip())
    
    def notify_trade_closed(self, symbol: str, trade_type: str, 
                           entry_price: float, exit_price: float,
                           profit: float, reason: str = "",
                           account_name: str = ""):
        """Notify when a trade is closed."""
        clean_symbol = self._clean_symbol(symbol)
        emoji = "✅" if profit >= 0 else "❌"
        profit_sign = "+" if profit >= 0 else ""
        acc_line = f"<b>Account:</b> {account_name}\n" if account_name else ""
        msg = f"""
{emoji} <b>TRADE CLOSED</b>

{acc_line}<b>Symbol:</b> {clean_symbol}
<b>Type:</b> {trade_type.upper()}
<b>Entry:</b> {entry_price:.5f}
<b>Exit:</b> {exit_price:.5f}
<b>P/L:</b> {profit_sign}${profit:.2f}
<b>Reason:</b> {reason}
"""
        self.send_message(msg.strip())
    
    def notify_error(self, error_message: str):
        """Notify on critical errors."""
        msg = f"""
⚠️ <b>BOT ERROR</b>

{error_message}
"""
        self.send_message(msg.strip())
    
    def notify_bot_started(self, symbol: str, risk_percent: float, account_id: str = ""):
        """Notify when bot starts."""
        # Clean comma-separated symbols
        clean_symbols = ", ".join([self._clean_symbol(s.strip()) for s in symbol.split(',')])
        
        acc_line = f"<b>Account:</b> {account_id}\n" if account_id else ""
        msg = f"""
🤖 <b>BOT STARTED</b>

{acc_line}<b>Symbol:</b> {clean_symbols}
<b>Risk:</b> {risk_percent}%
<b>Status:</b> Running
"""
        self.send_message(msg.strip())
    
    def notify_bot_stopped(self):
        """Notify when bot stops."""
        msg = "🛑 <b>BOT STOPPED</b>"
        self.send_message(msg)
    
    def notify_daily_summary(self, total_trades: int, wins: int, losses: int, 
                            total_pnl: float, balance: float):
        """Send daily trading summary."""
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        profit_sign = "+" if total_pnl >= 0 else ""
        emoji = "📈" if total_pnl >= 0 else "📉"
        
        msg = f"""
{emoji} <b>DAILY SUMMARY</b>

<b>Total Trades:</b> {total_trades}
<b>Wins/Losses:</b> {wins}/{losses}
<b>Win Rate:</b> {win_rate:.1f}%
<b>P/L:</b> {profit_sign}${total_pnl:.2f}
<b>Balance:</b> ${balance:.2f}
"""
        self.send_message(msg.strip())
    
    def notify_signal_detected(self, symbol: str, signal: str, signal_type: str,
                               price: float, rsi: float = 0, adx: float = 0,
                               is_active_symbol: bool = False,
                               strength: str = "", strength_score: int = 0,
                               strength_factors: list = None,
                               sl: float = 0, tp: float = 0):
        """Notify when a signal is detected on any symbol."""
        emoji = "🟢" if signal.upper() == "BUY" else "🔴"
        trade_status = "✅ TRADING" if is_active_symbol else "👀 SIGNAL ONLY"
        
        # Format strength factors
        factors_text = ""
        if strength_factors:
            factors_text = "\n".join([f"  • {f}" for f in strength_factors[:3]])
        
        # Format SL/TP display
        sl_tp_text = ""
        if sl > 0 and tp > 0:
            sl_tp_text = f"\n<b>SL:</b> {sl:.5f}\n<b>TP:</b> {tp:.5f}"
        
        clean_symbol = self._clean_symbol(symbol)
        msg = f"""
{emoji} <b>SIGNAL DETECTED</b>

<b>Symbol:</b> {clean_symbol}
<b>Signal:</b> {signal.upper()} ({signal_type})
<b>Strength:</b> {strength} ({strength_score}/100)
<b>Price:</b> {price:.5f}{sl_tp_text}
<b>RSI:</b> {rsi:.1f} | <b>ADX:</b> {adx:.1f}
<b>Status:</b> {trade_status}
"""
        if factors_text:
            msg += f"\n<b>Factors:</b>\n{factors_text}"
        
        self.send_message(msg.strip())



# Singleton instance
telegram_service = TelegramService()
