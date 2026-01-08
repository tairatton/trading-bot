"""FastAPI Web Dashboard for Trading Bot."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime
import os

from config import settings
from services import mt5_service, data_service, trading_service

# Create FastAPI app
app = FastAPI(title="Trading Bot Dashboard", version="1.0.0")

# Setup templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(templates_dir))

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    account = {}
    if mt5_service.connected:
        account = mt5_service.get_account_info()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "bot_status": trading_service.get_status(),
        "account": account,
        "settings": {
            "risk_percent": settings.RISK_PERCENT,
            "available_symbols": settings.AVAILABLE_SYMBOLS,
            "active_symbol": settings.SYMBOL,
            "bot_name": settings.BOT_NAME,
            "bot_type": settings.BOT_TYPE,
            "bot_badge_color": "#ff6b35" if settings.BOT_TYPE == "PROP_FIRM" else "#00d4ff",
            "enable_signal_alerts": settings.ENABLE_SIGNAL_ALERTS,
        },
        "mt5_connected": mt5_service.connected
    })


@app.post("/api/connect")
async def connect_mt5():
    """Connect to MT5."""
    try:
        result = mt5_service.connect()
        return JSONResponse({
            "success": result,
            "message": "Connected to Exness" if result else "Connection failed - MT5 not available"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error: {str(e)}"
        })


@app.post("/api/disconnect")
async def disconnect_mt5():
    """Disconnect from MT5."""
    mt5_service.disconnect()
    return JSONResponse({"success": True, "message": "Disconnected"})


@app.get("/api/positions")
async def get_positions():
    """Get open positions (single account)."""
    if not mt5_service.connected:
        return JSONResponse({"positions": []})

    positions = mt5_service.get_display_positions("")
    return JSONResponse({"positions": positions})


@app.post("/api/start")
async def start_bot():
    """Start the trading bot."""
    if not mt5_service.connected:
        return JSONResponse({"success": False, "message": "MT5 not connected"})
    
    result = trading_service.start(mt5_service, data_service)
    return JSONResponse(result)


@app.get("/api/last_error")
async def get_last_error():
    """Get last trading error."""
    error = trading_service.get_last_error() if hasattr(trading_service, 'get_last_error') else None
    return JSONResponse({"error": error})


@app.post("/api/stop")
async def stop_bot():
    """Stop the trading bot."""
    result = trading_service.stop()
    return JSONResponse(result)


@app.post("/api/test_buy")
async def test_buy():
    """Test place a buy order (0.01 lot)."""
    if not mt5_service.connected:
        return JSONResponse({"success": False, "error": "MT5 not connected"})
    
    result = mt5_service.place_order(
        order_type="buy",
        volume=0.01,
        sl=0,
        tp=0,
        comment="Test_Buy"
    )
    return JSONResponse(result)


@app.post("/api/test_sell")
async def test_sell():
    """Test place a sell order (0.01 lot)."""
    if not mt5_service.connected:
        return JSONResponse({"success": False, "error": "MT5 not connected"})
    
    result = mt5_service.place_order(
        order_type="sell",
        volume=0.01,
        sl=0,
        tp=0,
        comment="Test_Sell"
    )
    return JSONResponse(result)


@app.post("/api/pause")
async def pause_bot():
    """Pause the trading bot."""
    result = trading_service.pause()
    return JSONResponse(result)


@app.post("/api/resume")
async def resume_bot():
    """Resume the trading bot."""
    result = trading_service.resume()
    return JSONResponse(result)


@app.post("/api/close_position/{ticket}")
async def close_position(ticket: int):
    """Close a specific position by ticket ID."""
    if not mt5_service.connected:
        return JSONResponse({"success": False, "error": "MT5 not connected"})
    result = mt5_service.close_position(ticket)
    return JSONResponse(result)


@app.get("/api/history")
async def get_history():
    """Get recent trade history from MT5."""
    if not mt5_service.connected:
        return JSONResponse({"deals": []})
    deals = mt5_service.get_history_deals()
    return JSONResponse({"deals": deals})


@app.get("/api/status")
async def get_status():
    """Get bot status and account info (single account)."""
    status = trading_service.get_status()

    mt5_info = mt5_service.get_account_info() if mt5_service.connected else {}

    # Calculate closed P/L from history deals (last 7 days)
    today_pnl = 0.0
    if mt5_service.connected:
        deals = mt5_service.get_history_deals(days=7)
        for deal in deals:
            today_pnl += deal.get("profit", 0)

    # Get detailed metrics
    dash_metrics = trading_service.get_dashboard_metrics(mt5_service)

    return {
        "bot_status": "running" if status["running"] else "paused" if status["paused"] else "stopped",
        "account": {
            "balance": mt5_info.get("balance", 0),
            "equity": mt5_info.get("equity", 0),
            "profit": mt5_info.get("profit", 0),
            "currency": "USD",
            "account_count": 1 if mt5_service.connected else 0
        },
        "stats": status["stats"],
        "trade_history": status["trade_history"],
        "account_count": 1 if mt5_service.connected else 0,
        "today_pnl": today_pnl,
        "metrics": dash_metrics  # New field for frontend
    }


@app.post("/api/settings")
async def update_settings(
    risk_percent: float = Form(...),
    symbol: str = Form("EURUSDm"),
    enable_signal_alerts: bool = Form(False)
):
    """Update trading settings."""
    settings.RISK_PERCENT = risk_percent
    settings.SYMBOL = symbol
    settings.ENABLE_SIGNAL_ALERTS = enable_signal_alerts
    
    return JSONResponse({
        "success": True,
        "message": "Settings updated",
        "settings": {
            "risk_percent": settings.RISK_PERCENT,
            "symbol": settings.SYMBOL,
            "enable_signal_alerts": settings.ENABLE_SIGNAL_ALERTS
        }
    })


@app.get("/api/signal")
async def get_current_signal():
    """Get current trading signal."""
    if not mt5_service.connected:
        return JSONResponse({"signal": "none", "reason": "MT5 not connected"})
    
    # Fetch and process data for current symbol
    df = data_service.fetch_data_from_mt5(mt5_service, symbol=settings.SYMBOL, count=300)
    if df.empty:
        return JSONResponse({"signal": "none", "reason": "No data"})
    
    df = data_service.calculate_indicators(df)
    df = data_service.generate_signals(df)
    signal_info = data_service.get_latest_signal()
    
    # Check if can trade
    can_trade, reason = trading_service.can_trade(datetime.utcnow())
    
    return JSONResponse({
        **signal_info,
        "can_trade": can_trade,
        "reason": reason,
        "time": signal_info.get("time", datetime.now()).isoformat() if signal_info.get("time") else None
    })


# Cache for backtest data
cached_backtest_df = None
cached_backtest_time = None
CACHE_FILE = "data/backtest_cache.csv"

@app.post("/api/backtest")
async def run_backtest():
    """Run backtest and return forecast results."""
    global cached_backtest_df, cached_backtest_time
    
    try:
        from services.strategy import run_backtest as do_backtest, calculate_indicators, generate_signals
        from datetime import timedelta
        import os
        
        # Use cached data if available (cache for 5 minutes)
        now = datetime.now()
        cache_valid = (
            cached_backtest_df is not None and 
            cached_backtest_time is not None and 
            (now - cached_backtest_time) < timedelta(minutes=5)
        )
        
        if cache_valid:
            df = cached_backtest_df.copy()
        else:
            # Try to load from file if MT5 not connected
            if not mt5_service.connected:
                if os.path.exists(CACHE_FILE):
                    import pandas as pd
                    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
                    print(f"[Backtest] Using cached file: {CACHE_FILE}")
                else:
                    return JSONResponse({"success": False, "error": "MT5 not connected and no cached data"})
            else:
                # Fetch data from MT5
                df = data_service.fetch_data_from_mt5(mt5_service, count=5000)
                if df.empty:
                    return JSONResponse({"success": False, "error": "No data available"})
                
                # Calculate indicators and signals
                df = calculate_indicators(df)
                df = generate_signals(df)
                
                # Save to file for offline use
                os.makedirs("data", exist_ok=True)
                df.to_csv(CACHE_FILE)
                print(f"[Backtest] Saved data to: {CACHE_FILE}")
            
            # Update memory cache
            cached_backtest_df = df.copy()
            cached_backtest_time = now
        
        # Get actual account balance for realistic forecast
        account = mt5_service.get_account_info()
        initial_balance = account.get('balance')
        
        # Run backtest with current risk setting
        risk_decimal = settings.RISK_PERCENT / 100  # Convert 2% to 0.02
        results = do_backtest(df, initial_balance=initial_balance, risk_per_trade=risk_decimal)
        
        return JSONResponse({
            "success": True,
            "total_return": results.get("total_return", 0),
            "profit_factor": results.get("profit_factor", 0),
            "win_rate": results.get("win_rate", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "total_trades": results.get("total_trades", 0),
            "wins": results.get("winning_trades", 0),
            "final_balance": results.get("final_balance", initial_balance),
            "net_profit": results.get("final_balance", initial_balance) - initial_balance,
            "risk_used": settings.RISK_PERCENT
        })
        
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


# Web package init
def create_app():
    """Create FastAPI app instance."""
    return app

