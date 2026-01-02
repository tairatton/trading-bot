"""
Trading Bot - Main Entry Point

Usage:
    python main.py          # Start web dashboard
    python main.py --demo   # Demo mode (no MT5 required)
"""
import sys
import argparse
import uvicorn

# Add current directory to path
sys.path.insert(0, '.')

from config import settings
from web import app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    parser.add_argument('--host', default=settings.WEB_HOST, help='Host address')
    parser.add_argument('--port', type=int, default=settings.WEB_PORT, help='Port number')
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"  TRADING BOT - {settings.SYMBOL}")
    print("=" * 50)
    print(f"  Dashboard: http://{args.host}:{args.port}")
    print(f"  Risk: {settings.RISK_PERCENT}% per trade")
    print(f"  Session: {settings.SESSION_START_UTC}:00 - {settings.SESSION_END_UTC}:00 UTC")
    print("=" * 50)
    
    if args.demo:
        print("\n  [!] Running in DEMO mode (no MT5 connection)")
    
    print("\n  Press Ctrl+C to stop\n")
    
    # Run FastAPI server
    uvicorn.run(
        "web:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
