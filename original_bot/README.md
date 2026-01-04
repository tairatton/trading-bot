# Original Trading Bot

## Purpose
High return trading for personal account (NOT for Prop Firm)

## Configuration
- **Symbols:** EURUSD, USDCAD, USDJPY (3 symbols)
- **Risk per Trade:** 0.5%
- **Total Risk:** 1.5% (when all open)
- **Expected Monthly:** ~9%
- **Max Drawdown:** ~12%

## Setup
1. Edit `.env` file with your MT5 credentials
2. Install dependencies: `pip install -r requirements.txt`
3. Run bot: `python main.py`

## Backtest
```bash
python backtest/backtest.py
```

## Output Files
- `backtest_trades.csv` - Trade history
- `equity_curve.csv` - Equity curve data
