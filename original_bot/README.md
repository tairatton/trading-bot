# Original Trading Bot

## Purpose
High return trading for personal account (NOT for Prop Firm)

## Configuration
- **Symbols:** EURUSD, USDCAD, USDCHF (3 symbols)
- **Risk per Trade:** 0.70% (Aggressive)
- **Total Risk:** ~2.1% (when all open)
- **Expected Return:** ~365% per year 🚀
- **Max Drawdown:** ~19.85% (Medium/High Risk)

## Critical Warnings ⚠️
> **SLIPPAGE ALERT:** This strategy relies on a thin edge (Profit Factor 1.15).
> You **MUST** use a **RAW SPREAD / ECN** account.
> If using a Standard account (high spread), the bot may lose money!

## Backtest Results (1 Year)
- **Net Profit:** +365.65%
- **Profit Factor:** 1.15
- **Max Daily DD:** 9.48% (Do NOT use for Prop Firms)

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
