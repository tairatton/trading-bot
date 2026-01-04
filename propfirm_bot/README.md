# Prop Firm Trading Bot

## Purpose
Safe trading for Prop Firm Challenge (The5ers, FTMO, etc.)

## Configuration
- **Symbol:** EURUSD only
- **Risk per Trade:** 0.45%
- **Expected Monthly:** ~6.7%
- **Max Drawdown:** ~13.7%
- **Max Daily Loss:** ~3.2% (< 5% limit)

## Prop Firm Limits
| Limit | Requirement | Our Config |
|-------|-------------|------------|
| Daily Loss | 5% | 3.2% ✅ |
| Max Loss | $1,000 | ~$43 ✅ |
| Phase 1 Target | 8% | ~6.7%/month |
| Phase 2 Target | 5% | ~6.7%/month |

## Setup
1. Create account with prop firm (The5ers, etc.)
2. Edit `.env` file with your prop firm MT5 credentials
3. Install dependencies: `pip install -r requirements.txt`
4. Run bot: `python main.py`

## Backtest
```bash
python backtest/backtest.py
```

## Output Files
- `propfirm_backtest_trades.csv` - Trade history
- `propfirm_equity_curve.csv` - Equity curve
- `propfirm_daily_pnl.csv` - Daily P&L breakdown

## Timeline to Pass Challenge
- Phase 1 (8%): ~2 months
- Phase 2 (5%): ~1 month
- **Total: ~3 months to funded**
