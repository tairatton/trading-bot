# Prop Firm Trading Bot

## Purpose
Safe trading for Prop Firm Challenge (The5ers, FTMO, etc.)

## Configuration (Golden Setup 🏆)
- **Symbols:** EURUSD, USDCAD, USDCHF (3 Pairs)
- **Risk per Trade:** 0.09% (per pair)
- **Expected Return:** ~30% per year (Passes Challenge!)
- **Max Drawdown:** ~3.56% (Very Safe)
- **Max Daily Loss:** ~1.55% (Far below 5% limit)

## Prop Firm Limits
| Limit | Requirement | Our Config |
|-------|-------------|------------|
| Daily Loss | 5% | 1.55% ✅ |
| Max Loss | 10% | 3.56% ✅ |
| Phase 1 Target | 8% | Passed in ~4 months |
| **Slippage** | - | **Must use RAW/ECN Account** |

## Setup
1. Create account with prop firm (The5ers, etc.)
2. Edit `.env` file with your prop firm MT5 credentials
3. Install dependencies: `pip install -r requirements.txt`
4. Run bot: `python main.py`

## Backtest Results (1 Year)
- **Net Profit:** +30.18%
- **Profit Factor:** 1.22
- **Success Rate:** 56.1%
- **Stagnation:** Max 168 days flat

## Timeline to Pass Challenge
- **Estimate: ~3-5 months to funded** (Safe & Steady)
