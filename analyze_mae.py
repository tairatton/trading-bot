import pandas as pd
import glob
import os

def analyze_mae(dir_path, bot_name):
    file_path = os.path.join(dir_path, "trades.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # Filter only valid trades
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] <= 0]
    
    # MAE is stored as negative or positive? In my code:
    # mae = (pos["lowest"] - pos["entry_price"]) for Buy -> Negative value usually
    # mae = (pos["entry_price"] - pos["highest"]) for Sell -> Negative value usually
    # Let's check absolute MAE for analysis
    
    # Wait, my previous code:
    # mae = (pos["lowest"] - pos["entry_price"]) * lots * 100000 -> This is PnL value of MAE
    # So it represents the max floating loss in $.
    
    avg_mae_win = winners['mae'].abs().mean()
    avg_mae_loss = losers['mae'].abs().mean()
    
    avg_mfe_win = winners['mfe'].abs().mean()
    
    # E-Ratio (MFE / MAE)
    # How much excursion in our favor vs against us?
    e_ratio = avg_mfe_win / avg_mae_win if avg_mae_win != 0 else 0
    
    print(f"--- Analysis for {bot_name} ---")
    print(f"Total Trades: {len(df)}")
    print(f"Win Rate: {len(winners)/len(df)*100:.2f}%")
    print(f"\n[MAE Analysis - Entry Efficiency]")
    print(f"Average MAE (Winners): ${avg_mae_win:.2f}")
    print(f"  -> Meaning: On average, winning trades went against us ${avg_mae_win:.2f} before hitting TP.")
    print(f"Average MAE (Losers):  ${avg_mae_loss:.2f}")
    print(f"\n[MFE Analysis - Excursion]")
    print(f"Average MFE (Winners): ${avg_mfe_win:.2f}")
    print(f"E-Ratio (MFE/MAE):     {e_ratio:.2f}")
    print(f"  -> Meaning: For every $1 of drawdown endured, we got ${e_ratio:.2f} of potential profit.")
    print("\n------------------------------------------------\n")

if __name__ == "__main__":
    analyze_mae("backtest_results_propfirm", "Prop Firm Bot (Safe)")
    analyze_mae("backtest_results_original", "Original Bot (Aggressive)")
