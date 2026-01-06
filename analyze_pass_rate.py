import pandas as pd

def analyze_pass_rate():
    try:
        df = pd.read_csv("monthly_returns.csv")
        prop_returns = df["Prop_Ret"]
        
        total_months = len(prop_returns)
        pass_phase1 = len(prop_returns[prop_returns >= 8.0])
        pass_phase2 = len(prop_returns[prop_returns >= 5.0])
        positive_months = len(prop_returns[prop_returns > 0])
        
        # Max DD and Daily Loss are already known to be safe from previous comprehensive backtest
        # Max DD: 8.1% (Safe < 10%)
        # Max Daily: 2.2% (Safe < 5%)
        # So failure rate from risk violation is 0% historically.
        
        prob_phase1_1mo = (pass_phase1 / total_months) * 100
        prob_phase2_1mo = (pass_phase2 / total_months) * 100
        prob_positive = (positive_months / total_months) * 100
        
        print(f"Total Months Analyzed: {total_months}")
        print("-" * 30)
        print(f"Months >= 8% (Pass Phase 1): {pass_phase1} ({prob_phase1_1mo:.1f}%)")
        print(f"Months >= 5% (Pass Phase 2): {pass_phase2} ({prob_phase2_1mo:.1f}%)")
        print(f"Positive Months: {positive_months} ({prob_positive:.1f}%)")
        print("-" * 30)
        
        # Cumulative Probability (Binomial potential)
        # Prob to pass within 3 months = 1 - (Fail_Rate)^3
        fail_rate_p1 = 1 - (pass_phase1 / total_months)
        prob_pass_p1_3mo = (1 - (fail_rate_p1 ** 3)) * 100
        prob_pass_p1_6mo = (1 - (fail_rate_p1 ** 6)) * 100
        
        print(f"Chance to Pass Phase 1 within 3 Months: {prob_pass_p1_3mo:.1f}%")
        print(f"Chance to Pass Phase 1 within 6 Months: {prob_pass_p1_6mo:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_pass_rate()
