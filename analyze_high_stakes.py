import pandas as pd
import numpy as np
import random

def analyze_high_stakes_simulation():
    # Load actual monthly returns frequency
    df = pd.read_csv("monthly_returns.csv")
    returns_pool = df["Prop_Ret"].tolist()  # List of % returns (e.g. 1.5, -0.2, 5.0)
    
    SIMULATIONS = 100000
    START_BALANCE = 10000.0
    PROFIT_TARGET = 10800.0  # +8%
    LOSS_LIMIT = 9000.0      # -10% Static (High Stakes)
    # Note: If trailing, we would track HWM. Let's do Static first as it's common for Step 1.
    # The 5%ers High Stakes Step 1 has 10% Max Drawdown? 
    # Correction: High Stakes Max Loss is 10% (Relative/Scaling? No, usually static or absolute for High Stakes).
    # Let's assume simplest "Static $9,000 floor" which is common for "Balance-based" standard accounts.
    # Actually High Stakes is "10% of initial balance" -> Static.
    
    pass_count = 0
    fail_count = 0
    
    print(f"Running {SIMULATIONS} simulations for High Stakes Challenge...")
    print(f"Goal: ${PROFIT_TARGET:,.0f} | Floor: ${LOSS_LIMIT:,.0f}")
    
    for _ in range(SIMULATIONS):
        balance = START_BALANCE
        # Simulate until pass or fail
        while balance < PROFIT_TARGET and balance > LOSS_LIMIT:
            # Sample a random month from history
            ret_pct = random.choice(returns_pool)
            pnl = balance * (ret_pct / 100)
            balance += pnl
            
            # Safety break for infinite loops (though unlikely given positive expectancy)
            if balance > 100000: # Super pass
                break
        
        if balance >= PROFIT_TARGET:
            pass_count += 1
        else:
            fail_count += 1
            
    success_rate = (pass_count / SIMULATIONS) * 100
    print("-" * 40)
    print(f"Simulations: {SIMULATIONS}")
    print(f"Passed:      {pass_count}")
    print(f"Failed:      {fail_count}")
    print(f"Success Probability: {success_rate:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    analyze_high_stakes_simulation()
