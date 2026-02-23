import pandas as pd
from engine import backtest, ACCOUNT_SIZE, RISK_PERCENT, RISK_DOLLARS

# Load CSV
df = pd.read_csv("data/EURUSD_March2025_1min.csv")

# Run backtest
results = backtest(df)

if len(results) == 0:
    print("No trades found.")
else:
    print(f"Account: ${ACCOUNT_SIZE:,.2f}  |  Risk per trade: {RISK_PERCENT}% (${RISK_DOLLARS:.2f})\n")
    print(f"{'Date':<12} {'Dir':<6} {'Entry':<10} {'Stop':<10} {'½TP':<10} {'Full TP':<10} {'R:R':<28} {'P%':>6} {'P$':>8}  W/L")
    print("-" * 110)

    total_profit_dollars = 0
    wins = losses = partial_wins = breakevens = 0

    for r in results:
        print(
            f"{str(r['date']):<12} "
            f"{r['direction']:<6} "
            f"{r['entry']:<10} "
            f"{r['stop']:<10} "
            f"{r['half_tp']:<10} "
            f"{r['full_tp']:<10} "
            f"{r['R:R']:<28} "
            f"{r['profit_percent']:>5}%  "
            f"${r['profit_dollars']:>7}  "
            f"{r['W/L']}"
        )
        total_profit_dollars += r['profit_dollars']

        if   r['W/L'] == 'W':  wins          += 1
        elif r['W/L'] == 'L':  losses        += 1
        elif r['W/L'] == 'PW': partial_wins  += 1
        elif r['W/L'] == 'BE': breakevens    += 1

    total_trades       = len(results)
    total_profit_pct   = round((total_profit_dollars / ACCOUNT_SIZE) * 100, 2)
    final_account      = round(ACCOUNT_SIZE + total_profit_dollars, 2)

    print("-" * 110)
    print(f"\nSummary")
    print(f"  Trades:          {total_trades}  ({wins}W / {partial_wins}PW / {losses}L / {breakevens}BE)")
    print(f"  Total P&L:       ${round(total_profit_dollars, 2):>8}  ({total_profit_pct}%)")
    print(f"  Final Account:   ${final_account:>10,.2f}")