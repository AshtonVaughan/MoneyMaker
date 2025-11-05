"""
Simple calculation: How much you make with 2 lots at different win rates
"""

# Trading parameters
LOTS = 2  # 2 lots = 200,000 units
PIP_VALUE = 10  # $10 per pip for 2 lots
LEVERAGE = 100
SPREAD_PIPS = 0.5  # Raw spread (low spread times, ECN/STP broker)
TAKE_PROFIT_PIPS = 2.0  # Gross TP
STOP_LOSS_PIPS = 1.5  # Gross SL
TRADES_PER_DAY = 50
INITIAL_BALANCE = 10000

# Calculate net after spread (raw spread, minimal cost)
NET_WIN_PIPS = TAKE_PROFIT_PIPS - SPREAD_PIPS  # 1.5 pips net
NET_LOSS_PIPS = STOP_LOSS_PIPS + SPREAD_PIPS  # 2.0 pips net

# Dollar amounts per trade
WIN_AMOUNT = NET_WIN_PIPS * PIP_VALUE * LEVERAGE  # $500
LOSS_AMOUNT = NET_LOSS_PIPS * PIP_VALUE * LEVERAGE  # $3,000

print("="*70)
print("RETURNS CALCULATION - 2 Lots, Different Win Rates")
print("="*70)
print(f"\nSetup:")
print(f"  Lots: {LOTS}")
print(f"  Pip Value: ${PIP_VALUE} per pip")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Spread: {SPREAD_PIPS} pips")
print(f"  Take Profit: {TAKE_PROFIT_PIPS} pips (gross)")
print(f"  Stop Loss: {STOP_LOSS_PIPS} pips (gross)")
print(f"  Net Win: {NET_WIN_PIPS} pips = ${WIN_AMOUNT:,.0f} per win")
print(f"  Net Loss: {NET_LOSS_PIPS} pips = ${LOSS_AMOUNT:,.0f} per loss")
print(f"  Trades per day: {TRADES_PER_DAY}")
print(f"  Starting balance: ${INITIAL_BALANCE:,}\n")

win_rates = [0.50, 0.55, 0.60, 0.65]

print("="*70)
print("RESULTS")
print("="*70)

for win_rate in win_rates:
    wins = int(TRADES_PER_DAY * win_rate)
    losses = TRADES_PER_DAY - wins
    
    daily_pnl = (wins * WIN_AMOUNT) - (losses * LOSS_AMOUNT)
    daily_return_pct = (daily_pnl / INITIAL_BALANCE) * 100
    
    balance_day1 = INITIAL_BALANCE + daily_pnl
    balance_day30 = INITIAL_BALANCE + (daily_pnl * 30)
    balance_year1 = INITIAL_BALANCE + (daily_pnl * 252)
    
    print(f"\n{int(win_rate*100)}% Win Rate:")
    print(f"  Wins: {wins} × ${WIN_AMOUNT:,} = ${wins * WIN_AMOUNT:,}")
    print(f"  Losses: {losses} × ${LOSS_AMOUNT:,} = ${losses * LOSS_AMOUNT:,}")
    print(f"  Daily P&L: ${daily_pnl:,.2f}")
    print(f"  Daily Return: {daily_return_pct:.2f}%")
    print(f"  Balance after 1 day: ${balance_day1:,.2f}")
    print(f"  Balance after 30 days: ${balance_day30:,.2f}")
    print(f"  Balance after 1 year: ${balance_year1:,.2f}")

print("\n" + "="*70)

