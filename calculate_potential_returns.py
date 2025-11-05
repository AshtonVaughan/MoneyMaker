"""
Calculate potential returns based on backtest results
"""

import pandas as pd
import numpy as np
import config

# Trading parameters
LOTS = 2  # 2 lots = 200,000 units
PIP_VALUE = 10  # $10 per pip for 2 lots
LEVERAGE = 100
SPREAD_PIPS = 0.5  # Raw spread
TAKE_PROFIT_PIPS = 2.0
STOP_LOSS_PIPS = 1.5
TRADES_PER_DAY = 50  # Estimate based on 3-second updates

# Net after spread
NET_WIN_PIPS = TAKE_PROFIT_PIPS - SPREAD_PIPS  # 1.5 pips net
NET_LOSS_PIPS = STOP_LOSS_PIPS + SPREAD_PIPS  # 2.0 pips net

# Dollar amounts per trade
WIN_AMOUNT = NET_WIN_PIPS * PIP_VALUE * LEVERAGE
LOSS_AMOUNT = NET_LOSS_PIPS * PIP_VALUE * LEVERAGE

print("="*70)
print("POTENTIAL RETURNS CALCULATION")
print("="*70)
print(f"\nTrading Parameters:")
print(f"  Lots: {LOTS}")
print(f"  Leverage: {LEVERAGE}x")
print(f"  TP: {TAKE_PROFIT_PIPS} pips (net: {NET_WIN_PIPS} pips)")
print(f"  SL: {STOP_LOSS_PIPS} pips (net: {NET_LOSS_PIPS} pips)")
print(f"  Spread: {SPREAD_PIPS} pips")
print(f"  Win per trade: ${WIN_AMOUNT:,.2f}")
print(f"  Loss per trade: ${LOSS_AMOUNT:,.2f}")
print(f"  Risk/Reward: {NET_WIN_PIPS/NET_LOSS_PIPS:.2f}")

# Load backtest results
try:
    df = pd.read_csv("backtest_results/backtest_20251106_035632.csv")
    
    # Calculate actual win rate from backtest
    # Use predictions where TP prob > threshold or SL prob > threshold
    tp_signals = df[df['tp_probability'] > config.TP_PROBABILITY_THRESHOLD]
    sl_signals = df[df['sl_probability'] > config.SL_PROBABILITY_THRESHOLD]
    
    # For TP signals (long trades)
    tp_wins = tp_signals[tp_signals['tp_hit'] == 1]
    tp_losses = tp_signals[tp_signals['sl_hit'] == 1]
    
    # For SL signals (avoid trades or short trades)
    # This is trickier - we'll focus on TP signals for long trades
    
    total_tp_trades = len(tp_signals)
    tp_win_rate = len(tp_wins) / total_tp_trades if total_tp_trades > 0 else 0
    
    print(f"\nBacktest Results (from model predictions):")
    print(f"  Total TP signals: {total_tp_trades}")
    print(f"  TP hits: {len(tp_wins)}")
    print(f"  SL hits (losses): {len(tp_losses)}")
    print(f"  Estimated win rate: {tp_win_rate:.2%}")
    
    # Calculate returns
    if total_tp_trades > 0:
        wins = len(tp_wins)
        losses = len(tp_losses)
        
        daily_profit = (wins * WIN_AMOUNT) - (losses * LOSS_AMOUNT)
        daily_profit_normalized = daily_profit * (TRADES_PER_DAY / total_tp_trades)
        
        # Annualized (assuming 252 trading days)
        annual_profit = daily_profit_normalized * 252
        
        print(f"\nEstimated Returns (scaled to {TRADES_PER_DAY} trades/day):")
        print(f"  Daily profit: ${daily_profit_normalized:,.2f}")
        print(f"  Weekly profit: ${daily_profit_normalized * 5:,.2f}")
        print(f"  Monthly profit: ${daily_profit_normalized * 20:,.2f}")
        print(f"  Annual profit: ${annual_profit:,.2f}")
        
        # With $10,000 initial balance
        INITIAL_BALANCE = 10000
        daily_return_pct = (daily_profit_normalized / INITIAL_BALANCE) * 100
        annual_return_pct = (annual_profit / INITIAL_BALANCE) * 100
        
        print(f"\nReturns on ${INITIAL_BALANCE:,} account:")
        print(f"  Daily return: {daily_return_pct:.2f}%")
        print(f"  Annual return: {annual_return_pct:.2f}%")
        
        # Calculate at different win rates for comparison
        print(f"\n{'='*70}")
        print("SCENARIO ANALYSIS - Different Win Rates")
        print("="*70)
        
        for win_rate in [0.50, 0.52, 0.55, 0.60, 0.65, 0.70]:
            trades_per_day = TRADES_PER_DAY
            wins_per_day = trades_per_day * win_rate
            losses_per_day = trades_per_day * (1 - win_rate)
            
            daily_pnl = (wins_per_day * WIN_AMOUNT) - (losses_per_day * LOSS_AMOUNT)
            annual_pnl = daily_pnl * 252
            annual_return = (annual_pnl / INITIAL_BALANCE) * 100
            
            print(f"  {win_rate:.0%} win rate: ${daily_pnl:,.2f}/day | ${annual_pnl:,.2f}/year ({annual_return:.1f}%)")
    
    else:
        print("\nNo TP signals found in backtest - cannot estimate returns")
        print("Using correlation-based estimate...")
        
        # Based on correlations
        tp_corr = 0.244
        sl_corr = 0.175
        
        # Rough estimate: if correlation is weak, assume ~50-52% win rate
        estimated_win_rate = 0.51
        
        trades_per_day = TRADES_PER_DAY
        wins_per_day = trades_per_day * estimated_win_rate
        losses_per_day = trades_per_day * (1 - estimated_win_rate)
        
        daily_pnl = (wins_per_day * WIN_AMOUNT) - (losses_per_day * LOSS_AMOUNT)
        annual_pnl = daily_pnl * 252
        annual_return = (annual_pnl / INITIAL_BALANCE) * 100
        
        print(f"\nEstimated Returns (based on {estimated_win_rate:.0%} win rate):")
        print(f"  Daily profit: ${daily_pnl:,.2f}")
        print(f"  Annual profit: ${annual_pnl:,.2f}")
        print(f"  Annual return: {annual_return:.2f}%")
        
except FileNotFoundError:
    print("\nBacktest file not found. Using theoretical calculations...")
    
    # Based on backtest correlations (weak signals)
    # Estimate conservative win rate around 51-52%
    estimated_win_rate = 0.515
    
    trades_per_day = TRADES_PER_DAY
    wins_per_day = trades_per_day * estimated_win_rate
    losses_per_day = trades_per_day * (1 - estimated_win_rate)
    
    daily_pnl = (wins_per_day * WIN_AMOUNT) - (losses_per_day * LOSS_AMOUNT)
    annual_pnl = daily_pnl * 252
    
    print(f"\nEstimated Returns (based on {estimated_win_rate:.0%} win rate):")
    print(f"  Daily profit: ${daily_pnl:,.2f}")
    print(f"  Annual profit: ${annual_pnl:,.2f}")
    print(f"  Annual return: {(annual_pnl / 10000) * 100:.2f}%")

print("\n" + "="*70)
print("WARNING: These are estimates based on backtest performance.")
print("Actual results may vary significantly due to:")
print("  - Market conditions")
print("  - Model limitations (weak correlations)")
print("  - Execution quality")
print("  - Slippage and costs")
print("="*70)

