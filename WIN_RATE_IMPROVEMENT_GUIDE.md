# How to Get 60-65% Win Rate

## Improvements Made

### 1. Advanced Features Added (20+ new features)
- **Order Flow**: Price momentum, momentum strength, trend strength
- **Market Microstructure**: Spread indicators, wicks, body size
- **Price Action Patterns**: Bullish/bearish engulfing patterns
- **Support/Resistance**: Near support/resistance detection
- **Time-based Features**: Market session indicators (London, NY, overlap)
- **Mean Reversion**: Distance from MA, mean reversion signals
- **Volatility**: Expansion/contraction indicators
- **Momentum Divergence**: RSI-price divergence detection
- **Bollinger Squeeze**: Compression/expansion signals
- **Volume Confirmation**: Volume-price confirmation

### 2. Improved Label Engineering
- Uses momentum direction (70% of TP target) to break ties
- Considers which level hit first when both TP/SL hit
- More accurate labels for model training

### 3. Training Improvements
- More epochs (100 instead of 50)
- Lower learning rate (0.0003) for fine-tuning
- More training data (trying to get 50,000 bars = ~35 days)

### 4. Model Architecture
- Already optimized with regularization
- Dropout, L2 regularization, label smoothing
- Temperature scaling for calibration

## Expected Results

With these improvements, you should see:
- **More features**: ~50+ features (was 33)
- **Better signal quality**: Momentum-based features capture scalping patterns
- **Session awareness**: Time-based features capture volatility patterns
- **Improved win rate**: Target 60-65% (was 43%)

## Next Steps

1. **Wait for training to complete** (~10-15 minutes)
2. **Run backtest**: `python backtest.py`
3. **Check correlations**: Should be > 0.3 (was 0.24/0.17)
4. **Check win rate**: Should be 60%+ (was 43%)

## If Win Rate Still Low

1. **Get more data**: Try to get 3+ months of data
2. **Feature selection**: Remove noisy features
3. **Ensemble models**: Combine multiple models
4. **Trade filtering**: Only trade in high-probability setups
5. **Adjust TP/SL**: Maybe 2.5 TP / 1.5 SL works better

## Key Metrics to Watch

- **TP Correlation**: Should be > 0.3
- **SL Correlation**: Should be > 0.3  
- **Win Rate**: Should be 60%+
- **Average Probabilities**: Should be 60-80% (not 99%+)

