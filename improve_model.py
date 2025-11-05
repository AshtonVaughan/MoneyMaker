"""
Script to improve model performance by:
1. Getting more training data
2. Using better features
3. Training with improved labels
4. Adding trade filtering
"""

import os
from datetime import datetime, timedelta
from trainer import ModelTrainer
from mt5_connector import MT5Connector
import config

def get_more_training_data():
    """Get maximum available historical data"""
    connector = MT5Connector()
    if not connector.connect():
        raise ConnectionError("Failed to connect to MT5")
    
    print("Downloading maximum available historical data...")
    
    # Try to get as much data as possible
    import MetaTrader5 as mt5
    
    # Try different methods to get maximum data
    max_bars = 50000  # Try to get ~35 days of 1-minute data
    
    rates = mt5.copy_rates_from_pos(
        config.MT5_SYMBOL,
        config.MT5_TIMEFRAME_M1,
        0,
        max_bars
    )
    
    connector.disconnect()
    
    if rates is None or len(rates) == 0:
        print("Warning: Could not get more data")
        return None
    
    print(f"Downloaded {len(rates)} bars")
    return rates

def train_improved_model():
    """Train model with all improvements"""
    print("="*70)
    print("TRAINING IMPROVED MODEL FOR 60-65% WIN RATE")
    print("="*70)
    
    trainer = ModelTrainer()
    
    # Train with more data and better settings
    print("\nTraining with improved features and more data...")
    model = trainer.train_from_mt5(months=None, epochs=100)  # More epochs
    
    print("\n" + "="*70)
    print("Training completed!")
    print("Next: Run backtest to verify improved performance")
    print("="*70)
    
    return model

if __name__ == "__main__":
    train_improved_model()

