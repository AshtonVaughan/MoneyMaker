"""
Configuration settings for MT5 Live Scalping System
"""

# MT5 Settings
MT5_SYMBOL = "EURUSD"
MT5_TIMEFRAME_M1 = 1  # 1-minute bars
MT5_TIMEFRAME_TICKS = 0  # Tick data

# Trading Parameters
TAKE_PROFIT_PIPS = 2.0
STOP_LOSS_PIPS = 1.5
SPREAD_PIPS = 0.5  # Raw spread assumption

# Data Collection
LIVE_UPDATE_INTERVAL_SECONDS = 3
HISTORICAL_MONTHS = 3
SEQUENCE_LENGTH = 60  # 60 timesteps (1 hour of 1-minute data)

# Model Settings
MODEL_INPUT_FEATURES = None  # Will be set after feature engineering
BATCH_SIZE = 64  # Increased batch size for stability
EPOCHS = 100  # More epochs for better learning
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0003  # Lower learning rate for fine-tuning

# Paths
MODELS_DIR = "models"
LOGS_DIR = "logs"
DATA_DIR = "data"

# Prediction Thresholds
TP_PROBABILITY_THRESHOLD = 0.6
SL_PROBABILITY_THRESHOLD = 0.6
MIN_CONFIDENCE = 0.55

