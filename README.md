# MT5 Live Scalping System

A deep learning-based scalping system for EUR/USD that uses LSTM neural networks to predict Take Profit (TP) and Stop Loss (SL) hit probabilities in real-time.

## Features

- **Live Data Collection**: Fetches EUR/USD tick data from MT5 every 3 seconds
- **Deep Learning Model**: LSTM neural network for sequence pattern recognition
- **TP/SL Prediction**: Predicts probability of hitting TP (2 pips) vs SL (1.5 pips)
- **Forward Testing**: Logs predictions and signals without executing trades
- **Real-time Processing**: Continuous prediction loop with live market data

## Requirements

- MetaTrader 5 terminal installed and running
- Python 3.8+
- MT5 account (demo or live)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install MetaTrader 5 terminal from [MetaQuotes](https://www.metatrader5.com/)

3. (Optional) Create `.env` file for MT5 credentials:
```bash
cp .env.example .env
# Edit .env with your MT5 credentials
```

If you don't provide credentials, the system will use the currently logged-in MT5 terminal.

## Usage

### Step 1: Train the Model

First, train the LSTM model on historical data:

```bash
python trainer.py
```

This will:
- Download 3 months of historical EUR/USD data from MT5
- Calculate technical indicators
- Create training sequences
- Train the LSTM model
- Save the model to `models/` directory

### Step 2: Run Live Predictions

Start the live prediction system:

```bash
python main.py
```

Or with MT5 credentials:
```bash
python main.py --login YOUR_LOGIN --password YOUR_PASSWORD --server YOUR_SERVER
```

The system will:
- Connect to MT5
- Load the trained model
- Fetch live tick data every 3 seconds
- Generate TP/SL probability predictions
- Log all predictions to CSV files in `logs/` directory

### Step 3: View Results

Check the log files in `logs/` directory for all predictions and signals.

## Configuration

Edit `config.py` to customize:

- `TAKE_PROFIT_PIPS`: TP level (default: 2.0 pips)
- `STOP_LOSS_PIPS`: SL level (default: 1.5 pips)
- `LIVE_UPDATE_INTERVAL_SECONDS`: Update frequency (default: 3 seconds)
- `SEQUENCE_LENGTH`: Input sequence length (default: 60 timesteps)
- `TP_PROBABILITY_THRESHOLD`: Minimum TP probability for BUY signal (default: 0.6)
- `SL_PROBABILITY_THRESHOLD`: Minimum SL probability for SELL signal (default: 0.6)

## Project Structure

```
MoneyMaker/
├── config.py                  # Configuration settings
├── mt5_connector.py           # MT5 API integration
├── data_processor.py          # Feature engineering
├── model.py                   # LSTM model architecture
├── trainer.py                 # Training pipeline
├── live_predictor.py          # Real-time predictions
├── forward_test_logger.py    # Signal logging
├── main.py                    # Main application
├── models/                    # Saved model checkpoints
├── logs/                      # Prediction logs
└── requirements.txt           # Dependencies
```

## How It Works

1. **Data Collection**: MT5 connector fetches live tick data every 3 seconds
2. **Feature Engineering**: Technical indicators are calculated (RSI, MACD, Bollinger Bands, etc.)
3. **Sequence Creation**: Last 60 timesteps are used as input to the LSTM
4. **Prediction**: Model outputs two probabilities: P(TP hit) and P(SL hit)
5. **Signal Generation**: BUY/SELL/HOLD signals based on probability thresholds
6. **Logging**: All predictions are logged to CSV for forward testing

## Model Architecture

- **Input**: 60 timesteps × N features (technical indicators)
- **LSTM Layers**: 2 layers (64 and 32 units) with dropout
- **Dense Layers**: 32 and 16 units with dropout
- **Output**: 2 probabilities [P(TP), P(SL)]

## Cloud GPU Training

For training on large datasets (1M+ bars) with GPU acceleration:

1. **Download Historical Data**:
   ```bash
   python download_historical_data.py --bars 1000000 --sources dukascopy
   ```

2. **Upload to Cloud GPU** (Vast.ai, RunPod, etc.)

3. **Train on GPU**:
   ```bash
   python train_gpu.py --data data/historical/eurusd_historical_YYYYMMDD.h5 --epochs 200
   ```

See `CLOUD_GPU_SETUP.md` for detailed instructions.

## Notes

- The system does NOT execute trades - it only logs predictions
- Make sure MT5 terminal is running before starting the system
- For large datasets, use cloud GPU training (see CLOUD_GPU_SETUP.md)
- Model requires retraining periodically for best performance
- All predictions are logged for analysis and forward testing

## Troubleshooting

**MT5 Connection Failed**:
- Ensure MT5 terminal is installed and running
- Check if MT5 terminal is logged in
- Verify credentials in `.env` file

**Model Not Found**:
- Run `trainer.py` first to train a model
- Check `models/` directory for `.h5` files

**Prediction Errors**:
- Ensure enough historical data is available
- Check that MT5 terminal has internet connection
- Verify symbol name (EURUSD) is available in MT5

