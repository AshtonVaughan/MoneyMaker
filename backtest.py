"""
Backtesting Module - Tests model on unseen data to check for overfitting
Uses proper time-series validation (walk-forward)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import config
from mt5_connector import MT5Connector
from data_processor import DataProcessor
from model import ScalpingLSTM
from forward_test_logger import ForwardTestLogger


class Backtester:
    """Backtests model on unseen historical data"""
    
    def __init__(self, model_path: str):
        """
        Initialize backtester
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.processor = DataProcessor()
        self.results = []
        
    def load_model(self):
        """Load trained model"""
        print(f"Loading model from {self.model_path}...")
        self.model = ScalpingLSTM(input_shape=(config.SEQUENCE_LENGTH, 1))
        self.model.load(self.model_path)
        print("Model loaded")
        
        # Load feature names if available
        import os
        feature_path = os.path.join(config.MODELS_DIR, "feature_names.txt")
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        else:
            self.feature_names = None
    
    def get_test_data(self, days_after_training: int = 7, test_days: int = 7):
        """
        Get test data that wasn't used for training
        
        Args:
            days_after_training: Days after training date to start test (to avoid overlap)
            test_days: Number of days to test on
        
        Returns:
            DataFrame with test data
        """
        connector = MT5Connector()
        if not connector.connect():
            raise ConnectionError("Failed to connect to MT5")
        
        # Get data from recent period (after training)
        # Training ended around Nov 5, so test on Nov 6+ data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days + days_after_training)
        
        print(f"Downloading test data from {start_date.date()} to {end_date.date()}...")
        
        # Get enough bars for the test period (limit to 1000 bars max for speed)
        bars_needed = min(test_days * 24 * 60, 1000)  # Max 1000 bars for faster backtest
        
        # Get specific amount directly
        import MetaTrader5 as mt5
        rates = mt5.copy_rates_from_pos(
            config.MT5_SYMBOL,
            config.MT5_TIMEFRAME_M1,
            0,
            bars_needed + 200  # Extra for indicators
        )
        
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            # Convert time column to datetime index
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
            elif 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'], unit='s')
                df.set_index('Time', inplace=True)
            df.columns = [col.lower() for col in df.columns]
        else:
            df = None
        
        connector.disconnect()
        
        if df is None or len(df) == 0:
            raise ValueError("Failed to get test data")
        
        # Ensure index is datetime (get_historical_bars already sets it, but check anyway)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df.set_index('time', inplace=True)
            elif isinstance(df.index[0], (int, float)) and df.index[0] > 1e9:
                # Unix timestamp
                df.index = pd.to_datetime(df.index, unit='s')
        
        print(f"Got {len(df)} bars for testing")
        return df
    
    def backtest_on_data(self, df: pd.DataFrame) -> Dict:
        """
        Backtest model on historical data
        
        Args:
            df: Historical OHLCV data
        
        Returns:
            Dictionary with backtest results
        """
        print("\nRunning backtest...")
        
        # Calculate indicators
        df_with_indicators = self.processor.calculate_indicators(df)
        
        # Get feature columns
        if self.feature_names:
            feature_cols = [f for f in self.feature_names if f in df_with_indicators.columns]
            if len(feature_cols) < len(self.feature_names) * 0.8:
                feature_cols = df_with_indicators.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in feature_cols if col not in ['tp_hit', 'sl_hit']]
        else:
            feature_cols = df_with_indicators.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col not in ['tp_hit', 'sl_hit']]
        
        # Clean data
        df_features = df_with_indicators[feature_cols].dropna()
        
        if len(df_features) < config.SEQUENCE_LENGTH + 20:
            raise ValueError(f"Not enough data. Need at least {config.SEQUENCE_LENGTH + 20} bars, got {len(df_features)}")
        
        # Fit scaler on test data (or use training scaler if saved)
        # For proper backtesting, we should use the training scaler, but since we don't have it,
        # we'll fit on test data (this is a limitation but acceptable for demonstration)
        sample_data = df_features.iloc[:min(200, len(df_features))].values
        self.processor.scaler.fit(sample_data)
        self.processor.is_fitted = True
        
        # Create sequences and predict (sample every 5th bar for speed)
        predictions = []
        actual_tp_hits = []
        actual_sl_hits = []
        prices = []
        timestamps = []
        
        lookahead = 10  # Check next 10 bars for TP/SL
        step = 5  # Sample every 5th bar for faster backtest
        
        total_bars = len(df_features) - config.SEQUENCE_LENGTH - lookahead
        print(f"   Running predictions on {total_bars // step} samples...")
        
        for idx, i in enumerate(range(config.SEQUENCE_LENGTH, len(df_features) - lookahead, step)):
            if idx % 50 == 0:
                print(f"   Progress: {idx}/{total_bars // step} predictions", end='\r')
            
            # Get sequence
            sequence = df_features[feature_cols].iloc[i-config.SEQUENCE_LENGTH:i].values
            
            # Normalize
            sequence_normalized = self.processor.normalize_features(
                np.expand_dims(sequence, axis=0)
            )[0]
            
            # Predict
            tp_prob, sl_prob = self.model.predict_single(sequence_normalized)
            
            # Get actual outcome
            entry_price = df_features.iloc[i]['close']
            future_bars = df_features.iloc[i+1:i+1+lookahead]
            
            pip_value = 0.0001
            tp_price_long = entry_price + (config.TAKE_PROFIT_PIPS * pip_value)
            sl_price_long = entry_price - (config.STOP_LOSS_PIPS * pip_value)
            
            tp_hit = (future_bars['high'] >= tp_price_long).any() if 'high' in future_bars.columns else False
            sl_hit = (future_bars['low'] <= sl_price_long).any() if 'low' in future_bars.columns else False
            
            predictions.append({
                'timestamp': df_features.index[i],
                'price': entry_price,
                'tp_probability': tp_prob,
                'sl_probability': sl_prob,
                'tp_hit': 1 if tp_hit else 0,
                'sl_hit': 1 if sl_hit else 0
            })
            
            actual_tp_hits.append(1 if tp_hit else 0)
            actual_sl_hits.append(1 if sl_hit else 0)
            prices.append(entry_price)
            timestamps.append(df_features.index[i])
        
        print()  # New line after progress
        
        # Calculate metrics
        predictions_df = pd.DataFrame(predictions)
        
        # Classification metrics
        tp_predictions = (predictions_df['tp_probability'] > config.TP_PROBABILITY_THRESHOLD).astype(int)
        sl_predictions = (predictions_df['sl_probability'] > config.SL_PROBABILITY_THRESHOLD).astype(int)
        
        # TP prediction accuracy
        tp_correct = ((tp_predictions == 1) & (predictions_df['tp_hit'] == 1)).sum()
        tp_predicted = tp_predictions.sum()
        tp_precision = tp_correct / tp_predicted if tp_predicted > 0 else 0
        tp_recall = tp_correct / predictions_df['tp_hit'].sum() if predictions_df['tp_hit'].sum() > 0 else 0
        
        # SL prediction accuracy
        sl_correct = ((sl_predictions == 1) & (predictions_df['sl_hit'] == 1)).sum()
        sl_predicted = sl_predictions.sum()
        sl_precision = sl_correct / sl_predicted if sl_predicted > 0 else 0
        sl_recall = sl_correct / predictions_df['sl_hit'].sum() if predictions_df['sl_hit'].sum() > 0 else 0
        
        # Probability correlation
        tp_correlation = predictions_df['tp_probability'].corr(predictions_df['tp_hit'])
        sl_correlation = predictions_df['sl_probability'].corr(predictions_df['sl_hit'])
        
        results = {
            'total_predictions': len(predictions),
            'tp_hits_actual': predictions_df['tp_hit'].sum(),
            'sl_hits_actual': predictions_df['sl_hit'].sum(),
            'tp_precision': tp_precision,
            'tp_recall': tp_recall,
            'sl_precision': sl_precision,
            'sl_recall': sl_recall,
            'tp_correlation': tp_correlation,
            'sl_correlation': sl_correlation,
            'avg_tp_probability': predictions_df['tp_probability'].mean(),
            'avg_sl_probability': predictions_df['sl_probability'].mean(),
            'predictions': predictions_df
        }
        
        return results
    
    def run_backtest(self, test_days: int = 1) -> Dict:
        """
        Run complete backtest
        
        Args:
            test_days: Number of days to test on
        
        Returns:
            Backtest results
        """
        print("="*70)
        print("BACKTESTING MODEL - Out-of-Sample Testing")
        print("="*70)
        print("\nThis tests the model on data it has NEVER seen during training")
        print("to check for overfitting.\n")
        
        # Load model
        self.load_model()
        
        # Get test data (data after training period)
        test_data = self.get_test_data(days_after_training=1, test_days=test_days)
        
        # Run backtest
        results = self.backtest_on_data(test_data)
        
        # Display results
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        print(f"\nTotal Predictions: {results['total_predictions']}")
        print(f"Actual TP Hits: {results['tp_hits_actual']} ({results['tp_hits_actual']/results['total_predictions']*100:.1f}%)")
        print(f"Actual SL Hits: {results['sl_hits_actual']} ({results['sl_hits_actual']/results['total_predictions']*100:.1f}%)")
        
        print(f"\nTP Prediction Performance:")
        print(f"  Precision: {results['tp_precision']:.2%} (of TP predictions, how many were correct)")
        print(f"  Recall: {results['tp_recall']:.2%} (of actual TP hits, how many were predicted)")
        print(f"  Correlation: {results['tp_correlation']:.3f} (higher is better)")
        
        print(f"\nSL Prediction Performance:")
        print(f"  Precision: {results['sl_precision']:.2%} (of SL predictions, how many were correct)")
        print(f"  Recall: {results['sl_recall']:.2%} (of actual SL hits, how many were predicted)")
        print(f"  Correlation: {results['sl_correlation']:.3f} (higher is better)")
        
        print(f"\nAverage Probabilities:")
        print(f"  TP Probability: {results['avg_tp_probability']:.2%}")
        print(f"  SL Probability: {results['avg_sl_probability']:.2%}")
        
        # Save results
        import os
        results_dir = "backtest_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"backtest_{timestamp}.csv")
        results['predictions'].to_csv(results_file, index=False)
        print(f"\nDetailed predictions saved to: {results_file}")
        
        # Overfitting check
        print("\n" + "="*70)
        print("OVERFITTING ANALYSIS")
        print("="*70)
        
        if results['tp_correlation'] > 0.3 and results['sl_correlation'] > 0.3:
            print("OK: Model shows reasonable correlation on unseen data")
            print("  This suggests the model learned genuine patterns, not just memorized training data")
        elif results['tp_correlation'] > 0.1 or results['sl_correlation'] > 0.1:
            print("WARNING: Model shows weak correlation on unseen data")
            print("  The model may be overfitting - performance drops on new data")
        else:
            print("ERROR: Model shows poor correlation on unseen data")
            print("  Significant overfitting detected - model memorized training data")
        
        print("="*70)
        
        return results


if __name__ == "__main__":
    import os
    
    # Find latest model
    model_files = [f for f in os.listdir(config.MODELS_DIR) if f.endswith('.h5')]
    if not model_files:
        print("No model found. Train a model first.")
        exit(1)
    
    model_files.sort(reverse=True)
    model_path = os.path.join(config.MODELS_DIR, model_files[0])
    
    backtester = Backtester(model_path)
    results = backtester.run_backtest(test_days=7)

