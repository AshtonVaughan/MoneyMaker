"""
Live Prediction System - Real-time predictions every 3 seconds
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Tuple
import config
from mt5_connector import MT5Connector
from data_processor import DataProcessor
from model import ScalpingLSTM
from forward_test_logger import ForwardTestLogger


class LivePredictor:
    """Handles live predictions from MT5 data"""
    
    def __init__(
        self,
        model: ScalpingLSTM,
        processor: DataProcessor,
        logger: ForwardTestLogger = None
    ):
        """
        Initialize live predictor
        
        Args:
            model: Trained LSTM model
            processor: Fitted data processor
            logger: Forward test logger (optional)
        """
        self.model = model
        self.processor = processor
        self.logger = logger
        
        # Data buffer for sequences
        self.data_buffer = deque(maxlen=config.SEQUENCE_LENGTH + 10)
        self.last_sequence = None
        self.feature_names = None
        
        # Load feature names if available
        self._load_feature_names()
    
    def _load_feature_names(self):
        """Load feature names from saved model"""
        import os
        feature_path = os.path.join(config.MODELS_DIR, "feature_names.txt")
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.feature_names)} feature names")
    
    def update_buffer(self, tick_data: dict) -> bool:
        """
        Update data buffer with new tick
        
        Args:
            tick_data: Dictionary with tick data (time, bid, ask, etc.)
        
        Returns:
            True if buffer is ready for prediction
        """
        # Convert tick to OHLC format (using bid price)
        current_price = tick_data['bid']
        current_time = tick_data['time']
        
        # Create a minimal bar from tick
        bar_data = {
            'time': current_time,
            'open': current_price,
            'high': tick_data.get('ask', current_price),
            'low': current_price,
            'close': current_price,
            'tick_volume': tick_data.get('volume', 1)
        }
        
        self.data_buffer.append(bar_data)
        
        # Need at least SEQUENCE_LENGTH + enough for indicators
        return len(self.data_buffer) >= config.SEQUENCE_LENGTH + 50
    
    def prepare_sequence(self) -> Optional[np.ndarray]:
        """
        Prepare sequence from buffer for prediction
        
        Returns:
            Normalized sequence array or None if not enough data
        """
        if len(self.data_buffer) < config.SEQUENCE_LENGTH + 50:
            return None
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.data_buffer))
        df.set_index('time', inplace=True)
        
        # Calculate indicators
        df_with_indicators = self.processor.calculate_indicators(df)
        
        # Select features (same as training)
        if self.feature_names:
            # Use same features as training
            available_features = [f for f in self.feature_names if f in df_with_indicators.columns]
            if len(available_features) < len(self.feature_names) * 0.8:
                # Fallback: use all numeric features
                feature_cols = df_with_indicators.select_dtypes(include=[np.number]).columns.tolist()
            else:
                feature_cols = available_features
        else:
            # Fallback: use all numeric features
            feature_cols = df_with_indicators.select_dtypes(include=[np.number]).columns.tolist()
        
        # Get last SEQUENCE_LENGTH rows
        df_features = df_with_indicators[feature_cols].dropna()
        
        if len(df_features) < config.SEQUENCE_LENGTH:
            return None
        
        sequence = df_features.iloc[-config.SEQUENCE_LENGTH:].values
        
        # Normalize
        sequence_normalized = self.processor.normalize_features(
            np.expand_dims(sequence, axis=0)
        )[0]
        
        self.last_sequence = sequence_normalized
        return sequence_normalized
    
    def predict(self, sequence: np.ndarray = None) -> Tuple[float, float]:
        """
        Predict TP/SL probabilities
        
        Args:
            sequence: Input sequence (optional, uses last prepared sequence if None)
        
        Returns:
            Tuple of (TP_probability, SL_probability)
        """
        if sequence is None:
            sequence = self.last_sequence
            if sequence is None:
                raise ValueError("No sequence available. Call prepare_sequence() first.")
        
        # Add batch dimension
        X = np.expand_dims(sequence, axis=0)
        
        # Predict
        predictions = self.model.predict(X)
        
        tp_prob = float(predictions[0][0])
        sl_prob = float(predictions[0][1])
        
        return tp_prob, sl_prob
    
    def process_tick(self, tick_data: dict) -> Optional[dict]:
        """
        Process a new tick and generate prediction
        
        Args:
            tick_data: Dictionary with tick data
        
        Returns:
            Dictionary with prediction results or None if not ready
        """
        # Update buffer
        buffer_ready = self.update_buffer(tick_data)
        
        if not buffer_ready:
            return None
        
        # Prepare sequence
        sequence = self.prepare_sequence()
        if sequence is None:
            return None
        
        # Predict
        try:
            tp_prob, sl_prob = self.predict(sequence)
            
            # Get signal
            from forward_test_logger import ForwardTestLogger
            signal_info = ForwardTestLogger().get_signal(tp_prob, sl_prob)
            
            result = {
                'timestamp': tick_data['time'],
                'price': tick_data['bid'],
                'tp_probability': tp_prob,
                'sl_probability': sl_prob,
                'action': signal_info['action'],
                'confidence': signal_info['confidence'],
                'reason': signal_info.get('reason', '')
            }
            
            # Log if logger available
            if self.logger:
                self.logger.log_prediction(
                    timestamp=result['timestamp'],
                    current_price=result['price'],
                    tp_probability=result['tp_probability'],
                    sl_probability=result['sl_probability'],
                    suggested_action=result['action'],
                    confidence=result['confidence']
                )
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None


if __name__ == "__main__":
    # Test with sample data
    print("LivePredictor test - requires trained model")
    print("Run trainer.py first to create a model")

