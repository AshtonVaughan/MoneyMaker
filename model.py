"""
LSTM Model Architecture for TP/SL Probability Prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import config


class ScalpingLSTM:
    """LSTM model for predicting TP/SL hit probabilities"""
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units: list = [32, 16],  # Reduced complexity
        dropout_rate: float = 0.5,  # Increased dropout
        learning_rate: float = None,
        l2_reg: float = 0.001,  # L2 regularization
        temperature: float = 1.5  # Temperature scaling for calibration
    ):
        """
        Initialize LSTM model
        
        Args:
            input_shape: (sequence_length, num_features)
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate (defaults to config.LEARNING_RATE)
            l2_reg: L2 regularization strength
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.l2_reg = l2_reg
        self.temperature = temperature
        self.model = None
        
    def build_model(self) -> keras.Model:
        """Build the LSTM model architecture with regularization"""
        model = keras.Sequential()
        
        # L2 regularizer
        l2_reg = keras.regularizers.l2(self.l2_reg)
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # LSTM layers with regularization
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate * 0.5,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg
            ))
        
        # Dense layers with regularization (simplified)
        model.add(layers.Dense(16, activation='relu', kernel_regularizer=l2_reg))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(8, activation='relu', kernel_regularizer=l2_reg))
        model.add(layers.Dropout(self.dropout_rate * 0.5))
        
        # Output layer: 2 probabilities [P(TP hit), P(SL hit)]
        # Using softmax-like constraint: probabilities should sum to reasonable values
        model.add(layers.Dense(2, activation='sigmoid', kernel_regularizer=l2_reg))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        return model
    
    def get_model(self) -> keras.Model:
        """Get or build model"""
        if self.model is None:
            self.build_model()
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        self.model.summary()
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def predict(self, X: tf.Tensor) -> np.ndarray:
        """
        Predict TP/SL probabilities
        
        Args:
            X: Input sequences (batch_size, sequence_length, num_features)
        
        Returns:
            Array of shape (batch_size, 2) with [TP_prob, SL_prob]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or build_model() first.")
        return self.model.predict(X, verbose=0)
    
    def predict_single(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        Predict for a single sequence with temperature scaling
        
        Args:
            sequence: Single sequence (sequence_length, num_features)
        
        Returns:
            Tuple of (TP_probability, SL_probability)
        """
        # Add batch dimension
        X = np.expand_dims(sequence, axis=0)
        predictions = self.predict(X)
        
        # Apply temperature scaling to calibrate probabilities
        # Convert to logits, scale, then back to probabilities
        tp_logit = np.log(predictions[0][0] / (1 - predictions[0][0] + 1e-10))
        sl_logit = np.log(predictions[0][1] / (1 - predictions[0][1] + 1e-10))
        
        tp_scaled = 1 / (1 + np.exp(-tp_logit / self.temperature))
        sl_scaled = 1 / (1 + np.exp(-sl_logit / self.temperature))
        
        return float(tp_scaled), float(sl_scaled)


def create_model(
    sequence_length: int = None,
    num_features: int = None,
    lstm_units: list = None,
    dropout_rate: float = None
) -> ScalpingLSTM:
    """
    Factory function to create model
    
    Args:
        sequence_length: Length of input sequences
        num_features: Number of features
        lstm_units: LSTM layer units
        dropout_rate: Dropout rate
    
    Returns:
        ScalpingLSTM instance
    """
    seq_len = sequence_length or config.SEQUENCE_LENGTH
    lstm_units = lstm_units or [64, 32]
    dropout = dropout_rate or 0.3
    
    model = ScalpingLSTM(
        input_shape=(seq_len, num_features),
        lstm_units=lstm_units,
        dropout_rate=dropout
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model(sequence_length=60, num_features=30)
    model.summary()
    
    # Test prediction
    import numpy as np
    test_input = np.random.randn(1, 60, 30)
    predictions = model.predict(test_input)
    print(f"\nTest prediction shape: {predictions.shape}")
    print(f"TP prob: {predictions[0][0]:.4f}, SL prob: {predictions[0][1]:.4f}")

