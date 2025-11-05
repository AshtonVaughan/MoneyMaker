"""
Training pipeline for LSTM scalping model
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import config
from mt5_connector import MT5Connector
from data_processor import DataProcessor
from model import ScalpingLSTM, create_model


class ModelTrainer:
    """Handles model training pipeline"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.model = None
        
    def load_training_data(self, months: int = None) -> tuple:
        """
        Load and prepare training data from MT5
        
        Args:
            months: Number of months of data (defaults to config.HISTORICAL_MONTHS)
        
        Returns:
            Tuple of (X_train, y_train, feature_names)
        """
        print("Connecting to MT5...")
        connector = MT5Connector()
        if not connector.connect():
            raise ConnectionError("Failed to connect to MT5")
        
        print(f"Downloading {months or config.HISTORICAL_MONTHS} months of historical data...")
        df = connector.get_historical_bars(months=months)
        connector.disconnect()
        
        if df is None or len(df) == 0:
            raise ValueError("Failed to download historical data")
        
        print(f"Processing {len(df)} bars...")
        X, y, features = self.processor.prepare_training_data(df)
        
        print(f"Created {len(X)} sequences with {len(features)} features")
        print(f"TP hits: {y[:, 0].sum()}, SL hits: {y[:, 1].sum()}")
        
        return X, y, features
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_features: int,
        validation_split: float = None,
        epochs: int = None,
        batch_size: int = None,
        save_path: str = None
    ) -> ScalpingLSTM:
        """
        Train the LSTM model
        
        Args:
            X: Training sequences
            y: Training labels
            num_features: Number of features
            validation_split: Validation split ratio
            epochs: Number of epochs
            batch_size: Batch size
            save_path: Path to save model
        
        Returns:
            Trained model
        """
        val_split = validation_split or config.VALIDATION_SPLIT
        epochs_count = epochs or config.EPOCHS
        batch = batch_size or config.BATCH_SIZE
        
        # Apply label smoothing to prevent overconfidence
        label_smoothing = 0.1
        y_smooth = y * (1 - 2 * label_smoothing) + label_smoothing
        print(f"Applied label smoothing: {label_smoothing}")
        
        # Create model
        print(f"Building model with input shape: {X.shape[1:]}")
        self.model = create_model(
            sequence_length=X.shape[1],
            num_features=num_features
        )
        model = self.model.get_model()
        
        # Create save directory
        if save_path is None:
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(config.MODELS_DIR, f"scalping_model_{timestamp}.h5")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,  # More aggressive early stopping
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model with smoothed labels
        print("\nStarting training...")
        history = model.fit(
            X, y_smooth,  # Use smoothed labels
            batch_size=batch,
            epochs=epochs_count,
            validation_split=val_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        self.model.load(save_path)
        
        # Evaluate
        print("\nEvaluating model...")
        val_loss, val_acc, val_prec, val_rec = model.evaluate(
            X[int(len(X) * (1 - val_split)):],
            y[int(len(y) * (1 - val_split)):],
            verbose=0
        )
        
        print(f"\nValidation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  Precision: {val_prec:.4f}")
        print(f"  Recall: {val_rec:.4f}")
        
        return self.model
    
    def train_from_mt5(
        self,
        months: int = None,
        epochs: int = None,
        batch_size: int = None
    ) -> ScalpingLSTM:
        """
        Complete training pipeline from MT5 data
        
        Args:
            months: Number of months of data
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Trained model
        """
        # Load data
        X, y, features = self.load_training_data(months=months)
        
        # Train model
        model = self.train(
            X, y,
            num_features=len(features),
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save feature names for later use
        feature_path = os.path.join(config.MODELS_DIR, "feature_names.txt")
        with open(feature_path, 'w') as f:
            f.write('\n'.join(features))
        print(f"\nFeature names saved to {feature_path}")
        
        return model


if __name__ == "__main__":
    trainer = ModelTrainer()
    
    print("="*70)
    print("TRAINING SCALPING MODEL")
    print("="*70)
    
    try:
        model = trainer.train_from_mt5(months=3, epochs=50)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

