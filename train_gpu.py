"""
GPU-Optimized Training Script for Cloud GPU (RTX 5090)
This script loads data from HDF5 files and trains efficiently on GPU
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from trainer import ModelTrainer
from data_processor import DataProcessor
import config

# GPU Configuration
def setup_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  {gpu.name}")
        
        # Enable memory growth to avoid OOM
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
        
        # Enable mixed precision for faster training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision training enabled (FP16)")
        
        return True
    else:
        print("No GPU found - using CPU")
        return False


def load_historical_data(data_path: str) -> pd.DataFrame:
    """
    Load historical data from HDF5 or CSV
    
    Args:
        data_path: Path to data file (.h5 or .csv)
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        try:
            df = pd.read_hdf(data_path, key='data')
        except ImportError:
            raise ImportError("pytables required for HDF5. Install with: pip install tables")
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Loaded {len(df):,} bars")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def train_on_large_dataset(
    data_path: str,
    epochs: int = 100,
    batch_size: int = 128,  # Larger batch for GPU
    validation_split: float = 0.2
):
    """
    Train model on large historical dataset
    
    Args:
        data_path: Path to historical data file
        epochs: Number of training epochs
        batch_size: Batch size (larger for GPU)
        validation_split: Validation split ratio
    """
    print("="*70)
    print("GPU TRAINING ON LARGE DATASET")
    print("="*70)
    
    # Setup GPU
    has_gpu = setup_gpu()
    
    if has_gpu:
        print("\nGPU Configuration:")
        print(f"  Batch size: {batch_size} (optimized for GPU)")
        print(f"  Mixed precision: Enabled")
    else:
        print("\nCPU Configuration:")
        print(f"  Batch size: {batch_size}")
    
    # Load data
    print("\nLoading historical data...")
    df = load_historical_data(data_path)
    
    # Process data
    print("\nProcessing data...")
    processor = DataProcessor()
    X, y, features = processor.prepare_training_data(df)
    
    print(f"\nTraining Data:")
    print(f"  Sequences: {len(X):,}")
    print(f"  Features: {len(features)}")
    print(f"  Sequence length: {X.shape[1]}")
    print(f"  TP hits: {y[:, 0].sum():,.0f}")
    print(f"  SL hits: {y[:, 1].sum():,.0f}")
    
    # Train model
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    trainer = ModelTrainer()
    trainer.processor = processor
    
    model = trainer.train(
        X=X,
        y=y,
        num_features=len(features),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model on GPU with large dataset")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to historical data file (.h5 or .csv)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128 for GPU)"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    train_on_large_dataset(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )

