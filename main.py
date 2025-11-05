"""
Main Application - Orchestrates MT5 Live Scalping System
"""

import os
import sys
import time
import signal
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
import config
import MetaTrader5 as mt5
from mt5_connector import MT5Connector
from data_processor import DataProcessor
from model import ScalpingLSTM
from live_predictor import LivePredictor
from forward_test_logger import ForwardTestLogger


class ScalpingSystem:
    """Main application orchestrator"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        mt5_login: Optional[int] = None,
        mt5_password: Optional[str] = None,
        mt5_server: Optional[str] = None
    ):
        """
        Initialize scalping system
        
        Args:
            model_path: Path to trained model (if None, looks in models/ directory)
            mt5_login: MT5 account login
            mt5_password: MT5 account password
            mt5_server: MT5 server name
        """
        self.model_path = model_path
        self.mt5_login = mt5_login
        self.mt5_password = mt5_password
        self.mt5_server = mt5_server
        
        self.connector = None
        self.processor = None
        self.model = None
        self.predictor = None
        self.logger = None
        self.running = False
        
    def initialize(self):
        """Initialize all components"""
        print("="*70)
        print("INITIALIZING MT5 LIVE SCALPING SYSTEM")
        print("="*70)
        
        # Initialize logger
        print("\n1. Initializing logger...")
        self.logger = ForwardTestLogger()
        print(f"   Logger initialized. Log file: {self.logger.log_file}")
        
        # Connect to MT5
        print("\n2. Connecting to MT5...")
        self.connector = MT5Connector(
            login=self.mt5_login,
            password=self.mt5_password,
            server=self.mt5_server
        )
        if not self.connector.connect():
            raise ConnectionError("Failed to connect to MT5")
        print("   MT5 connected successfully")
        
        # Load model
        print("\n3. Loading model...")
        if self.model_path is None:
            # Find latest model
            models_dir = config.MODELS_DIR
            if not os.path.exists(models_dir):
                raise FileNotFoundError(
                    f"No models directory found. Train a model first using trainer.py"
                )
            
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
            if not model_files:
                raise FileNotFoundError(
                    f"No model files found in {models_dir}. Train a model first using trainer.py"
                )
            
            # Use most recent model
            model_files.sort(reverse=True)
            self.model_path = os.path.join(models_dir, model_files[0])
            print(f"   Using model: {self.model_path}")
        
        # Load model
        self.model = ScalpingLSTM(input_shape=(config.SEQUENCE_LENGTH, 1))  # Dummy shape
        self.model.load(self.model_path)
        print("   Model loaded successfully")
        
        # Initialize processor
        print("\n4. Initializing data processor...")
        self.processor = DataProcessor()
        
        # Load minimal historical data to fit scaler
        print("   Fitting scaler on historical data...")
        try:
            # Only need ~200 bars to fit scaler (enough for indicators + buffer)
            # Get last 200 bars directly
            rates = mt5.copy_rates_from_pos(config.MT5_SYMBOL, config.MT5_TIMEFRAME_M1, 0, 200)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df.columns = [col.lower() for col in df.columns]
                print(f"   Downloaded {len(df)} bars")
            else:
                df = None
            
            if df is not None and len(df) > 100:
                print(f"   Processing {len(df)} bars...")
                # Calculate indicators first
                df_with_indicators = self.processor.calculate_indicators(df)
                print("   Indicators calculated")
                
                # Get feature columns
                feature_cols = df_with_indicators.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in feature_cols if col not in ['tp_hit', 'sl_hit']]
                
                # Fit scaler on features (only need recent data)
                df_features = df_with_indicators[feature_cols].dropna()
                if len(df_features) > 50:
                    # Use last 100-200 bars for fitting (enough for stable scaling)
                    sample_size = min(200, len(df_features))
                    sample_data = df_features.iloc[-sample_size:].values
                    self.processor.scaler.fit(sample_data)
                    self.processor.is_fitted = True
                    print(f"   Processor fitted with {len(feature_cols)} features")
                else:
                    print("   Warning: Not enough data for fitting")
            else:
                print("   Warning: Limited historical data for fitting")
        except Exception as e:
            print(f"   Warning: Could not fit processor: {e}")
            import traceback
            traceback.print_exc()
        
        # Initialize predictor
        print("\n5. Initializing live predictor...")
        self.predictor = LivePredictor(
            model=self.model,
            processor=self.processor,
            logger=self.logger
        )
        print("   Predictor initialized")
        
        print("\n" + "="*70)
        print("SYSTEM READY")
        print("="*70)
        print(f"Symbol: {config.MT5_SYMBOL}")
        print(f"Update interval: {config.LIVE_UPDATE_INTERVAL_SECONDS} seconds")
        print(f"TP: {config.TAKE_PROFIT_PIPS} pips, SL: {config.STOP_LOSS_PIPS} pips")
        print("="*70 + "\n")
    
    def run(self):
        """Run the live prediction system"""
        if self.predictor is None:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        self.running = True
        
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\nShutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("Starting live prediction loop...")
        print("Press Ctrl+C to stop\n")
        
        prediction_count = 0
        
        try:
            while self.running:
                # Get live tick
                tick = self.connector.get_live_tick()
                
                if tick is None:
                    print("Warning: Failed to get tick data")
                    time.sleep(config.LIVE_UPDATE_INTERVAL_SECONDS)
                    continue
                
                # Process tick and get prediction
                result = self.predictor.process_tick(tick)
                
                if result:
                    prediction_count += 1
                    
                    # Print prediction
                    print(f"[{result['timestamp'].strftime('%H:%M:%S')}] "
                          f"Price: {result['price']:.5f} | "
                          f"TP Prob: {result['tp_probability']:.2%} | "
                          f"SL Prob: {result['sl_probability']:.2%} | "
                          f"Action: {result['action']} | "
                          f"Confidence: {result['confidence']:.2%}")
                
                # Wait for next update
                time.sleep(config.LIVE_UPDATE_INTERVAL_SECONDS)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError in prediction loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown system gracefully"""
        print("\nShutting down system...")
        
        # Save logs
        if self.logger:
            self.logger.save_logs()
            self.logger.print_summary()
        
        # Disconnect MT5
        if self.connector:
            self.connector.disconnect()
        
        print("System shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MT5 Live Scalping System')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--login', type=int, help='MT5 account login')
    parser.add_argument('--password', type=str, help='MT5 account password')
    parser.add_argument('--server', type=str, help='MT5 server name')
    
    args = parser.parse_args()
    
    # Load from environment if not provided
    from dotenv import load_dotenv
    load_dotenv()
    
    mt5_login = args.login or os.getenv('MT5_LOGIN')
    mt5_password = args.password or os.getenv('MT5_PASSWORD')
    mt5_server = args.server or os.getenv('MT5_SERVER')
    
    if mt5_login:
        mt5_login = int(mt5_login)
    
    # Create and run system
    system = ScalpingSystem(
        model_path=args.model,
        mt5_login=mt5_login,
        mt5_password=mt5_password,
        mt5_server=mt5_server
    )
    
    try:
        system.initialize()
        system.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

