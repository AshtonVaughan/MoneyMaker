"""
Forward Testing Logger - Logs predictions without executing trades
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import config


class ForwardTestLogger:
    """Logs predictions and signals for forward testing"""
    
    def __init__(self, log_dir: str = None):
        """
        Initialize logger
        
        Args:
            log_dir: Directory for log files (defaults to config.LOGS_DIR)
        """
        self.log_dir = log_dir or config.LOGS_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"forward_test_{timestamp}.csv")
        
        # Initialize log DataFrame
        self.logs = []
        
    def log_prediction(
        self,
        timestamp: datetime,
        current_price: float,
        tp_probability: float,
        sl_probability: float,
        suggested_action: str = None,
        confidence: float = None,
        additional_data: Dict = None
    ):
        """
        Log a prediction
        
        Args:
            timestamp: Prediction timestamp
            current_price: Current EUR/USD price
            tp_probability: Probability of TP hit
            sl_probability: Probability of SL hit
            suggested_action: Suggested action (buy/sell/hold)
            confidence: Confidence score
            additional_data: Additional data to log
        """
        # Determine action if not provided
        if suggested_action is None:
            if tp_probability > config.TP_PROBABILITY_THRESHOLD and tp_probability > sl_probability:
                suggested_action = "BUY"
            elif sl_probability > config.SL_PROBABILITY_THRESHOLD and sl_probability > tp_probability:
                suggested_action = "SELL"
            else:
                suggested_action = "HOLD"
        
        # Calculate confidence if not provided
        if confidence is None:
            confidence = max(tp_probability, sl_probability)
        
        log_entry = {
            'timestamp': timestamp,
            'price': current_price,
            'tp_probability': tp_probability,
            'sl_probability': sl_probability,
            'suggested_action': suggested_action,
            'confidence': confidence,
            'tp_threshold': config.TP_PROBABILITY_THRESHOLD,
            'sl_threshold': config.SL_PROBABILITY_THRESHOLD
        }
        
        # Add additional data if provided
        if additional_data:
            log_entry.update(additional_data)
        
        self.logs.append(log_entry)
        
        # Save to CSV periodically (every 10 entries)
        if len(self.logs) % 10 == 0:
            self.save_logs()
    
    def get_signal(
        self,
        tp_probability: float,
        sl_probability: float
    ) -> Dict[str, any]:
        """
        Generate trading signal from probabilities
        
        Args:
            tp_probability: TP hit probability
            sl_probability: SL hit probability
        
        Returns:
            Dictionary with signal information
        """
        confidence = max(tp_probability, sl_probability)
        
        if confidence < config.MIN_CONFIDENCE:
            return {
                'action': 'HOLD',
                'reason': 'Low confidence',
                'confidence': confidence
            }
        
        if tp_probability > config.TP_PROBABILITY_THRESHOLD and tp_probability > sl_probability:
            return {
                'action': 'BUY',
                'reason': f'TP probability {tp_probability:.2%} exceeds threshold',
                'confidence': confidence,
                'tp_prob': tp_probability,
                'sl_prob': sl_probability
            }
        elif sl_probability > config.SL_PROBABILITY_THRESHOLD and sl_probability > tp_probability:
            return {
                'action': 'SELL',
                'reason': f'SL probability {sl_probability:.2%} exceeds threshold',
                'confidence': confidence,
                'tp_prob': tp_probability,
                'sl_prob': sl_probability
            }
        else:
            return {
                'action': 'HOLD',
                'reason': 'Conflicting signals',
                'confidence': confidence,
                'tp_prob': tp_probability,
                'sl_prob': sl_probability
            }
    
    def save_logs(self):
        """Save logs to CSV file"""
        if not self.logs:
            return
        
        df = pd.DataFrame(self.logs)
        df.to_csv(self.log_file, index=False)
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics from logs
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.logs:
            return {}
        
        df = pd.DataFrame(self.logs)
        
        summary = {
            'total_predictions': len(df),
            'buy_signals': len(df[df['suggested_action'] == 'BUY']),
            'sell_signals': len(df[df['suggested_action'] == 'SELL']),
            'hold_signals': len(df[df['suggested_action'] == 'HOLD']),
            'avg_tp_probability': df['tp_probability'].mean(),
            'avg_sl_probability': df['sl_probability'].mean(),
            'avg_confidence': df['confidence'].mean(),
            'high_confidence_signals': len(df[df['confidence'] >= config.MIN_CONFIDENCE]),
            'log_file': self.log_file
        }
        
        return summary
    
    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary()
        if not summary:
            print("No logs to summarize")
            return
        
        print("\n" + "="*70)
        print("FORWARD TEST SUMMARY")
        print("="*70)
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"Buy Signals: {summary['buy_signals']}")
        print(f"Sell Signals: {summary['sell_signals']}")
        print(f"Hold Signals: {summary['hold_signals']}")
        print(f"\nAverage TP Probability: {summary['avg_tp_probability']:.2%}")
        print(f"Average SL Probability: {summary['avg_sl_probability']:.2%}")
        print(f"Average Confidence: {summary['avg_confidence']:.2%}")
        print(f"High Confidence Signals: {summary['high_confidence_signals']}")
        print(f"\nLog file: {summary['log_file']}")
        print("="*70)


if __name__ == "__main__":
    # Test logger
    logger = ForwardTestLogger()
    
    # Simulate some predictions
    import time
    for i in range(5):
        logger.log_prediction(
            timestamp=datetime.now(),
            current_price=1.1000 + i * 0.0001,
            tp_probability=0.65 + i * 0.05,
            sl_probability=0.35 - i * 0.05
        )
        time.sleep(0.1)
    
    logger.save_logs()
    logger.print_summary()

