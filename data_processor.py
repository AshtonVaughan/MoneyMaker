"""
Data preprocessing and feature engineering for LSTM model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import config


class DataProcessor:
    """Handles feature engineering and data preprocessing"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_scalers = {}
        self.is_fitted = False
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicator columns
        """
        data = df.copy()
        
        # Price features
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        data['rsi'] = self._calculate_rsi(data['close'], period=14)
        
        # MACD
        macd_data = self._calculate_macd(data['close'])
        data['macd'] = macd_data['macd']
        data['macd_signal'] = macd_data['signal']
        data['macd_hist'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(data['close'], period=20)
        data['bb_upper'] = bb_data['upper']
        data['bb_middle'] = bb_data['middle']
        data['bb_lower'] = bb_data['lower']
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # ATR (Average True Range)
        data['atr'] = self._calculate_atr(data, period=14)
        
        # Volume indicators
        if 'tick_volume' in data.columns:
            data['volume_sma'] = data['tick_volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['tick_volume'] / data['volume_sma']
        
        # Price change features
        data['returns'] = data['close'].pct_change()
        data['returns_5'] = data['close'].pct_change(5)
        data['returns_20'] = data['close'].pct_change(20)
        
        # Volatility
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Advanced Features for Scalping
        
        # 1. Order Flow Indicators
        data['price_momentum'] = data['close'].diff(5)  # 5-bar momentum
        data['momentum_strength'] = data['price_momentum'].abs()
        data['trend_strength'] = (data['close'] - data['sma_20']) / data['atr']
        
        # 2. Market Microstructure
        data['spread_indicator'] = (data['high'] - data['low']) / data['close']  # Normalized range
        data['wicks_upper'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['close']
        data['wicks_lower'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['close']
        data['body_size'] = abs(data['close'] - data['open']) / data['close']
        
        # 3. Price Action Patterns
        # Engulfing pattern strength
        data['bullish_engulf'] = ((data['close'] > data['open']) & 
                                  (data['close'].shift(1) < data['open'].shift(1)) &
                                  (data['close'] > data['open'].shift(1)) &
                                  (data['open'] < data['close'].shift(1))).astype(float)
        data['bearish_engulf'] = ((data['close'] < data['open']) & 
                                  (data['close'].shift(1) > data['open'].shift(1)) &
                                  (data['close'] < data['open'].shift(1)) &
                                  (data['open'] > data['close'].shift(1))).astype(float)
        
        # 4. Support/Resistance Levels
        data['near_support'] = ((data['low'] - data['low'].rolling(20).min()) / data['atr'] < 0.5).astype(float)
        data['near_resistance'] = ((data['high'].rolling(20).max() - data['high']) / data['atr'] < 0.5).astype(float)
        
        # 5. Time-based Features (for scalping - market hours matter)
        if isinstance(data.index, pd.DatetimeIndex):
            data['hour'] = data.index.hour
            data['minute'] = data.index.minute
            data['is_london_session'] = ((data['hour'] >= 8) & (data['hour'] < 16)).astype(float)
            data['is_ny_session'] = ((data['hour'] >= 13) & (data['hour'] < 21)).astype(float)
            data['is_overlap'] = ((data['hour'] >= 13) & (data['hour'] < 16)).astype(float)  # London-NY overlap
            data['is_volatile_hour'] = (data['hour'].isin([8, 9, 13, 14, 15])).astype(float)
        
        # 6. Mean Reversion Indicators
        data['distance_from_ma'] = (data['close'] - data['sma_20']) / data['atr']
        data['mean_reversion_signal'] = np.where(
            (data['distance_from_ma'] < -1) & (data['rsi'] < 30), 1,
            np.where((data['distance_from_ma'] > 1) & (data['rsi'] > 70), -1, 0)
        ).astype(float)
        
        # 7. Volatility Breakout Indicators
        data['volatility_expansion'] = (data['volatility'] > data['volatility'].rolling(20).mean() * 1.5).astype(float)
        data['volatility_contraction'] = (data['volatility'] < data['volatility'].rolling(20).mean() * 0.5).astype(float)
        
        # 8. Momentum Divergence
        data['rsi_momentum'] = data['rsi'].diff(5)
        data['price_rsi_divergence'] = np.sign(data['price_momentum']) != np.sign(data['rsi_momentum'])
        data['divergence_strength'] = data['price_rsi_divergence'].astype(float) * abs(data['rsi_momentum'])
        
        # 9. Price Compression/Expansion
        data['bb_squeeze'] = (data['bb_width'] < data['bb_width'].rolling(20).mean() * 0.5).astype(float)
        data['bb_expansion'] = (data['bb_width'] > data['bb_width'].rolling(20).mean() * 1.5).astype(float)
        
        # 10. Volume-Price Confirmation
        if 'volume_ratio' in data.columns:
            data['volume_price_confirmation'] = np.sign(data['returns']) * data['volume_ratio']
            data['high_volume_momentum'] = ((data['volume_ratio'] > 1.5) & (abs(data['returns']) > 0.0001)).astype(float)
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return {'macd': macd, 'signal': signal_line, 'histogram': histogram}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return {'upper': upper, 'middle': sma, 'lower': lower}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def create_sequences(
        self, 
        data: pd.DataFrame, 
        sequence_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        
        Args:
            data: DataFrame with features
            sequence_length: Length of sequences (defaults to config.SEQUENCE_LENGTH)
        
        Returns:
            Tuple of (X sequences, y labels)
        """
        seq_len = sequence_length or config.SEQUENCE_LENGTH
        
        # Select feature columns (exclude target and non-numeric)
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in ['tp_hit', 'sl_hit']]
        
        # Drop NaN rows
        data_clean = data[feature_cols].dropna()
        
        if len(data_clean) < seq_len + 1:
            raise ValueError(f"Not enough data. Need at least {seq_len + 1} rows, got {len(data_clean)}")
        
        # Create sequences
        X, y = [], []
        for i in range(seq_len, len(data_clean)):
            X.append(data_clean.iloc[i-seq_len:i].values)
            # Labels will be set separately based on TP/SL logic
            y.append([0, 0])  # Placeholder
        
        return np.array(X), np.array(y)
    
    def create_labels(
        self, 
        data: pd.DataFrame, 
        tp_pips: float = None,
        sl_pips: float = None,
        lookahead_bars: int = 10
    ) -> pd.DataFrame:
        """
        Create target labels: probability of hitting TP vs SL
        
        Args:
            data: DataFrame with OHLCV data
            tp_pips: Take profit in pips
            sl_pips: Stop loss in pips
            lookahead_bars: Number of bars to look ahead
        
        Returns:
            DataFrame with tp_hit and sl_hit columns
        """
        tp = tp_pips or config.TAKE_PROFIT_PIPS
        sl = sl_pips or config.STOP_LOSS_PIPS
        
        result = data.copy()
        result['tp_hit'] = 0
        result['sl_hit'] = 0
        
        pip_value = 0.0001  # For EURUSD
        
        for i in range(len(result) - lookahead_bars):
            entry_price = result.iloc[i]['close']
            entry_idx = result.index[i]
            
            # Check next N bars
            future_bars = result.iloc[i+1:i+1+lookahead_bars]
            
            # IMPROVED: Use momentum-based labeling for better win rate
            
            # Check for TP hit (long position)
            tp_price_long = entry_price + (tp * pip_value)
            tp_hit_long = (future_bars['high'] >= tp_price_long).any()
            
            # Check for SL hit (long position)
            sl_price_long = entry_price - (sl * pip_value)
            sl_hit_long = (future_bars['low'] <= sl_price_long).any()
            
            # IMPROVED: Check momentum direction (70% of TP target = favorable)
            price_change = (future_bars['close'].iloc[-1] - entry_price) / entry_price
            momentum_favorable_long = price_change > (tp * pip_value * 0.7)
            momentum_unfavorable_long = price_change < -(sl * pip_value * 0.7)
            
            # Check for TP hit (short position)
            tp_price_short = entry_price - (tp * pip_value)
            tp_hit_short = (future_bars['low'] <= tp_price_short).any()
            
            # Check for SL hit (short position)
            sl_price_short = entry_price + (sl * pip_value)
            sl_hit_short = (future_bars['high'] >= sl_price_short).any()
            
            momentum_favorable_short = price_change < -(tp * pip_value * 0.7)
            momentum_unfavorable_short = price_change > (sl * pip_value * 0.7)
            
            # IMPROVED: Use momentum to break ties and improve labels
            # If both TP and SL hit, check which came first OR use momentum
            if tp_hit_long and sl_hit_long:
                # Both hit - check which came first
                tp_idx = future_bars.index[(future_bars['high'] >= tp_price_long)].min() if tp_hit_long else None
                sl_idx = future_bars.index[(future_bars['low'] <= sl_price_long)].min() if sl_hit_long else None
                if tp_idx is not None and sl_idx is not None:
                    if future_bars.index.get_loc(tp_idx) < future_bars.index.get_loc(sl_idx):
                        tp_hit_long = True
                        sl_hit_long = False
                    else:
                        tp_hit_long = False
                        sl_hit_long = True
            elif momentum_favorable_long and not sl_hit_long:
                # Strong upward momentum without hitting SL = TP win
                tp_hit_long = True
                sl_hit_long = False
            elif momentum_unfavorable_long and not tp_hit_long:
                # Strong downward momentum without hitting TP = SL loss
                tp_hit_long = False
                sl_hit_long = True
            
            # Set labels (favor long positions for scalping)
            if tp_hit_long or tp_hit_short:
                result.loc[entry_idx, 'tp_hit'] = 1
            if sl_hit_long or sl_hit_short:
                result.loc[entry_idx, 'sl_hit'] = 1
        
        return result
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame,
        sequence_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare complete training dataset
        
        Args:
            df: Raw OHLCV DataFrame
            sequence_length: Sequence length for LSTM
        
        Returns:
            Tuple of (X sequences, y labels, feature names)
        """
        # Calculate indicators
        df_with_indicators = self.calculate_indicators(df)
        
        # Create labels
        df_with_labels = self.create_labels(df_with_indicators)
        
        # Select feature columns
        feature_cols = df_with_labels.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in ['tp_hit', 'sl_hit']]
        
        # Drop NaN
        df_clean = df_with_labels[feature_cols + ['tp_hit', 'sl_hit']].dropna()
        
        seq_len = sequence_length or config.SEQUENCE_LENGTH
        
        # Create sequences
        X, y = [], []
        for i in range(seq_len, len(df_clean)):
            X.append(df_clean[feature_cols].iloc[i-seq_len:i].values)
            y.append([
                df_clean.iloc[i]['tp_hit'],
                df_clean.iloc[i]['sl_hit']
            ])
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X_normalized = X_normalized.reshape(X.shape)
        
        self.is_fitted = True
        
        return X_normalized, y, feature_cols
    
    def normalize_features(self, data: np.ndarray) -> np.ndarray:
        """Normalize features using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call prepare_training_data first.")
        
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        normalized = self.scaler.transform(data_reshaped)
        return normalized.reshape(original_shape)


if __name__ == "__main__":
    # Test with sample data
    processor = DataProcessor()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 1.1000,
        'high': np.random.randn(1000).cumsum() + 1.1005,
        'low': np.random.randn(1000).cumsum() + 1.0995,
        'close': np.random.randn(1000).cumsum() + 1.1000,
        'tick_volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # Test indicators
    df_with_indicators = processor.calculate_indicators(sample_data)
    print(f"Features created: {len(df_with_indicators.columns)}")
    print(df_with_indicators.columns.tolist())
    
    # Test sequence creation
    try:
        X, y, features = processor.prepare_training_data(sample_data)
        print(f"\nSequences shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Feature count: {len(features)}")
    except Exception as e:
        print(f"Error: {e}")

