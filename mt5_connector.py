"""
MT5 API Connector for live data fetching and historical data download
"""

import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
import config


class MT5Connector:
    """Handles MT5 connection and data fetching"""
    
    def __init__(self, login: int = None, password: str = None, server: str = None):
        """
        Initialize MT5 connector
        
        Args:
            login: MT5 account login (optional, can use logged-in terminal)
            password: MT5 account password (optional)
            server: MT5 server name (optional)
        """
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        # If credentials provided, login
        if self.login and self.password and self.server:
            authorized = mt5.login(self.login, password=self.password, server=self.server)
            if not authorized:
                print(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
        
        # Verify connection
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            mt5.shutdown()
            return False
        
        self.connected = True
        print(f"Connected to MT5. Account: {account_info.login}, Balance: {account_info.balance}")
        return True
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        print("Disconnected from MT5")
    
    def get_live_tick(self) -> Optional[dict]:
        """
        Get latest tick data for EURUSD
        
        Returns:
            Dictionary with tick data or None if error
        """
        if not self.connected:
            if not self.connect():
                return None
        
        tick = mt5.symbol_info_tick(config.MT5_SYMBOL)
        if tick is None:
            print(f"Failed to get tick data: {mt5.last_error()}")
            return None
        
        return {
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'spread': tick.ask - tick.bid
        }
    
    def get_historical_bars(
        self, 
        months: int = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Optional[pd.DataFrame]:
        """
        Download historical 1-minute bars
        
        Args:
            months: Number of months to download (defaults to config.HISTORICAL_MONTHS)
            start_date: Start date (optional)
            end_date: End date (optional, defaults to now)
        
        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not self.connected:
            if not self.connect():
                return None
        
        # Set dates
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            months = months or config.HISTORICAL_MONTHS
            start_date = end_date - timedelta(days=months * 30)
        
        # Ensure dates are timezone-naive (MT5 requirement)
        if start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
        
        # Calculate number of bars needed (approximately)
        bars_count = 30000  # Default to ~30 days of 1-minute data
        if months:
            bars_count = int(months * 30 * 24 * 60)  # Approximate bars
        
        # Try multiple methods to get data
        rates = None
        
        # Method 1: copy_rates_from_pos (simplest, gets recent bars)
        try:
            rates = mt5.copy_rates_from_pos(
                config.MT5_SYMBOL,
                config.MT5_TIMEFRAME_M1,
                0,  # Start from current bar
                bars_count
            )
            if rates is not None and len(rates) > 0:
                print(f"Downloaded {len(rates)} bars using copy_rates_from_pos")
        except:
            pass
        
        # Method 2: copy_rates_from (with date)
        if rates is None or len(rates) == 0:
            try:
                recent_date = datetime.now() - timedelta(days=min(30, months * 30 if months else 30))
                rates = mt5.copy_rates_from(
                    config.MT5_SYMBOL,
                    config.MT5_TIMEFRAME_M1,
                    recent_date,
                    bars_count
                )
                if rates is not None and len(rates) > 0:
                    print(f"Downloaded {len(rates)} bars using copy_rates_from")
            except Exception as e:
                print(f"copy_rates_from error: {e}")
        
        # Method 3: copy_rates_range (last resort)
        if rates is None or len(rates) == 0:
            try:
                # Ensure timezone-naive dates
                start_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
                end_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
                rates = mt5.copy_rates_range(
                    config.MT5_SYMBOL,
                    config.MT5_TIMEFRAME_M1,
                    start_naive,
                    end_naive
                )
                if rates is not None and len(rates) > 0:
                    print(f"Downloaded {len(rates)} bars using copy_rates_range")
            except Exception as e:
                print(f"copy_rates_range error: {e}")
        
        if rates is None or len(rates) == 0:
            error_msg = mt5.last_error()
            print(f"All methods failed. Last error: {error_msg}")
            print(f"Trying with smaller bar count...")
            # Try with smaller count
            rates = mt5.copy_rates_from_pos(config.MT5_SYMBOL, config.MT5_TIMEFRAME_M1, 0, 10000)
        
        if rates is None or len(rates) == 0:
            print(f"Failed to get historical data: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        print(f"Downloaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        return df
    
    def stream_live_data(self, callback, interval_seconds: int = None):
        """
        Stream live tick data at specified interval
        
        Args:
            callback: Function to call with tick data: callback(tick_data)
            interval_seconds: Update interval (defaults to config.LIVE_UPDATE_INTERVAL_SECONDS)
        """
        if not self.connected:
            if not self.connect():
                return
        
        interval = interval_seconds or config.LIVE_UPDATE_INTERVAL_SECONDS
        
        print(f"Starting live data stream (every {interval} seconds)...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                tick = self.get_live_tick()
                if tick:
                    callback(tick)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopping data stream...")
        finally:
            self.disconnect()


if __name__ == "__main__":
    # Test connection
    connector = MT5Connector()
    if connector.connect():
        # Test live tick
        tick = connector.get_live_tick()
        print(f"Live tick: {tick}")
        
        # Test historical data
        df = connector.get_historical_bars(months=1)
        if df is not None:
            print(f"\nHistorical data shape: {df.shape}")
            print(df.head())
        
        connector.disconnect()

