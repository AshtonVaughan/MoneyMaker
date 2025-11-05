"""
Download large amounts of historical EUR/USD data from multiple sources
Supports: Dukascopy (free), MT5, and CSV import
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import sys
from typing import Optional
import config

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("Warning: MetaTrader5 not available. MT5 download disabled.")


class HistoricalDataDownloader:
    """Download historical forex data from various sources"""
    
    def __init__(self, output_dir: str = "data/historical"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def download_from_dukascopy(
        self,
        symbol: str = "EURUSD",
        start_date: datetime = None,
        end_date: datetime = None,
        timeframe: str = "M1"  # M1, M5, M15, M30, H1, H4, D1
    ) -> pd.DataFrame:
        """
        Download historical data from Dukascopy (free, no API key needed)
        
        Dukascopy provides tick data and OHLCV bars for free
        Website: https://www.dukascopy.com/swiss/english/marketwatch/historical/
        """
        print(f"Downloading {symbol} data from Dukascopy...")
        print("This may take a while for large datasets...")
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 2)  # 2 years
        
        if end_date is None:
            end_date = datetime.now()
        
        # Dukascopy uses different symbol format
        symbol_map = {
            "EURUSD": "EURUSD",
            "GBPUSD": "GBPUSD",
            "USDJPY": "USDJPY"
        }
        
        dukascopy_symbol = symbol_map.get(symbol, symbol)
        
        # Dukascopy timeframe mapping
        timeframe_map = {
            "M1": "M1",
            "M5": "M5",
            "M15": "M15",
            "M30": "M30",
            "H1": "H1",
            "H4": "H4",
            "D1": "D1"
        }
        
        tf = timeframe_map.get(timeframe, "M1")
        
        all_data = []
        current_date = start_date
        
        print(f"Downloading from {start_date.date()} to {end_date.date()}...")
        
        # Dukascopy provides data in monthly chunks
        while current_date < end_date:
            year = current_date.year
            month = current_date.month
            
            # Dukascopy URL format
            url = (
                f"https://www.dukascopy.com/datafeed/{dukascopy_symbol}/{year}/{month:02d}/"
                f"{month:02d}{year % 100:02d}_{dukascopy_symbol}_{tf}.csv.gz"
            )
            
            try:
                    print(f"  Downloading {year}-{month:02d}...", end=" ", flush=True)
                    response = requests.get(url, timeout=30, stream=True)
                    
                    if response.status_code == 200:
                        # Save compressed file
                        filename = os.path.join(
                            self.output_dir,
                            f"{dukascopy_symbol}_{tf}_{year}{month:02d}.csv.gz"
                        )
                        with open(filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        # Read and parse
                        try:
                            df = pd.read_csv(filename, compression='gzip', header=None)
                            df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                            
                            # Combine date and time
                            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                            df.set_index('datetime', inplace=True)
                            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                            df.columns = ['open', 'high', 'low', 'close', 'tick_volume']
                            
                            all_data.append(df)
                            print(f"OK ({len(df)} bars)")
                        except Exception as e:
                            print(f"ERROR parsing: {e}")
                            if os.path.exists(filename):
                                os.remove(filename)
                    else:
                        print(f"NOT AVAILABLE (status {response.status_code})")
                
                time.sleep(0.5)  # Be polite to server
                
            except Exception as e:
                print(f"ERROR: {e}")
            
            # Move to next month
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)
        
        if not all_data:
            print("No data downloaded from Dukascopy")
            return None
        
        # Combine all data
        df_combined = pd.concat(all_data, ignore_index=False)
        df_combined = df_combined.sort_index()
        df_combined = df_combined.drop_duplicates()
        
        print(f"\nDownloaded {len(df_combined):,} bars from Dukascopy")
        return df_combined
    
    def download_from_mt5(
        self,
        symbol: str = "EURUSD",
        months: int = 12
    ) -> Optional[pd.DataFrame]:
        """Download from MT5 (if available)"""
        if not MT5_AVAILABLE:
            print("MT5 not available")
            return None
        
        print(f"Downloading {symbol} data from MT5...")
        
        connector = MT5Connector()
        if not connector.connect():
            print("Failed to connect to MT5")
            return None
        
        try:
            df = connector.get_historical_bars(months=months)
            connector.disconnect()
            
            if df is not None:
                print(f"Downloaded {len(df):,} bars from MT5")
            return df
        except Exception as e:
            print(f"Error downloading from MT5: {e}")
            connector.disconnect()
            return None
    
    def combine_and_save(
        self,
        dataframes: list,
        filename: str = "eurusd_historical.h5"
    ):
        """Combine multiple dataframes and save to HDF5 (efficient format)"""
        if not dataframes:
            print("No data to save")
            return
        
        print("\nCombining datasets...")
        df_combined = pd.concat(dataframes, ignore_index=False)
        df_combined = df_combined.sort_index()
        df_combined = df_combined.drop_duplicates()
        
        # Remove any remaining duplicates by index
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Save to HDF5 (fast, compressed)
        df_combined.to_hdf(output_path, key='data', mode='w', complevel=9)
        
        # Also save CSV for compatibility
        csv_path = output_path.replace('.h5', '.csv')
        df_combined.to_csv(csv_path)
        
        print(f"\nSaved {len(df_combined):,} bars to:")
        print(f"  HDF5: {output_path}")
        print(f"  CSV:  {csv_path}")
        print(f"\nDate range: {df_combined.index.min()} to {df_combined.index.max()}")
        
        return df_combined


def download_large_dataset(
    target_bars: int = 1000000,
    sources: list = ["dukascopy", "mt5"]
):
    """
    Download large dataset targeting specific number of bars
    
    Args:
        target_bars: Target number of bars (1M = ~2 years of 1-minute data)
        sources: List of sources to try
    """
    downloader = HistoricalDataDownloader()
    all_dataframes = []
    
    # Calculate date range needed
    # 1 minute bars: ~1,440 per day, ~525,600 per year
    years_needed = max(2, int(target_bars / 525600) + 1)
    start_date = datetime.now() - timedelta(days=years_needed * 365)
    
    print("="*70)
    print(f"DOWNLOADING HISTORICAL DATA")
    print("="*70)
    print(f"Target: {target_bars:,} bars (~{years_needed} years)")
    print(f"Sources: {', '.join(sources)}")
    print("="*70 + "\n")
    
    # Try Dukascopy first (free, most data)
    if "dukascopy" in sources:
        try:
            df_duka = downloader.download_from_dukascopy(
                symbol="EURUSD",
                start_date=start_date,
                timeframe="M1"
            )
            if df_duka is not None:
                all_dataframes.append(df_duka)
        except Exception as e:
            print(f"Dukascopy download failed: {e}")
    
    # Try MT5 as supplement
    if "mt5" in sources and MT5_AVAILABLE:
        try:
            df_mt5 = downloader.download_from_mt5(
                symbol="EURUSD",
                months=12
            )
            if df_mt5 is not None:
                all_dataframes.append(df_mt5)
        except Exception as e:
            print(f"MT5 download failed: {e}")
    
    # Combine and save
    if all_dataframes:
        df_final = downloader.combine_and_save(
            all_dataframes,
            filename=f"eurusd_historical_{datetime.now().strftime('%Y%m%d')}.h5"
        )
        
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"Total bars: {len(df_final):,}")
        print(f"Data saved to: {downloader.output_dir}/")
        print("="*70)
        
        return df_final
    else:
        print("\nNo data downloaded. Check your internet connection and try again.")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download historical forex data")
    parser.add_argument(
        "--bars",
        type=int,
        default=1000000,
        help="Target number of bars (default: 1,000,000)"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["dukascopy"],
        choices=["dukascopy", "mt5"],
        help="Data sources to use"
    )
    
    args = parser.parse_args()
    
    download_large_dataset(target_bars=args.bars, sources=args.sources)

