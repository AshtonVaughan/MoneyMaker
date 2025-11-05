"""
Simplified data downloader with multiple sources
Fixes encoding issues and provides alternatives
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


def download_from_mt5_max(symbol="EURUSD", max_bars=100000):
    """Download maximum available data from MT5"""
    if not MT5_AVAILABLE:
        print("MT5 not available")
        return None
    
    print(f"Downloading from MT5 (target: {max_bars:,} bars)...")
    
    import MetaTrader5 as mt5
    
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return None
    
    all_data = []
    
    # Try multiple methods to get more data
    # Method 1: copy_rates_from_pos (most recent)
    print("  Method 1: Recent bars...", end=" ", flush=True)
    rates1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, min(max_bars, 100000))
    if rates1 is not None and len(rates1) > 0:
        df1 = pd.DataFrame(rates1)
        df1['time'] = pd.to_datetime(df1['time'], unit='s')
        df1.set_index('time', inplace=True)
        df1.columns = [col.lower() for col in df1.columns]
        all_data.append(df1)
        print(f"OK ({len(df1):,} bars)")
    
    # Method 2: Try to get older data by date range
    if len(all_data) > 0 and len(all_data[0]) < max_bars:
        print("  Method 2: Older data by date range...", end=" ", flush=True)
        try:
            # Get date from oldest bar
            oldest_date = all_data[0].index.min()
            # Try to get data 6 months earlier
            start_date = oldest_date - pd.Timedelta(days=180)
            
            rates2 = mt5.copy_rates_range(
                symbol,
                mt5.TIMEFRAME_M1,
                start_date,
                oldest_date
            )
            if rates2 is not None and len(rates2) > 0:
                df2 = pd.DataFrame(rates2)
                df2['time'] = pd.to_datetime(df2['time'], unit='s')
                df2.set_index('time', inplace=True)
                df2.columns = [col.lower() for col in df2.columns]
                all_data.append(df2)
                print(f"OK ({len(df2):,} bars)")
            else:
                print("SKIP (no data)")
        except Exception as e:
            print(f"SKIP ({e})")
    
    mt5.shutdown()
    
    if not all_data:
        print("No data received from MT5")
        return None
    
    # Combine
    df = pd.concat(all_data, ignore_index=False)
    df = df.sort_index().drop_duplicates()
    
    print(f"Total downloaded: {len(df):,} bars from MT5")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def download_from_yahoo_finance(symbol="EURUSD=X", period="2y", interval="1m"):
    """
    Download from Yahoo Finance (free, no API key)
    Note: Limited to ~60 days of 1-minute data
    """
    try:
        import yfinance as yf
        print(f"Downloading from Yahoo Finance ({period})...")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df is None or len(df) == 0:
            print("No data from Yahoo Finance")
            return None
        
        # Rename columns
        df.columns = [col.lower() for col in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.rename(columns={'volume': 'tick_volume'}, inplace=True)
        
        print(f"Downloaded {len(df):,} bars from Yahoo Finance")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
        return None
    except Exception as e:
        print(f"Yahoo Finance error: {e}")
        return None


def download_from_dukascopy_fixed(symbol="EURUSD", start_date=None, end_date=None):
    """
    Fixed Dukascopy downloader with better error handling
    """
    print("Downloading from Dukascopy...")
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=180)  # 6 months
    
    if end_date is None:
        end_date = datetime.now()
    
    all_data = []
    current_date = start_date
    
    while current_date < end_date:
        year = current_date.year
        month = current_date.month
        
        # Dukascopy URL - try different formats
        url1 = f"https://www.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{month:02d}{year % 100:02d}_{symbol}_M1.csv.gz"
        url2 = f"https://www.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{month:02d}{year % 100:02d}_{symbol}_bid_M1.csv.gz"
        
        downloaded = False
        for url in [url1, url2]:
            try:
                print(f"  Trying {year}-{month:02d}...", end=" ", flush=True)
                response = requests.get(url, timeout=10, stream=True)
                
                if response.status_code == 200:
                    filename = f"data/historical/{symbol}_M1_{year}{month:02d}.csv.gz"
                    os.makedirs("data/historical", exist_ok=True)
                    
                    with open(filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Parse
                    df = pd.read_csv(filename, compression='gzip', header=None)
                    if len(df.columns) >= 6:
                        df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                        df.set_index('datetime', inplace=True)
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        df.columns = ['open', 'high', 'low', 'close', 'tick_volume']
                        all_data.append(df)
                        print(f"OK ({len(df)} bars)")
                        downloaded = True
                        break
            except Exception as e:
                continue
        
        if not downloaded:
            print("SKIP")
        
        # Next month
        if month == 12:
            current_date = datetime(year + 1, 1, 1)
        else:
            current_date = datetime(year, month + 1, 1)
        
        time.sleep(0.3)  # Rate limiting
    
    if all_data:
        df_combined = pd.concat(all_data, ignore_index=False)
        df_combined = df_combined.sort_index().drop_duplicates()
        print(f"\nTotal: {len(df_combined):,} bars")
        return df_combined
    
    return None


def download_and_save(target_bars=100000):
    """Download from multiple sources and combine"""
    print("="*70)
    print("DOWNLOADING HISTORICAL DATA")
    print("="*70)
    print(f"Target: {target_bars:,} bars\n")
    
    os.makedirs("data/historical", exist_ok=True)
    all_data = []
    
    # Method 1: MT5 (most reliable if available)
    if MT5_AVAILABLE:
        df_mt5 = download_from_mt5_max("EURUSD", max_bars=min(target_bars, 50000))
        if df_mt5 is not None:
            all_data.append(df_mt5)
    
    # Method 2: Yahoo Finance (easy, but limited)
    df_yahoo = download_from_yahoo_finance("EURUSD=X", period="2y", interval="1m")
    if df_yahoo is not None:
        all_data.append(df_yahoo)
    
    # Method 3: Dukascopy (free, lots of data)
    df_duka = download_from_dukascopy_fixed("EURUSD")
    if df_duka is not None:
        all_data.append(df_duka)
    
    # Combine
    if all_data:
        print("\nCombining datasets...")
        df_final = pd.concat(all_data, ignore_index=False)
        df_final = df_final.sort_index().drop_duplicates()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d")
        csv_path = f"data/historical/eurusd_historical_{timestamp}.csv"
        
        # Save CSV (always works)
        df_final.to_csv(csv_path)
        print(f"CSV:  {csv_path}")
        
        # Save HDF5 if pytables available (optional, faster)
        try:
            h5_path = f"data/historical/eurusd_historical_{timestamp}.h5"
            df_final.to_hdf(h5_path, key='data', mode='w', complevel=9)
            print(f"HDF5: {h5_path}")
        except ImportError:
            print("HDF5 not available (install pytables for faster loading)")
        except Exception as e:
            print(f"HDF5 save failed: {e}")
        
        print(f"\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"Total bars: {len(df_final):,}")
        print(f"HDF5: {h5_path}")
        print(f"CSV:  {csv_path}")
        print(f"Date range: {df_final.index.min()} to {df_final.index.max()}")
        print("="*70)
        
        return df_final
    else:
        print("\nNo data downloaded. Trying alternative method...")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download historical forex data")
    parser.add_argument("--bars", type=int, default=100000, help="Target bars")
    args = parser.parse_args()
    
    download_and_save(target_bars=args.bars)

