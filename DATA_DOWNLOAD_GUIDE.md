# Data Download Guide

## Current Status

✅ **MT5 Download: WORKING**
- Downloads ~50,000 bars (~2 months of 1-minute data)
- Saved to: `data/historical/eurusd_historical_YYYYMMDD.csv`

## Quick Start

```bash
# Download from MT5 (recommended, works now)
python download_data_simple.py --bars 100000
```

This downloads maximum available data from MT5 and saves to CSV.

## Getting 1 Million Bars

### Option 1: Multiple MT5 Downloads (Recommended)

MT5 typically provides 1-3 months of data. To get more:

1. **Keep MT5 running and download periodically**
   - Run download script weekly/monthly
   - Combine multiple CSV files
   - Script will automatically deduplicate

2. **Use MT5's built-in data export**
   - Open MT5 terminal
   - Right-click chart → Save as Report
   - Exports can be imported

### Option 2: Dukascopy (Free, Requires Fix)

Dukascopy provides free historical data but URLs need updating:

**Manual Download:**
1. Go to: https://www.dukascopy.com/swiss/english/marketwatch/historical/
2. Select EUR/USD, M1 timeframe
3. Download monthly CSV files
4. Place in `data/historical/` folder
5. Use `combine_csv_files.py` (create this if needed)

**Programmatic Download:**
- Dukascopy URLs may require authentication or different format
- Check their API documentation for updates

### Option 3: Paid Data Providers

- **FXCM**: API access, historical data
- **OANDA**: API with historical data (requires account)
- **FX Data Shop**: Paid historical data

### Option 4: Cloud GPU Providers

Many cloud GPU providers offer data:
- **Vast.ai**: Some instances have MT5 pre-installed
- **RunPod**: Can install MT5 and download on GPU instance
- **Paperspace**: Pre-configured trading environments

## Combining Multiple Files

If you have multiple CSV files:

```python
import pandas as pd
import glob

files = glob.glob("data/historical/*.csv")
dfs = [pd.read_csv(f, index_col=0, parse_dates=True) for f in files]
df_combined = pd.concat(dfs, ignore_index=False)
df_combined = df_combined.sort_index().drop_duplicates()
df_combined.to_csv("data/historical/eurusd_combined.csv")
```

## File Formats

- **CSV**: Universal, works everywhere, slower to load
- **HDF5**: Fast, compressed, requires `pytables` package
  - Install: `pip install tables` (may require C++ compiler on Windows)

## Current Data

You now have:
- **File**: `data/historical/eurusd_historical_20251106.csv`
- **Bars**: ~50,000
- **Date Range**: Sep 17, 2025 to Nov 5, 2025
- **Size**: ~5-10 MB

## For Cloud GPU Training

1. **Upload CSV file** to cloud storage (Google Drive, Dropbox, etc.)
2. **Download on GPU instance**:
   ```bash
   wget <your-file-url> -O data/historical/eurusd_historical.csv
   ```
3. **Train**:
   ```bash
   python train_gpu.py --data data/historical/eurusd_historical.csv --epochs 200
   ```

## Troubleshooting

**"No data downloaded"**
- Check MT5 is running and logged in
- Verify symbol name (EURUSD)
- Check internet connection

**"Dukascopy not working"**
- URLs may have changed
- Try manual download from their website
- Check firewall/antivirus blocking

**"File too large for GitHub"**
- Use Git LFS: `git lfs track "*.csv"`
- Or upload to cloud storage instead
- CSV files > 100MB should use cloud storage

## Next Steps

1. ✅ Use current MT5 data (50K bars) for initial training
2. 🔄 Download more data over time from MT5
3. 📊 Train on GPU with available data
4. 🔍 Look for better data sources if needed

