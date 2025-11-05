# Cloud GPU Training Setup (RTX 5090)

## Quick Start

### 1. Download Historical Data (Local)

```bash
# Download 1 million bars (~2 years of 1-minute data)
python download_historical_data.py --bars 1000000 --sources dukascopy

# Or download from multiple sources
python download_historical_data.py --bars 1000000 --sources dukascopy mt5
```

This will create:
- `data/historical/eurusd_historical_YYYYMMDD.h5` (HDF5 format, fast)
- `data/historical/eurusd_historical_YYYYMMDD.csv` (CSV format, compatible)

### 2. Upload to Cloud Storage

**Option A: Google Drive / Dropbox**
- Upload the `.h5` file to cloud storage
- Download on GPU instance

**Option B: GitHub LFS (for smaller files)**
```bash
git lfs install
git lfs track "*.h5"
git add data/historical/*.h5
git commit -m "Add historical data"
git push
```

**Option C: Direct Upload**
- Use `scp` or cloud provider's upload tool
- Example: `scp data/historical/*.h5 user@gpu-instance:/path/to/data/`

### 3. Setup on GPU Instance

```bash
# Clone repository
git clone <your-repo-url>
cd MoneyMaker

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 4. Train on GPU

```bash
# Train with default settings
python train_gpu.py --data data/historical/eurusd_historical_YYYYMMDD.h5

# Train with custom settings
python train_gpu.py \
    --data data/historical/eurusd_historical_YYYYMMDD.h5 \
    --epochs 200 \
    --batch-size 256 \
    --validation-split 0.2
```

## Cloud GPU Providers

### Recommended Providers:

1. **Vast.ai** (Cheapest)
   - RTX 4090/5090: ~$0.50-1.00/hour
   - Setup: Upload your code, select GPU, run

2. **RunPod** (User-friendly)
   - RTX 4090/5090: ~$0.60-1.20/hour
   - Pre-configured ML templates

3. **Paperspace Gradient** (Professional)
   - RTX 4090: ~$1.50/hour
   - Full Jupyter notebook support

4. **Google Colab Pro** (Easy but slower)
   - Free tier: T4 GPU
   - Pro: V100/A100
   - Good for testing

### Example: Vast.ai Setup

1. Go to https://vast.ai
2. Create account, add credits
3. Click "Create" → "Jupyter Notebook"
4. Select RTX 5090 or RTX 4090
5. Upload code or clone from GitHub
6. Run training script

## Expected Training Time

- **1M bars** (~100K sequences):
  - RTX 5090: ~30-60 minutes for 100 epochs
  - RTX 4090: ~60-90 minutes for 100 epochs
  - CPU: ~6-12 hours for 100 epochs

## Optimizations

The `train_gpu.py` script includes:
- ✅ Mixed precision (FP16) for 2x speedup
- ✅ Large batch sizes (128-256) for GPU efficiency
- ✅ Memory growth to prevent OOM
- ✅ HDF5 data loading (fast I/O)

## Data Sources

### Dukascopy (Free, Recommended)
- **URL**: https://www.dukascopy.com/swiss/english/marketwatch/historical/
- **Data**: Tick data + OHLCV bars
- **History**: 10+ years available
- **Format**: CSV/GZ compressed
- **Limitation**: Monthly chunks, requires sequential download

### MT5 (If you have account)
- **Data**: Real-time + historical
- **History**: Depends on broker (usually 1-3 years)
- **Format**: Direct API access

### Other Sources
- **OANDA API**: Requires API key, good for recent data
- **Forex.com**: Requires account
- **Investing.com**: Web scraping (not recommended)

## Tips

1. **Start Small**: Test with 100K bars first, then scale up
2. **Monitor GPU**: Use `nvidia-smi` to check GPU usage
3. **Save Checkpoints**: Models auto-save, but keep backups
4. **Cost Control**: Set budget limits on cloud providers
5. **Data Validation**: Verify data quality before long training runs

## Troubleshooting

**Out of Memory (OOM)**
- Reduce batch size: `--batch-size 64`
- Use gradient accumulation
- Reduce sequence length in config.py

**Slow Training**
- Check GPU utilization: `nvidia-smi`
- Enable mixed precision (already enabled)
- Use SSD storage for data

**Data Download Fails**
- Check internet connection
- Try different months (some may not be available)
- Use MT5 as backup source

## File Structure

```
MoneyMaker/
├── download_historical_data.py  # Download script
├── train_gpu.py                 # GPU training script
├── data/
│   └── historical/              # Downloaded data
│       ├── eurusd_historical_*.h5
│       └── eurusd_historical_*.csv
├── models/                       # Trained models
└── requirements.txt             # Dependencies
```

