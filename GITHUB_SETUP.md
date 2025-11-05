# GitHub Repository Setup Guide

## Quick Setup

### 1. Initialize Git Repository

```bash
# Run setup script (creates directories and .gitkeep files)
bash setup_github.sh

# Or manually:
git init
git add .
git commit -m "Initial commit: MT5 Scalping System with GPU Training"
```

### 2. Create GitHub Repository

1. Go to https://github.com/new
2. Create new repository (don't initialize with README)
3. Copy the repository URL

### 3. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/MoneyMaker.git
git branch -M main
git push -u origin main
```

## File Structure

```
MoneyMaker/
├── README.md                      # Main documentation
├── CLOUD_GPU_SETUP.md            # GPU training guide
├── GITHUB_SETUP.md               # This file
├── WIN_RATE_IMPROVEMENT_GUIDE.md # Model improvement guide
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
│
├── config.py                     # Configuration
├── mt5_connector.py              # MT5 API wrapper
├── data_processor.py             # Feature engineering
├── model.py                      # LSTM model architecture
├── trainer.py                    # Training pipeline
├── train_gpu.py                  # GPU-optimized training
├── download_historical_data.py   # Data downloader
├── live_predictor.py             # Real-time predictions
├── forward_test_logger.py        # Prediction logger
├── main.py                       # Main application
├── backtest.py                   # Backtesting tool
│
├── data/                         # Data directory (gitignored)
│   └── historical/               # Historical data files
├── models/                       # Trained models (gitignored)
├── logs/                         # Prediction logs (gitignored)
└── backtest_results/             # Backtest results (gitignored)
```

## What's Included vs Excluded

### ✅ Included (GitHub)
- All Python source code
- Configuration files
- Documentation (README, guides)
- Requirements.txt
- Setup scripts

### ❌ Excluded (Git LFS or Cloud Storage)
- Model files (`.h5`) - too large
- Historical data (`.h5`, `.csv`) - very large
- Log files
- Environment files (`.env`)

## Large Files Strategy

### Option 1: Git LFS (GitHub Large File Storage)

For files > 100MB:
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git lfs track "*.csv.gz"

# Add and commit
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

**Note**: Git LFS has quotas (1GB free on GitHub)

### Option 2: Cloud Storage (Recommended)

**Best Practice**: Store data on cloud storage, not GitHub

1. **Download data locally**:
   ```bash
   python download_historical_data.py --bars 1000000
   ```

2. **Upload to cloud**:
   - Google Drive / Dropbox (share link)
   - AWS S3 / Google Cloud Storage
   - OneDrive / iCloud

3. **Download on GPU instance**:
   ```bash
   wget <cloud-storage-url> -O data/historical/eurusd_historical.h5
   ```

### Option 3: Release Assets

For model files, use GitHub Releases:
1. Create a release
2. Upload `.h5` files as release assets
3. Download when needed

## Cloud GPU Workflow

### Complete Workflow:

1. **Local Machine**:
   ```bash
   # Download data
   python download_historical_data.py --bars 1000000
   
   # Upload to cloud storage (Google Drive, etc.)
   # Commit code to GitHub
   git add .
   git commit -m "Add GPU training support"
   git push
   ```

2. **Cloud GPU Instance** (Vast.ai, RunPod, etc.):
   ```bash
   # Clone repository
   git clone https://github.com/YOUR_USERNAME/MoneyMaker.git
   cd MoneyMaker
   
   # Download data from cloud storage
   wget <your-data-url> -O data/historical/eurusd_historical.h5
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Train on GPU
   python train_gpu.py --data data/historical/eurusd_historical.h5 --epochs 200
   ```

3. **Download Trained Model**:
   ```bash
   # From GPU instance, download model
   scp user@gpu-instance:/path/to/models/scalping_model_*.h5 ./
   ```

## Repository Description

Suggested GitHub repository description:
```
Deep Learning EUR/USD Scalping System - LSTM-based TP/SL prediction with MT5 integration. 
Includes GPU training support for large datasets (1M+ bars). Forward testing only - no live trading.
```

## Tags/Settings

- **Topics**: `machine-learning`, `forex`, `lstm`, `trading`, `mt5`, `deep-learning`, `scalping`
- **License**: MIT (or your choice)
- **Visibility**: Private (recommended) or Public

## CI/CD (Optional)

Add GitHub Actions for automated testing:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/
```

## Security Notes

- ✅ `.env` is gitignored (never commit credentials)
- ✅ `.gitignore` excludes sensitive files
- ⚠️ Never commit MT5 passwords or API keys
- ⚠️ Review `config.py` - ensure no hardcoded secrets

## Next Steps

1. ✅ Run `bash setup_github.sh`
2. ✅ Initialize git: `git init`
3. ✅ Create GitHub repository
4. ✅ Push code: `git push -u origin main`
5. ✅ Download data locally
6. ✅ Upload data to cloud storage
7. ✅ Setup GPU instance
8. ✅ Train model on GPU

Your repository is now ready for cloud GPU training!

