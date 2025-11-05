# GPU Instance Setup Guide

## After Cloning Repository

```bash
# Navigate into the cloned directory
cd MoneyMaker

# Pull latest updates (if repository already exists)
git pull origin master

# Install dependencies
pip install -r requirements.txt

# Verify setup
python check_setup.py

# Start training
bash quick_start_gpu.sh
```

## If Repository Already Exists

If you get "destination path already exists":

```bash
# Navigate into existing directory
cd MoneyMaker

# Pull latest updates
git pull origin master

# If you get merge conflicts, reset to latest:
git fetch origin
git reset --hard origin/master
```

## Quick Commands Reference

```bash
# Check current directory
pwd

# List files
ls -la

# Navigate to MoneyMaker
cd MoneyMaker

# Pull updates
git pull origin master

# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Verify setup
python check_setup.py

# Start training
python train_gpu.py --data eurusd_historical_20251106.csv --epochs 200 --batch-size 256
```

