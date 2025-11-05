"""
Quick setup verification script for GPU instance
"""

import sys
import os

print("="*70)
print("SETUP VERIFICATION")
print("="*70)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check GPU
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU Status: OK")
        for gpu in gpus:
            print(f"  {gpu.name}")
        print(f"  TensorFlow version: {tf.__version__}")
    else:
        print("\nGPU Status: NOT FOUND (using CPU)")
except ImportError:
    print("\nTensorFlow: NOT INSTALLED")

# Check data file
data_file = "data/historical/eurusd_historical_20251106.csv"
if os.path.exists(data_file):
    import pandas as pd
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"\nData File: OK")
    print(f"  File: {data_file}")
    print(f"  Bars: {len(df):,}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Size: {os.path.getsize(data_file) / 1024 / 1024:.2f} MB")
else:
    print(f"\nData File: NOT FOUND")
    print(f"  Expected: {data_file}")

# Check required files
required_files = [
    "train_gpu.py",
    "trainer.py",
    "data_processor.py",
    "model.py",
    "config.py"
]

print(f"\nRequired Files:")
all_ok = True
for file in required_files:
    exists = os.path.exists(file)
    status = "OK" if exists else "MISSING"
    print(f"  {file}: {status}")
    if not exists:
        all_ok = False

print("\n" + "="*70)
if all_ok:
    print("SETUP COMPLETE - Ready to train!")
    print("\nNext command:")
    print(f"  python train_gpu.py --data {data_file} --epochs 200 --batch-size 256")
else:
    print("SETUP INCOMPLETE - Check missing files")
print("="*70)

