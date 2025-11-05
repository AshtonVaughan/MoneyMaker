# Troubleshooting Guide

## Data File Not Found

If you get `FileNotFoundError: Data file not found`:

### Solution 1: Check file location
```bash
# Find the CSV file
find . -name "*.csv" -type f

# Or list files in current directory
ls -la *.csv
```

### Solution 2: Use absolute path
```bash
# Find full path
find ~ -name "eurusd_historical_*.csv" 2>/dev/null

# Use full path in command
python train_gpu.py --data /full/path/to/eurusd_historical_20251106.csv --epochs 200
```

### Solution 3: Copy file to expected location
```bash
# Create directory if needed
mkdir -p data/historical

# Copy file
cp eurusd_historical_20251106.csv data/historical/

# Or if file is in root directory
cp /root/eurusd_historical_20251106.csv data/historical/
```

### Solution 4: Use filename only (if in current directory)
```bash
# If CSV is in MoneyMaker directory
python train_gpu.py --data eurusd_historical_20251106.csv --epochs 200
```

## GPU Not Detected

If you see "No GPU found - using CPU":

1. **Check GPU availability:**
   ```bash
   nvidia-smi
   ```

2. **Verify TensorFlow GPU support:**
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

3. **Install GPU-enabled TensorFlow:**
   ```bash
   pip install tensorflow[and-cuda]
   # Or for TensorFlow 2.x with CUDA
   pip install tensorflow-gpu
   ```

## CUDA Compatibility Warning

If you see: `TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0`

This is just a warning - TensorFlow will JIT-compile CUDA kernels. First run will be slower (30+ minutes), but subsequent runs will be fast.

**Solution:** Wait for first compilation, or use pre-built TensorFlow with CUDA support.

## Import Errors

### MetaTrader5 not found
- **OK on Linux/GPU**: MT5 is Windows-only, not needed for GPU training
- Scripts handle this automatically

### pytables not found
- **OK**: HDF5 is optional, CSV files work fine
- Install if needed: `pip install tables` (requires C++ compiler)

## Out of Memory (OOM)

If training crashes with OOM error:

1. **Reduce batch size:**
   ```bash
   python train_gpu.py --data file.csv --batch-size 64
   ```

2. **Reduce sequence length** (edit `config.py`):
   ```python
   SEQUENCE_LENGTH = 30  # Instead of 60
   ```

3. **Use gradient accumulation** (not implemented yet)

## Slow Training

If training is very slow:

1. **Check GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Verify mixed precision is enabled** (check output)

3. **Increase batch size** (if memory allows):
   ```bash
   --batch-size 512
   ```

4. **Use fewer features** (edit `data_processor.py`)

## Data Loading Issues

### CSV parsing errors:
- Check file encoding: `file -i filename.csv`
- Check column names match expected format
- Verify datetime column is parseable

### HDF5 errors:
- Install pytables: `pip install tables`
- Or use CSV format instead

## Quick Fixes

```bash
# Check current directory
pwd

# List all files
ls -la

# Find data files
find . -name "*.csv" -o -name "*.h5"

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"
```

