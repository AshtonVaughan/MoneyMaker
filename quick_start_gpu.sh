#!/bin/bash
# Quick start script for GPU training

echo "============================================================"
echo "QUICK START: GPU TRAINING"
echo "============================================================"

# Check setup
echo ""
echo "1. Verifying setup..."
python check_setup.py

echo ""
echo "2. Starting GPU training..."
echo ""

# Find data file
DATA_FILE=$(find . -name "eurusd_historical_*.csv" -type f | head -1)

if [ -z "$DATA_FILE" ]; then
    echo "ERROR: No data file found!"
    echo "Please download data first or specify path manually"
    exit 1
fi

echo "Using data file: $DATA_FILE"
echo ""

# Start training
python train_gpu.py \
    --data "$DATA_FILE" \
    --epochs 200 \
    --batch-size 256 \
    --validation-split 0.2

echo ""
echo "============================================================"
echo "Training complete! Check models/ directory for saved model"
echo "============================================================"

