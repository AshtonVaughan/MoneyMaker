#!/bin/bash
# Setup script for GitHub repository

echo "Setting up MoneyMaker for GitHub..."

# Create necessary directories
mkdir -p data/historical
mkdir -p models
mkdir -p logs
mkdir -p backtest_results

# Create .gitkeep files for empty directories
touch data/.gitkeep
touch data/historical/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep
touch backtest_results/.gitkeep

# Initialize git LFS for large files (if needed)
if command -v git-lfs &> /dev/null; then
    git lfs install
    git lfs track "*.h5"
    git lfs track "*.csv.gz"
    echo "Git LFS initialized for large files"
else
    echo "Warning: git-lfs not installed. Large data files will not be tracked."
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. git init"
echo "2. git add ."
echo "3. git commit -m 'Initial commit'"
echo "4. Create repository on GitHub"
echo "5. git remote add origin <your-repo-url>"
echo "6. git push -u origin main"

