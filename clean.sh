#!/usr/bin/env bash
# =====================================================================
# clean_py_cache.sh
# ---------------------------------------------------------------------
# Removes Python cache, compiled files, and common deep learning artifacts
# =====================================================================

echo "🧹 Cleaning Python and DL cache files..."

# 1. Remove Python cache folders and compiled files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name "*.pyd" -delete 2>/dev/null

# 2. Remove Jupyter Notebook checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# 3. Remove build and distribution folders
rm -rf build/ dist/ *.egg-info 2>/dev/null

# 4. Remove virtual environment cache (optional — uncomment if needed)
# rm -rf .venv venv env 2>/dev/null

# 5. Remove dataset/model cache from common DL frameworks
rm -rf ~/.cache/torch 2>/dev/null
rm -rf ~/.torch 2>/dev/null
rm -rf ~/.cache/huggingface 2>/dev/null
rm -rf ~/.cache/transformers 2>/dev/null
rm -rf ~/.keras 2>/dev/null
rm -rf ~/.fastai 2>/dev/null
rm -rf ~/.nv 2>/dev/null

# 6. Remove macOS and system-generated junk files
find . -type f -name ".DS_Store" -delete 2>/dev/null

# 7. Optional: clear temporary files
rm -rf /tmp/*python* 2>/dev/null
rm -rf /tmp/*torch* 2>/dev/null

echo "✅ Done! Python cache and DL artifacts removed."
