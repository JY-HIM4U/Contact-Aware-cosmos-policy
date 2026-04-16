#!/bin/bash
# Full pipeline: download → check → preprocess → train → evaluate
set -e

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${CONTACT_AWARE_WM:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

cd "$PROJECT_DIR"

echo "============================================"
echo "  RH20T World Model — Full Pipeline"
echo "============================================"

echo ""
echo "=== Step 1: Download RH20T cfg3 ==="
$PYTHON data/download.py

echo ""
echo "=== Step 2: Check tasks ==="
$PYTHON data/check_tasks.py

echo ""
echo "=== Step 3: Preprocess episodes ==="
$PYTHON data/preprocess.py

echo ""
echo "=== Step 4: Train image_only model ==="
$PYTHON train.py --condition image_only --epochs 50 --seed 42

echo ""
echo "=== Step 5: Train image_ft model ==="
$PYTHON train.py --condition image_ft --epochs 50 --seed 42

echo ""
echo "=== Step 6: Evaluate ==="
$PYTHON evaluate.py

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results in: $PROJECT_DIR/results/"
echo "============================================"
