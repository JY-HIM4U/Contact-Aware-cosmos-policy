#!/bin/bash
# Run all ablation variants for LIBERO world model.
# Assumes F/T extraction has been done for the target suite.
set -e

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${CONTACT_AWARE_WM:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_DIR"

SUITE=${SUITE:-spatial}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-64}

echo "============================================"
echo "  LIBERO World Model Ablations (suite=$SUITE)"
echo "============================================"

# Variant 1: Baseline (1-frame, no anchor, no FT)
echo ""
echo "=== Variant 1: Baseline (1-frame, no FT) ==="
$PYTHON train_libero.py --suite $SUITE --model_variant baseline \
    --context_frames 1 --epochs $EPOCHS --batch_size $BATCH --seed 42

# Variant 2: Baseline + FT (1-frame)
echo ""
echo "=== Variant 2: Baseline + FT (1-frame) ==="
$PYTHON train_libero.py --suite $SUITE --model_variant baseline \
    --use_ft --context_frames 1 --epochs $EPOCHS --batch_size $BATCH --seed 42

# Variant 3: Improved 4-frame anchor, no FT
echo ""
echo "=== Variant 3: Improved 4-frame anchor, no FT ==="
$PYTHON train_libero.py --suite $SUITE --model_variant improved \
    --use_anchor --context_frames 4 --context_noise_std 0.05 \
    --epochs $EPOCHS --batch_size $BATCH --seed 42

# Variant 4: Improved 4-frame anchor + FT (separate path)
echo ""
echo "=== Variant 4: Improved 4-frame anchor + FT ==="
$PYTHON train_libero.py --suite $SUITE --model_variant improved \
    --use_ft --use_anchor --context_frames 4 --context_noise_std 0.05 \
    --epochs $EPOCHS --batch_size $BATCH --seed 42

# Variant 5: Phase 2 fine-tuning on Variant 4
echo ""
echo "=== Variant 5: Phase 2 (fine-tune Variant 4) ==="
$PYTHON train_libero.py --suite $SUITE --model_variant improved \
    --use_ft --use_anchor --context_frames 4 --context_noise_std 0.05 \
    --phase 2 --epochs 50 --batch_size 32 --lr 1e-5 \
    --resume_from checkpoints/libero_improved_ft_ctx4_anchor/best_val.pt \
    --seed 42

echo ""
echo "============================================"
echo "  All ablations complete!"
echo "  Run: python evaluate_libero.py --suite $SUITE"
echo "============================================"
