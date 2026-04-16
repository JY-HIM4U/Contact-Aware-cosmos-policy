#!/bin/bash
# Flow matching pipeline: train → evaluate → generate videos
# Assumes data is already preprocessed (run run.sh first for data steps).
set -e

PYTHON="${PYTHON:-python}"
PROJECT_DIR="${CONTACT_AWARE_WM:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

cd "$PROJECT_DIR"

EPOCHS=${EPOCHS:-100}
ODE_STEPS=${ODE_STEPS:-20}
CHUNK_SIZE=${CHUNK_SIZE:-4}
CTX_FRAMES=${CTX_FRAMES:-4}
NOISE_STD=${NOISE_STD:-0.05}

echo "============================================"
echo "  RH20T World Model — Flow Matching Pipeline"
echo "  context_frames=$CTX_FRAMES  anchor=yes"
echo "  noise_std=$NOISE_STD  epochs=$EPOCHS"
echo "============================================"

echo ""
echo "=== Step 1: Train FM image_only (context + anchor + noise) ==="
$PYTHON train_fm.py --condition image_only --epochs $EPOCHS --batch_size 32 \
    --lr 1e-4 --ode_steps $ODE_STEPS --chunk_size $CHUNK_SIZE \
    --context_frames $CTX_FRAMES --use_anchor --context_noise_std $NOISE_STD \
    --seed 42

echo ""
echo "=== Step 2: Train FM image_ft (context + anchor + noise) ==="
$PYTHON train_fm.py --condition image_ft --epochs $EPOCHS --batch_size 32 \
    --lr 1e-4 --ode_steps $ODE_STEPS --chunk_size $CHUNK_SIZE \
    --context_frames $CTX_FRAMES --use_anchor --context_noise_std $NOISE_STD \
    --seed 42

echo ""
echo "=== Step 3: Evaluate all models ==="
$PYTHON evaluate_fm.py

echo ""
echo "=== Step 4: Generate rollout videos ==="
$PYTHON generate_video.py --ode_steps $ODE_STEPS --num_samples 5

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results in: $PROJECT_DIR/results/"
echo "============================================"
