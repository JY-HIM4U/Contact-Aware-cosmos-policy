#!/usr/bin/env bash
# Prepare data for the libero_90 single-task winedrawer experiment
# (KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet).
#
# Run this once on the cluster after copying the LIBERO raw HDF5s over.
# Sets up:
#   - symlinked single-task HDF5 dir at $BASE_DATASETS_DIR/LIBERO-Cosmos-Policy/libero_90_winedrawer/
#   - F/T NPZ extraction at  $BASE_DATASETS_DIR/libero_90_winedrawer_with_ft/
#   - F/T normalization stats (dataset_stats_p1p99.json) inside the F/T dir
#
# Requires:
#   - LIBERO raw at $LIBERO_DATA_ROOT (e.g. ~/data/libero_raw/libero_90)
#   - $BASE_DATASETS_DIR set (e.g. ~/data)
#   - dataset_statistics.json: copy from libero_90_stove/ (action/proprio norm
#     are LIBERO-wide; per-task stats are not strictly required for training).

set -euo pipefail

TASK_NAME="KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet"
TASK_IDX_DEFAULT=26  # libero_90 task index (verified locally) for KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet
TASK_IDX="${TASK_IDX:-$TASK_IDX_DEFAULT}"

BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-$HOME/data}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$BASE_DATASETS_DIR/libero_raw/libero_90}"

DATA_DIR="$BASE_DATASETS_DIR/LIBERO-Cosmos-Policy/libero_90_winedrawer"
FT_DIR="$BASE_DATASETS_DIR/libero_90_winedrawer_with_ft"
STOVE_DIR="$BASE_DATASETS_DIR/LIBERO-Cosmos-Policy/libero_90_stove"

mkdir -p "$DATA_DIR"
ln -sf "$LIBERO_DATA_ROOT/${TASK_NAME}_demo.hdf5" "$DATA_DIR/${TASK_NAME}_demo.hdf5"

# Reuse global LIBERO action/proprio normalization stats if a sibling stove dir exists
if [[ -f "$STOVE_DIR/dataset_statistics.json" ]]; then
    cp -n "$STOVE_DIR/dataset_statistics.json" "$DATA_DIR/dataset_statistics.json"
    [[ -f "$STOVE_DIR/dataset_statistics_post_norm.json" ]] && \
        cp -n "$STOVE_DIR/dataset_statistics_post_norm.json" "$DATA_DIR/dataset_statistics_post_norm.json"
fi

# Use whatever python is active. On x86 dev machines, run via:
#   uv run --extra cu128 bash prepare_libero90_winedrawer.sh
# On aarch64 (GH200) the uv-pinned flash-attn has no wheel; activate the
# cosmos-policy conda env first and run this script directly.
PY="${PY:-python}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$(dirname "$SCRIPT_DIR")"

"$PY" "$SCRIPT_DIR/extract_ft_libero.py" \
    --suite libero_90 \
    --task_idx "$TASK_IDX" \
    --output_dir "$FT_DIR"

"$PY" "$SCRIPT_DIR/compute_ft_stats.py" "$FT_DIR"

echo "[prepare_libero90_winedrawer] Done."
echo "  HDF5 dir : $DATA_DIR"
echo "  F/T dir  : $FT_DIR"
