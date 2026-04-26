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

if [[ -z "${BASE_DATASETS_DIR:-}" ]]; then
    echo "ERROR: set BASE_DATASETS_DIR to the dir holding LIBERO-Cosmos-Policy/ and libero_raw/" >&2
    echo "  e.g.  BASE_DATASETS_DIR=/data/clear/robot-simulation/data bash $0" >&2
    exit 1
fi

# LIBERO_DATA_ROOT must point at the directory containing the libero_90 *_demo.hdf5 files.
# Default mirrors the local-dev layout ($BASE_DATASETS_DIR/libero_raw/libero_90).
# IMPORTANT: must be exported so the python child (paths.py) inherits it.
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$BASE_DATASETS_DIR/libero_raw/libero_90}"

DATA_DIR="$BASE_DATASETS_DIR/LIBERO-Cosmos-Policy/libero_90_winedrawer"
FT_DIR="$BASE_DATASETS_DIR/libero_90_winedrawer_with_ft"
STOVE_DIR="$BASE_DATASETS_DIR/LIBERO-Cosmos-Policy/libero_90_stove"
TASK_HDF5="$LIBERO_DATA_ROOT/${TASK_NAME}_demo.hdf5"

if [[ ! -f "$TASK_HDF5" ]]; then
    echo "ERROR: task HDF5 not found at $TASK_HDF5" >&2
    echo "  Set LIBERO_DATA_ROOT to the dir containing libero_90 *_demo.hdf5 files." >&2
    exit 1
fi

mkdir -p "$DATA_DIR"
ln -sf "$TASK_HDF5" "$DATA_DIR/${TASK_NAME}_demo.hdf5"

# Reuse global LIBERO action/proprio normalization stats if a sibling stove dir exists
if [[ -f "$STOVE_DIR/dataset_statistics.json" ]]; then
    cp -n "$STOVE_DIR/dataset_statistics.json" "$DATA_DIR/dataset_statistics.json"
    [[ -f "$STOVE_DIR/dataset_statistics_post_norm.json" ]] && \
        cp -n "$STOVE_DIR/dataset_statistics_post_norm.json" "$DATA_DIR/dataset_statistics_post_norm.json"
fi

# Use whatever python is active. On x86 dev machines, run via:
#   uv run --extra cu128 bash prepare_libero90_winedrawer.sh
# On aarch64 (GH200) the uv-pinned flash-attn has no wheel; activate the
# cosmos-policy uv venv first and run this script directly.
PY="${PY:-python}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$(dirname "$SCRIPT_DIR")"

"$PY" "$SCRIPT_DIR/extract_ft_libero.py" \
    --suite libero_90 \
    --task_idx "$TASK_IDX" \
    --hdf5_dir "$LIBERO_DATA_ROOT" \
    --output_dir "$FT_DIR"

"$PY" "$SCRIPT_DIR/compute_ft_stats.py" "$FT_DIR"

echo "[prepare_libero90_winedrawer] Done."
echo "  HDF5 dir : $DATA_DIR"
echo "  F/T dir  : $FT_DIR"
