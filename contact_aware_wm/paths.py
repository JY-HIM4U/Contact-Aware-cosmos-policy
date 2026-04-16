"""Centralized path resolution for contact-aware world model scripts.

Paths are resolved from environment variables with sensible defaults:

  CONTACT_AWARE_WM      project root (default: parent of this file's dir)
  LIBERO_DATA_ROOT      raw LIBERO libero_90 hdf5 dir
                        (default: ~/LIBERO/datasets/libero_90/libero_90)
  DATA_ROOT             F/T-augmented dataset dir
                        (default: $CONTACT_AWARE_WM/data/libero_90_with_ft)
  CHECKPOINTS_ROOT      checkpoints dir
                        (default: $CONTACT_AWARE_WM/checkpoints)
  RESULTS_ROOT          outputs dir
                        (default: $CONTACT_AWARE_WM/results)

Set these in your shell (e.g., in ~/.bashrc) or via an env-file:

  export CONTACT_AWARE_WM=$HOME/cosmos-policy/contact_aware_wm
  export LIBERO_DATA_ROOT=$HOME/LIBERO/datasets/libero_90/libero_90
  export DATA_ROOT=/mnt/data/libero_90_with_ft
  export CHECKPOINTS_ROOT=/mnt/checkpoints/contact_aware_wm
  export RESULTS_ROOT=$CONTACT_AWARE_WM/results
"""
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

CONTACT_AWARE_WM = os.environ.get("CONTACT_AWARE_WM") or _THIS_DIR
LIBERO_DATA_ROOT = os.environ.get("LIBERO_DATA_ROOT") or os.path.expanduser(
    "~/LIBERO/datasets/libero_90/libero_90"
)
DATA_ROOT = os.environ.get("DATA_ROOT") or os.path.join(
    CONTACT_AWARE_WM, "data", "libero_90_with_ft"
)
CHECKPOINTS_ROOT = os.environ.get("CHECKPOINTS_ROOT") or os.path.join(
    CONTACT_AWARE_WM, "checkpoints"
)
RESULTS_ROOT = os.environ.get("RESULTS_ROOT") or os.path.join(
    CONTACT_AWARE_WM, "results"
)
