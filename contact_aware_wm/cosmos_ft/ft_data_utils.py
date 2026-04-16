"""Utilities for loading force/torque (F/T) data alongside Cosmos LIBERO samples.

Cosmos's LIBERODataset loads demos from HDF5 files. Our F/T data was extracted
by extract_ft_libero.py into one NPZ per demo, grouped by task name under
libero_90_with_ft/. This module provides the glue to:

  1) find the NPZ that corresponds to an (hdf5_file, demo_key) pair
  2) load the ft_wrist array aligned to the demo's timesteps
  3) build a chunked, log-scaled, normalized, contact-augmented F/T tensor
     that is drop-in compatible with the FTEncoder in ft_modules.py

Naming convention
-----------------
HDF5 file:  {TASK_NAME}_demo.hdf5   e.g. LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_demo.hdf5
Demo key:   data/demo_{i}           inside the HDF5

NPZ layout: libero_90_with_ft/{TASK_NAME}/demo_{i}.npz
            where TASK_NAME is the same prefix as the HDF5 (without "_demo.hdf5").
"""

from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import json
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# NPZ discovery + loading
# --------------------------------------------------------------------------- #

def hdf5_to_task_name(hdf5_path: str) -> str:
    """Turn '.../LIVING_ROOM_SCENE1_pick..._demo.hdf5' into 'LIVING_ROOM_SCENE1_pick...'."""
    base = os.path.basename(hdf5_path)
    base = re.sub(r"_demo\.hdf5$", "", base)
    return base


def demo_key_to_index(demo_key: str) -> int:
    """'data/demo_7' or 'demo_7' -> 7"""
    m = re.search(r"demo_(\d+)", demo_key)
    if m is None:
        raise ValueError(f"Cannot parse demo index from {demo_key!r}")
    return int(m.group(1))


def ft_npz_path(ft_root: str, hdf5_path: str, demo_key: str) -> str:
    task = hdf5_to_task_name(hdf5_path)
    idx = demo_key_to_index(demo_key)
    return os.path.join(ft_root, task, f"demo_{idx}.npz")


def load_ft_for_demo(
    ft_root: str,
    hdf5_path: str,
    demo_key: str,
) -> Optional[np.ndarray]:
    """Return (T, 6) ft_wrist for this demo, or None if no NPZ exists."""
    path = ft_npz_path(ft_root, hdf5_path, demo_key)
    if not os.path.exists(path):
        return None
    with np.load(path) as d:
        return d["ft_wrist"].astype(np.float32)


# --------------------------------------------------------------------------- #
# Normalization stats
# --------------------------------------------------------------------------- #

def load_ft_stats(
    ft_root: str,
    suite: str = "all",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ft_mean, ft_std from dataset_stats_{suite}.json.

    Falls back to per-channel unit normalization if the file is missing.
    Returns (mean[6], std[6]).
    """
    stats_path = os.path.join(ft_root, f"dataset_stats_{suite}.json")
    if not os.path.exists(stats_path):
        return np.zeros(6, np.float32), np.ones(6, np.float32)
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.asarray(stats.get("ft_mean", [0.0] * 6), dtype=np.float32)
    std = np.asarray(stats.get("ft_std", [1.0] * 6), dtype=np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


# --------------------------------------------------------------------------- #
# Chunking + preprocessing (keeps parity with ft_modules.log_scale_ft)
# --------------------------------------------------------------------------- #

def _log_scale(ft: np.ndarray) -> np.ndarray:
    return np.sign(ft) * np.log1p(np.abs(ft))


def get_ft_chunk_with_padding(
    ft: np.ndarray,
    relative_step_idx: int,
    chunk_size: int,
    num_steps: int,
) -> np.ndarray:
    """Return (chunk_size, 6) F/T chunk starting at relative_step_idx.

    Mirrors cosmos's `get_action_chunk_with_padding`: if the window runs
    off the end of the episode, repeat the last valid reading.
    """
    out = np.zeros((chunk_size, 6), dtype=np.float32)
    for k in range(chunk_size):
        t = relative_step_idx + k
        if t >= num_steps:
            t = num_steps - 1
        out[k] = ft[t]
    return out


def preprocess_ft_chunk(
    ft_chunk: np.ndarray,
    ft_mean: np.ndarray,
    ft_std: np.ndarray,
    apply_log_scale: bool = True,
) -> np.ndarray:
    """(T, 6) raw F/T -> (T, 6) log-scaled + z-scored."""
    if apply_log_scale:
        ft_chunk = _log_scale(ft_chunk)
        # Also log-scale the mean/std so normalization stays meaningful
        ft_mean = _log_scale(ft_mean)
        ft_std = np.maximum(_log_scale(np.abs(ft_std)), 1e-6)
    return (ft_chunk - ft_mean) / ft_std


def build_ft_sample(
    ft: np.ndarray,
    relative_step_idx: int,
    future_step_idx: int,
    chunk_size: int,
    ft_mean: np.ndarray,
    ft_std: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build the F/T fields to attach to a cosmos sample dict.

    Returns:
        current_ft_chunk: (chunk_size, 6) preprocessed F/T ending at current step
        future_ft_chunk:  (chunk_size, 6) preprocessed F/T starting at future step
        current_ft_last:  (6,) last preprocessed F/T reading (handy target)
        future_ft_first:  (6,) first F/T at future step (regression target)
    """
    num_steps = len(ft)
    cur = get_ft_chunk_with_padding(ft, relative_step_idx, chunk_size, num_steps)
    fut = get_ft_chunk_with_padding(ft, future_step_idx, chunk_size, num_steps)

    cur_n = preprocess_ft_chunk(cur, ft_mean, ft_std)
    fut_n = preprocess_ft_chunk(fut, ft_mean, ft_std)

    return {
        "current_ft_chunk": cur_n,
        "future_ft_chunk": fut_n,
        "current_ft_last": cur_n[-1],
        "future_ft_first": fut_n[0],
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ft_root", default=os.path.join(CONTACT_AWARE_WM, "data/libero_90_with_ft"))
    ap.add_argument(
        "--hdf5",
        default=os.path.join(LIBERO_DATA_ROOT,
            "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_demo.hdf5"),
    )
    ap.add_argument("--demo", default="data/demo_0")
    args = ap.parse_args()

    print(f"Task name from hdf5:  {hdf5_to_task_name(args.hdf5)}")
    print(f"Demo index:           {demo_key_to_index(args.demo)}")
    print(f"Expected NPZ:         {ft_npz_path(args.ft_root, args.hdf5, args.demo)}")

    ft = load_ft_for_demo(args.ft_root, args.hdf5, args.demo)
    print(f"F/T loaded:           shape={ft.shape}, dtype={ft.dtype}")
    print(f"F/T range:            min={ft.min(0)} max={ft.max(0)}")

    mean, std = load_ft_stats(args.ft_root, suite="all")
    print(f"Stats mean:           {mean}")
    print(f"Stats std:            {std}")

    sample = build_ft_sample(
        ft=ft,
        relative_step_idx=20,
        future_step_idx=36,
        chunk_size=16,
        ft_mean=mean,
        ft_std=std,
    )
    for k, v in sample.items():
        print(f"  {k:<20s} shape={v.shape}  mean={v.mean():.3f}  std={v.std():.3f}")
