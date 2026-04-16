#!/usr/bin/env python3
"""
Explore the LIBERO-90 dataset structure.

Prints:
  - Total tasks, demos per task, episode length statistics
  - Observation image shapes and action shapes
  - Total transitions across the entire dataset
  - Saves a sanity-check PNG grid of sample frames from 5 random tasks

Usage:
    python explore_libero.py [--data_root /path/to/libero_90]
"""

from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# LIBERO-90 task suite groupings (from LIBERO paper)
LIBERO_SUITES: Dict[str, List[str]] = {
    "spatial": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    ],
    # Will be auto-detected from filenames
}


def discover_tasks(data_root: str) -> Dict[str, str]:
    """Map task_name -> hdf5_path for all LIBERO-90 tasks.

    Args:
        data_root: path containing .hdf5 files

    Returns:
        dict mapping task name (filename stem) to full path
    """
    tasks = {}
    for f in sorted(os.listdir(data_root)):
        if f.endswith(".hdf5") or f.endswith("_demo.hdf5"):
            name = f.replace("_demo.hdf5", "").replace(".hdf5", "")
            tasks[name] = os.path.join(data_root, f)
    return tasks


def analyze_task(hdf5_path: str) -> Dict:
    """Analyze a single LIBERO task HDF5 file.

    Returns dict with: n_demos, episode_lengths, obs_shapes, action_shape, action_stats
    """
    info = {
        "n_demos": 0,
        "episode_lengths": [],
        "obs_keys": [],
        "obs_shapes": {},
        "action_shape": None,
        "action_stats": None,
        "total_transitions": 0,
    }

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return info

        demos = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        info["n_demos"] = len(demos)

        all_actions = []

        for i, demo_key in enumerate(demos):
            demo = f["data"][demo_key]

            # Episode length
            T = demo["actions"].shape[0]
            info["episode_lengths"].append(T)
            info["total_transitions"] += T

            # Observation shapes (from first demo only)
            if i == 0:
                info["action_shape"] = demo["actions"].shape[1:]
                if "obs" in demo:
                    info["obs_keys"] = sorted(demo["obs"].keys())
                    for k in demo["obs"].keys():
                        info["obs_shapes"][k] = demo["obs"][k].shape[1:]

            # Collect actions for stats (subsample for speed)
            if i < 10:
                all_actions.append(demo["actions"][:])

        if all_actions:
            acts = np.concatenate(all_actions, axis=0)
            info["action_stats"] = {
                "mean": acts.mean(axis=0),
                "std": acts.std(axis=0),
                "min": acts.min(axis=0),
                "max": acts.max(axis=0),
            }

    return info


def save_sample_grid(data_root: str, tasks: Dict[str, str], save_path: str,
                     n_tasks: int = 5, seed: int = 42) -> None:
    """Save a grid of sample frames from random tasks.

    Args:
        data_root: LIBERO data root
        tasks: task_name -> hdf5_path mapping
        save_path: where to save the PNG
        n_tasks: number of tasks to sample
        seed: random seed
    """
    rng = random.Random(seed)
    task_names = sorted(tasks.keys())
    sampled = rng.sample(task_names, min(n_tasks, len(task_names)))

    fig, axes = plt.subplots(n_tasks, 4, figsize=(14, 3.5 * n_tasks))
    if n_tasks == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Frame 0", "Frame T//4", "Frame T//2", "Frame T-1"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, task_name in enumerate(sampled):
        hdf5_path = tasks[task_name]
        with h5py.File(hdf5_path, "r") as f:
            demo = f["data/demo_0"]
            imgs = demo["obs/agentview_rgb"][:]  # (T, H, W, 3)
            T = len(imgs)

            indices = [0, T // 4, T // 2, T - 1]
            for col, idx in enumerate(indices):
                axes[row, col].imshow(imgs[idx])
                axes[row, col].axis("off")

        # Short task name for label
        short_name = task_name.replace("KITCHEN_SCENE", "KS").replace("LIVING_ROOM_SCENE", "LR")
        if len(short_name) > 50:
            short_name = short_name[:47] + "..."
        axes[row, 0].set_ylabel(short_name, fontsize=8, rotation=0, labelpad=120, va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explore] Saved sample grid: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Explore LIBERO-90 dataset")
    parser.add_argument("--data_root", type=str,
                        default=LIBERO_DATA_ROOT,
                        help="Path to LIBERO-90 HDF5 files")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Where to save output files")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Discover tasks
    tasks = discover_tasks(args.data_root)
    print(f"[explore] Found {len(tasks)} tasks in {args.data_root}")

    if len(tasks) == 0:
        print("[explore] ERROR: No HDF5 files found!")
        return

    # Analyze all tasks
    all_lengths = []
    total_demos = 0
    total_transitions = 0
    task_infos = {}

    print(f"\n{'Task':<70} | {'Demos':>5} | {'T min':>5} | {'T mean':>6} | {'T max':>5}")
    print("-" * 105)

    for task_name, hdf5_path in tasks.items():
        info = analyze_task(hdf5_path)
        task_infos[task_name] = info

        total_demos += info["n_demos"]
        total_transitions += info["total_transitions"]
        all_lengths.extend(info["episode_lengths"])

        short_name = task_name[:68]
        if info["episode_lengths"]:
            lens = info["episode_lengths"]
            print(f"{short_name:<70} | {info['n_demos']:>5} | {min(lens):>5} | "
                  f"{np.mean(lens):>6.0f} | {max(lens):>5}")
        else:
            print(f"{short_name:<70} | {info['n_demos']:>5} | {'N/A':>5} | {'N/A':>6} | {'N/A':>5}")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print(f"LIBERO-90 Dataset Summary")
    print(f"{'=' * 70}")
    print(f"Total tasks:        {len(tasks)}")
    print(f"Total demos:        {total_demos}")
    print(f"Total transitions:  {total_transitions:,}")
    print(f"Demos per task:     {total_demos / len(tasks):.0f}")
    print(f"Episode lengths:    min={min(all_lengths)}, "
          f"mean={np.mean(all_lengths):.0f}, "
          f"max={max(all_lengths)}, "
          f"std={np.std(all_lengths):.0f}")

    # Observation and action info from first task
    first_info = next(iter(task_infos.values()))
    print(f"\nObservation keys: {first_info['obs_keys']}")
    for k, shape in first_info["obs_shapes"].items():
        print(f"  {k}: {shape}")
    print(f"Action shape: {first_info['action_shape']}")

    if first_info["action_stats"] is not None:
        stats = first_info["action_stats"]
        dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
        print(f"\nAction statistics (first 10 demos of first task):")
        for i, name in enumerate(dim_names):
            print(f"  {name:>8}: mean={stats['mean'][i]:>7.4f}  "
                  f"std={stats['std'][i]:>6.4f}  "
                  f"range=[{stats['min'][i]:>6.3f}, {stats['max'][i]:>6.3f}]")

    # Identify LIBERO suites by scene prefix
    scene_counts = defaultdict(int)
    for name in tasks:
        parts = name.split("_")
        if len(parts) >= 2:
            scene = "_".join(parts[:2])
            scene_counts[scene] += 1

    print(f"\nScene distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f"  {scene}: {count} tasks")

    # Save sample grid
    save_sample_grid(args.data_root, tasks,
                     os.path.join(args.save_dir, "libero_sample_grid.png"))

    print(f"\n[explore] Done.")


if __name__ == "__main__":
    main()
