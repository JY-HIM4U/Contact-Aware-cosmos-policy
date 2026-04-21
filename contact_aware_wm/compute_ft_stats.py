#!/usr/bin/env python3
"""
Compute global F/T min/max statistics from extracted NPZ files and write
dataset_stats_all.json (the format expected by LIBERODataset with use_ft=True).

Usage:
    python compute_ft_stats.py ~/data/libero_spatial_with_ft
"""

import argparse
import json
import os
import sys

import numpy as np


def compute_stats(ft_dir: str) -> dict:
    all_ft = []
    n_demos = 0
    n_tasks = 0

    for task in sorted(os.listdir(ft_dir)):
        task_path = os.path.join(ft_dir, task)
        if not os.path.isdir(task_path):
            continue
        npz_files = sorted(f for f in os.listdir(task_path) if f.endswith(".npz"))
        if not npz_files:
            continue
        n_tasks += 1
        for npz_name in npz_files:
            data = np.load(os.path.join(task_path, npz_name))
            all_ft.append(data["ft_wrist"].astype(np.float32))
            n_demos += 1

    if not all_ft:
        raise ValueError(f"No NPZ files found in {ft_dir}")

    ft_all = np.concatenate(all_ft, axis=0)  # (N_total, 6)
    ft_min = ft_all.min(axis=0).tolist()
    ft_max = ft_all.max(axis=0).tolist()
    ft_mean = ft_all.mean(axis=0).tolist()
    ft_std = ft_all.std(axis=0).tolist()

    print(f"Tasks:       {n_tasks}")
    print(f"Demos:       {n_demos}")
    print(f"Transitions: {len(ft_all):,}")
    dim_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    print(f"\n{'dim':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
    for i, name in enumerate(dim_names):
        print(f"{name:>4}  {ft_min[i]:>10.4f}  {ft_max[i]:>10.4f}  {ft_mean[i]:>10.4f}  {ft_std[i]:>10.4f}")

    return {"ft_min": ft_min, "ft_max": ft_max, "ft_mean": ft_mean, "ft_std": ft_std}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ft_dir", help="Directory with per-task NPZ subdirectories")
    args = parser.parse_args()

    ft_dir = os.path.expanduser(args.ft_dir)
    if not os.path.isdir(ft_dir):
        print(f"Error: {ft_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(ft_dir)
    out_path = os.path.join(ft_dir, "dataset_stats_all.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
