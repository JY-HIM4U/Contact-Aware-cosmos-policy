"""Offline F/T preprocessor: copies an *_with_ft/ dataset dir to a new location with
the ft_wrist field filtered through CausalFtFilter (causal median3 + butter 3 Hz).
Recomputes p1/p99 stats at the end so the training pipeline can pick them up.

Usage:
    python contact_aware_wm/preprocess_ft.py \
        --src ~/data/libero_10_with_ft \
        --dst ~/data/libero_10_with_ft_filtered
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from cosmos_policy.datasets.ft_filter import CausalFtFilter


def main(src: str, dst: str) -> None:
    src = os.path.expanduser(src)
    dst = os.path.expanduser(dst)
    os.makedirs(dst, exist_ok=True)

    npz_files = sorted(glob.glob(f"{src}/**/*.npz", recursive=True))
    print(f"Found {len(npz_files)} NPZ files under {src}")
    filt = CausalFtFilter()

    all_ft_filtered = []
    for i, f in enumerate(npz_files):
        rel = os.path.relpath(f, src)
        out_path = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        d = np.load(f)
        filtered = filt.apply_offline(d["ft_wrist"])
        other = {k: d[k] for k in d.files if k != "ft_wrist"}
        np.savez_compressed(out_path, ft_wrist=filtered.astype(np.float32), **other)
        all_ft_filtered.append(filtered)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(npz_files)} done")

    # Recompute stats on the FILTERED data so training sees consistent ranges
    cat = np.concatenate(all_ft_filtered, axis=0)
    stats = {
        "ft_min": np.percentile(cat, 1, axis=0).tolist(),
        "ft_max": np.percentile(cat, 99, axis=0).tolist(),
        "ft_mean": cat.mean(0).tolist(),
        "ft_std": cat.std(0).tolist(),
    }
    with open(os.path.join(dst, "dataset_stats_all.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(dst, "dataset_stats_p1p99.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {dst}/dataset_stats_p1p99.json (and _all.json)")
    for i, ch in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
        print(f"  {ch}: p1={stats['ft_min'][i]:10.3f}  p99={stats['ft_max'][i]:10.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()
    main(args.src, args.dst)
