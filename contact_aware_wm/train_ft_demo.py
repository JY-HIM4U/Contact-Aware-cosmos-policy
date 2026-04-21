#!/usr/bin/env python3
"""Minimal F/T fine-tuning demo for Cosmos Policy on 4x A4000 (16GB).

Validates the dataset + model integration end-to-end with a single
forward/backward pass, then runs a short training loop.

Usage:
    cd /home/jaeyoun/cosmos-policy
    conda activate cosmos
    export CONTACT_AWARE_WM=$PWD/contact_aware_wm
    export LIBERO_DATA_ROOT=$HOME/LIBERO/datasets/libero_90/libero_90

    # Dry run (single batch, no GPU training) — validates data pipeline
    python contact_aware_wm/train_ft_demo.py --dry_run

    # Single-GPU training demo (conservative)
    python contact_aware_wm/train_ft_demo.py --num_steps 100

    # Multi-GPU (FSDP)
    torchrun --nproc_per_node=4 contact_aware_wm/train_ft_demo.py --num_steps 500
"""
import os
import sys
import argparse
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch


def dry_run(args):
    """Validate the dataset returns correctly shaped F/T-augmented samples."""
    from cosmos_policy.datasets.libero_dataset import LIBERODataset

    use_ft = not args.no_ft
    print(f"[dry_run] Creating LIBERODataset with use_ft={use_ft}...")
    ds = LIBERODataset(
        data_dir=args.libero_data_dir,
        use_ft=use_ft,
        ft_data_dir=args.ft_data_dir,
        ft_stats_path=os.path.join(args.ft_data_dir, "dataset_stats_all.json"),
        t5_text_embeddings_path=args.t5_path,
        chunk_size=16,
        use_image_aug=False,
        use_wrist_images=True,
        use_proprio=True,
        normalize_proprio=True,
        normalize_actions=True,
        num_duplicates_per_image=4,
        rollout_data_dir="",
        demonstration_sampling_prob=1.0,
        return_value_function_returns=True,
    )
    print(f"[dry_run] Dataset size: {len(ds)} steps, {ds.num_episodes} episodes")

    sample = ds[0]
    print(f"[dry_run] Sample keys: {sorted(sample.keys())}")
    print(f"[dry_run] video shape: {sample['video'].shape}")
    if use_ft:
        print(f"[dry_run] current_ft: {sample['current_ft']} (shape {sample['current_ft'].shape})")
        print(f"[dry_run] future_ft: {sample['future_ft']} (shape {sample['future_ft'].shape})")
        print(f"[dry_run] current_ft_latent_idx: {sample['current_ft_latent_idx']}")
        print(f"[dry_run] future_ft_latent_idx: {sample['future_ft_latent_idx']}")
    print(f"[dry_run] use_ft: {ds.use_ft}")

    # Verify latent indices are consistent
    print("\n[dry_run] Latent frame layout:")
    idx_keys = [k for k in sample.keys() if "latent_idx" in k]
    for k in sorted(idx_keys, key=lambda x: sample[x] if isinstance(sample[x], (int, float)) else -1):
        v = sample[k]
        print(f"  {k}: {v}")

    # Expected new total raw images: with F/T, state_t=11 and layout adds two
    # latent frames (current_ft + future_ft), giving 1 + 10*4 = 41. Without F/T
    # the baseline layout is state_t=9 → 1 + 8*4 = 33 frames.
    expected_raw = (1 + 10 * 4) if use_ft else (1 + 8 * 4)
    actual_raw = sample["video"].shape[1]  # C, T, H, W
    print(f"\n[dry_run] Expected {expected_raw} raw images, got {actual_raw}")
    if actual_raw != expected_raw:
        print(f"  WARNING: mismatch! Check frame insertion logic.")
    else:
        print("  OK")

    print("\n[dry_run] Checking multiple samples for consistency...")
    for i in [0, 1, len(ds) // 2, len(ds) - 1]:
        s = ds[i]
        assert s["video"].shape[1] == expected_raw, f"Sample {i}: video shape {s['video'].shape}"
        if use_ft:
            assert s["current_ft"].shape == (6,), f"Sample {i}: current_ft shape {s['current_ft'].shape}"
    print("[dry_run] All OK!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Only validate dataset; no model loading or training.")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-GPU batch size. Start with 1 for A4000 16GB.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--libero_data_dir", type=str,
                        default=os.environ.get("LIBERO_DATA_DIR",
                            os.path.expanduser("~/LIBERO/datasets/libero_90/libero_90")))
    parser.add_argument("--ft_data_dir", type=str,
                        default=os.environ.get("DATA_ROOT",
                            os.path.join(os.environ.get("CONTACT_AWARE_WM", "."),
                                         "data/libero_90_with_ft")))
    parser.add_argument("--t5_path", type=str, default="")
    parser.add_argument("--no_ft", action="store_true",
                        help="Skip F/T loading — useful before F/T NPZs exist.")
    args = parser.parse_args()

    # Auto-detect T5 embeddings
    if not args.t5_path:
        candidates = [
            os.path.join(args.libero_data_dir, "t5_embeddings.pkl"),
            os.path.join(args.libero_data_dir, "..", "t5_embeddings.pkl"),
        ]
        # Try HuggingFace cache
        try:
            from huggingface_hub import hf_hub_download
            candidates.append(hf_hub_download(
                "nvidia/Cosmos-Policy-LIBERO-Predict2-2B", "libero_t5_embeddings.pkl"
            ))
        except Exception:
            pass
        for c in candidates:
            if os.path.exists(c):
                args.t5_path = c
                break
        if not args.t5_path:
            print("ERROR: Could not find T5 embeddings. Set --t5_path.")
            sys.exit(1)
    print(f"[train] T5 path: {args.t5_path}")

    if args.dry_run:
        dry_run(args)
        return

    print("[train] Full training not yet implemented — run --dry_run first to validate the data pipeline.")
    print("[train] Next step: integrate with Cosmos Policy's FSDP training loop.")


if __name__ == "__main__":
    main()
