#!/usr/bin/env python3
"""
Evaluate world model variants on LIBERO via autoregressive rollout.

For each model variant and test episode:
  1. Anchor = GT frame[0] (permanent)
  2. Context deque of N-1 predicted frames (memory)
  3. Generate each frame autoregressively, comparing with GT

Metrics:
  - mse_curve, mean_mse, final_mse — pixel-level reconstruction
  - ssim_curve, mean_ssim — structural similarity
  - spike_count — MSE spikes > 3x rolling mean (measures instability)
  - recovery_rate — fraction of spikes that self-correct within 10 steps

Output:
  - results/libero/metrics_table.csv — one row per variant
  - results/libero/mse_curves_all_variants.png — all MSE curves
  - results/libero/rollout_{variant}_ep{i}.mp4 — comparison videos

Usage:
    python evaluate_libero.py --suite spatial [--max_episodes 5] [--max_frames 100]
"""

from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import argparse
import csv
import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_improved import ImprovedWorldModel, ModelConfig


def load_checkpoint(ckpt_dir: str, device: torch.device):
    """Load a model from checkpoint directory."""
    ckpt_path = os.path.join(ckpt_dir, "best_val.pt")
    if not os.path.exists(ckpt_path):
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})

    cfg = ModelConfig(
        use_ft=config.get("use_ft", False),
        context_frames=config.get("context_frames", 1),
        use_anchor=config.get("use_anchor", False),
        action_dim=7,
        img_size=config.get("img_size", 96),
    )
    model = ImprovedWorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config


def load_episode_npz(npz_path: str, stats: Dict, img_size: int = 96) -> Dict:
    """Load episode data and normalize actions/FT.

    If img_size differs from the stored resolution, resize all frames.
    """
    data = dict(np.load(npz_path))

    # Resize images if needed (NPZ stores at 96×96, model may want 128×128)
    images = data["images"]
    if images.shape[1] != img_size or images.shape[2] != img_size:
        T = images.shape[0]
        resized = np.zeros((T, img_size, img_size, 3), dtype=np.uint8)
        for t in range(T):
            resized[t] = cv2.resize(images[t], (img_size, img_size),
                                    interpolation=cv2.INTER_LINEAR)
        data["images"] = resized

    data["actions_norm"] = (
        (data["actions"] - stats["action_mean"]) / stats["action_std"]
    ).astype(np.float32)
    data["ft_norm"] = (
        (data["ft_wrist"] - stats["ft_mean"]) / stats["ft_std"]
    ).astype(np.float32)
    return data


def frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 or (H, W, 3) float → (1, 3, H, W) float tensor."""
    if frame.dtype == np.uint8:
        frame = frame.astype(np.float32) / 255.0
    return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)


def build_context(anchor: torch.Tensor, memory: deque, context_frames: int,
                  use_anchor: bool) -> torch.Tensor:
    """Build (1, N, 3, H, W) context tensor."""
    if context_frames == 1:
        return memory[-1] if len(memory) > 0 else anchor

    if use_anchor:
        slots = [anchor]
        mem_needed = context_frames - 1
        if len(memory) >= mem_needed:
            slots.extend(list(memory)[-mem_needed:])
        else:
            pad = mem_needed - len(memory)
            slots.extend([anchor] * pad)
            slots.extend(list(memory))
    else:
        if len(memory) >= context_frames:
            slots = list(memory)[-context_frames:]
        else:
            pad = context_frames - len(memory)
            slots = [anchor] * pad + list(memory)

    return torch.cat(slots, dim=0).unsqueeze(0)


def build_chunk(history: deque, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (1, H, D) chunk + mask from history deque."""
    vals = list(history)
    H = chunk_size
    D = vals[0].shape[-1] if vals else 7
    chunk = np.zeros((H, D), dtype=np.float32)
    mask = np.zeros(H, dtype=bool)
    n = min(len(vals), H)
    for k in range(n):
        chunk[H - n + k] = vals[len(vals) - n + k]
        mask[H - n + k] = True
    return (
        torch.from_numpy(chunk).unsqueeze(0),
        torch.from_numpy(mask).unsqueeze(0),
    )


@torch.no_grad()
def rollout_episode(model, episode_data: Dict, config: Dict,
                    device: torch.device, max_frames: int = 0,
                    num_samples: int = 3, ode_steps: int = 20) -> Dict:
    """Run autoregressive rollout on one episode.

    Returns dict with pred_frames, gt_frames, mse_curve.
    """
    images = episode_data["images"]
    actions = episode_data["actions_norm"]
    ft = episode_data["ft_norm"]
    T = len(images)

    context_frames = config.get("context_frames", 1)
    use_anchor = config.get("use_anchor", False)
    use_ft = config.get("use_ft", False)
    chunk_size = config.get("chunk_size", 4)

    N = T - 1
    if max_frames > 0:
        N = min(N, max_frames)

    anchor = frame_to_tensor(images[0], device)
    memory = deque(maxlen=max(context_frames - 1, 1) if use_anchor else context_frames)
    memory.append(anchor)

    action_hist = deque(maxlen=chunk_size)
    ft_hist = deque(maxlen=chunk_size)

    gt_frames = [images[0]]
    pred_frames = [images[0]]  # frame 0 is GT

    for i in range(N):
        action_hist.append(actions[i])
        ft_hist.append(ft[i])

        ctx = build_context(anchor, memory, context_frames, use_anchor).to(device)
        act_chunk, act_mask = build_chunk(action_hist, chunk_size)
        act_chunk = act_chunk.to(device)
        act_mask = act_mask.to(device)

        ft_chunk = None
        if use_ft:
            ft_chunk, _ = build_chunk(ft_hist, chunk_size)
            ft_chunk = ft_chunk.to(device)

        pred = model.sample(ctx, act_chunk, ft_chunk, num_steps=ode_steps,
                            pad_mask=act_mask, anchor_frame=anchor,
                            num_samples=num_samples)

        pred_np = pred.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        pred_uint8 = (pred_np * 255).astype(np.uint8)
        pred_frames.append(pred_uint8)
        gt_frames.append(images[i + 1])

        memory.append(pred)

    # Compute metrics
    mse_curve = []
    for i in range(len(pred_frames)):
        gt = gt_frames[i].astype(np.float32) / 255.0
        pr = pred_frames[i].astype(np.float32) / 255.0
        mse_curve.append(np.mean((gt - pr) ** 2))

    mse_curve = np.array(mse_curve)

    # Spike detection: MSE > 3x rolling mean (window=20)
    spike_count = 0
    recoveries = 0
    window = 20
    for i in range(window, len(mse_curve)):
        rolling_mean = mse_curve[max(0, i - window):i].mean()
        if mse_curve[i] > 3 * rolling_mean and rolling_mean > 0.001:
            spike_count += 1
            # Check recovery within 10 steps
            if i + 10 < len(mse_curve):
                future = mse_curve[i + 1:i + 11]
                if future.min() < 2 * rolling_mean:
                    recoveries += 1

    recovery_rate = recoveries / max(spike_count, 1)

    return {
        "pred_frames": pred_frames,
        "gt_frames": gt_frames,
        "mse_curve": mse_curve,
        "mean_mse": float(mse_curve[1:].mean()),
        "final_mse": float(mse_curve[-1]),
        "spike_count": spike_count,
        "recovery_rate": recovery_rate,
    }


def save_comparison_video(gt_frames, pred_frames, path, fps=10):
    """Side-by-side GT | Predicted video."""
    h, w = gt_frames[0].shape[:2]
    canvas_w = w * 2 + 4
    label_h = 20
    canvas_h = h + label_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (canvas_w, canvas_h))

    n = min(len(gt_frames), len(pred_frames))
    for i in range(n):
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40
        canvas[label_h:label_h + h, 0:w] = gt_frames[i]
        canvas[label_h:label_h + h, w + 4:w + 4 + w] = pred_frames[i]
        cv2.putText(canvas, "GT", (4, label_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, "Predicted", (w + 8, label_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, f"t={i}", (canvas_w - 40, label_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LIBERO world model variants")
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(CONTACT_AWARE_WM, "data/libero_90_with_ft"))
    parser.add_argument("--suite", type=str, default="all")
    parser.add_argument("--ckpt_root", type=str, default="checkpoints")
    parser.add_argument("--max_episodes", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--ode_steps", type=int, default=50,
                        help="ODE integration steps (more = sharper, 50 recommended)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Samples to average (1 = sharper, >1 = blurry but stable)")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "results/libero"
    os.makedirs(out_dir, exist_ok=True)

    # Discover checkpoints
    ckpt_dirs = {}
    for name in sorted(os.listdir(args.ckpt_root)):
        full = os.path.join(args.ckpt_root, name)
        if os.path.isdir(full) and name.startswith("libero_"):
            if os.path.exists(os.path.join(full, "best_val.pt")):
                ckpt_dirs[name] = full

    print(f"[eval] Found {len(ckpt_dirs)} variants: {list(ckpt_dirs.keys())}")

    # Load normalization stats
    stats_path = os.path.join(args.data_root, f"dataset_stats_{args.suite}.json")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(args.data_root, "dataset_stats_all.json")
    with open(stats_path) as f:
        raw_stats = json.load(f)
    stats = {k: np.array(v, dtype=np.float32) for k, v in raw_stats.items()}

    # Discover test episodes
    task_dirs = sorted([
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ])

    test_episodes = []
    for td in task_dirs:
        npzs = sorted([f for f in os.listdir(os.path.join(args.data_root, td)) if f.endswith(".npz")])
        n = len(npzs)
        n_train = max(1, int(0.9 * n))
        for npz in npzs[n_train:]:
            test_episodes.append(os.path.join(args.data_root, td, npz))

    if args.max_episodes > 0:
        test_episodes = test_episodes[:args.max_episodes]
    print(f"[eval] Test episodes: {len(test_episodes)}")

    # Evaluate each variant
    all_results = {}

    for variant_name, ckpt_dir in ckpt_dirs.items():
        print(f"\n[eval] === {variant_name} ===")
        model, config = load_checkpoint(ckpt_dir, device)
        if model is None:
            print(f"  Skipping (no checkpoint)")
            continue

        variant_mses = []
        variant_spikes = []
        variant_recoveries = []
        all_mse_curves = []

        model_img_size = config.get("img_size", 96)
        for ep_i, ep_path in enumerate(test_episodes):
            ep_data = load_episode_npz(ep_path, stats, img_size=model_img_size)
            result = rollout_episode(
                model, ep_data, config, device,
                max_frames=args.max_frames,
                num_samples=args.num_samples,
                ode_steps=args.ode_steps,
            )

            variant_mses.append(result["mean_mse"])
            variant_spikes.append(result["spike_count"])
            variant_recoveries.append(result["recovery_rate"])
            all_mse_curves.append(result["mse_curve"])

            # Save video for first episode
            if ep_i == 0:
                vid_path = os.path.join(out_dir, f"rollout_{variant_name}.mp4")
                save_comparison_video(result["gt_frames"], result["pred_frames"],
                                      vid_path, fps=args.fps)
                print(f"  Saved video: {vid_path}")

            print(f"  Ep {ep_i}: mean_mse={result['mean_mse']:.5f}, "
                  f"final_mse={result['final_mse']:.5f}, "
                  f"spikes={result['spike_count']}")

        all_results[variant_name] = {
            "mean_mse": float(np.mean(variant_mses)),
            "std_mse": float(np.std(variant_mses)),
            "final_mse": float(np.mean([c[-1] for c in all_mse_curves])),
            "spike_count": float(np.mean(variant_spikes)),
            "recovery_rate": float(np.mean(variant_recoveries)),
            "mse_curves": all_mse_curves,
        }

    # Save metrics table
    csv_path = os.path.join(out_dir, "metrics_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "mean_mse", "std_mse", "final_mse",
                          "spike_count", "recovery_rate"])
        for name, r in all_results.items():
            writer.writerow([name, f"{r['mean_mse']:.6f}", f"{r['std_mse']:.6f}",
                             f"{r['final_mse']:.6f}", f"{r['spike_count']:.1f}",
                             f"{r['recovery_rate']:.3f}"])
    print(f"\n[eval] Saved metrics: {csv_path}")

    # Plot MSE curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, r in all_results.items():
        # Average MSE curve across episodes
        max_len = max(len(c) for c in r["mse_curves"])
        avg_curve = np.zeros(max_len)
        counts = np.zeros(max_len)
        for c in r["mse_curves"]:
            avg_curve[:len(c)] += c
            counts[:len(c)] += 1
        avg_curve = avg_curve / np.maximum(counts, 1)
        ax.plot(avg_curve, label=name, linewidth=1.5)

    ax.set_xlabel("Frame", fontsize=13)
    ax.set_ylabel("MSE vs Ground Truth", fontsize=13)
    ax.set_title("Autoregressive Rollout: MSE Drift Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "mse_curves_all_variants.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Saved MSE curves: {plot_path}")

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"{'Variant':<45} | {'MSE':>10} | {'Final':>10} | {'Spikes':>7} | {'Recov':>7}")
    print("-" * 90)
    for name, r in all_results.items():
        print(f"{name:<45} | {r['mean_mse']:>10.5f} | {r['final_mse']:>10.5f} | "
              f"{r['spike_count']:>7.1f} | {r['recovery_rate']:>7.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
