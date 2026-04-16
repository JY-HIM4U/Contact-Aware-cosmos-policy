"""
Autoregressive video generation from a single first frame.

For each test episode:
  1. Start with ground-truth frame 0
  2. At each step, feed the *predicted* frame (not ground truth) back as input
     along with the ground-truth action (and optionally F/T)
  3. Generate the full rollout and compare with ground truth side-by-side

Supports multi-frame context and first-frame anchoring:
  - anchor: GT frame[0] is permanently stored and always injected as slot 0
  - memory: deque of N-1 most recent predicted frames
  - context = [anchor, memory_oldest, ..., memory_newest]

Usage:
    python generate_video.py [--ode_steps 20] [--num_samples 5]
"""

import argparse
import json
import os
from collections import deque

import cv2
import numpy as np
import torch

from model_fm import FlowMatchingWorldModel


def load_fm_model(condition, context_frames, use_anchor, device):
    """Load flow matching checkpoint, trying context-aware name first."""
    # Try context-specific checkpoint name
    tag = f"fm_{condition}"
    if context_frames > 1:
        tag += f"_ctx{context_frames}"
        if use_anchor:
            tag += "_anchor"

    ckpt_path = os.path.join("checkpoints", f"{tag}_best.pt")

    # Fallback to old naming convention
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("checkpoints", f"fm_{condition}_best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found for {tag}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ctx = ckpt.get("context_frames", 1)
    anc = ckpt.get("use_anchor", False)

    model = FlowMatchingWorldModel(
        condition=condition, context_frames=ctx, use_anchor=anc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ode_steps = ckpt.get("ode_steps", 20)
    chunk_size = ckpt.get("chunk_size", 4)
    print(f"[video] Loaded {tag} (epoch {ckpt['epoch']}, ctx={ctx}, anchor={anc}, "
          f"chunk={chunk_size})")
    return model, ode_steps, chunk_size, ctx, anc


def load_episode_data(episode_id):
    """Load all sequential data for a given episode."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
    data = np.load(os.path.join(data_dir, "samples.npz"))
    ep_ids = data["episode_ids"]

    mask = np.where(ep_ids == episode_id)[0]
    if len(mask) == 0:
        raise ValueError(f"Episode {episode_id} not found")

    with open(os.path.join(data_dir, "ft_stats.json")) as f:
        ft_stats = json.load(f)
    ft_mean = np.array(ft_stats["mean"], dtype=np.float32)
    ft_std = np.array(ft_stats["std"], dtype=np.float32)

    images_t = data["images_t"][mask]
    actions = data["actions"][mask]
    fts = data["fts"][mask]
    images_next = data["images_next"][mask]

    fts_norm = (fts - ft_mean) / ft_std
    gt_frames = list(images_t) + [images_next[-1]]

    return gt_frames, actions, fts_norm


def build_action_chunk(history, chunk_size):
    """Build a (1, H, D) tensor from a deque of past values."""
    H = chunk_size
    vals = list(history)
    D = vals[0].shape[-1] if len(vals) > 0 else 3
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


def frame_to_tensor(frame_uint8, device):
    """Convert (96, 96, 3) uint8 → (1, 3, 96, 96) float tensor."""
    f = frame_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).to(device)


def build_context_tensor(anchor_tensor, memory_deque, context_frames, use_anchor, device):
    """Build (1, N, 3, 96, 96) context tensor from anchor + memory deque.

    Args:
        anchor_tensor: (1, 3, 96, 96) — permanent GT first frame
        memory_deque: deque of (1, 3, 96, 96) tensors — recent predicted frames
        context_frames: N
        use_anchor: whether slot 0 is the anchor

    Returns:
        context: (1, N, 3, 96, 96) or (1, 3, 96, 96) when N=1
    """
    if context_frames == 1:
        # Single frame: just return the most recent memory frame
        if len(memory_deque) > 0:
            return memory_deque[-1]
        return anchor_tensor

    if use_anchor:
        # Slot 0 = anchor, slots 1..N-1 = memory
        memory_slots = context_frames - 1
        frames = [anchor_tensor]

        if len(memory_deque) >= memory_slots:
            frames.extend(list(memory_deque)[-memory_slots:])
        else:
            # Pad with anchor for missing early memory slots
            pad_count = memory_slots - len(memory_deque)
            frames.extend([anchor_tensor] * pad_count)
            frames.extend(list(memory_deque))
    else:
        # All N slots are memory
        frames = []
        if len(memory_deque) >= context_frames:
            frames = list(memory_deque)[-context_frames:]
        else:
            pad_count = context_frames - len(memory_deque)
            frames = [anchor_tensor] * pad_count + list(memory_deque)

    return torch.cat(frames, dim=0).unsqueeze(0)  # (1, N, 3, 96, 96)


@torch.no_grad()
def rollout(model, gt_frames, actions, fts, condition, ode_steps, chunk_size,
            context_frames, use_anchor, device, max_frames=0, num_samples=5):
    """Autoregressive rollout from first frame with anchor + memory."""
    N = len(actions)
    if max_frames > 0:
        N = min(N, max_frames)

    # Permanent anchor: GT frame[0], never replaced
    anchor_tensor = frame_to_tensor(gt_frames[0], device)

    # Memory deque: holds recent predicted frames
    memory_size = max(context_frames - 1, 1) if use_anchor else context_frames
    memory = deque(maxlen=memory_size)
    memory.append(anchor_tensor)  # Initialize with frame[0]

    predicted_frames = [gt_frames[0]]  # First frame is GT

    action_history = deque(maxlen=chunk_size)
    ft_history = deque(maxlen=chunk_size)

    for i in range(N):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Frame {i+1}/{N}")

        action_history.append(actions[i])
        ft_history.append(fts[i])

        action_chunk, action_mask = build_action_chunk(action_history, chunk_size)
        action_chunk = action_chunk.to(device)
        action_mask = action_mask.to(device)

        if condition == "image_ft":
            ft_chunk, _ = build_action_chunk(ft_history, chunk_size)
            ft_chunk = ft_chunk.to(device)
        else:
            ft_chunk = None

        # Build context: [anchor, memory_oldest, ..., memory_newest]
        context = build_context_tensor(
            anchor_tensor, memory, context_frames, use_anchor, device)

        pred_img = model.sample(
            context, action_chunk, ft_chunk,
            num_steps=ode_steps, pad_mask=action_mask,
            num_samples=num_samples)

        # Store prediction
        pred_np = pred_img.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        pred_uint8 = (pred_np * 255).astype(np.uint8)
        predicted_frames.append(pred_uint8)

        # Update memory with prediction (anchor stays permanent)
        memory.append(pred_img)

    return predicted_frames


def frames_to_video(frames, path, fps=10):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def make_comparison_video(gt_frames, pred_dict, path, fps=10):
    names = list(pred_dict.keys())
    ncols = 1 + len(names)
    h, w = gt_frames[0].shape[:2]
    label_h = 24
    frame_h = h + label_h
    canvas_w = w * ncols + (ncols - 1) * 2

    min_len = len(gt_frames)
    for preds in pred_dict.values():
        min_len = min(min_len, len(preds))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (canvas_w, frame_h))
    labels = ["Ground Truth"] + names

    for i in range(min_len):
        canvas = np.ones((frame_h, canvas_w, 3), dtype=np.uint8) * 40
        all_frames = [gt_frames[i]] + [pred_dict[n][i] for n in names]
        for j, (frame, label) in enumerate(zip(all_frames, labels)):
            x_off = j * (w + 2)
            canvas[label_h:label_h + h, x_off:x_off + w] = frame
            cv2.putText(canvas, label, (x_off + 4, label_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, f"t={i}", (canvas_w - 40, label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def compute_rollout_metrics(gt_frames, pred_frames):
    n = min(len(gt_frames), len(pred_frames))
    mses = []
    for i in range(n):
        gt = gt_frames[i].astype(np.float32) / 255.0
        pred = pred_frames[i].astype(np.float32) / 255.0
        mses.append(np.mean((gt - pred) ** 2))
    return np.array(mses)


def plot_rollout_metrics(metrics_dict, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, mses in metrics_dict.items():
        ax.plot(mses, label=name, linewidth=1.5)
    ax.set_xlabel("Frame", fontsize=13)
    ax.set_ylabel("MSE vs Ground Truth", fontsize=13)
    ax.set_title("Autoregressive Rollout: Prediction Drift Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(len(m) for m in metrics_dict.values()))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[video] Saved rollout metrics plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate autoregressive rollout videos")
    parser.add_argument("--ode_steps", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    # Model selection: specify which models to load
    parser.add_argument("--condition", type=str, default="image_ft",
                        choices=["image_only", "image_ft"])
    parser.add_argument("--context_frames", type=int, default=0,
                        help="Context frames (0 = auto-detect from checkpoint)")
    parser.add_argument("--use_anchor", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[video] Device: {device}")
    os.makedirs("results", exist_ok=True)

    # Try loading models with various configs
    models = {}

    # Try loading the old single-frame model as baseline
    for cond in ["image_only", "image_ft"]:
        try:
            m, ode, cs, ctx, anc = load_fm_model(cond, 1, False, device)
            label = f"FM-1frame ({cond.replace('image_', '')})"
            models[label] = (m, cond, ode, cs, ctx, anc)
        except FileNotFoundError:
            pass

    # Try loading multi-frame models
    for cond in ["image_only", "image_ft"]:
        for ctx_n in [4]:
            for anc in [True, False]:
                try:
                    m, ode, cs, ctx, anc_actual = load_fm_model(cond, ctx_n, anc, device)
                    tag_parts = [f"FM-{ctx}frame"]
                    if anc_actual:
                        tag_parts.append("anchor")
                    tag_parts.append(f"({cond.replace('image_', '')})")
                    label = " ".join(tag_parts)
                    models[label] = (m, cond, ode, cs, ctx, anc_actual)
                except FileNotFoundError:
                    pass

    if not models:
        print("[video] No models found!")
        return

    print(f"\n[video] Models loaded: {list(models.keys())}")

    # Find test episodes
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
    data = np.load(os.path.join(data_dir, "samples.npz"))
    ep_ids = data["episode_ids"]
    with open(os.path.join(data_dir, "splits.json")) as f:
        splits = json.load(f)
    test_idxs = set(splits["test"])
    test_episodes = sorted(set(ep_ids[i] for i in test_idxs))

    print(f"[video] Test episodes: {test_episodes}")

    for ep_id in test_episodes:
        print(f"\n{'='*60}")
        print(f"  Episode {ep_id}")
        print(f"{'='*60}")

        gt_frames, actions, fts = load_episode_data(ep_id)
        print(f"[video] Episode {ep_id}: {len(gt_frames)} GT frames, {len(actions)} actions")

        pred_dict = {}
        metrics_dict = {}

        for name, (model, condition, ode_steps, chunk_size, ctx, anc) in models.items():
            ode_s = args.ode_steps if args.ode_steps else ode_steps
            print(f"\n[video] Rolling out: {name} (ode={ode_s}, chunk={chunk_size}, "
                  f"ctx={ctx}, anchor={anc}, samples={args.num_samples})")

            pred_frames = rollout(
                model, gt_frames, actions, fts,
                condition=condition,
                ode_steps=ode_s,
                chunk_size=chunk_size,
                context_frames=ctx,
                use_anchor=anc,
                device=device,
                max_frames=args.max_frames,
                num_samples=args.num_samples,
            )
            pred_dict[name] = pred_frames

            mses = compute_rollout_metrics(gt_frames, pred_frames)
            metrics_dict[name] = mses
            print(f"  MSE: start={mses[1]:.5f}, mid={mses[len(mses)//2]:.5f}, "
                  f"end={mses[-1]:.5f}, mean={mses[1:].mean():.5f}")

            # Individual video
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            vid_path = f"results/rollout_ep{ep_id}_{safe_name}.mp4"
            frames_to_video(pred_frames, vid_path, fps=args.fps)
            print(f"[video] Saved: {vid_path}")

        # GT video
        gt_vid_path = f"results/rollout_ep{ep_id}_gt.mp4"
        frames_to_video(gt_frames, gt_vid_path, fps=args.fps)

        # Comparison video
        comp_path = f"results/rollout_ep{ep_id}_comparison.mp4"
        make_comparison_video(gt_frames, pred_dict, comp_path, fps=args.fps)
        print(f"[video] Saved comparison: {comp_path}")

        # Drift plot
        plot_path = f"results/rollout_ep{ep_id}_drift.png"
        plot_rollout_metrics(metrics_dict, plot_path)

    print("\n[video] All done!")


if __name__ == "__main__":
    main()
