#!/usr/bin/env python3
"""
Run NVIDIA Cosmos-Predict2-2B-Sample-Action-Conditioned on a LIBERO test episode.

This generates a video from:
  - First GT frame of the episode
  - Sequence of robot actions (7-DoF: dx, dy, dz, droll, dpitch, dyaw, gripper)

The Cosmos action-conditioned model was post-trained on the Bridge dataset (real
robot). We test it zero-shot on LIBERO simulated data to compare with our custom
flow matching world model.

Outputs:
  results/cosmos/cosmos_ep{ep_id}_pred.mp4 — Cosmos-generated video
  results/cosmos/cosmos_ep{ep_id}_gt.mp4 — Ground truth video
  results/cosmos/cosmos_ep{ep_id}_comparison.mp4 — Side-by-side
"""

from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import os
import sys
import argparse

# Set TOKENIZERS_PARALLELISM env var to avoid warnings with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress noisy warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import cv2
import mediapy as mp


def load_libero_episode(npz_path: str, hdf5_dir: str = None):
    """Load a LIBERO episode for Cosmos inference.

    Tries to load original 128×128 images from HDF5 (higher quality).
    Falls back to 96×96 NPZ images if HDF5 not found.

    Returns:
        first_frame: (H, W, 3) uint8 — initial conditioning frame
        actions: (T, 7) float32 — full action sequence (LIBERO 7D)
        gt_frames: list of (H, W, 3) uint8 — ground truth frames for comparison
    """
    data = np.load(npz_path)
    actions = data["actions"].astype(np.float32)  # (T, 7)

    # Try loading original 128×128 images from HDF5
    if hdf5_dir is None:
        hdf5_dir = LIBERO_DATA_ROOT

    # Extract task name and demo index from npz path
    task_name = os.path.basename(os.path.dirname(npz_path))
    demo_name = os.path.basename(npz_path).replace(".npz", "")
    demo_idx = int(demo_name.replace("demo_", ""))

    hdf5_path = os.path.join(hdf5_dir, f"{task_name}_demo.hdf5")
    if os.path.exists(hdf5_path):
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            demo_key = f"demo_{demo_idx}"
            if f"data/{demo_key}" in f:
                images = f[f"data/{demo_key}/obs/agentview_rgb"][:]  # (T, 128, 128, 3)
                print(f"[load] Using original HDF5 images: {images.shape}")
                first_frame = images[0]
                gt_frames = [images[i] for i in range(len(images))]
                return first_frame, actions, gt_frames

    # Fallback to NPZ (96×96)
    images = data["images"]
    print(f"[load] Using NPZ images: {images.shape}")
    first_frame = images[0]
    gt_frames = [images[i] for i in range(len(images))]
    return first_frame, actions, gt_frames


def libero_to_bridge_actions(libero_actions: np.ndarray) -> np.ndarray:
    """Convert LIBERO actions to the format expected by Cosmos action-conditioned.

    The Cosmos example script does: action_ee = raw_bridge_action[:, :6] * 20
    This means Bridge stored actions were divided by 20 for normalization,
    and Cosmos expects them BACK at original scale (multiplied by 20).

    LIBERO actions are 7D in [-1, 1]:
      [dx, dy, dz, droll, dpitch, dyaw, gripper]
    The OSC controller scales these by [0.05, 0.05, 0.05, 0.5, 0.5, 0.5].

    To match Bridge scale: multiply ee_delta by 20 (same as the example).
    Gripper: Bridge uses continuous width [0, 0.08], LIBERO uses [-1, 1].
    We map LIBERO gripper to Bridge range: (gripper + 1) / 2 * 0.08.

    Args:
        libero_actions: (T, 7) LIBERO actions in [-1, 1]

    Returns:
        bridge_actions: (T, 7) Bridge-format actions (scaled)
    """
    actions = libero_actions.copy()
    # LIBERO OSC controller maps: pos *= 0.05m, rot *= 0.5rad
    # Bridge stores raw ee displacement then example does * 20 to undo storage normalization
    # The model expects actions in Bridge's physical scale:
    #   pos: ~[-0.05, 0.05] meters → times 20 → [-1, 1] in model space
    #   rot: ~[-0.5, 0.5] radians → times 20 → [-10, 10] in model space
    # LIBERO actions are in [-1, 1] controller space:
    #   pos actual = action[:3] * 0.05  → already ~[-0.05, 0.05]
    #   rot actual = action[3:6] * 0.5  → already ~[-0.5, 0.5]
    # So: multiply LIBERO pos by (0.05 * 20 = 1.0), rot by (0.5 * 20 = 10.0)
    actions[:, :3] = actions[:, :3] * 1.0     # pos: already correct scale
    actions[:, 3:6] = actions[:, 3:6] * 10.0  # rot: scale up
    # Gripper: LIBERO [-1, 1] → Bridge continuous [0, 0.08]
    actions[:, 6] = (actions[:, 6] + 1.0) / 2.0 * 0.08
    return actions


def setup_cosmos_pipeline(model_size: str = "2B", seed: int = 0):
    """Initialize the Cosmos action-conditioned pipeline."""
    print(f"[cosmos] Initializing pipeline (model_size={model_size})...")

    from cosmos_predict2.configs.action_conditioned.config import (
        get_cosmos_predict2_action_conditioned_pipeline,
    )
    from cosmos_predict2.pipelines.video2world_action import (
        Video2WorldActionConditionedPipeline,
    )
    from imaginaire.constants import get_cosmos_predict2_action_conditioned_checkpoint
    from imaginaire.utils import misc

    # Monkey-patch the video preprocess to handle single-frame input.
    # The original code does randint(0, original-expected) which fails when
    # original (1) < expected (13). We replicate the single frame instead.
    from cosmos_predict2.pipelines import video2world as v2w_mod

    _orig_normalize = v2w_mod.Video2WorldPipeline._normalize_video_databatch_inplace
    def patched_normalize(self, data_batch, input_key=None):
        ik = self.input_video_key if input_key is None else input_key
        if ik in data_batch:
            v = data_batch[ik]
            if v.ndim == 5:  # (B, C, T, H, W)
                expected = self.tokenizer.get_pixel_num_frames(self.config.state_t)
                if v.shape[2] < expected:
                    # Replicate the (last) frame to match expected length
                    pad = expected - v.shape[2]
                    last = v[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
                    data_batch[ik] = torch.cat([v, last], dim=2)
        return _orig_normalize(self, data_batch, input_key)
    v2w_mod.Video2WorldPipeline._normalize_video_databatch_inplace = patched_normalize

    misc.set_random_seed(seed=seed, by_rank=True)

    # Optimize cuDNN
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    config = get_cosmos_predict2_action_conditioned_pipeline(
        model_size=model_size, resolution="480", fps="4"
    )
    # Disable guardrail and prompt refiner (don't need them for action-conditioned)
    config.guardrail_config.enabled = False
    config.prompt_refiner_config.enabled = False

    dit_path = get_cosmos_predict2_action_conditioned_checkpoint(
        model_size=model_size, resolution="480", fps="4"
    )
    print(f"[cosmos] DiT checkpoint: {dit_path}")

    pipe = Video2WorldActionConditionedPipeline.from_config(
        config=config,
        dit_path=dit_path,
        use_text_encoder=False,  # action-conditioned doesn't need text
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        load_prompt_refiner=False,
    )

    print(f"[cosmos] Pipeline ready.")
    return pipe


def resize_frame(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize an image to target shape, preserving aspect via center crop if needed."""
    h, w = frame.shape[:2]
    if h == target_h and w == target_w:
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def prepare_first_frame(first_frame: np.ndarray, target_h: int = 480,
                        target_w: int = 640) -> np.ndarray:
    """Resize the first frame to Cosmos's expected resolution.

    The action-conditioned model was trained on Bridge data at 640x480 (4:3).
    LIBERO frames are square (96x96 or 128x128). We resize to fill the
    target resolution by stretching (no black bars).
    """
    return cv2.resize(first_frame, (target_w, target_h),
                      interpolation=cv2.INTER_LANCZOS4)


def run_inference(
    pipe,
    first_frame: np.ndarray,
    actions: np.ndarray,
    chunk_size: int = 12,
    autoregressive: bool = False,
    guidance: float = 7.0,
    seed: int = 0,
) -> torch.Tensor:
    """Run Cosmos action-conditioned inference.

    Args:
        pipe: the loaded pipeline
        first_frame: (H, W, 3) uint8 — initial frame
        actions: (T, 7) — action sequence
        chunk_size: number of action steps per generation chunk (Cosmos: 12)
        autoregressive: if True, chain multiple chunks for longer videos

    Returns:
        video: torch.Tensor (1, 3, T, H, W) in [-1, 1]
    """
    # Resize to Cosmos's expected resolution
    first_frame = prepare_first_frame(first_frame)
    print(f"[cosmos] First frame shape (after resize): {first_frame.shape}")
    print(f"[cosmos] Actions shape: {actions.shape}")
    print(f"[cosmos] Chunk size: {chunk_size}, autoregressive: {autoregressive}")

    if autoregressive:
        video_chunks = []
        current_frame = first_frame
        n_chunks = len(actions) // chunk_size
        for i in range(0, n_chunks * chunk_size, chunk_size):
            print(f"  Chunk {i//chunk_size + 1}/{n_chunks}")
            chunk_actions = actions[i:i + chunk_size]
            video = pipe(
                current_frame,
                chunk_actions,
                num_conditional_frames=1,
                guidance=guidance,
                seed=seed + i,
            )
            # Use last frame as next chunk's input
            last = ((video[0, :, -1].permute(1, 2, 0).cpu().float().numpy() / 2 + 0.5)
                    .clip(0, 1) * 255).astype(np.uint8)
            current_frame = last
            video_chunks.append(video)
        # Concatenate, dropping the duplicate first frame from subsequent chunks
        video = torch.cat(
            [video_chunks[0]] + [c[:, :, :-1] for c in video_chunks[1:]], dim=2
        )
    else:
        # Single chunk: only first chunk_size action steps
        video = pipe(
            first_frame,
            actions[:chunk_size],
            num_conditional_frames=1,
            guidance=guidance,
            seed=seed,
        )

    return video


def video_tensor_to_uint8(video: torch.Tensor) -> np.ndarray:
    """Convert (1, 3, T, H, W) tensor in [-1, 1] to (T, H, W, 3) uint8."""
    v = video[0].cpu().float().numpy()  # (3, T, H, W)
    v = np.transpose(v, (1, 2, 3, 0))   # (T, H, W, 3)
    v = (v / 2 + 0.5).clip(0, 1) * 255
    return v.astype(np.uint8)


def write_video(frames, path: str, fps: int = 4):
    """Write list of (H, W, 3) uint8 frames to MP4."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[cosmos] Saved: {path}")


def make_comparison_video(gt_frames, cosmos_frames, our_frames, path: str, fps: int = 4):
    """Side-by-side comparison: [GT | Cosmos | Ours]."""
    h_gt, w_gt = gt_frames[0].shape[:2]
    h_co, w_co = cosmos_frames[0].shape[:2]
    h_ou, w_ou = our_frames[0].shape[:2] if our_frames else (h_gt, w_gt)

    # Resize all to a common height
    H = max(h_gt, h_co, h_ou)
    def resize_to_h(frames, h):
        out = []
        for f in frames:
            scale = h / f.shape[0]
            new_w = int(f.shape[1] * scale)
            out.append(cv2.resize(f, (new_w, h), interpolation=cv2.INTER_AREA))
        return out

    gt_r = resize_to_h(gt_frames, H)
    co_r = resize_to_h(cosmos_frames, H)
    if our_frames:
        ou_r = resize_to_h(our_frames, H)
    else:
        ou_r = []

    n_frames = min(len(gt_r), len(co_r))
    if ou_r:
        n_frames = min(n_frames, len(ou_r))

    label_h = 24
    gap = 4
    canvas_w = gt_r[0].shape[1] + co_r[0].shape[1] + (ou_r[0].shape[1] if ou_r else 0)
    if ou_r:
        canvas_w += gap * 2
    else:
        canvas_w += gap
    canvas_h = H + label_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (canvas_w, canvas_h))

    for i in range(n_frames):
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40
        x = 0
        # GT
        canvas[label_h:label_h + H, x:x + gt_r[i].shape[1]] = gt_r[i]
        cv2.putText(canvas, "Ground Truth", (x + 4, label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        x += gt_r[i].shape[1] + gap
        # Cosmos
        canvas[label_h:label_h + H, x:x + co_r[i].shape[1]] = co_r[i]
        cv2.putText(canvas, "Cosmos-Predict2-2B", (x + 4, label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        x += co_r[i].shape[1] + gap
        # Ours (if provided)
        if ou_r:
            canvas[label_h:label_h + H, x:x + ou_r[i].shape[1]] = ou_r[i]
            cv2.putText(canvas, "Our Custom Model", (x + 4, label_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[cosmos] Saved comparison: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run Cosmos on LIBERO test episode")
    parser.add_argument("--episode_npz", type=str,
                        default=os.path.join(DATA_ROOT,
                                "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket/demo_45.npz"))
    parser.add_argument("--our_video", type=str,
                        default=os.path.join(RESULTS_ROOT, "libero",
                                "rollout_libero_improved_ft_ctx4_anchor.mp4"),
                        help="Path to our custom model's video for comparison (optional)")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(CONTACT_AWARE_WM, "results/cosmos"))
    parser.add_argument("--chunk_size", type=int, default=12)
    parser.add_argument("--autoregressive", action="store_true",
                        help="Chain multiple chunks for longer videos")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ep_name = os.path.basename(args.episode_npz).replace(".npz", "")
    print(f"[cosmos] Episode: {ep_name}")

    # Load LIBERO data
    first_frame, actions, gt_frames = load_libero_episode(args.episode_npz)
    print(f"[cosmos] Loaded: first_frame={first_frame.shape}, "
          f"actions={actions.shape}, gt_frames={len(gt_frames)}")

    # Convert action format
    bridge_actions = libero_to_bridge_actions(actions)

    # Setup pipeline
    pipe = setup_cosmos_pipeline()

    # Run inference
    video = run_inference(
        pipe, first_frame, bridge_actions,
        chunk_size=args.chunk_size,
        autoregressive=args.autoregressive,
        guidance=args.guidance,
        seed=args.seed,
    )

    print(f"[cosmos] Generated video shape: {video.shape}")
    cosmos_frames = list(video_tensor_to_uint8(video))

    # Save Cosmos video
    cosmos_path = os.path.join(args.out_dir, f"cosmos_{ep_name}_pred.mp4")
    write_video(cosmos_frames, cosmos_path, fps=args.fps)

    # Save GT video
    gt_path = os.path.join(args.out_dir, f"cosmos_{ep_name}_gt.mp4")
    write_video(gt_frames[:len(cosmos_frames)], gt_path, fps=args.fps)

    # Load our model's video (if available) for comparison
    our_frames = []
    if args.our_video and os.path.exists(args.our_video):
        try:
            cap = cv2.VideoCapture(args.our_video)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                our_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            print(f"[cosmos] Loaded our model's video: {len(our_frames)} frames")
        except Exception as e:
            print(f"[cosmos] Failed to load our video: {e}")

    # Comparison video
    comp_path = os.path.join(args.out_dir, f"cosmos_{ep_name}_comparison.mp4")
    make_comparison_video(
        gt_frames[:len(cosmos_frames)], cosmos_frames, our_frames,
        comp_path, fps=args.fps,
    )

    print(f"\n[cosmos] Done! Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
