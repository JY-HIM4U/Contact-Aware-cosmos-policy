#!/usr/bin/env python3
"""
Run Cosmos-Policy-LIBERO-Predict2-2B on LIBERO tasks.

This model was trained ON LIBERO data and produces 224×224 predictions.
It generates: actions, future images (wrist + third-person), and values.

We use it to:
1. Roll out a LIBERO task in the sim
2. At each step, query the model for future image predictions
3. Save the predicted vs actual future frames as a comparison video
"""
from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import h5py
import numpy as np
import torch

# Cosmos Policy imports
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.experiments.robot.cosmos_utils import (
    COSMOS_IMAGE_SIZE,
    COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    get_action,
    get_latent_indices_from_model_config,
    prepare_images_for_model,
)
from cosmos_policy.datasets.dataset_utils import apply_jpeg_compression_np, resize_images
from cosmos_policy.utils.utils import duplicate_array
from cosmos_policy.constants import ACTION_DIM, PROPRIO_DIM


def load_cosmos_policy_model(ckpt_dir: str, device: str = "cuda"):
    """Load the Cosmos-Policy LIBERO model.

    Uses the same loading path as the official eval script:
      load_model_from_checkpoint(experiment_name, s3_checkpoint_dir, config_file)
    """
    print(f"[cosmos-policy] Loading model from {ckpt_dir}...")

    # Pass HF repo ID — the loader will resolve the checkpoint path
    model, config = load_model_from_checkpoint(
        experiment_name="cosmos_predict2_2b_480p_libero__inference_only",
        s3_checkpoint_dir="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        config_file="cosmos_policy/config/config.py",
        load_ema_to_reg=False,
    )
    model.eval()
    model = model.to(device)

    # Load dataset stats
    stats_path = os.path.join(ckpt_dir, "libero_dataset_statistics.json")
    with open(stats_path) as f:
        dataset_stats = json.load(f)

    # Load T5 text embeddings
    t5_path = os.path.join(ckpt_dir, "libero_t5_embeddings.pkl")
    with open(t5_path, "rb") as f:
        t5_embeddings = pickle.load(f)

    # Init the global T5 cache for get_action()
    from cosmos_policy.experiments.robot.cosmos_utils import init_t5_text_embeddings_cache
    init_t5_text_embeddings_cache(t5_path)

    print(f"[cosmos-policy] Model loaded. T5 embeddings: {len(t5_embeddings)} tasks")
    return model, config, dataset_stats, t5_embeddings


def load_libero_episode_from_hdf5(hdf5_path: str, demo_idx: int = 0):
    """Load a LIBERO episode directly from HDF5 (original 128×128 images)."""
    with h5py.File(hdf5_path, "r") as f:
        demo = f[f"data/demo_{demo_idx}"]
        images_agentview = demo["obs/agentview_rgb"][:]      # (T, 128, 128, 3)
        images_wrist = demo["obs/eye_in_hand_rgb"][:]        # (T, 128, 128, 3)
        actions = demo["actions"][:].astype(np.float32)       # (T, 7)
        ee_pos = demo["obs/ee_pos"][:].astype(np.float32)     # (T, 3)
        ee_ori = demo["obs/ee_ori"][:].astype(np.float32)     # (T, 3)
        gripper = demo["obs/gripper_states"][:].astype(np.float32)  # (T, 2)
        joint_states = demo["obs/joint_states"][:].astype(np.float32)  # (T, 7)

    # Build proprio: [gripper_joints(2), ee_pos(3), ee_quat(4)]
    # For simplicity, use ee_pos + ee_ori + gripper[:, :1]
    # Actual Cosmos-Policy LIBERO uses 9D proprio
    proprio = np.concatenate([gripper, ee_pos, np.zeros((len(ee_pos), 4), dtype=np.float32)], axis=1)[:, :9]

    return {
        "agentview": images_agentview,
        "wrist": images_wrist,
        "actions": actions,
        "proprio": proprio,
    }


def generate_future_predictions(model, config, dataset_stats, t5_embeddings,
                                episode_data, task_name, num_steps=50,
                                device="cuda"):
    """Roll through an episode, predicting future images at each step.

    Returns lists of predicted and actual future frames.
    """
    images_agent = episode_data["agentview"]
    images_wrist = episode_data["wrist"]
    actions = episode_data["actions"]
    proprios = episode_data["proprio"]
    T = min(len(images_agent) - 16, num_steps)  # Need 16 steps ahead for chunk

    # Get T5 embedding for this task
    t5_emb = None
    for key, emb in t5_embeddings.items():
        if task_name.lower().replace("_", " ") in key.lower().replace("_", " "):
            t5_emb = emb
            break
    if t5_emb is None:
        # Use first available
        t5_emb = list(t5_embeddings.values())[0]
        print(f"[cosmos-policy] WARNING: No matching T5 embedding for '{task_name}', using fallback")
    else:
        print(f"[cosmos-policy] Found T5 embedding for task")

    # Prepare config-like object for get_action
    class FakeCfg:
        suite = "libero"
        use_wrist_image = True
        use_proprio = True
        normalize_proprio = True
        unnormalize_actions = True
        trained_with_image_aug = True
        use_jpeg_compression = True
        flip_images = True
        num_wrist_images = 1
        ar_future_prediction = True
        ar_value_prediction = False
        use_variance_scale = False

    cfg = FakeCfg()

    predicted_frames = []
    gt_frames = []

    print(f"[cosmos-policy] Generating {T} future predictions...")
    for t in range(0, T, 16):  # Step by action chunk size
        if t % 16 == 0:
            print(f"  Step {t}/{T}")

        # Current observation
        obs = {
            "primary_image": images_agent[t],
            "wrist_image": images_wrist[t],
            "proprio": proprios[t],
        }

        try:
            result = get_action(
                cfg=cfg,
                model=model,
                dataset_stats=dataset_stats,
                obs=obs,
                task_label_or_embedding=t5_emb,
                seed=t,
                num_denoising_steps_action=5,
            )

            # Extract future image predictions
            if "future_primary_image" in result:
                pred_frame = result["future_primary_image"]
                if isinstance(pred_frame, torch.Tensor):
                    pred_frame = pred_frame.cpu().numpy()
                if pred_frame.dtype != np.uint8:
                    pred_frame = (pred_frame * 255).clip(0, 255).astype(np.uint8)
                predicted_frames.append(pred_frame)
            else:
                predicted_frames.append(images_agent[t])  # fallback

        except Exception as e:
            print(f"  Error at step {t}: {e}")
            predicted_frames.append(images_agent[t])

        # GT future frame (16 steps ahead)
        future_t = min(t + 16, len(images_agent) - 1)
        gt_frames.append(images_agent[future_t])

    return predicted_frames, gt_frames


def write_comparison_video(gt_frames, pred_frames, path, fps=2):
    """Side-by-side comparison video at 224x224."""
    n = min(len(gt_frames), len(pred_frames))
    if n == 0:
        print("[cosmos-policy] No frames to write")
        return

    h, w = gt_frames[0].shape[:2]
    h2, w2 = pred_frames[0].shape[:2]
    H = max(h, h2)

    label_h = 28
    gap = 4
    canvas_w = w + w2 + gap
    canvas_h = H + label_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (canvas_w, canvas_h))

    for i in range(n):
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30
        gt = cv2.resize(gt_frames[i], (w, H)) if gt_frames[i].shape[0] != H else gt_frames[i]
        pr = cv2.resize(pred_frames[i], (w2, H)) if pred_frames[i].shape[0] != H else pred_frames[i]

        canvas[label_h:label_h+H, 0:w] = gt
        canvas[label_h:label_h+H, w+gap:w+gap+w2] = pr

        cv2.putText(canvas, f"GT (t+16)", (4, label_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, f"Cosmos Predicted", (w+gap+4, label_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        cv2.putText(canvas, f"step {i}", (canvas_w-60, label_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[cosmos-policy] Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str,
                        default="/tmp/cosmos-predict2/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    parser.add_argument("--hdf5_path", type=str,
                        default=os.path.join(LIBERO_DATA_ROOT,
                                "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_demo.hdf5"))
    parser.add_argument("--demo_idx", type=int, default=45)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(CONTACT_AWARE_WM, "results/cosmos_policy"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model, config, dataset_stats, t5_embeddings = load_cosmos_policy_model(args.ckpt_dir)

    # Load episode
    print(f"[cosmos-policy] Loading episode from {args.hdf5_path}, demo {args.demo_idx}")
    episode_data = load_libero_episode_from_hdf5(args.hdf5_path, args.demo_idx)
    print(f"[cosmos-policy] Episode: agentview={episode_data['agentview'].shape}, "
          f"wrist={episode_data['wrist'].shape}, actions={episode_data['actions'].shape}")

    # Extract task name from filename
    task_name = os.path.basename(args.hdf5_path).replace("_demo.hdf5", "")

    # Generate predictions
    pred_frames, gt_frames = generate_future_predictions(
        model, config, dataset_stats, t5_embeddings,
        episode_data, task_name, num_steps=args.num_steps,
    )

    # Save comparison video
    out_path = os.path.join(args.out_dir,
                            f"cosmos_policy_{task_name[:50]}_demo{args.demo_idx}.mp4")
    write_comparison_video(gt_frames, pred_frames, out_path)

    # Also save individual frames
    frame_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for i in range(min(5, len(pred_frames))):
        cv2.imwrite(f"{frame_dir}/pred_{i}.png",
                    cv2.cvtColor(pred_frames[i], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{frame_dir}/gt_{i}.png",
                    cv2.cvtColor(gt_frames[i], cv2.COLOR_RGB2BGR))

    print(f"\n[cosmos-policy] Done! {len(pred_frames)} predictions saved to {args.out_dir}")


if __name__ == "__main__":
    main()
