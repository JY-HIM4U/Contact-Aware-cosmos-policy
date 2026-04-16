#!/usr/bin/env python3
"""
Run Cosmos-Policy-LIBERO as a TRUE AUTOREGRESSIVE WORLD MODEL.

Instead of getting fresh observations from the simulator every 16 steps,
this script feeds the model's OWN predicted future images back as input.
Only frame 0 comes from the real simulator — everything after is imagined.

This is the fair comparison with our custom flow matching world model.

Flow:
  t=0:   REAL obs from sim → model → actions + predicted future images
  t=16:  Feed PREDICTED images back → model → actions + next future predictions
  t=32:  Feed PREDICTED images back → model → ...
  (No simulator involved after t=0)

Usage:
    cd /tmp/cosmos-policy
    python $CONTACT_AWARE_WM/run_cosmos_policy_autoregressive.py \
        --task_idx 0 --num_autoreg_steps 5 --seed 195
"""
from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import cv2
import h5py
import numpy as np
import torch

from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.experiments.robot.cosmos_utils import (
    COSMOS_IMAGE_SIZE,
    get_action,
    get_future_state_prediction,
    get_future_images_from_generated_samples,
    get_latent_indices_from_model_config,
    init_t5_text_embeddings_cache,
    prepare_images_for_model,
)
from cosmos_policy.datasets.dataset_utils import resize_images


def load_model_and_data(ckpt_repo="nvidia/Cosmos-Policy-LIBERO-Predict2-2B"):
    """Load model, stats, and T5 embeddings using official get_model()."""
    print("[AR] Loading model...")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model, load_dataset_stats, download_hf_checkpoint
    )

    # Build a minimal config object that get_model expects
    class ModelCfg:
        config = "cosmos_predict2_2b_480p_libero__inference_only"
        ckpt_path = ckpt_repo
        config_file = "cosmos_policy/config/config.py"

    model, cosmos_config = get_model(ModelCfg())

    # Download and load stats + T5 embeddings
    from huggingface_hub import hf_hub_download
    stats_path = hf_hub_download(ckpt_repo, "libero_dataset_statistics.json")
    t5_path = hf_hub_download(ckpt_repo, "libero_t5_embeddings.pkl")

    dataset_stats = load_dataset_stats(stats_path)
    init_t5_text_embeddings_cache(t5_path)

    with open(t5_path, "rb") as f:
        t5_embeddings = pickle.load(f)

    print(f"[AR] Model loaded. {len(t5_embeddings)} task embeddings.")
    return model, cosmos_config, dataset_stats, t5_embeddings


def get_initial_obs_from_libero(task_suite_name="libero_spatial", task_idx=0,
                                seed=195):
    """Create a LIBERO environment and get the initial observation."""
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()
    suite = bench[task_suite_name]()

    task = suite.get_task(task_idx)
    task_name = suite.get_task_names()[task_idx]
    init_states = suite.get_task_init_states(task_idx)

    from libero.libero.envs import OffScreenRenderEnv
    env_args = {
        "bddl_file_name": suite.get_task_bddl_file_path(task_idx),
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_depths": False,
    }

    env = OffScreenRenderEnv(**env_args)
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    env.sim.set_state_from_flattened(init_states[seed % len(init_states)])
    env.sim.forward()
    # Step with dummy action to get proper observation dict
    dummy_action = np.zeros(7)
    obs, _, _, _ = env.step(dummy_action)

    # Get task description for T5
    task_description = task_name.replace("_", " ")

    return env, obs, task_description, task_name


class FakeConfig:
    """Mimics the eval config for get_action()."""
    suite = "libero"
    use_wrist_image = True
    use_proprio = True
    normalize_proprio = True
    unnormalize_actions = True
    trained_with_image_aug = True
    use_jpeg_compression = True
    flip_images = True
    num_wrist_images = 1
    num_third_person_images = 1
    use_third_person_image = True
    ar_future_prediction = True
    ar_value_prediction = False
    use_variance_scale = False
    chunk_size = 16
    num_open_loop_steps = 16
    model_family = "cosmos"
    search_depth = 1
    mask_current_state_action_for_value_prediction = False
    mask_future_state_for_qvalue_prediction = False


def extract_obs_from_raw(raw_obs, flip=True):
    """Extract observation dict from raw LIBERO env observation."""
    primary = raw_obs["agentview_image"]
    wrist = raw_obs["robot0_eye_in_hand_image"]
    if flip:
        primary = primary[::-1, :, :]  # LIBERO returns flipped images
        wrist = wrist[::-1, :, :]
    proprio = np.concatenate([
        raw_obs["robot0_gripper_qpos"],
        raw_obs["robot0_eef_pos"],
        raw_obs["robot0_eef_quat"],
    ])
    return {
        "primary_image": primary.copy(),
        "wrist_image": wrist.copy(),
        "proprio": proprio.copy(),
    }


def run_autoregressive_worldmodel(model, config, dataset_stats, t5_embeddings,
                                  env, initial_obs, task_description,
                                  num_autoreg_steps=5, seed=195):
    """Run the model autoregressively — feeding predictions back as input.

    Only the first observation (t=0) comes from the real sim.
    All subsequent observations are the model's own future predictions.

    Args:
        num_autoreg_steps: number of model queries (each produces 16-step chunk)
            Total frames = num_autoreg_steps * 16

    Returns:
        gt_frames: list of real sim frames (from actually running actions in sim)
        predicted_frames: list of model-imagined frames (autoregressive)
        actions_list: all predicted actions
    """
    cfg = FakeConfig()
    obs = extract_obs_from_raw(initial_obs)

    gt_frames = [obs["primary_image"].copy()]
    predicted_frames = [obs["primary_image"].copy()]  # t=0 is real for both
    all_actions = []

    current_obs = obs  # Start with real observation

    for step in range(num_autoreg_steps):
        print(f"[AR] Step {step+1}/{num_autoreg_steps} — "
              f"{'REAL obs' if step == 0 else 'PREDICTED obs (autoregressive)'}")

        # Query model
        action_return = get_action(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            obs=current_obs,
            task_label_or_embedding=task_description,
            seed=seed + step,
            num_denoising_steps_action=5,
            generate_future_state_and_value_in_parallel=False,
        )

        actions = action_return["actions"]  # list of (7,) or (16, 7) ndarray
        if isinstance(actions, list):
            actions = np.array(actions)
        all_actions.append(actions)

        # Get future image predictions from the generated latents
        latent_indices = action_return["latent_indices"]

        future_state = get_future_state_prediction(
            cfg=cfg,
            model=model,
            data_batch=action_return["data_batch"],
            generated_latent_with_action=action_return["generated_latent"],
            orig_clean_latent_frames=action_return["orig_clean_latent_frames"],
            future_proprio_latent_idx=latent_indices["future_proprio_latent_idx"],
            future_wrist_image_latent_idx=latent_indices["future_wrist_image_latent_idx"],
            future_wrist_image2_latent_idx=latent_indices.get("future_wrist_image2_latent_idx"),
            future_image_latent_idx=latent_indices["future_image_latent_idx"],
            future_image2_latent_idx=latent_indices.get("future_image2_latent_idx"),
            seed=seed + step,
            num_denoising_steps_future_state=1,
        )

        future_preds = future_state["future_image_predictions"]

        # Execute actions in REAL sim to get ground truth
        from cosmos_policy.experiments.robot.libero.run_libero_eval import get_libero_dummy_action
        for a_idx in range(len(actions)):
            raw_obs, reward, done, info = env.step(actions[a_idx])

        # Record GT frame (from real sim after executing actions)
        real_obs = extract_obs_from_raw(raw_obs)
        gt_frames.append(real_obs["primary_image"].copy())

        # Record PREDICTED frame (from model's imagination)
        if "future_image" in future_preds and future_preds["future_image"] is not None:
            pred_primary = future_preds["future_image"]
            if pred_primary.ndim == 4:
                pred_primary = pred_primary[0]  # Remove batch dim
            predicted_frames.append(pred_primary.copy())
        else:
            predicted_frames.append(current_obs["primary_image"])

        # KEY AUTOREGRESSIVE STEP: feed predictions back as next input
        # Instead of using real_obs (from sim), use the MODEL'S predictions
        current_obs = {
            "primary_image": future_preds.get("future_image", current_obs["primary_image"]),
            "wrist_image": future_preds.get("future_wrist_image", current_obs["wrist_image"]),
            "proprio": current_obs["proprio"],  # Keep last known proprio
        }
        # Handle batch dimension
        if current_obs["primary_image"].ndim == 4:
            current_obs["primary_image"] = current_obs["primary_image"][0]
        if current_obs["wrist_image"].ndim == 4:
            current_obs["wrist_image"] = current_obs["wrist_image"][0]

        print(f"  Actions shape: {actions.shape}, "
              f"Future image shape: {predicted_frames[-1].shape}")

    return gt_frames, predicted_frames, all_actions


def save_comparison_video(gt_frames, pred_frames, path, fps=2):
    """Side-by-side: GT (from real sim) | Predicted (autoregressive)."""
    n = min(len(gt_frames), len(pred_frames))
    h = max(gt_frames[0].shape[0], pred_frames[0].shape[0])
    w_gt = gt_frames[0].shape[1]
    w_pr = pred_frames[0].shape[1]

    label_h = 30
    gap = 4
    canvas_w = w_gt + gap + w_pr
    canvas_h = h + label_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (canvas_w, canvas_h))

    for i in range(n):
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30
        gt = cv2.resize(gt_frames[i], (w_gt, h)) if gt_frames[i].shape[0] != h else gt_frames[i]
        pr = cv2.resize(pred_frames[i], (w_pr, h)) if pred_frames[i].shape[0] != h else pred_frames[i]

        canvas[label_h:label_h+h, 0:w_gt] = gt
        canvas[label_h:label_h+h, w_gt+gap:w_gt+gap+w_pr] = pr

        cv2.putText(canvas, f"GT (real sim)", (4, label_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"Cosmos AUTOREGRESSIVE", (w_gt+gap+4, label_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(canvas, f"t={i*16}", (canvas_w-60, label_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[AR] Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_idx", type=int, default=0)
    parser.add_argument("--num_autoreg_steps", type=int, default=5,
                        help="Number of autoregressive model queries (5 = 80 sim steps)")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(CONTACT_AWARE_WM, "results/cosmos_policy_ar"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model, config, dataset_stats, t5_embeddings = load_model_and_data()

    # Get initial observation from real LIBERO sim
    env, initial_obs, task_desc, task_name = get_initial_obs_from_libero(
        task_idx=args.task_idx, seed=args.seed)
    print(f"[AR] Task: {task_desc}")
    print(f"[AR] Running {args.num_autoreg_steps} autoregressive steps "
          f"(= {args.num_autoreg_steps * 16} sim steps)")

    # Run autoregressive world model
    gt_frames, pred_frames, all_actions = run_autoregressive_worldmodel(
        model, config, dataset_stats, t5_embeddings,
        env, initial_obs, task_desc,
        num_autoreg_steps=args.num_autoreg_steps,
        seed=args.seed,
    )
    env.close()

    # Compute per-step MSE
    print(f"\n[AR] Results ({len(gt_frames)} GT frames, {len(pred_frames)} predicted):")
    for i in range(min(len(gt_frames), len(pred_frames))):
        gt = gt_frames[i].astype(np.float32) / 255.0
        pr = pred_frames[i].astype(np.float32) / 255.0
        if gt.shape != pr.shape:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0])).astype(np.float32)
            if pr.max() > 1.0:
                pr = pr / 255.0
        mse = np.mean((gt - pr) ** 2)
        print(f"  t={i*16:3d}: MSE={mse:.5f}")

    # Save comparison video
    short_name = task_name[:40].replace(" ", "_")
    vid_path = os.path.join(args.out_dir,
                            f"ar_worldmodel_{short_name}_seed{args.seed}.mp4")
    save_comparison_video(gt_frames, pred_frames, vid_path)

    # Save individual frames
    frame_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for i in range(len(pred_frames)):
        cv2.imwrite(f"{frame_dir}/gt_t{i*16}.png",
                    cv2.cvtColor(gt_frames[i], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{frame_dir}/pred_t{i*16}.png",
                    cv2.cvtColor(pred_frames[i], cv2.COLOR_RGB2BGR))

    print(f"\n[AR] Done! Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
