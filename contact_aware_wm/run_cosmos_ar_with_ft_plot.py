#!/usr/bin/env python3
"""
Run Cosmos-Policy-LIBERO as autoregressive world model AND record F/T data.

Generates per-task:
  - Side-by-side video (GT vs predicted)
  - Combined plot: video frames + F/T signals over time

Usage:
    cd /tmp/cosmos-policy
    python $CONTACT_AWARE_WM/run_cosmos_ar_with_ft_plot.py \
        --task_idx 7 --num_autoreg_steps 15
    # Or all tasks:
    python $CONTACT_AWARE_WM/run_cosmos_ar_with_ft_plot.py \
        --all_tasks --num_autoreg_steps 15
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
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_model, get_action, get_future_state_prediction,
    init_t5_text_embeddings_cache, load_dataset_stats,
)
from huggingface_hub import hf_hub_download


class FakeConfig:
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


def load_model():
    class ModelCfg:
        config = "cosmos_predict2_2b_480p_libero__inference_only"
        ckpt_path = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
        config_file = "cosmos_policy/config/config.py"

    model, cosmos_config = get_model(ModelCfg())
    stats_path = hf_hub_download("nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
                                  "libero_dataset_statistics.json")
    t5_path = hf_hub_download("nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
                               "libero_t5_embeddings.pkl")
    dataset_stats = load_dataset_stats(stats_path)
    init_t5_text_embeddings_cache(t5_path)
    with open(t5_path, "rb") as f:
        t5_embeddings = pickle.load(f)
    return model, dataset_stats, t5_embeddings


def create_env(task_idx, seed=195):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_name = suite.get_task_names()[task_idx]
    init_states = suite.get_task_init_states(task_idx)

    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_idx),
        camera_heights=256, camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"],
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_depths=False,
    )
    np.random.seed(seed)
    env.seed(seed)
    env.reset()
    env.sim.set_state_from_flattened(init_states[seed % len(init_states)])
    env.sim.forward()
    obs, _, _, _ = env.step(np.zeros(7))

    task_desc = task_name.replace("_", " ")
    return env, obs, task_desc, task_name


def extract_obs(raw_obs):
    primary = raw_obs["agentview_image"][::-1, :, :].copy()
    wrist = raw_obs["robot0_eye_in_hand_image"][::-1, :, :].copy()
    proprio = np.concatenate([
        raw_obs["robot0_gripper_qpos"],
        raw_obs["robot0_eef_pos"],
        raw_obs["robot0_eef_quat"],
    ])
    return {"primary_image": primary, "wrist_image": wrist, "proprio": proprio}


@torch.inference_mode()
def run_ar_with_ft(model, dataset_stats, env, initial_obs, task_desc,
                   num_steps=15, seed=195, proprio_mode="initial"):
    """Run autoregressive world model and record F/T from sim."""
    cfg = FakeConfig()
    obs = extract_obs(initial_obs)

    gt_frames = [obs["primary_image"].copy()]
    pred_frames = [obs["primary_image"].copy()]
    all_ft = [env.sim.data.sensordata[:6].copy().astype(np.float32)]
    all_mse = [0.0]
    timesteps = [0]

    current_obs = obs

    for step in range(num_steps):
        torch.cuda.empty_cache()
        action_return = get_action(
            cfg=cfg, model=model, dataset_stats=dataset_stats,
            obs=current_obs, task_label_or_embedding=task_desc,
            seed=seed + step, num_denoising_steps_action=5,
            generate_future_state_and_value_in_parallel=False,
        )

        actions = action_return["actions"]
        if isinstance(actions, list):
            actions = np.array(actions)

        # Get future predictions
        latent_indices = action_return["latent_indices"]
        future_state = get_future_state_prediction(
            cfg=cfg, model=model,
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
        if step == 0:
            fs_keys = list(future_state.keys()) if isinstance(future_state, dict) else []
            fp_keys = list(future_preds.keys()) if isinstance(future_preds, dict) else []
            print(f"  [debug] future_state keys: {fs_keys}")
            print(f"  [debug] future_preds keys: {fp_keys}")

        # Execute actions in real sim, recording F/T at each step
        step_ft = []
        for a in actions:
            raw_obs, _, _, _ = env.step(a)
            ft = env.sim.data.sensordata[:6].copy().astype(np.float32)
            step_ft.append(ft)

        # Record data
        real_obs = extract_obs(raw_obs)
        gt_frames.append(real_obs["primary_image"].copy())

        pred_img = future_preds.get("future_image", current_obs["primary_image"])
        if pred_img.ndim == 4:
            pred_img = pred_img[0]
        pred_frames.append(pred_img.copy())

        # MSE
        gt_f = gt_frames[-1].astype(np.float32) / 255.0
        pr_f = pred_frames[-1].astype(np.float32) / 255.0
        if gt_f.shape != pr_f.shape:
            pr_f = cv2.resize(pr_f, (gt_f.shape[1], gt_f.shape[0]))
            if pr_f.max() > 1.0:
                pr_f /= 255.0
        mse = np.mean((gt_f - pr_f) ** 2)
        all_mse.append(mse)

        # Average F/T over this chunk
        chunk_ft = np.mean(step_ft, axis=0)
        all_ft.append(chunk_ft)
        timesteps.append((step + 1) * 16)

        # Autoregressive: feed predictions back
        if proprio_mode == "predicted":
            # Key may live on future_state or inside future_image_predictions
            pred_proprio = (
                future_preds.get("future_proprio", None)
                if isinstance(future_preds, dict) else None
            )
            if pred_proprio is None:
                pred_proprio = future_state.get("future_proprio", None)
            if pred_proprio is None:
                pred_proprio = future_state.get("future_proprio_prediction", None)
            if pred_proprio is None:
                if step == 0:
                    print("  [warn] 'future_proprio' not in future_preds; "
                          "falling back to initial proprio.")
                next_proprio = current_obs["proprio"]
            else:
                if hasattr(pred_proprio, "detach"):
                    pred_proprio = pred_proprio.detach().cpu().numpy()
                pred_proprio = np.asarray(pred_proprio).squeeze()
                next_proprio = pred_proprio.astype(current_obs["proprio"].dtype)
        elif proprio_mode == "real":
            next_proprio = real_obs["proprio"]
        else:  # "initial"
            next_proprio = current_obs["proprio"]

        current_obs = {
            "primary_image": future_preds.get("future_image", current_obs["primary_image"]),
            "wrist_image": future_preds.get("future_wrist_image", current_obs["wrist_image"]),
            "proprio": next_proprio,
        }
        if current_obs["primary_image"].ndim == 4:
            current_obs["primary_image"] = current_obs["primary_image"][0]
        if current_obs["wrist_image"].ndim == 4:
            current_obs["wrist_image"] = current_obs["wrist_image"][0]

    return {
        "gt_frames": gt_frames,
        "pred_frames": pred_frames,
        "ft": np.array(all_ft),
        "mse": np.array(all_mse),
        "timesteps": np.array(timesteps),
    }


def create_combined_plot(result, task_name, save_path):
    """Create figure: top row = sampled video frames, bottom = F/T + MSE plots."""
    gt = result["gt_frames"]
    pred = result["pred_frames"]
    ft = result["ft"]
    mse = result["mse"]
    ts = result["timesteps"]
    n = len(gt)

    # Select frames to show (up to 6)
    frame_indices = [0]
    step = max(1, (n - 1) // 5)
    for i in range(1, 6):
        idx = min(i * step, n - 1)
        if idx not in frame_indices:
            frame_indices.append(idx)
    if n - 1 not in frame_indices:
        frame_indices.append(n - 1)
    n_frames = len(frame_indices)

    fig = plt.figure(figsize=(3.5 * n_frames, 14))
    gs = GridSpec(4, n_frames, figure=fig, height_ratios=[1, 1, 0.8, 0.8],
                  hspace=0.35, wspace=0.15)

    short_name = task_name.replace("_", " ")
    if len(short_name) > 60:
        short_name = short_name[:57] + "..."
    fig.suptitle(f"Autoregressive World Model: {short_name}", fontsize=14, fontweight="bold")

    # Row 1: GT frames
    for col, idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(gt[idx])
        ax.set_title(f"t={ts[idx]}", fontsize=10)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Ground Truth", fontsize=11, fontweight="bold")

    # Row 2: Predicted frames
    for col, idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs[1, col])
        frame = pred[idx]
        if frame.shape[:2] != gt[0].shape[:2]:
            frame = cv2.resize(frame, (gt[0].shape[1], gt[0].shape[0]))
        ax.imshow(frame)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Predicted (AR)", fontsize=11, fontweight="bold")

    # Row 3: Force/Torque plot
    ax_ft = fig.add_subplot(gs[2, :])
    colors_f = ["#e74c3c", "#2ecc71", "#3498db"]
    colors_t = ["#e67e22", "#9b59b6", "#1abc9c"]
    labels_f = ["Fx", "Fy", "Fz"]
    labels_t = ["Tx", "Ty", "Tz"]

    for i in range(3):
        ax_ft.plot(ts, ft[:, i], color=colors_f[i], label=labels_f[i], linewidth=1.5)
    for i in range(3):
        ax_ft.plot(ts, ft[:, 3 + i], color=colors_t[i], label=labels_t[i],
                   linewidth=1.5, linestyle="--")

    ax_ft.set_ylabel("Force (N) / Torque (Nm)", fontsize=11)
    ax_ft.set_xlabel("")
    ax_ft.legend(loc="upper right", ncol=6, fontsize=8)
    ax_ft.grid(True, alpha=0.3)
    ax_ft.set_title("End-Effector Force/Torque (from MuJoCo sim)", fontsize=11)

    # Shade contact regions (where |F| > 5N)
    force_mag = np.sqrt(ft[:, 0]**2 + ft[:, 1]**2 + ft[:, 2]**2)
    for i in range(len(force_mag)):
        if force_mag[i] > 5:
            ax_ft.axvspan(ts[max(0, i-1)], ts[min(len(ts)-1, i)],
                         alpha=0.15, color="red")

    # Row 4: MSE drift plot
    ax_mse = fig.add_subplot(gs[3, :])
    ax_mse.plot(ts, mse, color="#2c3e50", linewidth=2, marker="o", markersize=4)
    ax_mse.fill_between(ts, 0, mse, alpha=0.15, color="#3498db")
    ax_mse.set_xlabel("Simulation Timestep", fontsize=11)
    ax_mse.set_ylabel("MSE vs Ground Truth", fontsize=11)
    ax_mse.set_title("Autoregressive Prediction Drift", fontsize=11)
    ax_mse.grid(True, alpha=0.3)
    ax_mse.set_xlim(ts[0], ts[-1])

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def save_video(gt_frames, pred_frames, path, fps=2):
    n = min(len(gt_frames), len(pred_frames))
    h, w = gt_frames[0].shape[:2]
    pw = pred_frames[0].shape[1]
    label_h = 30
    gap = 4
    canvas_w = w + gap + pw
    canvas_h = h + label_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (canvas_w, canvas_h))
    for i in range(n):
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30
        gt = cv2.resize(gt_frames[i], (w, h)) if gt_frames[i].shape[:2] != (h, w) else gt_frames[i]
        pr = pred_frames[i]
        if pr.shape[:2] != (h, pw):
            pr = cv2.resize(pr, (pw, h))
        canvas[label_h:label_h+h, 0:w] = gt
        canvas[label_h:label_h+h, w+gap:w+gap+pw] = pr
        cv2.putText(canvas, "GT (sim)", (4, label_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "Cosmos AR", (w+gap+4, label_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(canvas, f"t={i*16}", (canvas_w-60, label_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def save_video_multi(gt_frames, mode_results, path, fps=2):
    """4-panel video: GT | AR(initial) | AR(predicted) | AR(real).

    mode_results: dict {mode_name: list_of_frames}, each same length as gt_frames.
    """
    modes = list(mode_results.keys())
    n = min(len(gt_frames), *[len(mode_results[m]) for m in modes])
    h, w = gt_frames[0].shape[:2]
    # use prediction frame width from first mode (usually matches GT after resize)
    pw = mode_results[modes[0]][0].shape[1]

    label_h = 30
    gap = 4
    n_panels = 1 + len(modes)
    canvas_w = w + (len(modes)) * (gap + pw)
    canvas_h = h + label_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (canvas_w, canvas_h))
    colors = {"initial": (100, 200, 255), "predicted": (100, 255, 150),
              "real": (255, 150, 100)}
    for i in range(n):
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30
        gt = gt_frames[i]
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h))
        canvas[label_h:label_h + h, 0:w] = gt
        cv2.putText(canvas, "GT (sim)", (4, label_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        x = w
        for m in modes:
            x += gap
            pr = mode_results[m][i]
            if pr.shape[:2] != (h, pw):
                pr = cv2.resize(pr, (pw, h))
            canvas[label_h:label_h + h, x:x + pw] = pr
            col = colors.get(m, (200, 200, 200))
            cv2.putText(canvas, f"AR [{m}]", (x + 4, label_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            x += pw

        cv2.putText(canvas, f"t={i * 16}", (canvas_w - 60, label_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_idx", type=int, default=7)
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--num_autoreg_steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(CONTACT_AWARE_WM, "results/cosmos_ar_ft"))
    parser.add_argument("--proprio_mode", type=str, default="initial",
                        choices=["initial", "predicted", "real"],
                        help="Proprio fed to policy each AR step: "
                             "'initial' = freeze at t=0 (original behavior), "
                             "'predicted' = use model's predicted future_proprio, "
                             "'real' = use sim proprio (oracle, for debugging).")
    parser.add_argument("--compare_modes", action="store_true",
                        help="Run all three proprio modes per task and save a "
                             "4-panel comparison video (GT + initial + predicted + real).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[AR+FT] Loading model...")
    model, dataset_stats, t5_embeddings = load_model()

    task_indices = list(range(10)) if args.all_tasks else [args.task_idx]
    # 'predicted' is intentionally omitted: upstream Cosmos does not expose a
    # decoded future_proprio, so 'predicted' silently falls back to 'initial'.
    modes_to_run = ["initial", "real"] if args.compare_modes else [args.proprio_mode]

    for task_idx in task_indices:
        print(f"\n{'='*60}")
        print(f"  Task {task_idx}")
        print(f"{'='*60}")

        mode_results = {}
        task_name = None
        for mode in modes_to_run:
            print(f"  --- proprio_mode={mode} ---")
            env, obs, task_desc, task_name = create_env(task_idx, args.seed)
            if mode == modes_to_run[0]:
                print(f"  {task_desc}")
            result = run_ar_with_ft(
                model, dataset_stats, env, obs, task_desc,
                num_steps=args.num_autoreg_steps, seed=args.seed,
                proprio_mode=mode,
            )
            env.close()
            mode_results[mode] = result
            print(f"  [{mode}] Final MSE: {result['mse'][-1]:.5f}")

        short = task_name[:50]
        if args.compare_modes:
            vid_path = os.path.join(args.out_dir,
                                    f"ar_compare_{short}_seed{args.seed}.mp4")
            frames_by_mode = {m: mode_results[m]["pred_frames"] for m in modes_to_run}
            # GT is identical across modes (same seed, same env); pick first
            save_video_multi(mode_results[modes_to_run[0]]["gt_frames"],
                             frames_by_mode, vid_path)
            print(f"  Saved 4-panel video: {vid_path}")
            # Save per-mode plot for the last-run mode only (keeps output manageable)
            for m in modes_to_run:
                plot_path = os.path.join(
                    args.out_dir,
                    f"ar_ft_plot_{m}_{short}_seed{args.seed}.png")
                create_combined_plot(mode_results[m], f"{task_name} [{m}]",
                                     plot_path)
        else:
            result = mode_results[modes_to_run[0]]
            vid_path = os.path.join(args.out_dir, f"ar_{short}_seed{args.seed}.mp4")
            save_video(result["gt_frames"], result["pred_frames"], vid_path)
            print(f"  Saved video: {vid_path}")
            plot_path = os.path.join(args.out_dir,
                                     f"ar_ft_plot_{short}_seed{args.seed}.png")
            create_combined_plot(result, task_name, plot_path)
            force_mag = np.sqrt(np.sum(result['ft'][:, :3] ** 2, axis=1))
            print(f"  Max force: {force_mag.max():.1f} N")

    print(f"\n[AR+FT] All done! Results: {args.out_dir}")


if __name__ == "__main__":
    main()
