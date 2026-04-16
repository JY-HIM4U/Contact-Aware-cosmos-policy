#!/usr/bin/env python3
"""Run autoregressive world model on contact-heavy LIBERO tasks."""
from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.insert(0, CONTACT_AWARE_WM)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch

from run_cosmos_ar_with_ft_plot import (
    load_model, run_ar_with_ft,
    save_video, extract_obs, FakeConfig,
)
from regenerate_ft_plots import create_combined_plot_with_joints


def run_task(model, dataset_stats, suite_name, task_idx, short_name,
             out_dir, seed=195, n_ar_steps=15):
    """Run one contact task end-to-end."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task_name = suite.get_task_names()[task_idx]
    init_states = suite.get_task_init_states(task_idx)
    task_desc = task_name.replace("_", " ")
    # Strip scene prefix (e.g. "KITCHEN SCENE3 ") so it matches T5 cache keys
    import re
    task_desc = re.sub(r'^KITCHEN SCENE\d+\s+', '', task_desc)
    print(f"  Task: {task_desc}")

    # Create env for AR world model run (needs camera images)
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

    # Run autoregressive world model
    result = run_ar_with_ft(
        model, dataset_stats, env, obs, task_desc,
        num_steps=n_ar_steps, seed=seed,
    )
    env.close()

    # Collect detailed F/T with the SAME actions the model predicted
    # (re-use the F/T from the AR run which was collected during env.step)
    ft_data = {
        "gripper_ft": result["ft"],
        "timesteps": result["timesteps"],
    }

    # Also need joint torques — re-run sim with scripted actions
    env2 = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_idx),
        camera_heights=128, camera_widths=128,
        camera_names=["agentview"],
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    np.random.seed(seed)
    env2.seed(seed)
    env2.reset()
    env2.sim.set_state_from_flattened(init_states[seed % len(init_states)])
    env2.sim.forward()

    # Collect joint torques
    all_jt = []
    all_gf = []
    all_gft = []
    for t in range(240):
        action = np.random.uniform(-0.3, 0.3, size=7).astype(np.float32)
        if t < 60:
            action[2] = -0.3; action[6] = -1.0
        elif t < 100:
            action[0] = 0.2; action[2] = -0.2; action[6] = -1.0
        elif t < 140:
            action[6] = 1.0
        else:
            action[2] = 0.3; action[6] = 1.0
        env2.step(action)
        all_jt.append(env2.sim.data.actuator_force[:7].copy().astype(np.float32))
        all_gf.append(env2.sim.data.actuator_force[7:9].copy().astype(np.float32))
        all_gft.append(env2.sim.data.sensordata[:6].copy().astype(np.float32))
    env2.close()

    ft_data_full = {
        "gripper_ft": np.array(all_gft),
        "joint_torques": np.array(all_jt),
        "gripper_fingers": np.array(all_gf),
        "timesteps": np.arange(240),
    }

    # Save video
    vid_path = os.path.join(out_dir, f"contact_{short_name}_seed{seed}.mp4")
    save_video(result["gt_frames"], result["pred_frames"], vid_path)

    # Save combined plot
    plot_path = os.path.join(out_dir, f"contact_{short_name}_seed{seed}.png")
    create_combined_plot_with_joints(
        ft_data_full, result["gt_frames"], result["pred_frames"],
        task_name, plot_path)

    # Summary
    force_mag = np.sqrt(np.sum(ft_data_full["gripper_ft"][:, :3]**2, axis=1))
    print(f"  Final MSE: {result['mse'][-1]:.5f}")
    print(f"  Max gripper force: {force_mag.max():.1f} N")
    print(f"  Max joint torques: {np.abs(ft_data_full['joint_torques']).max(axis=0)}")


def main():
    out_dir = os.path.join(CONTACT_AWARE_WM, "results/cosmos_ar_contact")
    os.makedirs(out_dir, exist_ok=True)

    print("[contact] Loading model...")
    model, dataset_stats, t5_embeddings = load_model()

    tasks = [
        ("libero_goal", 0, "open_middle_drawer"),
        ("libero_goal", 3, "open_drawer_put_bowl"),
        ("libero_goal", 5, "push_plate_to_stove"),
        ("libero_goal", 7, "turn_on_stove"),
        ("libero_10", 2, "turn_stove_put_moka"),
        ("libero_10", 3, "bowl_in_drawer_close"),
        ("libero_10", 9, "mug_in_microwave_close"),
    ]

    for suite_name, task_idx, short_name in tasks:
        # Skip tasks that already have results
        mp4_path = os.path.join(out_dir, f"contact_{short_name}_seed195.mp4")
        if os.path.exists(mp4_path):
            print(f"\n[skip] {short_name} already done: {mp4_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  {suite_name} [{task_idx}]: {short_name}")
        print(f"{'='*60}")
        try:
            torch.cuda.empty_cache()
            run_task(model, dataset_stats, suite_name, task_idx, short_name,
                     out_dir, seed=195, n_ar_steps=15)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    print(f"\n[contact] All done! Results: {out_dir}")


if __name__ == "__main__":
    main()
