#!/usr/bin/env python3
"""
Extract simulated F/T readings from LIBERO-90 by replaying demonstrations.

LIBERO's MuJoCo XML defines sensors:
  - gripper0_force_ee:  (3,) — end-effector force [Fx, Fy, Fz]
  - gripper0_torque_ee: (3,) — end-effector torque [Tx, Ty, Tz]

These are read from env.sim.data.sensordata (shape (6,)) at each timestep.

For each demo HDF5, replays the recorded states (not actions — faster and exact)
and extracts F/T at each timestep. Also extracts joint torques as an alternative.

Output format:
  libero_90_with_ft/
    {task_name}/
      demo_{i}.npz — {images, actions, ft_wrist, joint_torques, ee_pos, ee_ori}

Usage:
    # LIBERO-Spatial only (10 tasks × 50 demos)
    python extract_ft_libero.py --suite spatial

    # All 90 tasks
    python extract_ft_libero.py --suite all

    # Single task for testing
    python extract_ft_libero.py --task_idx 0 --max_demos 2
"""

from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm


# LIBERO suite → task index ranges (approximate; we detect by name prefix)
SUITE_PREFIXES = {
    "spatial": ["LIVING_ROOM_SCENE1", "LIVING_ROOM_SCENE2"],
    "object": ["LIVING_ROOM_SCENE3", "LIVING_ROOM_SCENE4"],
    "goal": ["LIVING_ROOM_SCENE5", "LIVING_ROOM_SCENE6"],
}


def get_libero_env_and_suite():
    """Import LIBERO and create benchmark suite."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()
    suite = bench["libero_90"]()
    return suite, OffScreenRenderEnv


def get_task_indices_for_suite(suite, suite_name: str) -> List[int]:
    """Get task indices belonging to a specific LIBERO suite."""
    if suite_name == "all":
        return list(range(suite.get_num_tasks()))

    prefixes = SUITE_PREFIXES.get(suite_name)
    if prefixes is None:
        raise ValueError(f"Unknown suite: {suite_name}. Choose from: {list(SUITE_PREFIXES.keys()) + ['all']}")

    task_names = suite.get_task_names()
    indices = []
    for i, name in enumerate(task_names):
        for prefix in prefixes:
            if name.startswith(prefix):
                indices.append(i)
                break
    return indices


def extract_ft_from_demo(
    env,
    hdf5_path: str,
    demo_key: str,
    img_size: int = 96,
) -> Optional[Dict[str, np.ndarray]]:
    """Replay a single demo through the env and extract F/T at each timestep.

    Instead of replaying actions (which may drift), we set the MuJoCo state
    directly from the saved `states` array, then read the sensor data.

    Args:
        env: LIBERO OffScreenRenderEnv
        hdf5_path: path to the HDF5 demo file
        demo_key: e.g. 'demo_0'
        img_size: resize images to this resolution

    Returns:
        dict with arrays: images, actions, ft_wrist, joint_torques, ee_pos, ee_ori
        or None if extraction fails
    """
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        states = demo["states"][:]           # (T, state_dim)
        actions = demo["actions"][:].astype(np.float32)  # (T, 7)
        images = demo["obs/agentview_rgb"][:]  # (T, 128, 128, 3)
        ee_pos = demo["obs/ee_pos"][:].astype(np.float32)  # (T, 3)
        ee_ori = demo["obs/ee_ori"][:].astype(np.float32)  # (T, 3)

    T = len(states)
    ft_wrist = np.zeros((T, 6), dtype=np.float32)
    joint_torques = np.zeros((T, 7), dtype=np.float32)

    # Resize images
    if img_size != 128:
        resized = np.zeros((T, img_size, img_size, 3), dtype=np.uint8)
        for t in range(T):
            resized[t] = cv2.resize(images[t], (img_size, img_size),
                                    interpolation=cv2.INTER_AREA)
        images = resized

    # Replay states through MuJoCo to extract F/T
    try:
        env.reset()
        for t in range(T):
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()

            # F/T from built-in sensors: gripper0_force_ee + gripper0_torque_ee
            sensordata = env.sim.data.sensordata.copy()
            if len(sensordata) >= 6:
                ft_wrist[t] = sensordata[:6].astype(np.float32)

            # Joint torques (first 7 = arm joints)
            qfrc = env.sim.data.qfrc_actuator.copy()
            joint_torques[t] = qfrc[:7].astype(np.float32)

    except Exception as e:
        print(f"  Warning: F/T extraction failed: {e}")
        return None

    return {
        "images": images,
        "actions": actions,
        "ft_wrist": ft_wrist,
        "joint_torques": joint_torques,
        "ee_pos": ee_pos,
        "ee_ori": ee_ori,
    }


def process_task(
    suite,
    OffScreenRenderEnv,
    task_idx: int,
    hdf5_dir: str,
    output_dir: str,
    img_size: int = 96,
    max_demos: int = 0,
) -> Dict:
    """Process all demos for a single task.

    Returns:
        stats dict with ft_mean, ft_std, n_demos, n_transitions
    """
    task_name = suite.get_task_names()[task_idx]
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    # Find the HDF5 file
    hdf5_path = os.path.join(hdf5_dir, f"{task_name}_demo.hdf5")
    if not os.path.exists(hdf5_path):
        # Try without _demo suffix
        hdf5_path = os.path.join(hdf5_dir, f"{task_name}.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"  Skipping {task_name}: HDF5 not found")
        return {}

    # Count demos
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])

    if max_demos > 0:
        demo_keys = demo_keys[:max_demos]

    # Create environment for this task
    env_args = {
        "bddl_file_name": suite.get_task_bddl_file_path(task_idx),
        "camera_heights": 128,
        "camera_widths": 128,
        "camera_names": ["agentview"],
        "has_renderer": False,
        "has_offscreen_renderer": False,  # We don't need rendering — just physics
        "use_camera_obs": False,
    }

    try:
        env = OffScreenRenderEnv(**env_args)
    except Exception as e:
        print(f"  Skipping {task_name}: env creation failed: {e}")
        return {}

    all_ft = []
    n_transitions = 0
    n_saved = 0

    for demo_key in tqdm(demo_keys, desc=f"  {task_name[:40]}", leave=False):
        demo_idx = int(demo_key.replace("demo_", ""))
        out_path = os.path.join(task_output_dir, f"demo_{demo_idx}.npz")

        # Skip if already processed
        if os.path.exists(out_path):
            data = np.load(out_path)
            all_ft.append(data["ft_wrist"])
            n_transitions += len(data["ft_wrist"])
            n_saved += 1
            continue

        result = extract_ft_from_demo(env, hdf5_path, demo_key, img_size)
        if result is None:
            continue

        np.savez_compressed(out_path, **result)
        all_ft.append(result["ft_wrist"])
        n_transitions += len(result["ft_wrist"])
        n_saved += 1

    env.close()

    stats = {"task_name": task_name, "n_demos": n_saved, "n_transitions": n_transitions}

    if all_ft:
        ft_all = np.concatenate(all_ft, axis=0)
        stats["ft_mean"] = ft_all.mean(axis=0).tolist()
        stats["ft_std"] = ft_all.std(axis=0).tolist()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract F/T from LIBERO demos")
    parser.add_argument("--hdf5_dir", type=str,
                        default=LIBERO_DATA_ROOT)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(CONTACT_AWARE_WM, "data/libero_90_with_ft"))
    parser.add_argument("--suite", type=str, default="spatial",
                        choices=["spatial", "object", "goal", "all"],
                        help="Which LIBERO suite to process")
    parser.add_argument("--task_idx", type=int, default=-1,
                        help="Process a single task by index (-1 = use --suite)")
    parser.add_argument("--max_demos", type=int, default=0,
                        help="Max demos per task (0 = all)")
    parser.add_argument("--img_size", type=int, default=96)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    suite, OffScreenRenderEnv = get_libero_env_and_suite()
    total_tasks = suite.get_num_tasks()
    print(f"[extract_ft] LIBERO-90: {total_tasks} tasks")

    # Determine which tasks to process
    if args.task_idx >= 0:
        task_indices = [args.task_idx]
    else:
        task_indices = get_task_indices_for_suite(suite, args.suite)

    task_names = suite.get_task_names()
    print(f"[extract_ft] Processing {len(task_indices)} tasks (suite={args.suite})")
    for i in task_indices:
        print(f"  [{i}] {task_names[i]}")

    # Process tasks
    all_stats = []
    for task_idx in task_indices:
        print(f"\n[extract_ft] Task {task_idx}: {task_names[task_idx]}")
        stats = process_task(
            suite, OffScreenRenderEnv, task_idx,
            args.hdf5_dir, args.output_dir,
            img_size=args.img_size,
            max_demos=args.max_demos,
        )
        if stats:
            all_stats.append(stats)

    # Save summary statistics
    stats_path = os.path.join(args.output_dir, "extraction_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"F/T Extraction Summary")
    print(f"{'=' * 60}")
    total_demos = sum(s.get("n_demos", 0) for s in all_stats)
    total_trans = sum(s.get("n_transitions", 0) for s in all_stats)
    print(f"Tasks processed: {len(all_stats)}")
    print(f"Total demos:     {total_demos}")
    print(f"Total transitions: {total_trans:,}")

    # F/T statistics from first 3 tasks
    print(f"\nF/T Statistics (per task):")
    dim_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    for s in all_stats[:3]:
        if "ft_mean" in s:
            print(f"  {s['task_name'][:50]}:")
            for i, name in enumerate(dim_names):
                print(f"    {name}: mean={s['ft_mean'][i]:>8.4f}, std={s['ft_std'][i]:>8.4f}")

    print(f"\n[extract_ft] Stats saved to: {stats_path}")
    print(f"[extract_ft] Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
