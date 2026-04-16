#!/usr/bin/env python3
"""
Collect random and perturbed-expert rollouts for LIBERO tasks.

For each task:
  - 30 episodes of uniform random actions (clipped to valid range)
  - 20 episodes of perturbed expert actions (GT + N(0, 0.1))

Extracts F/T readings during collection (from MuJoCo sensordata).
Saves in same format as extract_ft_libero.py output.

Usage:
    python collect_random_rollouts_libero.py --suite spatial [--tasks_per_run 3]
"""

from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def get_libero_env_and_suite():
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    return suite, OffScreenRenderEnv


SUITE_PREFIXES = {
    "spatial": ["LIVING_ROOM_SCENE1", "LIVING_ROOM_SCENE2"],
    "object": ["LIVING_ROOM_SCENE3", "LIVING_ROOM_SCENE4"],
    "goal": ["LIVING_ROOM_SCENE5", "LIVING_ROOM_SCENE6"],
}


def get_task_indices(suite, suite_name: str):
    if suite_name == "all":
        return list(range(suite.get_num_tasks()))
    prefixes = SUITE_PREFIXES.get(suite_name, [])
    names = suite.get_task_names()
    return [i for i, n in enumerate(names) if any(n.startswith(p) for p in prefixes)]


def collect_episodes(env, init_states, n_episodes: int, max_steps: int,
                     action_fn, img_size: int = 96):
    """Collect episodes using a given action function.

    Args:
        env: LIBERO environment
        init_states: array of initial MuJoCo states
        n_episodes: how many to collect
        max_steps: max steps per episode
        action_fn: callable(obs, t) -> action (7D)
        img_size: resize images

    Returns:
        list of dicts {images, actions, ft_wrist, joint_torques, ee_pos, ee_ori}
    """
    episodes = []
    for ep in range(n_episodes):
        obs = env.reset()
        state_idx = ep % len(init_states)
        env.sim.set_state_from_flattened(init_states[state_idx])
        env.sim.forward()

        images, actions_list, ft_list, jt_list, eepos_list, eeori_list = [], [], [], [], [], []

        for t in range(max_steps):
            # Render
            img = env.sim.render(camera_name="agentview", height=128, width=128)[::-1]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            images.append(img)

            # Get proprioception
            ee_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
            ee_mat = env.sim.data.get_body_xmat("gripper0_eef").copy()
            ee_ori = np.zeros(3)  # simplified euler

            # F/T from sensors
            ft = env.sim.data.sensordata[:6].copy().astype(np.float32)
            jt = env.sim.data.qfrc_actuator[:7].copy().astype(np.float32)

            # Action
            action = action_fn(obs, t)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            obs, reward, done, info = env.step(action)

            actions_list.append(action)
            ft_list.append(ft)
            jt_list.append(jt)
            eepos_list.append(ee_pos.astype(np.float32))
            eeori_list.append(ee_ori.astype(np.float32))

            if done:
                break

        episodes.append({
            "images": np.array(images, dtype=np.uint8),
            "actions": np.array(actions_list, dtype=np.float32),
            "ft_wrist": np.array(ft_list, dtype=np.float32),
            "joint_torques": np.array(jt_list, dtype=np.float32),
            "ee_pos": np.array(eepos_list, dtype=np.float32),
            "ee_ori": np.array(eeori_list, dtype=np.float32),
        })

    return episodes


def main():
    parser = argparse.ArgumentParser(description="Collect random rollouts for LIBERO")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(CONTACT_AWARE_WM, "data/libero_90_with_ft"))
    parser.add_argument("--suite", type=str, default="spatial")
    parser.add_argument("--n_random", type=int, default=30)
    parser.add_argument("--n_perturbed", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--perturb_std", type=float, default=0.1)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    suite, OffScreenRenderEnv = get_libero_env_and_suite()
    task_indices = get_task_indices(suite, args.suite)
    task_names = suite.get_task_names()

    print(f"[collect] Suite: {args.suite}, tasks: {len(task_indices)}")
    print(f"[collect] {args.n_random} random + {args.n_perturbed} perturbed per task")

    for task_idx in task_indices:
        task_name = task_names[task_idx]
        task_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        print(f"\n[collect] Task {task_idx}: {task_name}")

        init_states = suite.get_task_init_states(task_idx)
        env_args = {
            "bddl_file_name": suite.get_task_bddl_file_path(task_idx),
            "camera_heights": 128,
            "camera_widths": 128,
            "camera_names": ["agentview"],
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": False,
        }

        try:
            env = OffScreenRenderEnv(**env_args)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Random rollouts
        print(f"  Collecting {args.n_random} random episodes...")
        random_fn = lambda obs, t: np.random.uniform(-1, 1, size=7).astype(np.float32)
        random_eps = collect_episodes(
            env, init_states, args.n_random, args.max_steps, random_fn, args.img_size)

        for i, ep in enumerate(random_eps):
            path = os.path.join(task_dir, f"random_{i}.npz")
            np.savez_compressed(path, **ep)

        # Perturbed expert rollouts (using saved expert states)
        # Load first expert demo's actions as reference
        import h5py
        hdf5_dir = LIBERO_DATA_ROOT
        hdf5_path = os.path.join(hdf5_dir, f"{task_name}_demo.hdf5")

        if os.path.exists(hdf5_path):
            print(f"  Collecting {args.n_perturbed} perturbed-expert episodes...")
            with h5py.File(hdf5_path, "r") as f:
                demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])
                expert_actions_list = []
                for dk in demo_keys[:args.n_perturbed]:
                    expert_actions_list.append(f["data"][dk]["actions"][:].astype(np.float32))

            for i in range(min(args.n_perturbed, len(expert_actions_list))):
                expert_actions = expert_actions_list[i]

                def perturbed_fn(obs, t, ea=expert_actions, std=args.perturb_std):
                    if t < len(ea):
                        return ea[t] + np.random.normal(0, std, size=7).astype(np.float32)
                    return np.random.uniform(-1, 1, size=7).astype(np.float32)

                eps = collect_episodes(env, init_states, 1, args.max_steps,
                                       perturbed_fn, args.img_size)
                if eps:
                    path = os.path.join(task_dir, f"perturbed_{i}.npz")
                    np.savez_compressed(path, **eps[0])

        env.close()
        n_total = len([f for f in os.listdir(task_dir) if f.endswith(".npz")])
        print(f"  Total episodes for {task_name}: {n_total}")

    print(f"\n[collect] Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
