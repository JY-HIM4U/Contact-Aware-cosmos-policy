#!/bin/bash
# Run autoregressive world model + F/T plots on contact-heavy LIBERO tasks
set -e
cd /tmp/cosmos-policy
PYTHON=~/.conda/envs/cosmos/bin/python
SCRIPT="${CONTACT_AWARE_WM:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/run_cosmos_ar_with_ft_plot.py
OUT="${RESULTS_ROOT:-${CONTACT_AWARE_WM:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/results}/results/cosmos_ar_contact

# We need to modify the script to accept --suite argument
# Instead, let's run a Python one-liner for each task

$PYTHON -c "
import os, sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, os.environ.get("CONTACT_AWARE_WM", os.path.dirname(os.path.abspath(__file__))))
from run_cosmos_ar_with_ft_plot import (
    load_model, run_ar_with_ft, create_combined_plot_with_joints,
    save_video, collect_ft_and_torques, FakeConfig
)
from regenerate_ft_plots import create_env as create_env_spatial

import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

out_dir = '$OUT'
os.makedirs(out_dir, exist_ok=True)

print('[contact] Loading model...')
model, dataset_stats, t5_embeddings = load_model()

# Define contact tasks to run
tasks = [
    ('libero_goal', 0, 'open_the_middle_drawer_of_the_cabinet'),
    ('libero_goal', 3, 'open_the_top_drawer_and_put_the_bowl_inside'),
    ('libero_goal', 5, 'push_the_plate_to_the_front_of_the_stove'),
    ('libero_goal', 7, 'turn_on_the_stove'),
    ('libero_10', 2, 'turn_on_stove_and_put_moka_pot'),
    ('libero_10', 3, 'put_bowl_in_drawer_and_close'),
    ('libero_10', 9, 'put_mug_in_microwave_and_close'),
]

seed = 195
n_ar_steps = 15

for suite_name, task_idx, short_name in tasks:
    print(f'\n{\"=\"*60}')
    print(f'  {suite_name} task {task_idx}: {short_name}')
    print(f'{\"=\"*60}')

    # Create environment
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task_name = suite.get_task_names()[task_idx]
    init_states = suite.get_task_init_states(task_idx)

    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_idx),
        camera_heights=256, camera_widths=256,
        camera_names=['agentview', 'robot0_eye_in_hand'],
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_depths=False,
    )
    np.random.seed(seed)
    env.seed(seed)
    env.reset()
    env.sim.set_state_from_flattened(init_states[seed % len(init_states)])
    env.sim.forward()
    obs, _, _, _ = env.step(np.zeros(7))

    task_desc = task_name.replace('_', ' ')
    print(f'  Task description: {task_desc}')

    # Run autoregressive world model with F/T recording
    result = run_ar_with_ft(
        model, dataset_stats, env, obs, task_desc,
        num_steps=n_ar_steps, seed=seed,
    )
    env.close()

    # Also collect detailed F/T with scripted motions for the plot
    env2 = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_idx),
        camera_heights=128, camera_widths=128,
        camera_names=['agentview'],
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    np.random.seed(seed)
    env2.seed(seed)
    env2.reset()
    env2.sim.set_state_from_flattened(init_states[seed % len(init_states)])
    env2.sim.forward()

    from regenerate_ft_plots import collect_ft_and_torques
    ft_data = collect_ft_and_torques(env2, 240, seed)
    env2.close()

    # Create combined plot
    from regenerate_ft_plots import create_combined_plot_with_joints
    plot_path = os.path.join(out_dir, f'contact_{short_name}_seed{seed}.png')
    create_combined_plot_with_joints(
        ft_data, result['gt_frames'], result['pred_frames'],
        task_name, plot_path)

    # Save video
    vid_path = os.path.join(out_dir, f'contact_{short_name}_seed{seed}.mp4')
    save_video(result['gt_frames'], result['pred_frames'], vid_path)

    # Summary
    force_mag = np.sqrt(np.sum(ft_data['gripper_ft'][:, :3]**2, axis=1))
    jt_max = np.abs(ft_data['joint_torques']).max(axis=0)
    print(f'  Final MSE: {result[\"mse\"][-1]:.5f}')
    print(f'  Max gripper force: {force_mag.max():.1f} N')
    print(f'  Max joint torques: {[\"%.1f\"%x for x in jt_max]}')

print(f'\n[contact] All done! Results: {out_dir}')
" 2>&1
