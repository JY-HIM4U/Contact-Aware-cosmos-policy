#!/usr/bin/env python3
"""
Command the robot in the Cosmos world model with custom action sequences.

Instead of replaying GT actions from a LIBERO demo, this script lets you
specify ANY action sequence you want — random, scripted, or from a policy —
and asks Cosmos "what would the robot look like executing these commands?"

This is the world-model-as-simulator use case: you can test action sequences
without actually running them on the real robot.

Available pre-defined action patterns:
  - "forward": Move +X for N steps
  - "down": Move -Z for N steps (toward table)
  - "grasp": Lower gripper, close, lift
  - "circle": Move in a horizontal circle
  - "random": Uniform random actions in [-0.5, 0.5]
  - "zigzag": Alternating +X / -X movements
  - "pickup_then_lift": Lower → close gripper → lift up
  - "stop": All zeros (robot doesn't move)

Usage:
    python cosmos_command_robot.py --pattern grasp
    python cosmos_command_robot.py --pattern circle --num_steps 36 --autoregressive
    python cosmos_command_robot.py --pattern random --seed 42
"""

from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

# Reuse functions from run_cosmos_libero.py
import sys
sys.path.insert(0, CONTACT_AWARE_WM)
from run_cosmos_libero import (
    setup_cosmos_pipeline,
    prepare_first_frame,
    run_inference,
    video_tensor_to_uint8,
    write_video,
)


def make_action_sequence(pattern: str, num_steps: int = 12,
                         seed: int = 0) -> np.ndarray:
    """Create a custom action sequence for the robot.

    LIBERO actions are 7D: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    Each component is in [-1, 1]. Gripper: -1 = open, +1 = close.

    Returns:
        actions: (num_steps, 7) float32
    """
    rng = np.random.RandomState(seed)
    actions = np.zeros((num_steps, 7), dtype=np.float32)

    if pattern == "stop":
        # All zeros — robot stays still
        actions[:, 6] = -1.0  # gripper open

    elif pattern == "forward":
        actions[:, 0] = 0.5   # +X
        actions[:, 6] = -1.0

    elif pattern == "down":
        actions[:, 2] = -0.5  # -Z
        actions[:, 6] = -1.0

    elif pattern == "grasp":
        # Phase 1: lower (first third)
        n1 = num_steps // 3
        actions[:n1, 2] = -0.6  # move down
        actions[:n1, 6] = -1.0  # gripper open
        # Phase 2: close gripper (middle third)
        n2 = num_steps // 3
        actions[n1:n1+n2, 6] = 1.0  # gripper close
        # Phase 3: lift (last third)
        actions[n1+n2:, 2] = 0.6   # move up
        actions[n1+n2:, 6] = 1.0   # gripper stays closed

    elif pattern == "pickup_then_lift":
        # 4 phases: down, close, up, hold
        q = num_steps // 4
        actions[0*q:1*q, 2] = -0.7      # down
        actions[0*q:1*q, 6] = -1.0      # gripper open
        actions[1*q:2*q, 6] = 1.0       # close gripper
        actions[2*q:3*q, 2] = 0.7       # up
        actions[2*q:3*q, 6] = 1.0
        actions[3*q:, 6] = 1.0          # hold

    elif pattern == "circle":
        # Horizontal circle in XY plane
        for i in range(num_steps):
            theta = 2 * np.pi * i / num_steps
            actions[i, 0] = 0.4 * np.cos(theta)  # dx
            actions[i, 1] = 0.4 * np.sin(theta)  # dy
            actions[i, 6] = -1.0

    elif pattern == "random":
        actions = rng.uniform(-0.5, 0.5, size=(num_steps, 7)).astype(np.float32)
        # Bias gripper to mostly open
        actions[:, 6] = rng.choice([-1.0, 1.0], size=num_steps,
                                    p=[0.7, 0.3])

    elif pattern == "zigzag":
        for i in range(num_steps):
            actions[i, 0] = 0.5 if i % 4 < 2 else -0.5
            actions[i, 6] = -1.0

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return actions


def get_first_frame_from_episode(npz_path: str) -> np.ndarray:
    """Get the first frame from a LIBERO episode for use as conditioning."""
    data = np.load(npz_path)
    return data["images"][0]


def annotate_video(frames, pattern: str, num_steps: int):
    """Add label overlay to each frame showing the command."""
    annotated = []
    for i, f in enumerate(frames):
        f = f.copy()
        h, w = f.shape[:2]
        # Top banner
        cv2.rectangle(f, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(f, f"COMMAND: {pattern}", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        cv2.putText(f, f"step {i+1}/{len(frames)}", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        annotated.append(f)
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Command robot in Cosmos world model")
    parser.add_argument("--pattern", type=str, default="grasp",
                        choices=["stop", "forward", "down", "grasp",
                                 "pickup_then_lift", "circle", "random", "zigzag"])
    parser.add_argument("--first_frame_episode", type=str,
                        default=os.path.join(DATA_ROOT,
                                "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket/demo_45.npz"),
                        help="Episode to grab the initial frame from")
    parser.add_argument("--num_steps", type=int, default=12,
                        help="Number of action steps (12 = single chunk, 36 = 3 chunks)")
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(CONTACT_AWARE_WM, "results/cosmos/commands"))
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Get initial frame from a LIBERO episode (only the first frame matters)
    first_frame = get_first_frame_from_episode(args.first_frame_episode)
    print(f"[command] First frame from episode: {first_frame.shape}")

    # Build the custom action sequence
    actions = make_action_sequence(args.pattern, args.num_steps, args.seed)
    print(f"[command] Action sequence: pattern={args.pattern}, "
          f"shape={actions.shape}")
    print(f"[command] Action sample (first 3 steps):")
    for i in range(min(3, len(actions))):
        print(f"  step {i}: {actions[i]}")

    # Setup pipeline
    pipe = setup_cosmos_pipeline()

    # Generate
    print(f"[command] Generating video with command='{args.pattern}'...")
    video = run_inference(
        pipe, first_frame, actions,
        chunk_size=12,
        autoregressive=args.autoregressive,
        guidance=args.guidance,
        seed=args.seed,
    )
    print(f"[command] Generated video: {video.shape}")

    cosmos_frames = list(video_tensor_to_uint8(video))
    cosmos_frames = annotate_video(cosmos_frames, args.pattern, args.num_steps)

    out_path = os.path.join(args.out_dir, f"command_{args.pattern}_seed{args.seed}.mp4")
    write_video(cosmos_frames, out_path, fps=args.fps)

    print(f"\n[command] Done!")
    print(f"  Generated: {out_path}")
    print(f"  {len(cosmos_frames)} frames at {args.fps} fps "
          f"({len(cosmos_frames)/args.fps:.1f} seconds)")


if __name__ == "__main__":
    main()
