#!/usr/bin/env python3
"""
PyTorch Dataset for LIBERO-90 world model training.

Loads preprocessed NPZ files (from extract_ft_libero.py) with:
  images (T, 96, 96, 3), actions (T, 7), ft_wrist (T, 6),
  joint_torques (T, 7), ee_pos (T, 3), ee_ori (T, 3)

Returns per sample:
  anchor_frame:   (3, H, W)          — always frame[0] of episode (never noisy)
  context_frames: (N-1, 3, H, W)     — last N-1 frames before current
  target_frame:   (3, H, W)          — frame[t+1] to predict
  actions_chunk:  (chunk_size, 7)     — action history ending at t
  ft_chunk:       (chunk_size, 6)     — F/T history ending at t
  action_mask:    (chunk_size,)       — bool mask (True = valid)
  task_id:        int                 — which of 90 tasks

Episode boundary handling:
  - anchor_frame is always frame[0] (first GT frame of the episode)
  - context pads with copies of frame[0] at episode start
  - action/FT chunks pad with zeros at episode start

Normalization:
  - Images: [0, 255] uint8 → [0, 1] float32
  - Actions: zero-mean unit-variance (stats computed from training split)
  - F/T: zero-mean unit-variance (stats computed from training split)
  - Stats saved to / loaded from dataset_stats.json
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _resize_img(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize image if needed. img: (H, W, 3) uint8 or float32."""
    if img.shape[0] == target_size and img.shape[1] == target_size:
        return img
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


class LiberoDataset(Dataset):
    """LIBERO-90 dataset for flow matching world model training.

    Args:
        data_root: path to libero_90_with_ft/ directory
        suite: 'spatial', 'object', 'goal', 'all' — which tasks to include
        split: 'train' or 'val' (90/10 split within each task's demos)
        context_frames: number of frames in context window (including anchor)
        use_anchor: whether slot 0 is the episode's first frame
        use_ft: whether to include F/T readings
        chunk_size: action/FT history length
        img_size: resize images to this (should match extract_ft_libero.py)
        augment: whether to apply data augmentation
    """

    # LIBERO suite → task name prefixes
    SUITE_PREFIXES = {
        "spatial": ["LIVING_ROOM_SCENE1", "LIVING_ROOM_SCENE2"],
        "object": ["LIVING_ROOM_SCENE3", "LIVING_ROOM_SCENE4"],
        "goal": ["LIVING_ROOM_SCENE5", "LIVING_ROOM_SCENE6"],
    }

    def __init__(
        self,
        data_root: str,
        suite: str = "spatial",
        split: str = "train",
        context_frames: int = 4,
        use_anchor: bool = True,
        use_ft: bool = True,
        chunk_size: int = 4,
        img_size: int = 96,
        augment: bool = True,
    ):
        self.data_root = data_root
        self.suite = suite
        self.split = split
        self.context_frames = context_frames
        self.use_anchor = use_anchor and (context_frames > 1)
        self.use_ft = use_ft
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.augment = augment and (split == "train")

        # Discover tasks and load episodes
        self.episodes: List[Dict] = []  # list of {task_id, task_name, npz_path}
        self.samples: List[Tuple[int, int]] = []  # (episode_idx, timestep_t)
        self._load_episodes()

        # Compute or load normalization stats
        self.stats = self._get_or_compute_stats()

        print(f"[LiberoDataset] {split}: {len(self.samples)} samples from "
              f"{len(self.episodes)} episodes ({suite}), "
              f"ctx={context_frames}, anchor={self.use_anchor}, ft={use_ft}")

    def _get_task_names_for_suite(self) -> List[str]:
        """Get task directory names matching the selected suite."""
        all_tasks = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        ])

        if self.suite == "all":
            return all_tasks

        prefixes = self.SUITE_PREFIXES.get(self.suite)
        if prefixes is None:
            raise ValueError(f"Unknown suite: {self.suite}")

        return [t for t in all_tasks if any(t.startswith(p) for p in prefixes)]

    def _load_episodes(self) -> None:
        """Discover and index all episodes."""
        task_names = self._get_task_names_for_suite()
        task_name_to_id = {name: i for i, name in enumerate(task_names)}

        for task_name in task_names:
            task_dir = os.path.join(self.data_root, task_name)
            npz_files = sorted([f for f in os.listdir(task_dir) if f.endswith(".npz")])

            # 90/10 train/val split within each task
            n_total = len(npz_files)
            n_train = max(1, int(0.9 * n_total))

            if self.split == "train":
                selected = npz_files[:n_train]
            else:
                selected = npz_files[n_train:]

            for npz_file in selected:
                npz_path = os.path.join(task_dir, npz_file)
                ep_idx = len(self.episodes)

                # Load to get episode length (just the shape, not full data)
                with np.load(npz_path) as data:
                    T = data["images"].shape[0]

                self.episodes.append({
                    "task_id": task_name_to_id[task_name],
                    "task_name": task_name,
                    "npz_path": npz_path,
                    "length": T,
                })

                # Each timestep t where we can predict t+1 is a sample
                for t in range(T - 1):
                    self.samples.append((ep_idx, t))

    def _get_or_compute_stats(self) -> Dict[str, np.ndarray]:
        """Compute or load normalization statistics from training split."""
        stats_path = os.path.join(self.data_root, f"dataset_stats_{self.suite}.json")

        if os.path.exists(stats_path):
            with open(stats_path) as f:
                raw = json.load(f)
            return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

        # Only compute from training split
        if self.split != "train":
            raise FileNotFoundError(
                f"Stats not found at {stats_path}. Run training split first to compute stats."
            )

        print(f"[LiberoDataset] Computing normalization stats...")
        all_actions = []
        all_ft = []

        for ep in self.episodes:
            with np.load(ep["npz_path"]) as data:
                all_actions.append(data["actions"])
                if self.use_ft:
                    all_ft.append(data["ft_wrist"])

        actions_all = np.concatenate(all_actions, axis=0)
        stats = {
            "action_mean": actions_all.mean(axis=0).tolist(),
            "action_std": np.maximum(actions_all.std(axis=0), 1e-6).tolist(),
        }

        if all_ft:
            ft_all = np.concatenate(all_ft, axis=0)
            stats["ft_mean"] = ft_all.mean(axis=0).tolist()
            stats["ft_std"] = np.maximum(ft_all.std(axis=0), 1e-6).tolist()
        else:
            stats["ft_mean"] = [0.0] * 6
            stats["ft_std"] = [1.0] * 6

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[LiberoDataset] Stats saved to {stats_path}")

        return {k: np.array(v, dtype=np.float32) for k, v in stats.items()}

    def _load_episode_data(self, ep_idx: int) -> Dict[str, np.ndarray]:
        """Load full episode data from NPZ file."""
        return dict(np.load(self.episodes[ep_idx]["npz_path"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self.samples[idx]
        ep = self.episodes[ep_idx]
        data = self._load_episode_data(ep_idx)

        images = data["images"]    # (T, H, W, 3) uint8
        actions = data["actions"]  # (T, 7)
        ft = data["ft_wrist"]      # (T, 6)
        T = len(images)

        # --- Helper to load and resize a single frame ---
        S = self.img_size
        def load_frame(idx):
            return _resize_img(images[idx], S).astype(np.float32) / 255.0

        # --- Build context frames ---
        N = self.context_frames
        anchor_img = load_frame(0)  # Always frame[0]

        if N > 1 and self.use_anchor:
            context = np.zeros((N, S, S, 3), dtype=np.float32)
            context[0] = anchor_img
            memory_slots = N - 1
            for k in range(memory_slots):
                src_t = t - (memory_slots - 1 - k)
                if src_t >= 0:
                    context[1 + k] = load_frame(src_t)
                else:
                    context[1 + k] = anchor_img
        elif N > 1:
            context = np.zeros((N, S, S, 3), dtype=np.float32)
            for k in range(N):
                src_t = t - (N - 1 - k)
                if src_t >= 0:
                    context[k] = load_frame(src_t)
                else:
                    context[k] = anchor_img
        else:
            context = load_frame(t)

        target = load_frame(t + 1)

        # --- Build action/FT chunks ---
        H = self.chunk_size
        actions_chunk = np.zeros((H, 7), dtype=np.float32)
        ft_chunk = np.zeros((H, 6), dtype=np.float32)
        action_mask = np.zeros(H, dtype=bool)

        for k in range(H):
            src_t = t - (H - 1 - k)
            if 0 <= src_t < T:
                actions_chunk[k] = actions[src_t]
                ft_chunk[k] = ft[src_t]
                action_mask[k] = True

        # Normalize actions and F/T
        actions_chunk = (actions_chunk - self.stats["action_mean"]) / self.stats["action_std"]
        ft_chunk = (ft_chunk - self.stats["ft_mean"]) / self.stats["ft_std"]
        actions_chunk[~action_mask] = 0.0
        ft_chunk[~action_mask] = 0.0

        if not self.use_ft:
            ft_chunk = np.zeros_like(ft_chunk)

        # --- Augmentation (training only) ---
        if self.augment and np.random.random() < 0.5:
            # Horizontal flip
            if context.ndim == 4:
                context = context[:, :, ::-1, :].copy()
            else:
                context = context[:, ::-1, :].copy()
            anchor_img = anchor_img[:, ::-1, :].copy()
            target = target[:, ::-1, :].copy()
            # Flip y-component of action (index 1) and dpitch (index 4)
            actions_chunk[:, 1] = -actions_chunk[:, 1]
            actions_chunk[:, 4] = -actions_chunk[:, 4]
            ft_chunk[:, 1] = -ft_chunk[:, 1]
            ft_chunk[:, 4] = -ft_chunk[:, 4]

        # --- Convert to tensors ---
        if context.ndim == 4:
            context_t = torch.from_numpy(context).permute(0, 3, 1, 2)  # (N, 3, H, W)
        else:
            context_t = torch.from_numpy(context).permute(2, 0, 1)     # (3, H, W)

        return {
            "anchor_frame": torch.from_numpy(anchor_img).permute(2, 0, 1),  # (3, H, W)
            "context_imgs": context_t,
            "img_t": context_t[-1] if context_t.ndim == 4 else context_t,
            "target_frame": torch.from_numpy(target).permute(2, 0, 1),
            "img_next": torch.from_numpy(target).permute(2, 0, 1),  # alias
            "actions_chunk": torch.from_numpy(actions_chunk),
            "action": torch.from_numpy(actions_chunk),  # alias
            "ft_chunk": torch.from_numpy(ft_chunk),
            "ft": torch.from_numpy(ft_chunk),  # alias
            "action_mask": torch.from_numpy(action_mask),
            "pad_mask": torch.from_numpy(action_mask),  # alias for model compat
            "task_id": ep["task_id"],
        }
