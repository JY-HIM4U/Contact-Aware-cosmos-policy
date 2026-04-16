"""
PyTorch Dataset for the RH20T world model experiment.

Each sample: (context_imgs, action_chunk, ft_chunk) -> img_next

Supports:
  - Multi-frame context: return last N frames as conditioning (context_frames param)
  - First-frame anchoring: slot 0 of context is always the first GT frame of the
    episode, providing a stable reference for static scene elements (use_anchor param)
  - Action/F/T temporal chunks with episode-boundary padding
  - Backward compatibility: context_frames=1 returns single img_t as before
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")


class RH20TWorldModelDataset(Dataset):
    """
    Returns dict:
        context_imgs: (N, 3, 96, 96) float32 — multi-frame context window
                      When use_anchor=True: slot 0 = episode's first frame (anchor)
                      Remaining slots = recent past frames (memory)
                      When context_frames=1: squeezed to (3, 96, 96) as 'img_t'
        action:       (H, 3) float32 or (3,) when chunk_size=1
        ft:           (H, 6) float32 or (6,) when chunk_size=1
        pad_mask:     (H,) bool
        img_next:     (3, 96, 96) float32 in [0, 1]
    """

    def __init__(self, split="train", data_dir=None, augment=True, chunk_size=1,
                 context_frames=1, use_anchor=False):
        if data_dir is None:
            data_dir = DATA_DIR

        self.chunk_size = chunk_size
        self.context_frames = context_frames
        self.use_anchor = use_anchor and (context_frames > 1)

        # Load samples
        npz_path = os.path.join(data_dir, "samples.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Dataset not found at {npz_path}. Run data/preprocess.py first."
            )
        data = np.load(npz_path)

        # Load splits
        splits_path = os.path.join(data_dir, "splits.json")
        with open(splits_path) as f:
            splits = json.load(f)

        indices = splits[split]

        self.images_t = data["images_t"][indices]      # (N, 96, 96, 3) uint8
        self.actions = data["actions"][indices]          # (N, 3) float32
        self.fts = data["fts"][indices]                  # (N, 6) float32
        self.images_next = data["images_next"][indices]  # (N, 96, 96, 3) uint8

        # Episode IDs for boundary detection
        if "episode_ids" in data:
            self.episode_ids = data["episode_ids"][indices]  # (N,) int32
        else:
            self.episode_ids = np.zeros(len(indices), dtype=np.int32)

        # Precompute anchor frame index for each episode (first sample in this split)
        # Maps episode_id -> local index of that episode's first frame
        self._anchor_indices = {}
        for local_idx in range(len(self.episode_ids)):
            eid = self.episode_ids[local_idx]
            if eid not in self._anchor_indices:
                self._anchor_indices[eid] = local_idx

        # Load F/T normalization stats
        ft_stats_path = os.path.join(data_dir, "ft_stats.json")
        with open(ft_stats_path) as f:
            ft_stats = json.load(f)
        self.ft_mean = np.array(ft_stats["mean"], dtype=np.float32)
        self.ft_std = np.array(ft_stats["std"], dtype=np.float32)

        self.split = split
        self.augment = augment and (split == "train")

        print(f"[dataset] Loaded {split}: {len(self)} samples, "
              f"chunk_size={chunk_size}, context_frames={context_frames}, "
              f"use_anchor={self.use_anchor}")

    def __len__(self):
        return len(self.images_t)

    def _get_chunk(self, idx):
        """Get action/ft chunk of H past steps ending at idx (inclusive)."""
        H = self.chunk_size
        current_ep = self.episode_ids[idx]

        actions_chunk = np.zeros((H, 3), dtype=np.float32)
        fts_chunk = np.zeros((H, 6), dtype=np.float32)
        pad_mask = np.zeros(H, dtype=bool)

        for k in range(H):
            src_idx = idx - (H - 1 - k)
            if src_idx >= 0 and self.episode_ids[src_idx] == current_ep:
                actions_chunk[k] = self.actions[src_idx]
                fts_chunk[k] = self.fts[src_idx]
                pad_mask[k] = True

        return actions_chunk, fts_chunk, pad_mask

    def _get_context_frames(self, idx):
        """Get multi-frame context window ending at idx.

        Returns:
            context: (N, 96, 96, 3) uint8 — N context frames
              If use_anchor: slot 0 = first frame of episode (anchor)
                             slots 1..N-1 = most recent memory frames
              Else:          slots 0..N-1 = most recent N frames

        At episode boundaries, missing frames are padded by repeating the
        earliest available frame (which is the anchor if use_anchor=True).
        """
        N = self.context_frames
        current_ep = self.episode_ids[idx]
        anchor_idx = self._anchor_indices[current_ep]
        anchor_frame = self.images_t[anchor_idx]

        if self.use_anchor:
            # Slot 0 = anchor (always the first GT frame of the episode)
            # Slots 1..N-1 = most recent N-1 frames (memory)
            memory_slots = N - 1
            context = np.zeros((N, 96, 96, 3), dtype=np.uint8)
            context[0] = anchor_frame

            for k in range(memory_slots):
                # k=0 is oldest memory, k=memory_slots-1 is current (idx)
                src_idx = idx - (memory_slots - 1 - k)
                if src_idx >= 0 and self.episode_ids[src_idx] == current_ep:
                    context[1 + k] = self.images_t[src_idx]
                else:
                    # Pad with anchor frame at episode boundaries
                    context[1 + k] = anchor_frame
        else:
            # All N slots are recent frames (no dedicated anchor)
            context = np.zeros((N, 96, 96, 3), dtype=np.uint8)
            for k in range(N):
                src_idx = idx - (N - 1 - k)
                if src_idx >= 0 and self.episode_ids[src_idx] == current_ep:
                    context[k] = self.images_t[src_idx]
                else:
                    # Pad with first available frame
                    context[k] = anchor_frame

        return context

    def __getitem__(self, idx):
        img_next = self.images_next[idx].astype(np.float32) / 255.0

        # Get context frames or single frame
        if self.context_frames > 1:
            context = self._get_context_frames(idx).astype(np.float32) / 255.0
        else:
            context = self.images_t[idx].astype(np.float32) / 255.0

        # Get action/ft chunk
        actions_chunk, fts_chunk, pad_mask = self._get_chunk(idx)

        # Normalize F/T
        fts_chunk = (fts_chunk - self.ft_mean) / self.ft_std
        fts_chunk[~pad_mask] = 0.0

        # Random horizontal flip (training only)
        if self.augment and np.random.random() < 0.5:
            if self.context_frames > 1:
                context = context[:, :, ::-1, :].copy()  # flip W dim for (N, H, W, C)
            else:
                context = context[:, ::-1, :].copy()      # flip W dim for (H, W, C)
            img_next = img_next[:, ::-1, :].copy()
            actions_chunk[:, 1] = -actions_chunk[:, 1]
            fts_chunk[:, 1] = -fts_chunk[:, 1]
            fts_chunk[:, 4] = -fts_chunk[:, 4]

        # Convert to tensors
        if self.context_frames > 1:
            # (N, H, W, C) -> (N, C, H, W)
            context_t = torch.from_numpy(context).permute(0, 3, 1, 2)
        else:
            # (H, W, C) -> (C, H, W)
            context_t = torch.from_numpy(context).permute(2, 0, 1)

        img_next = torch.from_numpy(img_next).permute(2, 0, 1)
        action = torch.from_numpy(actions_chunk)
        ft = torch.from_numpy(fts_chunk)
        pad_mask = torch.from_numpy(pad_mask)

        # Backward compatibility: squeeze when chunk_size=1
        if self.chunk_size == 1:
            action = action.squeeze(0)
            ft = ft.squeeze(0)
            pad_mask = pad_mask.squeeze(0)

        return {
            "context_imgs": context_t,  # (N, 3, 96, 96) or (3, 96, 96) when N=1
            "img_t": context_t if self.context_frames == 1 else context_t[-1],
            "action": action,
            "ft": ft,
            "pad_mask": pad_mask,
            "img_next": img_next,
        }
