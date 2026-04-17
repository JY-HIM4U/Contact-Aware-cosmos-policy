"""LIBERODataset subclass that adds Force/Torque (F/T) as a latent-injected modality.

Adds two new latent frames to the sequence:
  - current_ft  (6D): F/T reading at time t     [conditioning]
  - future_ft   (6D): F/T reading at time t+K   [prediction target]

The new layout (state_t=11):
  0: blank | 1: proprio | 2: current_ft | 3: wrist | 4: primary
  5: action | 6: future_proprio | 7: future_ft | 8: future_wrist | 9: future_primary
  10: value

F/T data is loaded from pre-extracted NPZ files (one per demo) located under
`ft_data_dir/{task_folder}/demo_{i}.npz`, each containing an `ft_wrist` array
of shape (T, 6).
"""

import os
import json

import numpy as np

from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.datasets.dataset_utils import rescale_data
from cosmos_policy.utils.utils import duplicate_array


class LIBERODatasetFT(LIBERODataset):

    def __init__(
        self,
        *args,
        ft_data_dir: str = "",
        normalize_ft: bool = True,
        ft_stats_path: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ft_data_dir = ft_data_dir
        self.normalize_ft = normalize_ft

        # Filter episodes to only those with matching T5 embeddings
        if hasattr(self, "t5_text_embeddings"):
            valid_cmds = set(self.t5_text_embeddings.keys())
            removed = 0
            new_data = {}
            new_idx = 0
            for ep_idx in range(self.num_episodes):
                if self.data[ep_idx]["command"] in valid_cmds:
                    new_data[new_idx] = self.data[ep_idx]
                    new_idx += 1
                else:
                    removed += 1
            if removed:
                self.data = new_data
                self.num_episodes = new_idx
                self.num_steps = sum(ep["num_steps"] for ep in self.data.values())
                self._build_step_index_mapping()
                print(f"[LIBERODatasetFT] Removed {removed} episodes without T5 match")

        if not ft_data_dir:
            raise ValueError("ft_data_dir is required for LIBERODatasetFT")

        # Load F/T stats for normalization
        if ft_stats_path:
            stats_file = ft_stats_path
        else:
            stats_file = os.path.join(ft_data_dir, "dataset_stats_all.json")
        with open(stats_file) as f:
            ft_stats = json.load(f)
        self.ft_min = np.array(ft_stats["ft_min"], dtype=np.float32)
        self.ft_max = np.array(ft_stats["ft_max"], dtype=np.float32)

        # Attach F/T arrays to each episode in self.data
        n_matched = 0
        for ep_idx in range(self.num_episodes):
            ep = self.data[ep_idx]
            task_folder = self._command_to_task_folder(ep)
            if task_folder is None:
                ep["ft"] = np.zeros((ep["num_steps"], 6), dtype=np.float32)
                ep["has_ft"] = False
                continue
            demo_idx = self._episode_demo_idx(ep_idx)
            npz_path = os.path.join(ft_data_dir, task_folder, f"demo_{demo_idx}.npz")
            if os.path.exists(npz_path):
                ft = np.load(npz_path)["ft_wrist"].astype(np.float32)
                # Align length with episode
                T = ep["num_steps"]
                if len(ft) >= T:
                    ft = ft[:T]
                else:
                    ft = np.pad(ft, ((0, T - len(ft)), (0, 0)), mode="edge")
                ep["ft"] = ft
                ep["has_ft"] = True
                n_matched += 1
            else:
                ep["ft"] = np.zeros((ep["num_steps"], 6), dtype=np.float32)
                ep["has_ft"] = False

        print(f"[LIBERODatasetFT] Matched F/T for {n_matched}/{self.num_episodes} episodes")

        # Normalize F/T to [-1, +1]
        if self.normalize_ft:
            for ep_idx in range(self.num_episodes):
                ep = self.data[ep_idx]
                if ep["has_ft"]:
                    denom = self.ft_max - self.ft_min
                    denom = np.where(denom < 1e-8, 1.0, denom)
                    ep["ft"] = 2.0 * (ep["ft"] - self.ft_min) / denom - 1.0

    def _command_to_task_folder(self, ep):
        """Reconstruct the task folder name from the HDF5-derived command."""
        # The folder names look like LIVING_ROOM_SCENE1_pick_up_the_..._basket
        # Try matching by listing ft_data_dir
        cmd = ep["command"].replace(" ", "_")
        for entry in os.listdir(self.ft_data_dir):
            if not os.path.isdir(os.path.join(self.ft_data_dir, entry)):
                continue
            # Strip scene prefix for matching
            entry_words = entry.split("_")
            entry_cmd = []
            skip = True
            for w in entry_words:
                if "SCENE" in w:
                    skip = False
                    continue
                if skip:
                    continue
                entry_cmd.append(w)
            entry_joined = "_".join(entry_cmd)
            if entry_joined == cmd:
                return entry
        return None

    def _episode_demo_idx(self, ep_idx):
        """Infer which demo index within its HDF5 this episode corresponds to."""
        # Episodes are loaded in order: for each HDF5, demo_0, demo_1, ...
        # We track by counting episodes per unique suite/command pair
        target = self.data[ep_idx]
        count = 0
        for i in range(ep_idx):
            ep = self.data[i]
            if ep["command"] == target["command"] and ep["suite"] == target["suite"]:
                count += 1
        return count

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # Recover which episode + timestep this sample came from
        episode_idx, relative_step_idx = self._get_episode_and_step(idx)
        episode_data = self.data.get(episode_idx)

        if episode_data is None or "ft" not in episode_data:
            current_ft = np.zeros(6, dtype=np.float32)
            future_ft = np.zeros(6, dtype=np.float32)
            has_ft = False
        else:
            future_frame_idx = min(
                relative_step_idx + self.chunk_size,
                episode_data["num_steps"] - 1,
            )
            current_ft = episode_data["ft"][relative_step_idx].copy()
            future_ft = episode_data["ft"][future_frame_idx].copy()
            has_ft = episode_data.get("has_ft", False)

        # Insert two blank images into the video tensor for F/T latent frames.
        # The parent's __getitem__ built a video of shape (T_parent, H, W, C) or
        # (T_parent, C, H, W). We need to insert blank frames at the right positions.
        video = sample["video"]  # (T, C, H, W) or (T, H, W, C)
        blank = np.zeros_like(video[0:1])
        # Insert current_ft frame after current_proprio (index 1 → insert at 2)
        # Insert future_ft frame after future_proprio (shifts with each insertion)
        # Parent layout: 0:blank 1:proprio 2:wrist 3:primary 4:action 5:fut_proprio 6:fut_wrist 7:fut_primary 8:value
        # New layout:    0:blank 1:proprio 2:cur_ft 3:wrist 4:primary 5:action 6:fut_proprio 7:fut_ft 8:fut_wrist 9:fut_primary 10:value

        # Duplicate blank to match per-image duplication (parent uses num_duplicates_per_image=4)
        # But video is already concatenated — each "frame" is 4 raw images stacked
        # except index 0 which is a single frame.
        # Actually the parent stacks via np.concatenate along axis=0 before preprocess.
        # After preprocess, the shape is (T_total_raw_images, C, H, W).
        # Each modality occupies either 1 (blank padding) or num_duplicates (4) raw images.
        # We need to figure out the cumulative raw-image offsets.

        n_dup = self.num_duplicates_per_image  # typically 4
        # Parent sequence (raw image counts):
        # idx0: 1 (blank padding, NOT duplicated)
        # idx1: n_dup (proprio blank)
        # idx2: n_dup (wrist)
        # idx3: n_dup (primary)
        # idx4: n_dup (action blank)
        # idx5: n_dup (future proprio blank)
        # idx6: n_dup (future wrist)
        # idx7: n_dup (future primary)
        # idx8: n_dup (value blank)
        # Total raw: 1 + 8*n_dup = 33

        # We insert at raw-image position:
        #   current_ft: after proprio (after 1 + n_dup = 5 raw images)
        #   future_ft: after future_proprio (after 1 + 5*n_dup + n_dup = 1 + 6*n_dup with new cur_ft)
        #     Actually: 1(blank) + n_dup(proprio) + n_dup(cur_ft_NEW) + n_dup(wrist) + n_dup(primary) + n_dup(action) + n_dup(fut_proprio) = 1 + 6*n_dup

        insert_pos_cur_ft = 1 + n_dup  # after blank(1) + proprio(n_dup)
        blank_chunk = np.zeros_like(video[1:1 + n_dup])  # n_dup blank frames
        video = np.concatenate([
            video[:insert_pos_cur_ft],
            blank_chunk,
            video[insert_pos_cur_ft:],
        ], axis=0)

        # Now future_ft goes after future_proprio
        # Current layout in raw images:
        # 0: 1 blank | 1-4: proprio | 5-8: cur_ft | 9-12: wrist | 13-16: primary
        # 17-20: action | 21-24: fut_proprio | ... fut_ft goes here
        insert_pos_fut_ft = 1 + 6 * n_dup  # after fut_proprio
        video = np.concatenate([
            video[:insert_pos_fut_ft],
            blank_chunk,
            video[insert_pos_fut_ft:],
        ], axis=0)

        sample["video"] = video

        # Shift all parent latent indices that come after our insertions
        # Parent indices are in "latent frame" space (not raw image).
        # current_ft is new index 2, so everything >= 2 shifts +1
        # future_ft is new index 7 (after first shift), so everything >= 7 shifts +1
        def shift(key, threshold, amount=1):
            v = sample[key]
            if isinstance(v, int):
                if v >= 0 and v >= threshold:
                    sample[key] = v + amount
            elif hasattr(v, 'item'):
                val = v.item() if hasattr(v, 'item') else int(v)
                if val >= 0 and val >= threshold:
                    sample[key] = val + amount

        # First insertion: current_ft at index 2
        for k in ["current_wrist_image_latent_idx", "current_image_latent_idx",
                   "action_latent_idx", "future_proprio_latent_idx",
                   "future_wrist_image_latent_idx", "future_image_latent_idx",
                   "value_latent_idx"]:
            shift(k, 2)

        current_ft_latent_idx = 2

        # Second insertion: future_ft after future_proprio
        # After first shift, future_proprio is at index 6
        future_ft_latent_idx = sample["future_proprio_latent_idx"] + 1 if isinstance(
            sample["future_proprio_latent_idx"], int
        ) else sample["future_proprio_latent_idx"].item() + 1

        for k in ["future_wrist_image_latent_idx", "future_image_latent_idx",
                   "value_latent_idx"]:
            shift(k, future_ft_latent_idx)

        sample["current_ft"] = current_ft
        sample["future_ft"] = future_ft
        sample["current_ft_latent_idx"] = current_ft_latent_idx
        sample["future_ft_latent_idx"] = future_ft_latent_idx
        sample["has_ft"] = 1 if has_ft else 0

        return sample

    def _get_episode_and_step(self, idx):
        """Map global index to (episode_idx, relative_step_idx)."""
        if hasattr(self, '_step_to_episode_step'):
            return self._step_to_episode_step[idx]
        # Reconstruct from parent's step mapping
        cumulative = 0
        for ep_idx in range(self.num_episodes):
            ep_len = self.data[ep_idx]["num_steps"]
            if idx < cumulative + ep_len:
                return ep_idx, idx - cumulative
            cumulative += ep_len
        return self.num_episodes - 1, 0
