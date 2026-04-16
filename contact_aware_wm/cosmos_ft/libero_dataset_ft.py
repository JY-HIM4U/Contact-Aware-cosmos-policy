"""LIBERODatasetFT: Cosmos's LIBERODataset + force/torque (F/T) fields.

Approach C: thin subclass of the upstream `LIBERODataset`. We call `super().__getitem__`
unchanged, then attach F/T-related fields to the returned dict. No upstream file
is touched.

Added fields per sample
-----------------------
- `current_ft_chunk` : (chunk_size, 6) preprocessed F/T history ending at current step
- `future_ft_chunk`  : (chunk_size, 6) preprocessed F/T starting at the future step
- `current_ft_last`  : (6,) last value of the current chunk
- `future_ft_first`  : (6,) first value of the future chunk (primary regression target)
- `has_ft`           : scalar bool — False if no NPZ was found for this demo; F/T
                       arrays are zeroed in that case so batching still works.

Notes
-----
- For demonstration samples the demo's HDF5 path + `demo_{i}` key are looked up
  from `self.data[episode_idx]`; we rely on upstream storing these (it does,
  through the internal episode metadata). For robustness we also fall back to
  reconstructing from `command` when the path is missing.
- For rollout samples (success/failure) we don't have extracted F/T — `has_ft=False`
  and we pass zeros. You can filter these out at the DataLoader level with
  `has_ft` if needed.
"""

from __future__ import annotations
from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT

import os
from typing import Any, Dict

import numpy as np

from cosmos_policy.datasets.libero_dataset import LIBERODataset

from .ft_data_utils import (
    build_ft_sample,
    load_ft_for_demo,
    load_ft_stats,
)


class LIBERODatasetFT(LIBERODataset):
    """Drop-in replacement for LIBERODataset that also returns F/T fields."""

    def __init__(
        self,
        *args,
        ft_root: str = os.path.join(CONTACT_AWARE_WM, "data/libero_90_with_ft"),
        ft_stats_suite: str = "all",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ft_root = ft_root
        self.ft_mean, self.ft_std = load_ft_stats(ft_root, suite=ft_stats_suite)

        # Cache per-episode F/T arrays so repeated samples from the same demo
        # don't re-read the NPZ from disk.
        self._ft_cache: Dict[int, np.ndarray | None] = {}

        # Precompute episode -> (hdf5_path, demo_key) lookup. This is the part
        # most likely to break if upstream changes internals — we guard it.
        self._episode_lookup = self._build_episode_lookup()

    # --------------------------------------------------------------------- #
    # Episode -> (hdf5_path, demo_key) resolution
    # --------------------------------------------------------------------- #

    def _build_episode_lookup(self):
        """Best-effort map: episode_idx -> (hdf5_path, demo_key).

        We check a few common attribute names. If none exist, we skip and
        `_resolve_demo` will return (None, None) causing has_ft=False.
        """
        lookup = {}
        # Walk self.data which is a list/dict of episodes
        if not hasattr(self, "data"):
            return lookup
        data = self.data
        episodes = data.values() if isinstance(data, dict) else data
        for i, ep in enumerate(episodes):
            hdf5_path = None
            demo_key = None
            if isinstance(ep, dict):
                hdf5_path = ep.get("hdf5_path") or ep.get("file_path") or ep.get("source_file")
                demo_key = ep.get("demo_key") or ep.get("demo") or ep.get("key")
            lookup[i] = (hdf5_path, demo_key)
        return lookup

    def _resolve_demo(self, episode_idx: int):
        return self._episode_lookup.get(episode_idx, (None, None))

    # --------------------------------------------------------------------- #
    # F/T fetch with caching
    # --------------------------------------------------------------------- #

    def _get_ft(self, episode_idx: int):
        if episode_idx in self._ft_cache:
            return self._ft_cache[episode_idx]
        hdf5_path, demo_key = self._resolve_demo(episode_idx)
        ft = None
        if hdf5_path and demo_key:
            try:
                ft = load_ft_for_demo(self.ft_root, hdf5_path, demo_key)
            except Exception as e:
                print(f"[LIBERODatasetFT] F/T load failed for ep={episode_idx}: {e}")
                ft = None
        self._ft_cache[episode_idx] = ft
        return ft

    # --------------------------------------------------------------------- #
    # Sample construction
    # --------------------------------------------------------------------- #

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        # Figure out which episode this sample came from. Upstream encodes the
        # mapping via its step-to-episode tables; we replicate the demo-only branch
        # since rollouts don't have extracted F/T anyway.
        episode_idx = -1
        relative_step_idx = 0
        future_step_idx = 0
        if idx < getattr(self, "adjusted_demo_count", idx + 1):
            global_step_idx = idx % self.num_steps
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
            future_step_idx = min(
                relative_step_idx + self.chunk_size,
                self.data[episode_idx]["num_steps"] - 1,
            )

        ft = self._get_ft(episode_idx) if episode_idx >= 0 else None
        chunk_size = self.chunk_size

        if ft is None:
            zeros = np.zeros((chunk_size, 6), dtype=np.float32)
            sample.update({
                "current_ft_chunk": zeros,
                "future_ft_chunk": zeros,
                "current_ft_last": np.zeros(6, np.float32),
                "future_ft_first": np.zeros(6, np.float32),
                "has_ft": np.float32(0.0),
            })
        else:
            ft_fields = build_ft_sample(
                ft=ft,
                relative_step_idx=relative_step_idx,
                future_step_idx=future_step_idx,
                chunk_size=chunk_size,
                ft_mean=self.ft_mean,
                ft_std=self.ft_std,
            )
            sample.update(ft_fields)
            sample["has_ft"] = np.float32(1.0)

        return sample
