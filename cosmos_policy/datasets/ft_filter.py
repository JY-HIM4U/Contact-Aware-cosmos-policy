"""Causal streaming F/T filter used at both training (offline) and eval (streaming).

Pipeline: causal median-3 -> causal Butterworth low-pass (3 Hz, order 2).
The same class is used offline (apply_offline) and online (reset + step per episode)
so training and inference see the same signal distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class CausalFtFilter:
    def __init__(self, cutoff_hz: float = 3.0, fs_hz: float = 20.0, order: int = 2, n_channels: int = 6):
        self.sos = butter(order, cutoff_hz, btype="low", fs=fs_hz, output="sos")
        self.n_channels = n_channels
        self.reset()

    def reset(self) -> None:
        self._med_buf: list[np.ndarray] = []
        # Zero initial state: assumes signal starts at 0 (true at episode start — no contact yet).
        # Using sosfilt_zi (unit-step steady state) would cause a spurious step transient at t=0.
        zi_shape = sosfilt_zi(self.sos).shape
        self._zi = [np.zeros(zi_shape, dtype=np.float64) for _ in range(self.n_channels)]

    def step(self, ft_6d: np.ndarray) -> np.ndarray:
        ft_6d = np.asarray(ft_6d, dtype=np.float32)
        self._med_buf.append(ft_6d.copy())
        if len(self._med_buf) > 3:
            self._med_buf.pop(0)
        med = np.median(np.stack(self._med_buf, axis=0), axis=0)

        out = np.empty_like(ft_6d)
        for c in range(self.n_channels):
            y, self._zi[c] = sosfilt(self.sos, [med[c]], zi=self._zi[c])
            out[c] = y[0]
        return out

    def apply_offline(self, ft_seq: np.ndarray) -> np.ndarray:
        self.reset()
        out = np.empty_like(ft_seq, dtype=np.float32)
        for t in range(len(ft_seq)):
            out[t] = self.step(ft_seq[t])
        return out
