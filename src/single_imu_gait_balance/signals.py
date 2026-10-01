"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/signals.py
Description : Signal helpers: gap filling of marker trajectories, zero-phase
              Butterworth low-pass filters and time normalisation of one gait
              cycle.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

#: Samples per gait cycle used by the model (0, 1, ..., 100 % of the cycle).
N_POINTS = 101


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Start and end (exclusive) of the True stretches of a boolean array."""
    edges = np.diff(np.concatenate([[0], valid.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill_gaps(trajectory: np.ndarray, max_gap: int, min_segment: int = 3) -> np.ndarray:
    """Fill gaps of at most ``max_gap`` frames with a cubic spline.

    Visible samples stay unchanged; stretches shorter than ``min_segment``
    frames are treated as noise and removed.

    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Marker positions, NaN where missing.
    max_gap : int
        Longest gap (frames) that is filled; longer gaps stay NaN.
    """
    out = np.array(trajectory, dtype=float)
    valid = np.isfinite(out).all(axis=1)
    for start, stop in _runs(valid):
        if stop - start < min_segment:
            valid[start:stop] = False
    out[~valid] = np.nan
    runs = _runs(valid)
    blocks, block = [], runs[:1]
    for run in runs[1:]:  # neighbouring stretches with small gaps form one block
        if run[0] - block[-1][1] <= max_gap:
            block.append(run)
        else:
            blocks.append(block)
            block = [run]
    if block:
        blocks.append(block)
    for block in blocks:
        if len(block) < 2:
            continue
        frames = np.concatenate([np.arange(a, b) for a, b in block])
        span = np.arange(block[0][0], block[-1][1])
        missing = span[~valid[span]]
        out[missing] = CubicSpline(frames, out[frames], axis=0)(missing)
    return out


def lowpass(signal: np.ndarray, rate: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter along the first axis."""
    b, a = butter(order, cutoff / (rate / 2.0), btype="low")
    return filtfilt(b, a, signal, axis=0)


def lowpass_segments(trajectory: np.ndarray, rate: float, cutoff: float, order: int = 4) -> np.ndarray:
    """``lowpass`` applied to each visible stretch; stretches too short to filter are kept as they are."""
    b, a = butter(order, cutoff / (rate / 2.0), btype="low")
    pad = 3 * max(len(a), len(b))  # filtfilt's padding needs more samples than this
    out = np.array(trajectory, dtype=float)
    for start, stop in _runs(np.isfinite(out).all(axis=1)):
        if stop - start > pad:
            out[start:stop] = filtfilt(b, a, out[start:stop], axis=0)
    return out


def time_normalise(signal: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    """Linearly resample one cycle (first and last sample at 0 % and 100 %) to ``n_points`` samples.

    Parameters
    ----------
    signal : ndarray, shape (n_samples, n_channels)
    """
    signal = np.asarray(signal, dtype=float)
    x_in = np.linspace(0.0, 1.0, signal.shape[0])
    x_out = np.linspace(0.0, 1.0, n_points)
    return np.column_stack([np.interp(x_out, x_in, signal[:, c]) for c in range(signal.shape[1])])
