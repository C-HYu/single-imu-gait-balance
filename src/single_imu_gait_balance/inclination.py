"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/inclination.py
Description : The two balance variables: the inclination angle (IA) of the
              center-of-pressure-to-center-of-mass vector in the sagittal and
              frontal planes, and its rate of change (RCIA).
              - IA: u = (COM - COP) / |COM - COP|, t = Z x u, sagittal IA =
                asin(t . Y), frontal IA = s asin(t . X), with X the walking
                direction, Z vertical, Y = Z x X and s = +1 for a left and -1
                for a right reference limb (positive: COM ahead of the COP,
                and medial of it relative to the reference limb).
              - Ground-truth RCIA: derivative of a cubic smoothing spline fitted
                to the IA, smoothing chosen by generalised cross-validation.
              - Predicted RCIA: central differences of the predicted IA.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_smoothing_spline

from .gait import GaitCycle
from .signals import N_POINTS, time_normalize

VERTICAL = np.array([0.0, 0.0, 1.0])


def progression_frame(com: np.ndarray, cycle: GaitCycle) -> tuple[np.ndarray, np.ndarray]:
    """Walking direction X (COM displacement over the cycle, horizontal) and Y = Z x X (to the left)."""
    x = com[cycle.end] - com[cycle.start]
    x = x - (x @ VERTICAL) * VERTICAL
    x = x / np.linalg.norm(x)
    y = np.cross(VERTICAL, x)
    return x, y / np.linalg.norm(y)


def inclination_angles(com: np.ndarray, cop: np.ndarray, x: np.ndarray, y: np.ndarray,
                       side: str) -> np.ndarray:
    """Sagittal and frontal IA in degrees, shape (n_frames, 2).

    Parameters
    ----------
    com, cop : ndarray, shape (n_frames, 3)
        mm, laboratory frame (z vertical).
    x, y : ndarray, shape (3,)
        From :func:`progression_frame`.
    side : {"left", "right"}
        Reference limb (the one whose heel strikes start and end the cycle).
    """
    u = com - cop
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    t = np.cross(VERTICAL, u)
    sign = 1.0 if side == "left" else -1.0
    sagittal = np.degrees(np.arcsin(np.clip(t @ y, -1, 1)))
    frontal = sign * np.degrees(np.arcsin(np.clip(t @ x, -1, 1)))
    return np.column_stack([sagittal, frontal])


def rcia_from_ia(ia: np.ndarray, cycle_time: float, frame_rate: float, n_points: int = N_POINTS) -> np.ndarray:
    """Ground-truth RCIA (deg/s) at ``n_points`` samples of the cycle.

    A cubic smoothing spline with the smoothing parameter chosen by
    generalised cross-validation (Craven & Wahba 1979; the role of GCVSPL,
    Woltring 1986) is fitted to the IA on the motion-capture frames and
    differentiated.

    Parameters
    ----------
    ia : ndarray, shape (n_samples, 2)
        IA over one cycle, degrees.
    cycle_time : float
        s.
    frame_rate : float
        Hz of the motion capture; the IA is first brought to that frame grid.
    """
    n_frames = max(int(round(cycle_time * frame_rate)) + 1, 10)
    ia_frames = time_normalize(ia, n_frames)
    t_frames = np.linspace(0.0, cycle_time, n_frames)
    t_out = np.linspace(0.0, cycle_time, n_points)
    out = np.empty((n_points, ia.shape[1]))
    for c in range(ia.shape[1]):
        out[:, c] = make_smoothing_spline(t_frames, ia_frames[:, c]).derivative()(t_out)
    return out


def finite_difference_rcia(ia: np.ndarray, cycle_time: np.ndarray) -> np.ndarray:
    """RCIA (deg/s) of time-normalized IA by central differences (one-sided at the ends).

    Parameters
    ----------
    ia : ndarray, shape (n_cycles, n_points, 2)
    cycle_time : ndarray, shape (n_cycles,)
        s.
    """
    dt = np.asarray(cycle_time, dtype=float) / (ia.shape[1] - 1)
    return np.gradient(ia, axis=1) / dt[:, None, None]
