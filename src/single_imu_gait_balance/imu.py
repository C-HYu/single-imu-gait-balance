"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/imu.py
Description : Sacral IMU input of the model: sensor axes -> body axes, 15 Hz
              low-pass filter, cut to the gait cycle, mirroring of right-limb
              cycles, and time normalisation.
              Body axes: x anterior, y up, z to the right. Six channels:
              acceleration (m/s^2, including gravity) and angular velocity
              (rad/s) about x, y and z.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from .signals import N_POINTS, lowpass, time_normalise

#: Xsens MTw on the sacrum with its x axis up and its case facing backwards
#: (sensor y to the left, z posterior): body = (-z, x, -y).
XSENS_SACRUM = ("-z", "x", "-y")


def to_body_axes(raw: np.ndarray, axes: tuple[str, str, str]) -> np.ndarray:
    """Re-order and re-sign sensor axes into body axes.

    Parameters
    ----------
    raw : ndarray, shape (n_samples, 3)
        Sensor x, y, z.
    axes : tuple of str
        Sensor axis giving body x (anterior), y (up) and z (right), e.g. ("-z", "x", "-y").
    """
    columns = []
    for spec in axes:
        sign = -1.0 if spec.startswith("-") else 1.0
        columns.append(sign * raw[:, "xyz".index(spec[-1])])
    return np.column_stack(columns)


def mirror_right_limb(signals: np.ndarray) -> np.ndarray:
    """Reflect a right-limb cycle across the sagittal plane so it looks like a left-limb one.

    The medio-lateral acceleration (z) and the angular velocities about the
    anteroposterior (x) and vertical (y) axes change sign.
    """
    out = signals.copy()
    out[:, [2, 3, 4]] *= -1
    return out


def imu_cycle(acc: np.ndarray, gyro: np.ndarray, rate: float, start_s: float, end_s: float, side: str,
              cutoff: float = 15.0) -> np.ndarray:
    """Model input for one gait cycle, shape (101, 6).

    Parameters
    ----------
    acc, gyro : ndarray, shape (n_samples, 3)
        Body axes (see :func:`to_body_axes`), whole recording.
    rate : float
        Hz.
    start_s, end_s : float
        Heel strikes of the reference limb on the IMU time axis, s.
    side : {"left", "right"}
        Reference limb; right cycles are mirrored.
    """
    signals = lowpass(np.hstack([acc, gyro]), rate, cutoff)
    first, last = int(round(start_s * rate)), int(round(end_s * rate))
    if first < 0 or last >= signals.shape[0] or last - first < 4:
        raise ValueError(f"cycle {start_s:.2f}-{end_s:.2f} s outside the IMU recording")
    segment = signals[first:last + 1]
    if side == "right":
        segment = mirror_right_limb(segment)
    # Cubic interpolation to 501 points, then linear to 101 points.
    x = np.linspace(0.0, 1.0, segment.shape[0])
    fine = interp1d(x, segment, axis=0, kind="cubic")(np.linspace(0.0, 1.0, 501))
    return time_normalise(fine, N_POINTS)
