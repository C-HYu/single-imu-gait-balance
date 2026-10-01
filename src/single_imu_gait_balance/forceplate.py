"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/forceplate.py
Description : Ground reaction of the force plates in the laboratory frame and
              the center of pressure (COP) of the summed force, resampled to
              the marker frames.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .c3d import ForcePlate
from .signals import lowpass

GRAVITY = 9.81  # m/s^2


@dataclass
class GroundReaction:
    """Force-plate data at marker frames.

    Attributes
    ----------
    plate_force : ndarray, shape (n_frames, n_plates, 3)
        Force of each plate in the laboratory frame, N.
    total_force : ndarray, shape (n_frames, 3)
        N.
    cop : ndarray, shape (n_frames, 3)
        COP of the summed force on the plate surface, mm; NaN when unloaded.
    """

    plate_force: np.ndarray
    total_force: np.ndarray
    cop: np.ndarray


def plate_wrench(plate: ForcePlate, zero_samples: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Force (N) and moment about the laboratory origin (N mm) of one plate, laboratory frame.

    The mean of the first ``zero_samples`` samples (unloaded plate) is removed
    as the zero offset.
    """
    force = plate.force - plate.force[:zero_samples].mean(axis=0)
    moment = plate.moment - plate.moment[:zero_samples].mean(axis=0)
    force_lab = force @ plate.rotation.T
    moment_lab = moment @ plate.rotation.T + np.cross(plate.transducer_origin, force_lab)
    return force_lab, moment_lab


def ground_reaction(plates: list[ForcePlate], analog_rate: float, marker_rate: float, n_frames: int,
                    cutoff: float = 25.0, min_fz: float = 10.0) -> GroundReaction:
    """Summed ground reaction and its COP at the marker frame rate.

    Each plate is low-pass filtered (4th-order zero-phase Butterworth, 25 Hz)
    at the analog rate and down-sampled to the marker frames. The COP of the
    summed wrench lies on the plate surface at height h (the vertical-force
    weighted height of the loaded surfaces):

        COP_x = (h F_x - M_y) / F_z,   COP_y = (M_x + h F_y) / F_z,   COP_z = h.

    Parameters
    ----------
    plates : list of ForcePlate
        The plates walked on.
    analog_rate, marker_rate : float
        Hz.
    n_frames : int
        Marker frames of the recording.
    min_fz : float
        Below this total vertical force (N) the COP is undefined (NaN).
    """
    step = int(round(analog_rate / marker_rate))
    forces, moments = [], []
    for plate in plates:
        force, moment = plate_wrench(plate)
        forces.append(lowpass(force, analog_rate, cutoff)[::step])
        moments.append(lowpass(moment, analog_rate, cutoff)[::step])
    n = min(n_frames, min(f.shape[0] for f in forces))
    plate_force = np.full((n_frames, len(plates), 3), np.nan)
    plate_force[:n] = np.stack([f[:n] for f in forces], axis=1)
    total_moment = np.full((n_frames, 3), np.nan)
    total_moment[:n] = np.sum([m[:n] for m in moments], axis=0)
    total_force = plate_force.sum(axis=1)

    heights = np.array([p.center[2] for p in plates])
    fz_plates = plate_force[:, :, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        h = (fz_plates * heights).sum(axis=1) / fz_plates.sum(axis=1)
        fx, fy, fz = total_force.T
        cop = np.column_stack([(h * fx - total_moment[:, 1]) / fz, (total_moment[:, 0] + h * fy) / fz, h])
    cop[~(np.abs(fz) >= min_fz)] = np.nan
    return GroundReaction(plate_force, total_force, cop)
