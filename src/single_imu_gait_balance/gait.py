"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/gait.py
Description : Gait events: force-plate contacts and heel strikes from the heel
              marker (its most anterior position relative to the pelvis; Zeni
              et al., Gait Posture 2008;27:710-714).
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class GaitCycle:
    """Events of one gait cycle as marker-frame indices.

    ``start`` and ``end`` are successive heel strikes of the reference foot;
    ``cto``, ``chs`` and ``to`` are the contralateral toe-off, contralateral
    heel strike and reference toe-off.
    """

    start: int
    cto: int
    chs: int
    to: int
    end: int

    @property
    def frames(self) -> np.ndarray:
        """Frames of the cycle, both heel strikes included."""
        return np.arange(self.start, self.end + 1)

    def percent(self) -> dict[str, float]:
        """CTO, CHS and TO as % of the cycle."""
        length = self.end - self.start
        return {k: 100.0 * (getattr(self, k) - self.start) / length for k in ("cto", "chs", "to")}

    def is_ordered(self) -> bool:
        return self.start < self.cto < self.chs < self.to < self.end


def plate_contacts(plate_fz: np.ndarray, threshold: float) -> list[tuple[int, int] | None]:
    """(first loaded frame, first unloaded frame after the last loaded one) of each plate.

    Parameters
    ----------
    plate_fz : ndarray, shape (n_frames, n_plates)
        Vertical force of each plate, N.
    threshold : float
        Loading threshold, N.
    """
    contacts = []
    for fz in np.abs(np.nan_to_num(plate_fz)).T:
        loaded = np.flatnonzero(fz > threshold)
        contacts.append((int(loaded[0]), int(loaded[-1]) + 1) if loaded.size else None)
    return contacts


def heel_strikes(heel: np.ndarray, pelvis: np.ndarray, forward: np.ndarray,
                 min_distance: int = 40, prominence: float = 100.0) -> np.ndarray:
    """Frames where the heel is most anterior relative to the pelvis.

    Parameters
    ----------
    heel, pelvis : ndarray, shape (n_frames, 3)
        Heel marker and a pelvis point (e.g. the center of a sacral cluster), mm.
    forward : ndarray, shape (3,)
        Unit vector of the walking direction.
    min_distance : int
        Fewest frames between two strikes of the same foot.
    prominence : float
        mm.
    """
    signal = (heel - pelvis) @ forward
    peaks, _ = find_peaks(np.nan_to_num(signal, nan=-1e9), distance=min_distance, prominence=prominence)
    return np.asarray([p for p in peaks if np.isfinite(signal[max(p - 3, 0):p + 4]).all()], dtype=int)
