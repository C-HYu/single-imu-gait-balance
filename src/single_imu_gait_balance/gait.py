"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/gait.py
Description : Gait events: force-plate contacts, heel strikes from the heel
              marker (its most anterior position relative to the pelvis; Zeni
              et al., Gait Posture 2008;27:710-714), and one gait cycle from
              three or four floor plates walked over in a row.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-02
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .forceplate import GRAVITY


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


#: Tolerance (mm) beyond a plate edge for the heel marker (it sits behind and
#: above the heel's contact point) and for the forefoot point.
PLATE_MARGIN = {"HEE": 60.0, "MTH": 40.0}


def inside(plate, point: np.ndarray, margin: float) -> bool:
    """Is a point over the plate surface, within ``margin`` (mm) of its edges?"""
    low, high = plate.corners[:, :2].min(axis=0), plate.corners[:, :2].max(axis=0)
    return bool(np.all((point[:2] >= low - margin) & (point[:2] <= high + margin)))


def plate_cycle(plates: list, plate_fz: np.ndarray, body_mass: float, points: dict[str, np.ndarray],
                pelvis: np.ndarray, rate: float, strike_lead_s: float = 0.10) -> tuple[GaitCycle, str, dict]:
    """Gait cycle and reference limb from foot contacts on three or four floor plates in a row.

    With the plates ordered by first contact (p0, p1, p2, p3), the reference
    foot lands on p1 and the other foot on p0 and p2:
        start = first contact on p1,  CTO = last contact on p0 + 1,
        CHS   = first contact on p2,  TO  = last contact on p1 + 1,
        end   = first contact on p3 (four plates), or (three plates) the next
                heel strike of the reference heel found from the heel marker,
                corrected by the same method's offset from the plate contact
                at the start.
    Each foot must be on its plate early in stance and just before toe-off,
    and the body weight must be on the plates throughout the cycle.

    Parameters
    ----------
    plates : list of ForcePlate
        Floor plates.
    plate_fz : ndarray, shape (n_frames, n_plates)
        Vertical force of each plate at the marker frames, N.
    body_mass : float
        kg; contacts start at 3 % of body weight.
    points : dict of str to ndarray, shape (n_frames, 3)
        Needs LHEE, RHEE, LMTH and RMTH (heels and forefeet), mm.
    pelvis : ndarray, shape (n_frames, 3)
        A pelvis point (for the heel strikes with three plates), mm.
    rate : float
        Marker frame rate, Hz.
    strike_lead_s : float
        Largest lead of the heel-marker strike over the plate contact (three
        plates); a larger lead means that the heel landed beside the plate.

    Returns
    -------
    cycle, side ("left" or "right"), and details for the record.
    """
    def frames(seconds: float) -> int:
        return int(round(seconds * rate))

    weight = body_mass * GRAVITY
    contacts = plate_contacts(plate_fz, 0.03 * weight)
    order = sorted((i for i, c in enumerate(contacts) if c is not None), key=lambda i: contacts[i][0])[:4]
    if len(order) < 3:
        raise ValueError(f"need foot contacts on three or four force plates, found {len(order)}")
    for k in order:
        first, last = contacts[k]
        if (np.abs(plate_fz[first:last, k]) > 0.03 * weight).mean() < 0.95:
            raise ValueError("a plate was loaded twice (two foot contacts)")
    (a0, b0), (a1, b1), (a2, b2) = (contacts[i] for i in order[:3])
    p0, p1, p2 = (plates[i] for i in order[:3])
    n_frames = len(plate_fz)

    def foot_on(plate, side: str, frame: int) -> bool:
        frame = min(max(frame, 0), n_frames - 1)
        return all(inside(plate, points[f"{side}{m}"][frame], margin) for m, margin in PLATE_MARGIN.items()
                   if np.isfinite(points[f"{side}{m}"][frame]).all())

    # 80 ms after a strike (foot flat) and just before toe-off the two feet are
    # on different plates; at mid-stance the swinging foot passes over the plate.
    early, late = frames(0.08), frames(0.03)
    on = [s for s in "LR" if foot_on(p1, s, a1 + early)]
    if len(on) != 1:
        raise ValueError(f"cannot tell which foot is on the second plate ({on or 'none'})")
    ref, other = on[0], "R" if on[0] == "L" else "L"
    if not (foot_on(p0, other, b0 - late) and foot_on(p2, other, a2 + early) and foot_on(p1, ref, b1 - late)):
        raise ValueError("a foot is not fully on its plate")

    details: dict = {}
    if len(order) == 4:
        a3 = contacts[order[3]][0]
        if not foot_on(plates[order[3]], ref, a3 + early):
            raise ValueError("the reference foot is not fully on the fourth plate")
        cycle = GaitCycle(start=a1, cto=b0, chs=a2, to=b1, end=a3)
    else:
        forward = p2.center - p0.center
        forward[2] = 0.0
        forward /= np.linalg.norm(forward)
        strikes = {s: heel_strikes(points[f"{s}HEE"], pelvis, forward, min_distance=frames(0.4)) for s in "LR"}

        def strike_bias(side: str, plate_start: int) -> int:
            near = strikes[side][np.abs(strikes[side] - plate_start) <= frames(0.15)]
            if near.size == 0:
                raise ValueError(f"no {side} heel strike near its plate contact")
            bias = int(near[np.argmin(np.abs(near - plate_start))]) - plate_start
            if not -frames(strike_lead_s) <= bias <= frames(0.03):
                raise ValueError(f"{side} heel strike {bias:+d} frames from its plate contact (heel beside the plate)")
            return bias

        bias, bias_other = strike_bias(ref, a1), strike_bias(other, a2)
        later = strikes[ref][strikes[ref] > b1]
        if later.size == 0:
            raise ValueError("the next heel strike of the reference foot is not in the recording")
        cycle = GaitCycle(start=a1, cto=b0, chs=a2, to=b1, end=int(later[0]) - bias)
        details = {"strike_bias": bias, "strike_bias_other": bias_other}
    if not cycle.is_ordered():
        raise ValueError(f"gait events out of order: {cycle}")
    if not 0.6 <= (cycle.end - cycle.start) / rate <= 2.5:
        raise ValueError(f"implausible cycle time {(cycle.end - cycle.start) / rate:.2f} s")
    total = np.abs(plate_fz[cycle.start:cycle.end - 1].sum(axis=1))
    if total.min() < 0.5 * weight:
        raise ValueError(f"body weight not on the plates during the cycle (min {total.min() / weight:.2f} BW)")
    return cycle, ("left" if ref == "L" else "right"), details


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
