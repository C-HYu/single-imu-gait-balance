"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/com.py
Description : Whole-body center of mass (COM) from a 7-segment model: thighs,
              shanks, feet and head-arms-trunk, with Winter's segment mass
              fractions and COM positions (Winter DA, Biomechanics and Motor
              Control of Human Movement, 4th ed., Table 4.1).
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import numpy as np

#: (segment, proximal point, distal point, mass fraction, COM distance from the
#: proximal end as a fraction of the segment length). The mass fractions sum to 1.
SEGMENTS = (
    ("left_thigh", "LTRO", "LKJC", 0.100, 0.433),
    ("right_thigh", "RTRO", "RKJC", 0.100, 0.433),
    ("left_shank", "LKJC", "LMMA", 0.0465, 0.433),
    ("right_shank", "RKJC", "RMMA", 0.0465, 0.433),
    ("left_foot", "LLMA", "LMTH", 0.0145, 0.5),
    ("right_foot", "RLMA", "RMTH", 0.0145, 0.5),
    ("head_arms_trunk", "mid_TRO", "mid_SAP", 0.678, 0.626),
)

#: Points the model needs, for each side (prefix L or R):
#: TRO greater trochanter, KJC knee joint center, MMA / LMA medial / lateral
#: malleolus, MTH first metatarsal head, SAP acromion.
REQUIRED_POINTS = tuple(f"{s}{p}" for s in "LR" for p in ("TRO", "KJC", "MMA", "LMA", "MTH", "SAP"))


def body_com(points: dict[str, np.ndarray]) -> np.ndarray:
    """Whole-body COM.

    Parameters
    ----------
    points : dict of str to ndarray, shape (n_frames, 3)
        The points of :data:`REQUIRED_POINTS`, mm.

    Returns
    -------
    ndarray, shape (n_frames, 3)
        mm; NaN where a point is missing.
    """
    missing = [p for p in REQUIRED_POINTS if p not in points]
    if missing:
        raise KeyError(f"points missing for the COM model: {', '.join(missing)}")
    pts = dict(points)
    pts["mid_TRO"] = (points["LTRO"] + points["RTRO"]) / 2
    pts["mid_SAP"] = (points["LSAP"] + points["RSAP"]) / 2
    com = 0.0
    for _, proximal, distal, mass, ratio in SEGMENTS:
        com = com + mass * (pts[proximal] + ratio * (pts[distal] - pts[proximal]))
    return com
