"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/c3d.py
Description : Read marker trajectories and force-plate signals from a C3D file
              (ezc3d).
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ezc3d
import numpy as np


@dataclass
class ForcePlate:
    """Raw signals and geometry of one force plate (C3D FORCE_PLATFORM group).

    Attributes
    ----------
    corners : ndarray, shape (4, 3)
        Surface corners in the laboratory frame, mm. By the C3D convention
        corner 1 lies in the +x/+y quadrant of the plate frame, corner 2 in
        -x/+y, corner 3 in -x/-y and corner 4 in +x/-y.
    origin : ndarray, shape (3,)
        Vector from the centre of the plate surface to the transducer origin,
        in plate coordinates, mm.
    force : ndarray, shape (n_samples, 3)
        Force in plate coordinates, N (analog rate).
    moment : ndarray, shape (n_samples, 3)
        Moment about the transducer origin in plate coordinates, N mm.
    """

    corners: np.ndarray
    origin: np.ndarray
    force: np.ndarray
    moment: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        """Plate-to-laboratory rotation matrix (columns = plate axes)."""
        c = self.corners
        x = (c[0] + c[3]) / 2 - (c[1] + c[2]) / 2
        y = (c[0] + c[1]) / 2 - (c[2] + c[3]) / 2
        x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
        z = np.cross(x, y)
        return np.column_stack([x, y, z / np.linalg.norm(z)])

    @property
    def center(self) -> np.ndarray:
        """Centre of the plate surface in the laboratory frame, mm."""
        return self.corners.mean(axis=0)

    @property
    def transducer_origin(self) -> np.ndarray:
        """Point about which ``moment`` is measured, laboratory frame, mm."""
        return self.center + self.rotation @ self.origin


@dataclass
class C3DRecording:
    """Markers and force plates of one C3D file.

    Attributes
    ----------
    path : Path
    marker_rate, analog_rate : float
        Hz.
    markers : dict of str to ndarray, shape (n_frames, 3)
        Marker trajectories in mm, NaN where missing. Any "subject:" prefix
        is removed from the labels.
    plates : list of ForcePlate
    """

    path: Path
    marker_rate: float
    analog_rate: float
    markers: dict[str, np.ndarray]
    plates: list[ForcePlate] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return next(iter(self.markers.values())).shape[0] if self.markers else 0

    @property
    def samples_per_frame(self) -> int:
        """Analog samples per marker frame (e.g. 1000 Hz / 100 Hz = 10)."""
        return int(round(self.analog_rate / self.marker_rate))


def _values(group: dict, name: str, default=None):
    return group[name]["value"] if name in group else default


def read_c3d(path: str | Path) -> C3DRecording:
    """Read markers (mm) and force plates of type 2 or 4 (Fx, Fy, Fz, Mx, My, Mz)."""
    path = Path(path)
    c3d = ezc3d.c3d(str(path))
    params = c3d["parameters"]

    point = params["POINT"]
    units = str((_values(point, "UNITS", []) or ["mm"])[0]).strip().lower() or "mm"
    scale = {"mm": 1.0, "cm": 10.0, "m": 1000.0}.get(units)
    if scale is None:
        raise ValueError(f"{path.name}: unsupported POINT:UNITS {units!r}")
    labels = list(point["LABELS"]["value"])
    k = 2
    while f"LABELS{k}" in point:  # more than 255 points continue in LABELS2, LABELS3, ...
        labels += list(point[f"LABELS{k}"]["value"])
        k += 1
    labels = [str(label).split(":")[-1].strip() for label in labels]
    points = c3d["data"]["points"][:3].transpose(2, 1, 0) * scale  # (n_frames, n_markers, 3)
    markers = {}
    for i, label in enumerate(labels):
        if label and not label.startswith("*") and label not in markers:
            markers[label] = points[:, i, :].astype(float)

    analog_rate = float(params["ANALOG"]["RATE"]["value"][0])
    analogs = c3d["data"]["analogs"][0]  # (n_channels, n_samples), in N and N mm
    plates = []
    fp = params.get("FORCE_PLATFORM", {})
    n_plates = int(_values(fp, "USED", [0])[0]) if fp else 0
    if n_plates:
        types = np.atleast_1d(_values(fp, "TYPE"))
        corners = np.asarray(_values(fp, "CORNERS"), dtype=float).reshape(3, 4, n_plates, order="F")
        origins = np.asarray(_values(fp, "ORIGIN"), dtype=float).reshape(3, n_plates, order="F")
        channels = np.asarray(_values(fp, "CHANNEL"), dtype=int).reshape(-1, n_plates, order="F")
        for i in range(n_plates):
            if types[i] not in (2, 4):
                raise ValueError(f"{path.name}: force plate type {types[i]} is not supported")
            signals = analogs[channels[:6, i] - 1].T  # (n_samples, 6)
            if types[i] == 4:
                matrix = np.asarray(_values(fp, "CAL_MATRIX"), dtype=float).reshape(6, 6, -1, order="F")[:, :, i]
                signals = signals @ matrix.T
            plates.append(ForcePlate(corners[:, :, i].T * scale, origins[:, i] * scale,
                                     signals[:, :3], signals[:, 3:6]))
    return C3DRecording(path, float(point["RATE"]["value"][0]), analog_rate, markers, plates)
