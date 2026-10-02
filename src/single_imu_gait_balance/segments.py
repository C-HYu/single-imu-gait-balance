"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/segments.py
Description : Marker-based body model for laboratories that record a static
              (standing) calibration trial before the walking trials, as in
              Yu et al. (2023):
              - the marker set file (configs/markersets/*.json): which C3D
                label is which landmark, and which markers form rigid clusters;
              - the subject model from the static trial: mean marker positions
                of the quietest stretch, hip joint centers (Bell's regression)
                and ankle centers (midpoint of the malleoli);
              - segment optimization of a walking trial: every cluster is
                fitted as a rigid body to the measured markers, frame by frame,
                and its markers are replaced by their fitted positions, which
                also fills short marker gaps;
              - checks of the marker labels (auto-labelled data contain swaps
                that give a wrong COM without any error).
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-02
Last updated: 2026-10-02
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rigid import fit_rigid_robust

MARKERSETS = Path(__file__).resolve().parents[2] / "configs" / "markersets"
#: Landmarks every marker set must provide (L and R): the points of the COM
#: model (com.REQUIRED_POINTS, with the knee center from the epicondyles LFC
#: and MFC) plus the heel and the pelvis markers.
REQUIRED = tuple(f"{s}{p}" for s in "LR" for p in ("TRO", "LFC", "MFC", "MMA", "LMA", "MTH", "SAP", "HEE")) + (
    "LASI", "RASI", "LPSI", "RPSI")
PELVIS = ("LASI", "RASI", "LPSI", "RPSI")
#: Points computed from markers: name -> markers it is computed from.
DERIVED = {"RHJC": PELVIS, "LHJC": PELVIS, "RAJC": ("RLMA", "RMMA"), "LAJC": ("LLMA", "LMMA")}
#: Residual (mm) above which a cluster marker is left out of a frame's fit
#: (well-labelled markers stay below about 6 mm).
OUTLIER_LIMIT = 15.0


@dataclass
class MarkerSet:
    """Contents of a marker set file.

    Attributes
    ----------
    labels : dict of str to str
        Landmark name used here -> marker label in the C3D files.
    clusters : dict of str to list of str
        Segment -> landmarks (and derived points) fitted together as a rigid
        body, in fitting order. May be empty (no segment optimization).
    carried : dict of str to list of str
        Segment -> landmarks that move with it but are not used to fit it;
        only their missing frames are filled from the segment pose.
    """

    labels: dict[str, str]
    clusters: dict[str, list[str]]
    carried: dict[str, list[str]]

    def rename(self, markers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """C3D markers -> landmarks of this set (other markers are dropped)."""
        return {name: markers[label] for name, label in self.labels.items() if label in markers}


def load_markerset(name_or_path: str | Path) -> MarkerSet:
    """A marker set by name (configs/markersets/<name>.json) or path."""
    path = Path(name_or_path)
    if not path.is_file():
        path = MARKERSETS / f"{name_or_path}.json"
    if not path.is_file():
        available = ", ".join(p.stem for p in sorted(MARKERSETS.glob("*.json")))
        raise FileNotFoundError(f"marker set {name_or_path!r} not found (available: {available})")
    data = json.loads(path.read_text(encoding="utf-8"))
    markerset = MarkerSet(data["labels"], data.get("clusters", {}), data.get("carried", {}))
    missing = [m for m in REQUIRED if m not in markerset.labels]
    if missing:
        raise ValueError(f"{path.name}: the labels list lacks {', '.join(missing)}")
    return markerset


# =============================================================================
# Subject model from the static trial
# =============================================================================


def hip_joint_centers(lasi: np.ndarray, rasi: np.ndarray, lpsi: np.ndarray,
                      rpsi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right and left hip joint centers by Bell's regression.

    In the pelvis frame (x anterior, y superior, z to the right) the center
    lies 19 % posterior, 30 % inferior and 36 % lateral of the inter-ASIS
    distance from the mid-ASIS point (Bell AL, Pedersen DR, Brand RA,
    J Biomech 1990;23:617-621).

    Parameters
    ----------
    lasi, rasi, lpsi, rpsi : ndarray, shape (3,) or (n_frames, 3)
        mm.
    """
    mid_asis = (lasi + rasi) / 2
    x = mid_asis - (lpsi + rpsi) / 2
    y = np.cross(rasi - lpsi, lasi - rpsi)
    z = np.cross(x, y)
    x = np.cross(y, z)
    x, y, z = (v / np.linalg.norm(v, axis=-1, keepdims=True) for v in (x, y, z))
    width = np.linalg.norm(lasi - rasi, axis=-1)[..., None]
    base = mid_asis - 0.19 * width * x - 0.30 * width * y
    return base + 0.36 * width * z, base - 0.36 * width * z


def derived_point(name: str, points: dict[str, np.ndarray]) -> np.ndarray:
    """A point of :data:`DERIVED` from its markers."""
    if name.endswith("HJC"):
        right, left = hip_joint_centers(points["LASI"], points["RASI"], points["LPSI"], points["RPSI"])
        return right if name[0] == "R" else left
    side = name[0]
    return (points[f"{side}LMA"] + points[f"{side}MMA"]) / 2


@dataclass
class SubjectModel:
    """Static positions (mm) of the landmarks and derived points of one subject.

    They serve as the local coordinates of every cluster: the fitted poses map
    these points onto the measured ones, so no anatomical frame is needed.
    """

    points: dict[str, np.ndarray]
    static_frames: np.ndarray
    body_mass: float


def quiet_frames(markers: dict[str, np.ndarray], half_window: int = 10) -> np.ndarray:
    """The quietest fully visible stretch of a standing trial (2 x half_window + 1 frames).

    Its center is the frame where the most markers are visible and, among
    those, the markers move least.
    """
    stack = np.stack(list(markers.values()), axis=1)  # (n_frames, n_markers, 3)
    visible = np.isfinite(stack).all(axis=2).sum(axis=1)
    speed = np.linalg.norm(np.diff(stack, axis=0), axis=2)
    with warnings.catch_warnings():  # frames without any marker give NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmean(np.vstack([speed, speed[-1:]]), axis=1)
    kernel = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    smooth_visible = np.convolve(visible, kernel, mode="same")
    smooth_speed = np.convolve(np.nan_to_num(speed, nan=1e6), kernel, mode="same")
    best = smooth_visible >= smooth_visible.max() - 1e-9
    center = int(np.flatnonzero(best)[np.argmin(smooth_speed[best])])
    return np.arange(max(0, center - half_window), min(len(stack), center + half_window + 1))


def calibrate(markers: dict[str, np.ndarray], frames: np.ndarray, body_mass: float) -> SubjectModel:
    """Subject model: mean landmark positions over the static frames plus the derived points.

    Parameters
    ----------
    markers : dict of str to ndarray, shape (n_frames, 3)
        Static-trial landmarks (MarkerSet.rename), mm, NaN where missing.
    """
    points = {}
    with warnings.catch_warnings():  # markers never seen in these frames give NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, trajectory in markers.items():
            mean = np.nanmean(trajectory[frames], axis=0)
            if np.isfinite(mean).all():
                points[name] = mean
    missing = [m for m in REQUIRED if m not in points]
    if missing:
        raise ValueError(f"static trial: markers not visible: {', '.join(missing)}")
    for name in DERIVED:
        points[name] = derived_point(name, points)
    return SubjectModel(points, np.asarray(frames), body_mass)


# =============================================================================
# Walking trial
# =============================================================================


def segment_kinematics(model: SubjectModel, markers: dict[str, np.ndarray], markerset: MarkerSet,
                       outlier_limit: float | None = OUTLIER_LIMIT) -> dict[str, np.ndarray]:
    """Landmarks of a walking trial after segment optimization, plus the joint centers.

    Each cluster is fitted (in the order of the marker set) to the measured
    markers of every frame and its markers are replaced by their fitted
    positions. A derived point joins the fit of later clusters once the
    cluster that holds all its markers has been fitted (hip centers from the
    pelvis, ankle centers from the shanks). A marker whose residual exceeds
    ``outlier_limit`` (mm) is left out of that frame's fit.

    Returns
    -------
    dict of str to ndarray, shape (n_frames, 3)
        All landmarks, the hip and ankle centers (RHJC, LHJC, RAJC, LAJC) and
        the knee centers RKJC and LKJC (midpoint of the epicondyles), mm.
    """
    n_frames = next(iter(markers.values())).shape[0]
    out = {k: v.copy() for k, v in markers.items()}
    for segment, names in markerset.clusters.items():
        names = [n for n in names if n in model.points]
        local = np.stack([model.points[n] for n in names])
        measured = np.stack([out.get(n, np.full((n_frames, 3), np.nan)) for n in names], axis=1)
        rotation, translation, _ = fit_rigid_robust(local, measured, outlier_limit or np.inf)

        def place(point: str) -> np.ndarray:
            return np.einsum("fij,j->fi", rotation, model.points[point]) + translation

        for name in names:
            out[name] = place(name)
        for name in markerset.carried.get(segment, ()):
            if name in model.points:
                current = out.get(name, np.full((n_frames, 3), np.nan)).copy()
                gap = ~np.isfinite(current).all(axis=1)
                current[gap] = place(name)[gap]
                out[name] = current
        for name, sources in DERIVED.items():
            if name not in out and set(sources) <= set(names):
                out[name] = place(name)
    for name in DERIVED:  # not carried by any cluster: from the markers of each frame
        if name not in out:
            out[name] = derived_point(name, out)
    for side in "RL":
        out[f"{side}KJC"] = (out[f"{side}LFC"] + out[f"{side}MFC"]) / 2
    return out


def check_markers(model: SubjectModel, markers: dict[str, np.ndarray], frames: np.ndarray, markerset: MarkerSet,
                  residual_limit: float = 10.0, trunk_limit: float = 50.0, foot_limit: float = 20.0
                  ) -> tuple[list[str], list[str]]:
    """Check the labels of the markers used for the COM during ``frames``.

    - Residual: median distance of each cluster marker to its fitted position
      (outlier-robust fit); above ``residual_limit`` (mm) the label is wrong
      in much of the cycle.
    - Distances to the static trial: acromion to pelvis center
      (``trunk_limit``) and first metatarsal head to heel (``foot_limit``),
      median change in mm.

    Parameters
    ----------
    markers : dict of str to ndarray, shape (n_frames, 3)
        Gap-filled, filtered landmarks before segment optimization.

    Returns
    -------
    problems : list of str
        Reasons to reject the cycle (empty if it passes).
    notes : list of str
        Markers that the robust fit left out in more than 5 % of the frames.
    """
    problems, notes = [], []
    empty = np.full((len(frames), 3), np.nan)

    def now(name: str) -> np.ndarray:
        return markers[name][frames] if name in markers else empty

    for segment, names in markerset.clusters.items():
        names = [n for n in names if n in model.points and n in markers]  # derived points are not measured
        if len(names) < 3:
            continue
        local = np.stack([model.points[n] for n in names])
        measured = np.stack([now(n) for n in names], axis=1)
        rotation, translation, used = fit_rigid_robust(local, measured, OUTLIER_LIMIT)
        implied = np.einsum("fij,mj->fmi", rotation, local) + translation[:, None]
        visible = np.isfinite(measured).all(axis=2)
        error = np.where(used, np.linalg.norm(implied - np.nan_to_num(measured), axis=2), np.nan)
        for j, name in enumerate(names):
            if not visible[:, j].any():
                continue
            share = 1.0 - used[:, j].sum() / visible[:, j].sum()
            if share > 0.05:
                notes.append(f"{name} left out of the {segment} fit in {100 * share:.0f} % of frames")
            if np.isfinite(error[:, j]).any() and np.nanmedian(error[:, j]) > residual_limit:
                problems.append(f"{name}: fit residual {np.nanmedian(error[:, j]):.0f} mm ({segment})")

    pelvis_now = np.mean([now(n) for n in PELVIS], axis=0)
    pelvis_static = np.mean([model.points[n] for n in PELVIS], axis=0)
    for name, reference, limit in (("LSAP", None, trunk_limit), ("RSAP", None, trunk_limit),
                                   ("LMTH", "LHEE", foot_limit), ("RMTH", "RHEE", foot_limit)):
        ref_now = pelvis_now if reference is None else now(reference)
        ref_static = pelvis_static if reference is None else model.points[reference]
        if not np.isfinite(now(name)).all(axis=1).any():
            if reference is None:  # metatarsal heads are filled from the foot pose; acromions cannot be
                problems.append(f"{name}: not visible during the cycle")
            continue
        change = np.linalg.norm(now(name) - ref_now, axis=1) - np.linalg.norm(model.points[name] - ref_static)
        if np.isfinite(change).any() and np.nanmedian(np.abs(change)) > limit:
            problems.append(f"{name}: distance to {reference or 'the pelvis'} changed by "
                            f"{np.nanmedian(np.abs(change)):.0f} mm from the static trial")
    return problems, notes
