"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/rigid.py
Description : Least-squares rigid-body fit of a marker cluster (SVD method,
              Söderkvist & Wedin, J Biomech 1993;26:1473-1477), and filling of
              a point that moves with a cluster.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-02
"""

from __future__ import annotations

import numpy as np


def fit_rigid(local: np.ndarray, measured: np.ndarray, used: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation of a cluster in every frame.

    Missing markers (NaN) get zero weight; frames with fewer than three
    visible markers have no solution (NaN).

    Parameters
    ----------
    local : ndarray, shape (m, 3)
        Cluster in its reference position.
    measured : ndarray, shape (n_frames, m, 3)
        Measured positions, NaN where missing.
    used : ndarray of bool, shape (n_frames, m), optional
        Markers to fit in each frame (default: every visible one).

    Returns
    -------
    rotation : ndarray, shape (n_frames, 3, 3)
    translation : ndarray, shape (n_frames, 3)
        ``measured ~= local @ rotation.T + translation``.
    """
    visible = np.isfinite(measured).all(axis=2)
    weight = (visible if used is None else visible & used).astype(float)  # (n, m)
    count = weight.sum(axis=1)
    g = np.nan_to_num(measured)
    w = weight[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        local_mean = (w * local[None]).sum(axis=1) / count[:, None]
        g_mean = (w * g).sum(axis=1) / count[:, None]
    lc = (local[None] - local_mean[:, None]) * w
    gc = (g - g_mean[:, None]) * w
    u, _, vt = np.linalg.svd(np.nan_to_num(np.einsum("fmi,fmj->fij", lc, gc)))
    d = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", vt.transpose(0, 2, 1), u.transpose(0, 2, 1))))
    correction = np.stack([np.ones_like(d), np.ones_like(d), d], axis=1)
    rotation = np.einsum("fji,fj,fkj->fik", vt, correction, u)  # V diag(1, 1, d) U^T
    translation = g_mean - np.einsum("fij,fj->fi", rotation, local_mean)
    bad = count < 3
    rotation[bad], translation[bad] = np.nan, np.nan
    return rotation, translation


def fit_rigid_robust(local: np.ndarray, measured: np.ndarray, outlier_limit: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`fit_rigid` that leaves out a marker far from the fitted cluster.

    While the largest residual of a frame exceeds ``outlier_limit`` (mm) and
    more than three markers remain, that marker is dropped and the frame is
    fitted again. This removes a swapped or misplaced label from the solution.

    Returns
    -------
    rotation, translation : as :func:`fit_rigid`
    used : ndarray of bool, shape (n_frames, m)
        Markers that took part in the final fit.
    """
    used = np.isfinite(measured).all(axis=2)
    while True:
        rotation, translation = fit_rigid(local, measured, used)
        implied = np.einsum("fij,mj->fmi", rotation, local) + translation[:, None]
        residual = np.where(used, np.linalg.norm(implied - np.nan_to_num(measured), axis=2), 0.0)
        rows = np.flatnonzero((residual.max(axis=1) > outlier_limit) & (used.sum(axis=1) > 3))
        if rows.size == 0:
            return rotation, translation, used
        used[rows, np.argmax(residual[rows], axis=1)] = False


def carried_point(point: np.ndarray, cluster: np.ndarray, min_frames: int = 10) -> np.ndarray:
    """Fill the missing frames of a point that moves rigidly with a marker cluster.

    The point's position in the cluster frame is the median over the frames
    where both are seen; missing frames are rebuilt from the cluster pose.

    Parameters
    ----------
    point : ndarray, shape (n_frames, 3)
    cluster : ndarray, shape (n_frames, m, 3)
    """
    seen = np.isfinite(point).all(axis=1) & np.isfinite(cluster).all(axis=(1, 2))
    if seen.sum() < min_frames:
        return point
    rotation, translation = fit_rigid(cluster[np.flatnonzero(seen)[0]], cluster)
    local = np.einsum("fji,fj->fi", rotation[seen], point[seen] - translation[seen])  # R^T (p - t)
    rebuilt = np.einsum("fij,j->fi", rotation, np.median(local, axis=0)) + translation
    out = point.copy()
    gap = ~np.isfinite(out).all(axis=1)
    out[gap] = rebuilt[gap]
    return out
