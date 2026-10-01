"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/rigid.py
Description : Least-squares rigid-body fit of a marker cluster (SVD method,
              Söderkvist & Wedin, J Biomech 1993;26:1473-1477), and filling of
              a point that moves with a cluster.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import numpy as np


def fit_rigid(local: np.ndarray, measured: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation of a cluster in every frame.

    Missing markers (NaN) get zero weight; frames with fewer than three
    visible markers have no solution (NaN).

    Parameters
    ----------
    local : ndarray, shape (m, 3)
        Cluster in its reference position.
    measured : ndarray, shape (n_frames, m, 3)
        Measured positions, NaN where missing.

    Returns
    -------
    rotation : ndarray, shape (n_frames, 3, 3)
    translation : ndarray, shape (n_frames, 3)
        ``measured ~= local @ rotation.T + translation``.
    """
    weight = np.isfinite(measured).all(axis=2).astype(float)  # (n, m)
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
