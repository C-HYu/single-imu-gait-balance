"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/cycles.py
Description : The data format shared by every data set: one folder per gait
              cycle with
                cycle.npz         imu (101 x 6), ia (101 x 2), rcia (101 x 2)
                cycle.json        cycle_time_s plus descriptive fields
                ground_truth.csv  the IA and RCIA as text
                imu_input.csv     the IMU input as text
              and the in-memory set of many cycles used for training.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .signals import N_POINTS

IMU_COLUMNS = ["acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]
TRUTH_COLUMNS = ["sagittal_IA_deg", "frontal_IA_deg", "sagittal_RCIA_deg_s", "frontal_RCIA_deg_s"]


def write_cycle(folder: str | Path, imu: np.ndarray, ia: np.ndarray, rcia: np.ndarray, info: dict) -> Path:
    """Write one gait cycle.

    Parameters
    ----------
    imu : ndarray, shape (101, 6)
        Body axes (x anterior, y up, z right); acc m/s^2, gyro rad/s.
        Right-limb cycles mirrored.
    ia : ndarray, shape (101, 2)
        Sagittal and frontal IA, degrees.
    rcia : ndarray, shape (101, 2)
        Sagittal and frontal RCIA, degrees per second.
    info : dict
        Must contain ``cycle_time_s``; ``subject``, ``speed``, ``trial`` and
        anything else are kept for reference.
    """
    if "cycle_time_s" not in info:
        raise ValueError("info needs cycle_time_s")
    for name, array, width in (("imu", imu, 6), ("ia", ia, 2), ("rcia", rcia, 2)):
        if np.shape(array) != (N_POINTS, width):
            raise ValueError(f"{name} must have shape ({N_POINTS}, {width}), got {np.shape(array)}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    percent = np.linspace(0, 100, N_POINTS)
    np.savez_compressed(folder / "cycle.npz", imu=imu, ia=ia, rcia=rcia)
    (folder / "cycle.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    truth = pd.DataFrame(np.hstack([ia, rcia]), columns=TRUTH_COLUMNS)
    truth.insert(0, "gait_cycle_percent", percent)
    truth.round(6).to_csv(folder / "ground_truth.csv", index=False)
    signals = pd.DataFrame(imu, columns=IMU_COLUMNS)
    signals.insert(0, "gait_cycle_percent", percent)
    signals.round(6).to_csv(folder / "imu_input.csv", index=False)
    return folder


def find_cycles(root: str | Path) -> list[Path]:
    """Every cycle folder (one with cycle.npz and cycle.json) below ``root``, sorted."""
    root = Path(root)
    return sorted(p.parent for p in root.rglob("cycle.npz") if (p.parent / "cycle.json").is_file())


@dataclass
class CycleSet:
    """Many gait cycles in memory.

    Attributes
    ----------
    imu : ndarray, shape (n, 101, 6)
    ia, rcia : ndarray, shape (n, 101, 2)
    cycle_time : ndarray, shape (n,)
        s.
    meta : DataFrame, n rows
        ``name`` (folder relative to the set root) plus the fields of cycle.json.
    """

    imu: np.ndarray
    ia: np.ndarray
    rcia: np.ndarray
    cycle_time: np.ndarray
    meta: pd.DataFrame

    def __len__(self) -> int:
        return self.imu.shape[0]

    def subset(self, index) -> "CycleSet":
        index = np.asarray(index)
        return CycleSet(self.imu[index], self.ia[index], self.rcia[index], self.cycle_time[index],
                        self.meta.iloc[index].reset_index(drop=True))


def load_cycles(root: str | Path) -> CycleSet:
    """Read every cycle folder below ``root``."""
    root = Path(root)
    folders = find_cycles(root)
    if not folders:
        raise FileNotFoundError(f"no cycle folders (cycle.npz + cycle.json) below {root}")
    imu, ia, rcia, times, rows = [], [], [], [], []
    for folder in folders:
        with np.load(folder / "cycle.npz") as z:
            imu.append(z["imu"])
            ia.append(z["ia"])
            rcia.append(z["rcia"])
        info = json.loads((folder / "cycle.json").read_text(encoding="utf-8"))
        times.append(float(info["cycle_time_s"]))
        flat = {k: v for k, v in info.items() if not isinstance(v, (dict, list))}
        rows.append({"name": folder.relative_to(root).as_posix(), **flat})
    return CycleSet(np.stack(imu), np.stack(ia), np.stack(rcia), np.array(times), pd.DataFrame(rows))
