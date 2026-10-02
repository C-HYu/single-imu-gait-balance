"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/imu_files.py
Description : Reading the sacral sensor from an IMU recording, in sensor axes:
              - .mtb  Xsens MT Manager log (needs the Xsens Device API, see xsens.py);
              - .mat  one struct per sensor, named by the sensor ID, with fields
                      Acc (m/s^2) and Gyro (rad/s), n x 3 each;
              - .csv  one sensor: columns acc_x_m_s2, acc_y_m_s2, acc_z_m_s2,
                      gyro_x_rad_s, gyro_y_rad_s, gyro_z_rad_s (the layout
                      written by `gait-balance mtb2csv`).
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-02
Last updated: 2026-10-02
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

IMU_SUFFIXES = (".mtb", ".mat", ".csv")
CSV_COLUMNS = ["acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]


def _field(struct, name: str):
    """Struct field by name, ignoring case (Acc, acc, ACC)."""
    for key in getattr(struct, "_fieldnames", []):
        if key.lower() == name.lower():
            return getattr(struct, key)
    return None


def read_sensors(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every sensor in an IMU file: {sensor ID: (acc m/s^2, gyro rad/s)}, sensor axes."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mtb":
        from .xsens import read_mtb

        return {sensor: (d["acc"], d["gyro"]) for sensor, d in read_mtb(path).items()}
    if suffix == ".mat":
        sensors = {}
        for name, value in sio.loadmat(path, squeeze_me=True, struct_as_record=False).items():
            acc, gyro = _field(value, "Acc"), _field(value, "Gyro")
            if not name.startswith("__") and acc is not None and gyro is not None:
                sensors[name] = (np.asarray(acc, dtype=float), np.asarray(gyro, dtype=float))
        if not sensors:
            raise ValueError(f"{path.name}: no struct with fields Acc and Gyro")
        return sensors
    if suffix == ".csv":
        table = pd.read_csv(path)
        missing = [c for c in CSV_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"{path.name}: columns missing: {', '.join(missing)}")
        values = table[CSV_COLUMNS].to_numpy(dtype=float)
        return {path.stem: (values[:, :3], values[:, 3:])}
    raise ValueError(f"{path.name}: IMU files must be .mtb, .mat or .csv")


def read_sacral_imu(path: str | Path, sensor: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Acceleration and angular velocity of the sacral sensor, sensor axes.

    Parameters
    ----------
    sensor : str, optional
        Sensor ID (e.g. the Xsens device ID). Needed only when the file holds
        several sensors.

    Returns
    -------
    acc : ndarray, shape (n, 3), m/s^2
    gyro : ndarray, shape (n, 3), rad/s
    sensor : str
        ID of the sensor read.
    """
    sensors = read_sensors(path)
    if sensor:
        if sensor not in sensors:
            raise ValueError(f"{Path(path).name}: no sensor {sensor} (found {', '.join(sorted(sensors))})")
    elif len(sensors) == 1:
        sensor = next(iter(sensors))
    else:
        raise ValueError(f"{Path(path).name} holds several sensors ({', '.join(sorted(sensors))}): "
                         "give the ID of the sacral one")
    acc, gyro = (a.copy() for a in sensors[sensor])
    n = min(len(acc), len(gyro))
    acc, gyro = acc[:n], gyro[:n]
    for signal in (acc, gyro):  # isolated unreadable samples: linear interpolation
        bad = ~np.isfinite(signal).all(axis=1)
        if bad.all():
            raise ValueError(f"{Path(path).name}: sensor {sensor} has no valid samples")
        if bad.any():
            idx = np.arange(n)
            for j in range(3):
                signal[:, j] = np.interp(idx, idx[~bad], signal[~bad, j])
    return acc, gyro, sensor
