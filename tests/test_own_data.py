"""
Single IMU Gait Balance
-----------------------
File        : tests/test_own_data.py
Description : Tests on invented data for using one's own recordings: set
              division, IMU files, import of processed arrays, static
              calibration with segment optimization and label checks, and the
              gait cycle from four force plates.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-02
Last updated: 2026-10-02
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
from scipy.spatial.transform import Rotation

from single_imu_gait_balance.c3d import ForcePlate
from single_imu_gait_balance.cycles import load_cycles
from single_imu_gait_balance.datasets.arrays import import_arrays
from single_imu_gait_balance.gait import GaitCycle, plate_cycle
from single_imu_gait_balance.imu_files import CSV_COLUMNS, read_sacral_imu
from single_imu_gait_balance.segments import calibrate, check_markers, load_markerset, segment_kinematics
from single_imu_gait_balance.splits import read_split_file, set_from_list, split_by_trial


def test_split_by_trial_and_split_file(tmp_path):
    sets = split_by_trial(100, (0.8, 0.1, 0.1), seed=42)
    assert [sets.count(s) for s in ("training", "validation", "testing")] == [80, 10, 10]
    assert split_by_trial(100, (0.9, 0.0, 0.1)).count("validation") == 0
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["G1/trial01"], "val": ["trial02"], "test": ["trial03"]}))
    listed = read_split_file(path)
    assert set_from_list("trial01", listed) == "training"
    assert set_from_list("S1/trial02", listed) == "validation"
    assert set_from_list("trial04", listed) is None


def test_imu_files_mat_and_csv(tmp_path):
    rng = np.random.default_rng(0)
    acc, gyro = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    sio.savemat(tmp_path / "rec.mat", {"S1": {"Acc": acc, "Gyro": gyro}, "S2": {"Acc": acc * 2, "Gyro": gyro}})
    a, g, sensor = read_sacral_imu(tmp_path / "rec.mat", "S2")
    np.testing.assert_allclose(a, acc * 2)
    assert sensor == "S2"
    with pytest.raises(ValueError, match="several sensors"):
        read_sacral_imu(tmp_path / "rec.mat")
    pd.DataFrame(np.hstack([acc, gyro]), columns=CSV_COLUMNS).to_csv(tmp_path / "rec.csv", index=False)
    a, g, _ = read_sacral_imu(tmp_path / "rec.csv")
    np.testing.assert_allclose(g, gyro)


def test_import_arrays(tmp_path):
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 501)
    for i in range(20):
        folder = tmp_path / "raw" / f"S{i % 2}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / f"c{i:02d}_imu.npz", data=rng.normal(size=(501, 6)))
        ia = np.column_stack([5 * np.sin(2 * np.pi * t), 2 * np.cos(2 * np.pi * t), rng.normal(size=(501, 2))])
        np.savez(folder / f"c{i:02d}_ia.npz", data=ia, cycle_time_s=1.1)
    spec = {"imu_files": "*/*_imu.npz", "imu_key": "data", "ia_file": [["_imu.npz", "_ia.npz"]], "ia_key": "data",
            "cycle_time_key": "cycle_time_s", "frame_rate_hz": 100, "fields_from_path": {"subject": 0},
            "folder": "{subject}/{name}"}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    table = import_arrays(tmp_path / "raw", tmp_path / "spec.json", tmp_path / "out", split="80/10/10")
    assert (table.status == "used").all()
    training = load_cycles(tmp_path / "out" / "training")
    assert len(training) == 16 and set(training.meta.subject) == {"S0", "S1"}
    assert training.ia.shape == (16, 101, 2) and np.allclose(training.cycle_time, 1.1)


def _standing_markers() -> dict[str, np.ndarray]:
    """An invented standing posture (x forward, y left, z up), mm."""
    sides = {"L": 1, "R": -1}
    layout = {"ASI": (50, 120, 950), "PSI": (-100, 50, 970), "SAP": (0, 180, 1450), "TRO": (0, 170, 850),
              "THI": (20, 190, 700), "LFC": (0, 140, 500), "MFC": (0, 60, 500), "TT": (40, 100, 450),
              "SHA": (10, 130, 300), "LMA": (0, 120, 80), "MMA": (0, 60, 90), "HEE": (-60, 90, 40),
              "FOO": (60, 90, 60), "TOE": (150, 90, 30), "MTH": (120, 60, 30)}
    return {f"{s}{name}": np.array([x, sign * y, z], dtype=float)
            for s, sign in sides.items() for name, (x, y, z) in layout.items()}


def test_static_calibration_segment_fit_and_label_checks():
    markerset = load_markerset("yu2023")
    standing = _standing_markers()
    static = {k: np.tile(v, (40, 1)) for k, v in standing.items()}
    model = calibrate(static, np.arange(40), body_mass=70.0)
    assert model.points["RHJC"][1] < 0 < model.points["LHJC"][1]  # right hip on the right (-y)

    n = 120  # the whole body turns a little and moves forward
    rot = Rotation.from_euler("z", np.linspace(0, 0.3, n)[:, None]).as_matrix()
    shift = np.column_stack([np.linspace(0, 1500, n), np.zeros(n), np.zeros(n)])
    truth = {k: np.einsum("fij,j->fi", rot, v) + shift for k, v in standing.items()}
    walking = {k: v.copy() for k, v in truth.items()}
    walking["RTHI"][30:60] = np.nan
    walking["LMTH"][10:20] = np.nan
    points = segment_kinematics(model, walking, markerset)
    np.testing.assert_allclose(points["RTHI"], truth["RTHI"], atol=1e-6)
    np.testing.assert_allclose(points["LMTH"], truth["LMTH"], atol=1e-6)
    np.testing.assert_allclose(points["RKJC"], (truth["RLFC"] + truth["RMFC"]) / 2, atol=1e-6)
    assert check_markers(model, walking, np.arange(n), markerset)[0] == []

    walking["LSAP"] = truth["LASI"].copy()  # an acromion label on a pelvis marker
    problems, _ = check_markers(model, walking, np.arange(n), markerset)
    assert any(p.startswith("LSAP") for p in problems)


def _plate(x0: float) -> ForcePlate:
    """A 500 x 400 mm floor plate from x0 to x0 + 500 (corners in the C3D order)."""
    corners = np.array([[x0 + 500, 200, 0], [x0, 200, 0], [x0, -200, 0], [x0 + 500, -200, 0]], dtype=float)
    return ForcePlate(corners, np.zeros(3), np.zeros((10, 3)), np.zeros((10, 3)))


def test_gait_cycle_from_four_plates():
    n, rate = 260, 100.0
    plates = [_plate(500.0 * k) for k in range(4)]
    fz = np.zeros((n, 4))
    for k, (first, last) in enumerate(((0, 61), (50, 121), (110, 181), (170, 241))):
        fz[first:last, k] = 700.0
    frames = np.arange(n)
    left_x = np.where(frames < 85, 250.0, 1250.0)   # left foot on plates 1 and 3
    right_x = np.where(frames < 145, 750.0, 1750.0)  # right foot on plates 2 and 4
    points = {}
    for side, x, y in (("L", left_x, 100.0), ("R", right_x, -100.0)):
        points[f"{side}HEE"] = np.column_stack([x - 100, np.full(n, y), np.full(n, 40.0)])
        points[f"{side}MTH"] = np.column_stack([x + 100, np.full(n, y), np.full(n, 30.0)])
    cycle, side, _ = plate_cycle(plates, fz, 70.0, points, np.zeros((n, 3)), rate)
    assert cycle == GaitCycle(start=50, cto=61, chs=110, to=121, end=170) and side == "right"
