"""
Single IMU Gait Balance
-----------------------
File        : tests/test_core.py
Description : Tests on invented data: gap filling, COM, rigid fit, heel
              strikes, IA and RCIA, IMU axes and mirroring, gait-cycle folders,
              a short training run, a short Bayesian search and partial zip
              extraction.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from single_imu_gait_balance.com import REQUIRED_POINTS, body_com
from single_imu_gait_balance.cycles import load_cycles, write_cycle
from single_imu_gait_balance.download import extract_local
from single_imu_gait_balance.experiment import load_parts, train_and_test, tune
from single_imu_gait_balance.gait import GaitCycle, heel_strikes
from single_imu_gait_balance.imu import XSENS_SACRUM, mirror_right_limb, to_body_axes
from single_imu_gait_balance.inclination import inclination_angles, progression_frame, rcia_from_ia
from single_imu_gait_balance.rigid import carried_point
from single_imu_gait_balance.signals import fill_gaps


def test_fill_gaps_restores_a_smooth_line():
    t = np.linspace(0, 1, 100)
    truth = np.column_stack([t, t ** 2, np.sin(t)])
    gappy = truth.copy()
    gappy[40:50] = np.nan
    np.testing.assert_allclose(fill_gaps(gappy, max_gap=15), truth, atol=1e-3)
    assert np.isnan(fill_gaps(gappy, max_gap=5)[45]).all()


def test_com_of_identical_points_is_that_point():
    points = {name: np.tile([1.0, 2.0, 3.0], (5, 1)) for name in REQUIRED_POINTS}
    np.testing.assert_allclose(body_com(points), np.tile([1.0, 2.0, 3.0], (5, 1)))


def test_carried_point_fills_gaps():
    n = 120
    cluster_local = np.array([[0, 0, 0], [100, 0, 0], [0, 80, 0], [0, 0, 60]], dtype=float)
    point_local = np.array([30.0, -90.0, -40.0])
    rot = Rotation.from_euler("xyz", np.column_stack([0.3 * np.sin(np.linspace(0, 6, n)),
                                                      0.1 * np.cos(np.linspace(0, 6, n)),
                                                      np.linspace(0, 0.5, n)])).as_matrix()
    shift = np.column_stack([np.linspace(0, 1500, n), np.zeros(n), np.full(n, 900.0)])
    cluster = np.einsum("fij,mj->fmi", rot, cluster_local) + shift[:, None]
    truth = np.einsum("fij,j->fi", rot, point_local) + shift
    point = truth.copy()
    point[40:70] = np.nan
    np.testing.assert_allclose(carried_point(point, cluster), truth, atol=1e-6)


def test_heel_strikes_at_most_anterior_positions():
    t = np.arange(400)
    heel = np.zeros((400, 3))
    heel[:, 0] = 300 * np.cos(2 * np.pi * (t - 50) / 110)
    assert heel_strikes(heel, np.zeros((400, 3)), np.array([1.0, 0, 0])).tolist() == [50, 160, 270, 380]


def test_inclination_angle_signs():
    n = 11
    com = np.column_stack([np.linspace(0, 1000, n), np.zeros(n), np.full(n, 900.0)])
    cycle = GaitCycle(0, 2, 5, 7, n - 1)
    x, y = progression_frame(com, cycle)
    # COP 100 mm behind and 50 mm to the right of the COM (right = -y for walking along +x)
    cop = com + np.array([-100.0, -50.0, -900.0])
    ia = inclination_angles(com, cop, x, y, "left")
    np.testing.assert_allclose(ia[:, 0], np.degrees(np.arcsin(100 / np.linalg.norm([100, 50, 900]))), atol=1e-9)
    assert (ia[:, 0] > 0).all()                                            # COM ahead of the COP
    np.testing.assert_allclose(inclination_angles(com, cop, x, y, "right")[:, 1], -ia[:, 1])


def test_rcia_of_a_sine():
    t = np.linspace(0, 1.0, 101)
    ia = np.column_stack([5 * np.sin(2 * np.pi * t), 2 * np.cos(2 * np.pi * t)])
    rcia = rcia_from_ia(ia, 1.0, 100.0)
    expected = np.column_stack([10 * np.pi * np.cos(2 * np.pi * t), -4 * np.pi * np.sin(2 * np.pi * t)])
    assert np.max(np.abs(rcia - expected)[5:-5]) < 0.5  # deg/s, away from the ends


def test_body_axes_and_mirroring():
    raw = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(to_body_axes(raw, XSENS_SACRUM), [[-3.0, 1.0, -2.0]])
    np.testing.assert_allclose(mirror_right_limb(np.arange(6.0)[None]), [[0, 1, -2, -3, -4, 5]])


def _invented_cycles(root, n, rng):
    t = np.linspace(0, 1, 101)
    for i in range(n):
        phase = rng.uniform(0, 1)
        ia = np.column_stack([8 * np.sin(2 * np.pi * (t + phase)), 3 * np.cos(2 * np.pi * (t + phase))])
        imu = np.column_stack([np.sin(2 * np.pi * (t + phase) + k) for k in range(6)]) + rng.normal(0, 0.05, (101, 6))
        write_cycle(root / f"S{i % 3}" / f"trial{i:03d}", imu, ia, rcia_from_ia(ia, 1.0, 100.0),
                    {"cycle_time_s": 1.0, "subject": f"S{i % 3}"})


def test_cycle_folders_and_a_short_training(tmp_path):
    rng = np.random.default_rng(0)
    _invented_cycles(tmp_path / "training", 40, rng)
    _invented_cycles(tmp_path / "testing", 8, rng)
    data = load_cycles(tmp_path / "training")
    assert len(data) == 40 and data.meta.subject.nunique() == 3
    parts = load_parts(tmp_path / "training", None, tmp_path / "testing")
    config = {"hidden1": 8, "hidden2": 4, "dense_units": 16, "max_epochs": 3, "batch_size": 8, "lam": 0.01}
    table = train_and_test(parts, config, tmp_path / "run", [0], device="cpu")
    assert (tmp_path / "run" / "seed0" / "model.pt").exists() and (tmp_path / "run" / "report.md").exists()
    assert set(table.variable) == {"sagittal_IA", "frontal_IA", "sagittal_RCIA", "frontal_RCIA"}


def test_short_bayesian_search(tmp_path):
    rng = np.random.default_rng(1)
    _invented_cycles(tmp_path / "training", 30, rng)
    parts = load_parts(tmp_path / "training", None, None)
    space = {"_comment": "notes are allowed", "hidden1": {"type": "int", "low": 4, "high": 8, "step": 4},
             "lam": {"type": "float", "low": 0.001, "high": 1.0, "log": True}}
    best = tune(parts, space, tmp_path / "search", n_trials=2, max_epochs=2, patience=2,
                start_from={"hidden1": 8, "lam": 0.01}, device="cpu")
    assert set(best) >= {"hidden1", "lam", "max_epochs"} and (tmp_path / "search" / "best_config.json").exists()


def test_window_opens():
    import tkinter as tk

    import pytest

    from single_imu_gait_balance import gui
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    app = gui.App(root)
    assert app.notebook.index("end") == 4  # data, train & test, tuning, test a model
    for source in ("trials", "arrays", "public"):  # the three kinds of data on the first tab
        app.source.set(source)
        app._show_source()
    root.destroy()


def test_extract_selected_members(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("01/mocap/l_comf_01.c3d", b"walk")
        z.writestr("01/mocap/l_comf_01.x2d", b"raw camera data")
        z.writestr("01/imu_extracted/data_l_comf_01.mat", b"imu")
    pattern = r"^\d\d/(mocap/.*\.c3d|imu_extracted/.*\.mat)$"
    assert extract_local(archive, pattern, tmp_path / "out") == (2, 2)
    assert (tmp_path / "out" / "01" / "mocap" / "l_comf_01.c3d").read_bytes() == b"walk"
    assert not (tmp_path / "out" / "01" / "mocap" / "l_comf_01.x2d").exists()
    assert extract_local(archive, pattern, tmp_path / "out") == (2, 0)  # already there: nothing fetched
