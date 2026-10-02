"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/datasets/trials.py
Description : Your own recordings as gait cycles. One "trial folder" holds
                - a static (standing) C3D and a walking C3D, recorded with the
                  marker set of a marker set file (default: yu2023), with the
                  walking trial crossing three or four floor force plates;
                - the IMU recording of the same walk (.mtb, .mat or .csv, see
                  imu_files.py), started together with the motion capture;
                - optionally trial_info.json with any of: subject, group, speed,
                  body_mass_kg, static_c3d, walking_c3d, imu_file, sensor,
                  imu_offset_s.
              Every trial folder below a root folder becomes one gait-cycle
              folder (cycles.py), so the outputs can be used for training,
              testing, or simply read as input (imu_input.csv) and ground
              truth (ground_truth.csv).

              Steps for one trial (as in Yu et al., 2023):
              1. Static trial: quietest standing frames -> subject model
                 (segments.py); body mass from the plates unless given.
              2. Walking trial: marker gaps up to 0.15 s filled, 5 Hz low-pass,
                 segment optimization, 7-segment COM (com.py).
              3. Plates: 25 Hz low-pass, COP of the summed force; gait cycle
                 and reference limb from the plate contacts (gait.plate_cycle).
              4. Marker-label checks (segments.check_markers) and a left/right
                 check; failing trials are rejected with the reason.
              5. IA in the walking frame; RCIA from a GCV smoothing spline.
              6. IMU: sensor -> body axes, 15 Hz low-pass, cut at the same
                 heel strikes (shifted by imu_offset_s), right cycles mirrored.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-02
Last updated: 2026-10-02
"""

from __future__ import annotations

import json
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..c3d import read_c3d
from ..com import body_com
from ..cycles import write_cycle
from ..forceplate import GRAVITY, floor_plates, ground_reaction
from ..gait import plate_cycle
from ..imu import XSENS_SACRUM, imu_cycle, to_body_axes
from ..imu_files import IMU_SUFFIXES, read_sacral_imu
from ..inclination import inclination_angles, progression_frame, rcia_from_ia
from ..segments import PELVIS, calibrate, check_markers, load_markerset, quiet_frames, segment_kinematics
from ..signals import N_POINTS, fill_gaps, lowpass_segments, time_normalize
from ..splits import parse_fractions, read_split_file, set_from_list, split_by_trial

INFO_FILE = "trial_info.json"
STATIC_HINT = re.compile(r"static|subcali|calib|stand", re.IGNORECASE)
MAX_GAP_S = 0.15        # longest marker gap filled, s
MARKER_CUTOFF = 5.0     # Hz
INFO_KEYS = ("subject", "group", "speed", "body_mass_kg", "static_c3d", "walking_c3d", "imu_file", "sensor",
             "imu_offset_s")


@dataclass
class ImuSettings:
    """How the IMU recording relates to the motion capture.

    Attributes
    ----------
    sensor : str or None
        ID of the sacral sensor; needed only if the file holds several sensors.
    axes : tuple of str
        Sensor axes giving body x (anterior), y (up) and z (right); see imu.to_body_axes.
    rate : float
        Hz.
    offset_s : float
        Start of the IMU recording after the first C3D frame, s (0 when one
        trigger starts both systems).
    """

    sensor: str | None = None
    axes: tuple[str, str, str] = XSENS_SACRUM
    rate: float = 100.0
    offset_s: float = 0.0


@dataclass
class TrialFolder:
    folder: Path
    static_c3d: Path
    walking_c3d: Path
    imu_file: Path
    info: dict = field(default_factory=dict)


# =============================================================================
# Finding the files
# =============================================================================


def _pelvis_travel(c3d: Path, labels: list[str]) -> float:
    """Horizontal distance (mm) covered by the pelvis markers (all markers if those are missing)."""
    markers = read_c3d(c3d).markers
    names = [n for n in labels if n in markers] or list(markers)
    if not names:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmean(np.stack([markers[n][:, :2] for n in names]), axis=0)
        return float(np.linalg.norm(np.nanmax(center, axis=0) - np.nanmin(center, axis=0)))


def classify_c3d(files: list[Path], pelvis_labels: list[str]) -> tuple[Path, Path]:
    """(static, walking) C3D: a name with static/subcali/calib/stand marks the static trial;
    otherwise the pelvis travel decides (static < 300 mm, walking > 800 mm)."""
    if len(files) < 2:
        raise ValueError(f"expected a static and a walking C3D, found {len(files)}")
    travel = {p: _pelvis_travel(p, pelvis_labels) for p in files}
    hinted = [p for p in files if STATIC_HINT.search(p.stem)]
    static = hinted[0] if len(hinted) == 1 else min(files, key=lambda p: np.nan_to_num(travel[p], nan=np.inf))
    walking = [p for p in files if p != static and travel[p] > 800.0]
    described = ", ".join(f"{p.name} {travel[p]:.0f} mm" for p in files)
    if not travel[static] < 300.0:
        raise ValueError(f"no static C3D recognized (pelvis travel: {described})")
    if len(walking) != 1:
        raise ValueError(f"expected one walking C3D, found {len(walking)} (pelvis travel: {described})")
    return static, walking[0]


def is_trial_folder(folder: Path) -> bool:
    """At least two C3D files and an IMU file, or a trial_info.json that names them."""
    try:
        files = [p for p in folder.iterdir() if p.is_file()]
    except OSError:
        return False
    if (folder / INFO_FILE).is_file() and {"static_c3d", "walking_c3d", "imu_file"} <= set(_read_info(folder)):
        return True
    return sum(p.suffix.lower() == ".c3d" for p in files) >= 2 and any(p.suffix.lower() in IMU_SUFFIXES for p in files)


def find_trial_folders(root: str | Path) -> list[Path]:
    """Every trial folder below ``root`` (``root`` included), sorted."""
    root = Path(root)
    return [p for p in [root, *sorted(q for q in root.rglob("*") if q.is_dir())] if is_trial_folder(p)]


def _read_info(folder: Path) -> dict:
    path = folder / INFO_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def trial_files(folder: str | Path, pelvis_labels: list[str]) -> TrialFolder:
    """Locate the files of one trial folder (trial_info.json may name them)."""
    folder = Path(folder)
    info = _read_info(folder)
    if "sacrum_sensor" in info and "sensor" not in info:
        info["sensor"] = info["sacrum_sensor"]
    if "static_c3d" in info and "walking_c3d" in info:
        static, walking = folder / info["static_c3d"], folder / info["walking_c3d"]
    else:
        static, walking = classify_c3d(sorted(p for p in folder.iterdir() if p.suffix.lower() == ".c3d"),
                                       pelvis_labels)
    if "imu_file" in info:
        imu = folder / info["imu_file"]
    else:
        found = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMU_SUFFIXES),
                       key=lambda p: IMU_SUFFIXES.index(p.suffix.lower()))
        if not found:
            raise ValueError("no IMU file (.mtb, .mat or .csv)")
        imu = found[0]
    return TrialFolder(folder, static, walking, imu, {k: info[k] for k in INFO_KEYS if k in info})


# =============================================================================
# One trial
# =============================================================================


_MODELS: dict = {}


def subject_model(static_c3d: Path, markerset_name: str, body_mass: float | None = None):
    """Subject model of a static trial.

    Kept in memory: many trial folders share one static trial (also as copies
    or links, so the key is the file's name, size and time, not its path).
    """
    stat = static_c3d.stat()
    key = (static_c3d.name, stat.st_size, stat.st_mtime_ns, str(markerset_name), body_mass)
    if key not in _MODELS:
        if len(_MODELS) > 16:
            _MODELS.clear()
        _MODELS[key] = _calibrate_static(static_c3d, markerset_name, body_mass)
    return _MODELS[key]


def _calibrate_static(static_c3d: Path, markerset_name: str, body_mass: float | None):
    markerset = load_markerset(markerset_name)
    rec = read_c3d(static_c3d)
    markers = markerset.rename(rec.markers)
    frames = quiet_frames(markers)
    if body_mass is None:
        floor = floor_plates(rec.plates)
        if not floor:
            raise ValueError("static trial: no floor plates to weigh the subject; give body_mass_kg in trial_info.json")
        reaction = ground_reaction(floor, rec.analog_rate, rec.marker_rate, rec.n_frames, zero_offset=False)
        body_mass = float(np.nanmean(np.abs(reaction.total_force[frames, 2])) / GRAVITY)
        if not 20.0 <= body_mass <= 250.0:
            raise ValueError(f"static trial: body mass from the plates is {body_mass:.0f} kg; "
                             "give body_mass_kg in trial_info.json")
    return calibrate(markers, frames, body_mass)


def process_trial(files: TrialFolder, markerset_name: str = "yu2023", imu: ImuSettings | None = None) -> dict:
    """One trial folder -> one gait cycle: imu (101 x 6), ia and rcia (101 x 2) and descriptive fields."""
    imu = imu or ImuSettings()
    markerset = load_markerset(markerset_name)
    mass = files.info.get("body_mass_kg")
    model = subject_model(files.static_c3d, markerset_name, None if mass is None else float(mass))

    rec = read_c3d(files.walking_c3d)
    rate = rec.marker_rate
    max_gap = int(round(MAX_GAP_S * rate))
    markers = {k: lowpass_segments(fill_gaps(v, max_gap), rate, MARKER_CUTOFF)
               for k, v in markerset.rename(rec.markers).items()}
    points = segment_kinematics(model, markers, markerset)
    com = body_com(points)
    floor = floor_plates(rec.plates)
    if len(floor) < 3:
        raise ValueError(f"needs three or four floor force plates, found {len(floor)}")
    reaction = ground_reaction(floor, rec.analog_rate, rate, rec.n_frames)
    pelvis = np.mean([points[m] for m in PELVIS], axis=0)
    cycle, side, events = plate_cycle(floor, reaction.plate_force[:, :, 2], model.body_mass, points, pelvis, rate)
    frames = cycle.frames
    if frames[-1] >= rec.n_frames:
        raise ValueError("the cycle ends after the recording")
    problems, notes = check_markers(model, markers, frames, markerset)
    if problems:
        raise ValueError("marker labels: " + "; ".join(problems))
    if not np.isfinite(com[frames]).all():
        raise ValueError("COM undefined during the cycle (markers missing)")
    cop = reaction.cop[frames].copy()
    if not np.isfinite(cop[:-1]).all():
        raise ValueError("COP undefined during the cycle")
    if not np.isfinite(cop[-1]).all():
        cop[-1] = cop[-2]
    x, y = progression_frame(com, cycle)
    # A mirrored labelling (left markers on the right) would swap the reference
    # limb. Swapped acromions do not matter: the COM uses only their midpoint.
    right = np.cross(x, [0.0, 0.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for pair, what in ((("RKJC", "LKJC"), "knees"), (("RTRO", "LTRO"), "hips")):
            gap = np.nanmedian((points[pair[0]][frames] - points[pair[1]][frames]) @ right)
            if not gap > 50.0:
                raise ValueError(f"{what}: right minus left {gap:.0f} mm along the right axis (labels swapped)")
    ia = inclination_angles(com[frames], cop, x, y, side)
    cycle_time = (cycle.end - cycle.start) / rate

    acc, gyro, sensor = read_sacral_imu(files.imu_file, files.info.get("sensor") or imu.sensor)
    offset = float(files.info.get("imu_offset_s", imu.offset_s))
    signals = imu_cycle(to_body_axes(acc, imu.axes), to_body_axes(gyro, imu.axes), imu.rate,
                        cycle.start / rate - offset, cycle.end / rate - offset, side)
    return {"imu": signals, "ia": time_normalize(ia, N_POINTS), "rcia": rcia_from_ia(ia, cycle_time, rate),
            "cycle_time_s": round(cycle_time, 4), "reference_limb": side,
            "events_percent": {k.upper(): round(v, 2) for k, v in cycle.percent().items()},
            "body_mass_kg": round(model.body_mass, 2), "marker_rate_hz": rate,
            "static_c3d": files.static_c3d.name, "walking_c3d": files.walking_c3d.name,
            "imu_file": files.imu_file.name, "sensor": sensor, "imu_offset_s": offset,
            "plates": len(floor), "quality_notes": notes, **events}


# =============================================================================
# Many trial folders
# =============================================================================


def _job(job: dict) -> dict:
    row = {"folder": job["name"], "status": "rejected", "reason": ""}
    try:
        files = trial_files(job["path"], job["pelvis_labels"])
        row.update({k: files.info[k] for k in ("subject", "group", "speed") if k in files.info})
        return {**row, **process_trial(files, job["markerset"], job["imu"]), "status": "used"}
    except Exception as error:  # a rejected trial must not stop the conversion
        return {**row, "reason": str(error).splitlines()[0][:300]}


def _reason_kind(reason: str) -> str:
    """Reason without numbers and marker names, for counting."""
    if reason.startswith("marker labels"):
        return "marker labels do not fit the static trial (details in trials.csv)"
    return re.sub(r"[-+]?\d+(\.\d+)?", "#", reason)


def convert(root: str | Path, out_dir: str | Path, markerset: str = "yu2023", imu: ImuSettings | None = None,
            split: str | None = None, split_file: str | Path | None = None, seed: int = 42,
            workers: int | None = None) -> pd.DataFrame:
    """Convert every trial folder below ``root`` into gait-cycle folders under ``out_dir``.

    Each cycle keeps the trial folder's path relative to ``root``. With
    ``split`` ("80/10/10", by trial, random with ``seed``) or ``split_file``
    (JSON lists of trial names, see splits.py) the cycles go to
    out_dir/training, validation and testing; without, all go to out_dir.
    Returns the table of all trial folders (also out_dir/trials.csv).
    """
    root, out = Path(root), Path(out_dir)
    sets = load_markerset(markerset)  # fails early on a bad marker set file
    pelvis_labels = [sets.labels[m] for m in PELVIS]
    folders = find_trial_folders(root)
    if not folders:
        raise FileNotFoundError(f"no trial folders (two C3D files and an IMU file) below {root}")
    jobs = [{"path": str(f), "name": (f.relative_to(root).as_posix() if f != root else f.name),
             "markerset": markerset, "imu": imu or ImuSettings(), "pelvis_labels": pelvis_labels} for f in folders]
    workers = workers or min(4, os.cpu_count() or 1)
    print(f"{len(jobs)} trial folders to convert, {workers} processes", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, future in enumerate(as_completed([pool.submit(_job, job) for job in jobs]), 1):
            rows.append(future.result())
            if k % 20 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} converted", flush=True)

    used = sorted((r for r in rows if r["status"] == "used"), key=lambda r: r["folder"])
    if split_file:
        listed = read_split_file(split_file)
        for r in used:
            r["set"] = set_from_list(r["folder"], listed) or set_from_list(r["folder"].split("/")[-1], listed)
        for r in used:
            if r["set"] is None:
                r.update(status="skipped", reason="not listed in the split file")
        used = [r for r in used if r["status"] == "used"]
    elif split:
        for r, which in zip(used, split_by_trial(len(used), parse_fractions(split), seed)):
            r["set"] = which
    for r in used:
        info = {"cycle_time_s": r["cycle_time_s"], "trial": r["folder"],
                **{k: r[k] for k in ("subject", "group", "speed") if k in r},
                **{k: r[k] for k in ("reference_limb", "events_percent", "body_mass_kg", "marker_rate_hz",
                                     "static_c3d", "walking_c3d", "imu_file", "sensor", "imu_offset_s", "plates",
                                     "quality_notes", "strike_bias", "strike_bias_other") if k in r},
                "marker_set": Path(markerset).stem, "software": f"single-imu-gait-balance {__version__}"}
        folder = out / r["set"] / r["folder"] if r.get("set") else out / r["folder"]
        write_cycle(folder, r["imu"], r["ia"], r["rcia"], info)

    out.mkdir(parents=True, exist_ok=True)
    columns = ["folder", "subject", "group", "speed", "status", "reason", "set", "cycle_time_s", "reference_limb",
               "body_mass_kg", "imu_file"]
    table = pd.DataFrame([{k: r.get(k) for k in columns} for r in rows]).sort_values("folder").reset_index(drop=True)
    table.to_csv(out / "trials.csv", index=False)
    counts = table[table.status == "used"].set.value_counts()
    reasons = table[table.status != "used"].reason.map(_reason_kind).value_counts()
    split_text = (", ".join(f"{s} {counts.get(s, 0)}" for s in ("training", "validation", "testing"))
                  if split or split_file else "not split")
    lines = [f"Trial folders as gait-cycle folders (single-imu-gait-balance {__version__})", "",
             f"Source: {root}", f"Marker set: {markerset}", "",
             f"{len(used)} gait cycles from {len(table)} trial folders ({split_text}).",
             f"Not used: {int((table.status != 'used').sum())} (reasons in trials.csv):"]
    lines += [f"  {n:5d}  {reason}" for reason, n in reasons.items()]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[5:]), flush=True)
    return table
