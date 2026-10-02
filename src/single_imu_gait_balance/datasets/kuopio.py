"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/datasets/kuopio.py
Description : The Kuopio gait data set: download, and conversion of every
              walking trial into one gait cycle (see cycles.py).
              Source: Lavikainen J et al., Data in Brief 2024;56:110841,
              doi:10.1016/j.dib.2024.110841; data doi:10.5281/zenodo.10559504,
              license CC BY 4.0. It has 51 healthy adults walking overground
              at slow, comfortable and fast speed over three force plates;
              Vicon at 100 Hz and Xsens MTw IMUs at 100 Hz.

              Download: per participant only the walking-trial C3D files and
              the extracted IMU files (about 7 GB of the 23 GB archives), plus
              the participant table. Optionally also the raw .mtb recordings
              (0.5 GB): with the Xsens MT Software Suite installed (xsens.py),
              trials that lack an extracted IMU file are then used too.

              Conversion of one trial:
              - COM: 7-segment model (com.py). The data set has no trochanter,
                ASIS/PSIS or first-metatarsal-head markers, so the functional
                hip center replaces the trochanter and the midpoint of the
                hallux and 4th-toe markers replaces the metatarsal head; the
                knee center is the midpoint of the epicondyle markers.
              - COP: summed wrench of the three floor plates (forceplate.py).
              - Cycle: heel strike on the middle plate to the next heel strike
                of the same foot, which lands beyond the plates. That strike is
                found from the heel marker and corrected by the difference
                between the same method and the plate contact at the start.
                Trials where a foot is not fully on its plate are rejected.
              - IMU: pelvis sensor B42DA3 (sensor x up, y left, z posterior,
                as XSENS_SACRUM). Acceleration from calibratedAcceleration;
                angular velocity from the strapdown increments (sdi.dq x
                100 Hz), which equal the gyroscope output (or both from the
                .mtb through the Xsens Device API). The 0-1 sample lag
                to the motion capture is estimated per trial from the angular
                speed of the pelvis marker cluster.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-02
"""

from __future__ import annotations

import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial.transform import Rotation

from .. import __version__
from ..c3d import read_c3d
from ..com import body_com
from ..cycles import write_cycle
from ..download import download_file, extract_local, extract_remote
from ..forceplate import floor_plates, ground_reaction
from ..gait import plate_cycle
from ..imu import XSENS_SACRUM, imu_cycle, to_body_axes
from ..inclination import inclination_angles, progression_frame, rcia_from_ia
from ..rigid import carried_point, fit_rigid
from ..signals import N_POINTS, fill_gaps, lowpass, lowpass_segments, time_normalize
from ..splits import split_by_trial

NAME = "kuopio"
CITATION = ("Lavikainen J, Vartiainen P, Stenroth L, Karjalainen PA, Korhonen RK, Liukkonen MK, Mononen ME. "
            "Gait data from 51 healthy participants with motion capture, inertial measurement units, and computer "
            "vision. Data in Brief 2024;56:110841. doi:10.1016/j.dib.2024.110841. "
            "Data: doi:10.5281/zenodo.10559504 (CC BY 4.0).")
URL = "https://zenodo.org/api/records/10559504/files/{}/content"
FILES = {"readme.txt": "347ea503f7a19dd6b4f994b083cee6c3",
         "info_participants.xlsx": "013c2dc43224098bdd9ed94885146e50"}
ARCHIVES = {"measurement_data_1_to_17.zip": "edab84684570169b3f0d239b2d9fa629",
            "measurement_data_18_to_34.zip": "8728c009f535d6198a03e853acc21d21",
            "measurement_data_35_to_51.zip": "200c10674354ce2d052b14816f93f408"}
#: Archive members needed: walking-trial C3D files and extracted IMU files.
MEMBERS = r"^\d\d/(mocap/[lr]_(slow|comf|fast)_\d+\.c3d|imu_extracted/data_[lr]_(slow|comf|fast)_\d+\.mat)$"
#: Optional raw Xsens recordings (0.5 GB). Some trials have no extracted IMU
#: file; with the Xsens MT Software Suite installed their .mtb is used.
MEMBERS_MTB = r"^\d\d/imu/[lr]_(slow|comf|fast)_\d+\.mtb$"
TRIAL = re.compile(r"^[lr]_(slow|comf|fast)_\d+$")
SPEEDS = {"slow": "slow", "comf": "comfortable", "fast": "fast"}

RATE = 100.0                # Hz, motion capture and IMU
MAX_GAP = 15                # frames filled in marker gaps (0.15 s)
PELVIS_SENSOR = "B42DA3"
PELVIS_CLUSTER = ("Pelvis1", "Pelvis2", "Pelvis3", "Pelvis4")

#: Anatomical point -> marker label (checked from the marker geometry of every participant).
ROLES = {
    "RSAP": "Torso2", "LSAP": "Torso3",                                          # acromions
    "RMFC": "RFemur5", "RLFC": "RFemur6", "LMFC": "LFemur5", "LLFC": "LFemur6",  # epicondyles
    "RMMA": "RTibia5", "RLMA": "RTibia6", "LMMA": "LTibia5", "LLMA": "LTibia6",  # malleoli
    "RHEE": "RFoot1", "RHAL": "RFoot2", "RT4": "RFoot3",                        # heel, hallux, 4th toe
    "LHEE": "LFoot1", "LHAL": "LFoot2", "LT4": "LFoot3",
    "RHJC": "Pelvis_RFemur_score", "LHJC": "Pelvis_LFemur_score",                # functional joint centers
    "RKJC_f": "RFemur_RTibia_score", "LKJC_f": "LFemur_LTibia_score",
    "RAJC_f": "RTibia_RFoot_score", "LAJC_f": "LTibia_LFoot_score",
}
#: Participant 1 has no epicondyle or malleolus markers (functional centers are
#: used instead). Participant 2 has no acromion markers and cannot be used.
ROLE_OVERRIDES = {
    1: {"RMFC": None, "RLFC": None, "LMFC": None, "LLFC": None,
        "RMMA": None, "RLMA": None, "LMMA": None, "LLMA": None},
    2: {"RSAP": None, "LSAP": None},
}
#: Marker cluster that carries each functional joint center (fills its gaps).
CARRIERS = {"RHJC": PELVIS_CLUSTER, "LHJC": PELVIS_CLUSTER,
            "RKJC_f": tuple(f"RFemur{i}" for i in range(1, 5)), "LKJC_f": tuple(f"LFemur{i}" for i in range(1, 5)),
            "RAJC_f": tuple(f"RTibia{i}" for i in range(1, 5)), "LAJC_f": tuple(f"LTibia{i}" for i in range(1, 5))}


# =============================================================================
# Download
# =============================================================================


def download(raw_dir: str | Path, method: str = "partial", zip_dir: str | Path | None = None,
             with_mtb: bool = False) -> Path:
    """Fetch the files needed into ``raw_dir``.

    Parameters
    ----------
    method : {"partial", "full"}
        "partial" reads only the needed members of the remote archives (about
        7 GB); "full" downloads the three archives (23 GB) and unpacks them.
    zip_dir : path, optional
        Folder with the archives downloaded before; used instead of the network.
    with_mtb : bool
        Also fetch the raw Xsens recordings (see MEMBERS_MTB).
    """
    raw = Path(raw_dir)
    pattern = f"(?:{MEMBERS})|(?:{MEMBERS_MTB})" if with_mtb else MEMBERS
    for name, md5 in FILES.items():
        download_file(URL.format(name), raw / name, md5)
    for name, md5 in ARCHIVES.items():
        local = Path(zip_dir) / name if zip_dir else None
        if local is not None and local.exists():
            extract_local(local, pattern, raw / "participants")
        elif method == "full":
            extract_local(download_file(URL.format(name), raw / "archives" / name, md5), pattern, raw / "participants")
        else:
            extract_remote(URL.format(name), pattern, raw / "participants")
    return raw


# =============================================================================
# One trial
# =============================================================================


def read_pelvis_imu(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis acceleration (m/s^2) and angular velocity (rad/s), sensor axes.

    From an extracted IMU file (.mat: calibratedAcceleration and the strapdown
    increments sdi.dq) or, with the Xsens Device API, from the raw .mtb.
    """
    if Path(path).suffix.lower() == ".mtb":
        from ..xsens import read_mtb

        data = read_mtb(path)[PELVIS_SENSOR]
        acc, gyro = data["acc"].copy(), data["gyro"].copy()
    else:
        sensor = getattr(sio.loadmat(path, squeeze_me=True, struct_as_record=False)["sensors"], PELVIS_SENSOR)
        increments = np.atleast_1d(sensor.sdi)
        dq = np.full((len(increments), 4), np.nan)
        for i, item in enumerate(increments):
            value = np.asarray(getattr(item, "dq", []), dtype=float).ravel()
            if value.size == 4 and np.linalg.norm(value) > 0.5:
                dq[i] = value
        gyro = np.full((len(dq), 3), np.nan)
        ok = np.isfinite(dq).all(axis=1)
        gyro[ok] = Rotation.from_quat(dq[ok][:, [1, 2, 3, 0]]).as_rotvec() * RATE
        acc = np.asarray(sensor.calibratedAcceleration, dtype=float)[:len(dq)].copy()
    acc[~(np.abs(np.nan_to_num(acc)).sum(axis=1) > 0)] = np.nan  # rows that could not be read are all zeros
    n = len(acc)
    for signal in (acc, gyro):
        bad = ~np.isfinite(signal).all(axis=1)
        if bad.all():
            raise ValueError("pelvis IMU data are empty")
        idx = np.arange(n)
        for j in range(3):
            signal[:, j] = np.interp(idx, idx[~bad], signal[~bad, j])
    return acc, gyro


def anatomical_points(markers: dict[str, np.ndarray], subject: int) -> dict[str, np.ndarray]:
    """Points of the COM model (com.REQUIRED_POINTS) plus the heels."""
    roles = {**ROLES, **ROLE_OVERRIDES.get(subject, {})}

    def get(role: str) -> np.ndarray | None:
        label = roles.get(role)
        point = markers.get(label) if label else None
        if point is not None and role in CARRIERS and all(m in markers for m in CARRIERS[role]):
            point = carried_point(point, np.stack([markers[m] for m in CARRIERS[role]], axis=1))
        return point

    points, missing = {}, []
    for s in "RL":
        mfc, lfc, mma, lma = get(f"{s}MFC"), get(f"{s}LFC"), get(f"{s}MMA"), get(f"{s}LMA")
        knee = (mfc + lfc) / 2 if mfc is not None and lfc is not None else get(f"{s}KJC_f")
        if mma is None or lma is None:
            mma = lma = get(f"{s}AJC_f")
        hal, toe4 = get(f"{s}HAL"), get(f"{s}T4")
        wanted = {f"{s}TRO": get(f"{s}HJC"), f"{s}KJC": knee, f"{s}MMA": mma, f"{s}LMA": lma,
                  f"{s}MTH": (hal + toe4) / 2 if hal is not None and toe4 is not None else None,
                  f"{s}SAP": get(f"{s}SAP"), f"{s}HEE": get(f"{s}HEE")}
        for name, value in wanted.items():
            if value is None:
                missing.append(name)
            else:
                points[name] = value
    if missing:
        raise ValueError(f"markers missing for {', '.join(missing)}")
    return points


def imu_lag(cluster: np.ndarray, gyro: np.ndarray, max_lag: int = 5) -> tuple[int, float]:
    """IMU samples by which the IMU trails the mocap frames, from the angular speed of the pelvis cluster.

    Returns (lag, correlation); the correlation is NaN if it cannot be estimated.
    """
    ok = np.isfinite(cluster).all(axis=(1, 2))
    if ok.sum() < 60:
        return 0, float("nan")
    rotation, _ = fit_rigid(cluster[np.flatnonzero(ok)[ok.sum() // 2]], cluster)
    good = np.isfinite(rotation).all(axis=(1, 2))
    speed = np.full(len(rotation), np.nan)
    for i in np.flatnonzero(good[:-1] & good[1:]):
        speed[i] = np.linalg.norm(Rotation.from_matrix(rotation[i].T @ rotation[i + 1]).as_rotvec()) * RATE
    gyro_speed = np.linalg.norm(lowpass(gyro, RATE, 6.0), axis=1)
    best, n = (0, float("nan")), min(len(speed), len(gyro_speed))
    for lag in range(-max_lag, max_lag + 1):
        i0, i1 = max(0, -lag), min(n, n - lag)
        x, y = speed[i0:i1], gyro_speed[i0 + lag:i1 + lag]
        valid = np.isfinite(x)
        if valid.sum() > 60:
            r = float(np.corrcoef(x[valid], y[valid])[0, 1])
            if not r <= best[1]:
                best = (lag, r)
    return best


def process_trial(c3d_path: str | Path, imu_path: str | Path, subject: int, body_mass: float,
                  min_lag_r: float = 0.8) -> dict:
    """One walking trial -> one gait cycle: imu (101 x 6), ia and rcia (101 x 2) and descriptive fields.

    ``imu_path`` is the extracted IMU file (.mat) or the raw recording (.mtb).
    """
    rec = read_c3d(c3d_path)
    if abs(rec.marker_rate - RATE) > 1e-6:
        raise ValueError(f"marker rate {rec.marker_rate} Hz, expected {RATE}")
    floor = floor_plates(rec.plates)
    if len(floor) != 3:
        raise ValueError(f"expected three floor plates, found {len(floor)}")
    labels = {v for v in {**ROLES, **ROLE_OVERRIDES.get(subject, {})}.values() if v} | set(PELVIS_CLUSTER)
    labels |= {m for cluster in CARRIERS.values() for m in cluster}
    markers = {k: lowpass_segments(fill_gaps(v, MAX_GAP), RATE, 5.0) for k, v in rec.markers.items() if k in labels}
    points = anatomical_points(markers, subject)
    pelvis = np.mean([markers[m] for m in PELVIS_CLUSTER], axis=0)

    reaction = ground_reaction(floor, rec.analog_rate, rec.marker_rate, rec.n_frames)
    cycle, side, events = plate_cycle(floor, reaction.plate_force[:, :, 2], body_mass, points, pelvis, RATE)
    frames = cycle.frames
    if frames[-1] >= rec.n_frames:
        raise ValueError("the cycle ends after the recording")
    com = body_com(points)
    if not np.isfinite(com[frames]).all():
        raise ValueError("COM undefined during the cycle (markers missing)")
    cop = reaction.cop[frames].copy()
    if not np.isfinite(cop[:-1]).all():
        raise ValueError("COP undefined during the cycle")
    if not np.isfinite(cop[-1]).all():
        cop[-1] = cop[-2]
    x, y = progression_frame(com, cycle)
    right = np.cross(x, [0.0, 0.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for pair, what in ((("RSAP", "LSAP"), "acromions"), (("RKJC", "LKJC"), "knees"), (("RTRO", "LTRO"), "hips")):
            gap = np.nanmedian((points[pair[0]][frames] - points[pair[1]][frames]) @ right)
            if not gap > 50.0:
                raise ValueError(f"{what}: right minus left {gap:.0f} mm along the right axis (labels swapped)")
    ia = inclination_angles(com[frames], cop, x, y, side)
    cycle_time = (cycle.end - cycle.start) / RATE

    acc_raw, gyro_raw = read_pelvis_imu(imu_path)
    lag, lag_r = imu_lag(np.stack([markers[m] for m in PELVIS_CLUSTER], axis=1), gyro_raw)
    if not lag_r >= min_lag_r:
        lag = 0  # not identifiable: the nominal synchronisation
    imu = imu_cycle(to_body_axes(acc_raw, XSENS_SACRUM), to_body_axes(gyro_raw, XSENS_SACRUM), RATE,
                    (cycle.start + lag) / RATE, (cycle.end + lag) / RATE, side)
    return {"imu": imu, "ia": time_normalize(ia, N_POINTS), "rcia": rcia_from_ia(ia, cycle_time, RATE),
            "cycle_time_s": cycle_time, "reference_limb": side,
            "events_percent": {k.upper(): round(v, 2) for k, v in cycle.percent().items()},
            "imu_lag_samples": int(lag), "imu_lag_r": None if not np.isfinite(lag_r) else round(lag_r, 3), **events}


# =============================================================================
# The whole data set
# =============================================================================


def _job(job: dict) -> dict:
    try:
        return {**job, "status": "used", "reason": "",
                **process_trial(job["c3d"], job["imu"], job["subject"], job["body_mass"])}
    except Exception as error:  # a rejected trial must not stop the build
        return {**job, "status": "rejected", "reason": str(error).splitlines()[0][:200]}


def build(raw_dir: str | Path, out_dir: str | Path, workers: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Convert every walking trial and write training/validation/testing cycle folders (80/10/10 by trial).

    Returns the table of all trials (also written to out_dir/trials.csv).
    """
    from ..xsens import backend

    raw, out = Path(raw_dir), Path(out_dir)
    info = pd.read_excel(raw / "info_participants.xlsx").set_index("ID")
    xsens_ok = backend() is not None
    jobs, rows, from_mtb = [], [], 0
    for subject_dir in sorted((raw / "participants").glob("[0-9][0-9]")):
        subject = int(subject_dir.name)
        invalid = set(str(info.loc[subject, "Invalid_trials"]).split(","))
        for c3d in sorted((subject_dir / "mocap").glob("*.c3d")):
            if not TRIAL.match(c3d.stem):
                continue
            mat = subject_dir / "imu_extracted" / f"data_{c3d.stem}.mat"
            mtb = subject_dir / "imu" / f"{c3d.stem}.mtb"
            job = {"subject": subject, "trial": c3d.stem, "c3d": str(c3d), "body_mass": float(info.loc[subject, "Mass"]),
                   "imu": str(mat if mat.exists() or not mtb.exists() else mtb)}
            if c3d.stem in invalid:
                rows.append({**job, "status": "skipped", "reason": "listed as invalid by the data set authors"})
            elif mat.exists() or (mtb.exists() and xsens_ok):
                jobs.append(job)
                from_mtb += not mat.exists()
            elif mtb.exists():
                rows.append({**job, "status": "skipped",
                             "reason": "only an .mtb recording; needs the Xsens MT Software Suite (gait-balance xsens-check)"})
            else:
                rows.append({**job, "status": "skipped", "reason": "no IMU recording"})
    workers = workers or min(4, os.cpu_count() or 1)
    print(f"{len(jobs)} walking trials to convert ({from_mtb} from .mtb files; {len(rows)} skipped), "
          f"{workers} processes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, future in enumerate(as_completed([pool.submit(_job, job) for job in jobs]), 1):
            rows.append(future.result())
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} converted", flush=True)

    used = sorted((r for r in rows if r["status"] == "used"), key=lambda r: (r["subject"], r["trial"]))
    if not used:
        raise RuntimeError("no trial could be converted")
    for r, which in zip(used, split_by_trial(len(used), seed=seed)):
        r["set"] = which
        cycle_info = {"cycle_time_s": r["cycle_time_s"], "data_set": "Kuopio gait", "subject": f"S{r['subject']:02d}",
                      "trial": r["trial"], "speed": SPEEDS[r["trial"].split("_")[1]], **{k: r[k] for k in (
                          "reference_limb", "events_percent", "imu_lag_samples", "imu_lag_r", "strike_bias",
                          "strike_bias_other")}, "software": f"single-imu-gait-balance {__version__}"}
        write_cycle(out / r["set"] / f"S{r['subject']:02d}" / r["trial"], r["imu"], r["ia"], r["rcia"], cycle_info)

    table = pd.DataFrame([{k: r.get(k) for k in ("subject", "trial", "status", "reason", "set", "cycle_time_s",
                                                  "reference_limb", "imu_lag_samples")} for r in rows])
    table = table.sort_values(["subject", "trial"]).reset_index(drop=True)
    table.to_csv(out / "trials.csv", index=False)
    counts = table[table.status == "used"].set.value_counts()
    reasons = table[table.status == "rejected"].reason.str.replace(r"[-+]?\d+(\.\d+)?", "#", regex=True).value_counts()
    lines = [f"Kuopio gait data set as gait-cycle folders (single-imu-gait-balance {__version__})", "",
             f"Source: {CITATION}", "",
             f"{len(used)} gait cycles from {table[table.status == 'used'].subject.nunique()} participants; "
             f"split by trial (seed {seed}): training {counts.get('training', 0)}, "
             f"validation {counts.get('validation', 0)}, testing {counts.get('testing', 0)}.",
             f"Walking trials: {len(table)}; skipped {int((table.status == 'skipped').sum())}, "
             f"rejected {int((table.status == 'rejected').sum())} (reasons in trials.csv):"]
    lines += [f"  {n:5d}  {reason}" for reason, n in reasons.items()]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:]), flush=True)
    return table
