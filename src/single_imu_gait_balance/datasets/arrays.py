"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/datasets/arrays.py
Description : Gait cycles that were already processed elsewhere, stored as
              array files (.npz or .mat), one gait cycle per file or pair of
              files: the IMU input of the cycle and its IA, at any number of
              samples (they are resampled to 101). A small JSON "spec" file
              says where the arrays are:

              {
                "imu_files": "**/*_imu.npz",         glob below the root folder
                "imu_key": "data",                   array with the 6 IMU channels
                "imu_columns": [0, 1, 2, 3, 4, 5],   optional (default: the first 6)
                "ia_file": [["_imu.npz", "_ia.npz"]],
                                                     optional: replacements that turn the
                                                     IMU file's path into the IA file's
                                                     (default: the same file)
                "ia_key": "data",                    array with the IA (degrees)
                "ia_columns": [0, 1],                optional: sagittal, frontal
                "cycle_time_key": "cycle_time_s",    scalar in the IA (or IMU) file, s
                "frame_rate_hz": 100,                motion-capture rate of the IA
                "rcia_key": null,                    optional: stored RCIA (deg/s); else
                                                     computed as in the paper (GCV spline)
                "fields_from_path": {"subject": 0},  optional: fields from the folder
                                                     names (0 = first folder below root)
                "folder": "{subject}/{name}"         optional: output layout (default {name})
              }

              The IMU must already follow the model's convention: body axes
              (x anterior, y up, z right), acc in m/s^2, gyro in rad/s,
              15 Hz low-pass, right-limb cycles mirrored (imu.py).
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-02
Last updated: 2026-10-02
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from .. import __version__
from ..cycles import write_cycle
from ..inclination import rcia_from_ia
from ..signals import N_POINTS, time_normalize
from ..splits import parse_fractions, read_split_file, set_from_list, split_by_trial

CYCLE_TIME_LIMITS = (0.3, 4.0)  # s; outside: a file or event error


def read_spec(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in ("imu_files", "imu_key", "ia_key", "cycle_time_key", "frame_rate_hz") if k not in spec]
    if missing:
        raise ValueError(f"{Path(path).name}: missing {', '.join(missing)}")
    return spec


def _load(path: Path) -> dict:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if path.suffix.lower() == ".mat":
        return {k: v for k, v in sio.loadmat(path, squeeze_me=True).items() if not k.startswith("__")}
    raise ValueError(f"{path.name}: array files must be .npz or .mat")


def read_cycle(imu_path: Path, root: Path, spec: dict) -> dict:
    """One gait cycle: imu (101 x 6), ia and rcia (101 x 2), cycle time and descriptive fields."""
    rel = imu_path.relative_to(root).as_posix()
    ia_rel = rel
    for old, new in spec.get("ia_file", []):
        ia_rel = ia_rel.replace(old, new)
    ia_path = root / ia_rel
    if not ia_path.is_file():
        raise FileNotFoundError(f"no IA file {ia_rel}")
    imu_data = _load(imu_path)
    ia_data = imu_data if ia_path == imu_path else _load(ia_path)
    imu = np.asarray(imu_data[spec["imu_key"]], dtype=float)[:, spec.get("imu_columns", list(range(6)))]
    ia = np.asarray(ia_data[spec["ia_key"]], dtype=float)[:, spec.get("ia_columns", [0, 1])]
    key = spec["cycle_time_key"]
    cycle_time = float(np.ravel(ia_data[key] if key in ia_data else imu_data[key])[0])
    if imu.shape[1] != 6 or ia.shape[1] != 2:
        raise ValueError(f"expected 6 IMU and 2 IA columns, got {imu.shape[1]} and {ia.shape[1]}")
    if not (np.isfinite(imu).all() and np.isfinite(ia).all()):
        raise ValueError("NaN or infinite values")
    if not CYCLE_TIME_LIMITS[0] <= cycle_time <= CYCLE_TIME_LIMITS[1]:
        raise ValueError(f"cycle time {cycle_time:.3f} s")
    if spec.get("rcia_key"):
        rcia = time_normalize(np.asarray(ia_data[spec["rcia_key"]], dtype=float)[:, :2], N_POINTS)
    else:  # the paper's ground truth: GCV smoothing spline on the motion-capture frames
        rcia = rcia_from_ia(ia, cycle_time, float(spec["frame_rate_hz"]))
    parts = Path(rel).parts[:-1]
    fields = {name: parts[i] for name, i in spec.get("fields_from_path", {}).items() if i < len(parts)}
    name = ia_path.stem
    return {"name": name, "imu": time_normalize(imu, N_POINTS), "ia": time_normalize(ia, N_POINTS), "rcia": rcia,
            "cycle_time_s": cycle_time, "fields": fields,
            "folder": spec.get("folder", "{name}").format(name=name, **fields),
            "source_files": sorted({rel, ia_rel})}


def import_arrays(root: str | Path, spec_path: str | Path, out_dir: str | Path, split: str | None = None,
                  split_file: str | Path | None = None, seed: int = 42) -> pd.DataFrame:
    """Write one gait-cycle folder per array file (pair) found with the spec.

    With ``split`` ("80/10/10", by trial, random with ``seed``) or
    ``split_file`` (JSON lists of trial names) the cycles go to
    out_dir/training, validation and testing; without, all go to out_dir.
    Returns the table of all files (also out_dir/trials.csv).
    """
    root, out = Path(root), Path(out_dir)
    spec = read_spec(spec_path)
    files = sorted(root.glob(spec["imu_files"]))
    if not files:
        raise FileNotFoundError(f"no files matching {spec['imu_files']} below {root}")
    print(f"{len(files)} files to import", flush=True)
    rows, cycles, seen = [], [], set()
    for k, path in enumerate(files, 1):
        try:
            cycle = read_cycle(path, root, spec)
            if cycle["folder"] in seen:
                raise ValueError(f"a second cycle named {cycle['folder']} (add folders to the spec's \"folder\")")
            seen.add(cycle["folder"])
            cycles.append(cycle)
            rows.append({"trial": cycle["folder"], "status": "used", "reason": "", **cycle["fields"],
                         "cycle_time_s": cycle["cycle_time_s"]})
        except Exception as error:  # a bad file must not stop the import
            rows.append({"trial": path.relative_to(root).as_posix(), "status": "rejected",
                         "reason": str(error).splitlines()[0][:200]})
        if k % 100 == 0 or k == len(files):
            print(f"  {k}/{len(files)} imported", flush=True)
    if not cycles:
        raise RuntimeError("no gait cycle could be imported")
    by_trial = {r["trial"]: r for r in rows}
    if split_file:
        listed = read_split_file(split_file)
        for c in cycles:
            c["set"] = set_from_list(c["folder"], listed) or set_from_list(c["name"], listed)
            if c["set"] is None:
                by_trial[c["folder"]].update(status="skipped", reason="not listed in the split file")
        cycles = [c for c in cycles if c["set"] is not None]
    elif split:
        for c, which in zip(cycles, split_by_trial(len(cycles), parse_fractions(split), seed)):
            c["set"] = which
    for c in cycles:
        by_trial[c["folder"]]["set"] = c.get("set")
        info = {"cycle_time_s": c["cycle_time_s"], "trial": c["name"], **c["fields"],
                "source_files": c["source_files"], "spec": Path(spec_path).name,
                "rcia": "stored" if spec.get("rcia_key") else "GCV smoothing spline of the IA",
                "software": f"single-imu-gait-balance {__version__}"}
        folder = out / c["set"] / c["folder"] if c.get("set") else out / c["folder"]
        write_cycle(folder, c["imu"], c["ia"], c["rcia"], info)
    table = pd.DataFrame(rows)
    table.to_csv(out / "trials.csv", index=False)
    used = table[table.status == "used"]
    counts = used["set"].value_counts() if "set" in used else pd.Series(dtype=int)
    split_text = (", ".join(f"{s} {counts.get(s, 0)}" for s in ("training", "validation", "testing"))
                  if split or split_file else "not split")
    lines = [f"Array files as gait-cycle folders (single-imu-gait-balance {__version__})", "",
             f"Source: {root}", f"Spec: {spec_path}", "",
             f"{len(used)} gait cycles ({split_text}); not used: {len(table) - len(used)} (reasons in trials.csv)."]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[5:]), flush=True)
    return table
