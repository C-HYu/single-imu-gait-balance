"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/splits.py
Description : Division of gait cycles into training, validation and testing
              sets by trial (as in the paper: 80/10/10, random), or from a
              JSON file that lists the trials of each set.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-02
Last updated: 2026-10-02
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

SETS = ("training", "validation", "testing")
#: Other key names accepted in a split file.
ALIASES = {"train": "training", "val": "validation", "valid": "validation", "test": "testing"}


def parse_fractions(text: str) -> tuple[float, float, float]:
    """"80/10/10" -> (0.8, 0.1, 0.1)."""
    parts = [float(p) for p in str(text).replace(",", "/").split("/")]
    if len(parts) != 3 or min(parts) < 0 or sum(parts) <= 0:
        raise ValueError(f"split must look like 80/10/10 (training/validation/testing), got {text!r}")
    total = sum(parts)
    return parts[0] / total, parts[1] / total, parts[2] / total


def split_by_trial(n: int, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> list[str]:
    """Random set ("training", "validation" or "testing") for each of ``n`` trials.

    The held-out share is drawn first and then divided between validation and
    testing, so 80/10/10 matches the split of the paper's code.
    """
    sets = np.array(["training"] * n, dtype=object)
    train_share, val_share, test_share = fractions
    holdout = val_share + test_share
    if n == 0 or holdout <= 0:
        return sets.tolist()
    if train_share <= 0:
        rest = np.arange(n)
    else:
        _, rest = train_test_split(np.arange(n), test_size=holdout, random_state=seed)
    if val_share <= 0:
        sets[rest] = "testing"
    elif test_share <= 0:
        sets[rest] = "validation"
    else:
        val_idx, test_idx = train_test_split(rest, test_size=test_share / holdout, random_state=seed)
        sets[val_idx], sets[test_idx] = "validation", "testing"
    return sets.tolist()


def read_split_file(path: str | Path) -> dict[str, str]:
    """Trial name -> set, from a JSON file {"training": [...], "validation": [...], "testing": [...]}.

    The keys "train", "val" and "test" are accepted too. A listed name may
    carry leading folders ("Group/Trial12"); it matches a trial whose name or
    path ends with it.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    listed = {}
    for key, names in data.items():
        name = ALIASES.get(key, key)
        if name in SETS:
            for trial in names:
                listed[str(trial).replace("\\", "/")] = name
    if not listed:
        raise ValueError(f"{path}: no lists named training/validation/testing (or train/val/test)")
    return listed


def set_from_list(trial: str, listed: dict[str, str]) -> str | None:
    """Set of a trial (name or relative path) in a split file, or None if it is not listed."""
    trial = trial.replace("\\", "/")
    if trial in listed:
        return listed[trial]
    for name, which in listed.items():
        if trial.endswith("/" + name) or name.endswith("/" + trial):
            return which
    return None
