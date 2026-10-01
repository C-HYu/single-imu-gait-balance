"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/metrics.py
Description : RMSE and relative RMSE (rRMSE = RMSE / range of the measured
              curve x 100 %) of the four balance variables, and the values
              reported for the bi-GRU with weighted MSE in the paper.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inclination import finite_difference_rcia

#: Balance variables in reporting order, with units.
VARIABLES = (("sagittal_IA", "deg"), ("frontal_IA", "deg"), ("sagittal_RCIA", "deg/s"), ("frontal_RCIA", "deg/s"))
NAMES = [name for name, _ in VARIABLES]

#: Bi-GRU with weighted MSE in Yu et al. (2023): mean (SD) over the test trials.
PAPER = pd.DataFrame({"rmse_mean": [0.61, 0.46, 13.13, 6.38], "rmse_sd": [0.24, 0.21, 5.69, 1.98],
                      "rrmse_mean": [3.82, 5.33, 5.32, 4.01], "rrmse_sd": [1.53, 3.76, 2.17, 2.08]},
                     index=NAMES)


def cycle_errors(ia_true: np.ndarray, ia_pred: np.ndarray, rcia_true: np.ndarray,
                 cycle_time: np.ndarray) -> dict[str, np.ndarray]:
    """RMSE and rRMSE of every cycle, each shape (n_cycles, 4) in the order of :data:`VARIABLES`.

    The predicted RCIA is the finite difference of the predicted IA; the
    reference RCIA is the spline derivative of the measured IA.
    """
    truth = np.concatenate([ia_true, rcia_true], axis=2)
    pred = np.concatenate([ia_pred, finite_difference_rcia(ia_pred, cycle_time)], axis=2)
    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=1))
    span = truth.max(axis=1) - truth.min(axis=1)
    return {"rmse": rmse, "rrmse": 100.0 * rmse / np.where(span > 0, span, np.nan)}


def summarise(errors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean and SD over cycles, one row per balance variable."""
    return pd.DataFrame({"unit": [u for _, u in VARIABLES],
                         "rmse_mean": np.nanmean(errors["rmse"], axis=0),
                         "rmse_sd": np.nanstd(errors["rmse"], axis=0, ddof=1),
                         "rrmse_mean": np.nanmean(errors["rrmse"], axis=0),
                         "rrmse_sd": np.nanstd(errors["rrmse"], axis=0, ddof=1)}, index=NAMES)


def mean_rrmse(errors: dict[str, np.ndarray]) -> float:
    """Mean rRMSE (%) over the four variables: the validation score to minimise."""
    return float(np.nanmean(errors["rrmse"]))
