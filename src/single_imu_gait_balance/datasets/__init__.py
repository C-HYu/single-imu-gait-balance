"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/datasets/__init__.py
Description : Registry of the data sets that can be downloaded and converted.
              A data set module provides NAME, CITATION, download(raw_dir,
              method, zip_dir) and build(raw_dir, out_dir, workers); see
              docs/ADDING_A_DATASET.md.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from . import kuopio

DATASETS = {kuopio.NAME: kuopio}
