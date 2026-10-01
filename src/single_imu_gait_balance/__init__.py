"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/__init__.py
Description : Dynamic balance during gait from one sacral IMU. The ground truth
              is the inclination angle (IA) of the centre-of-pressure-to-
              centre-of-mass vector and its rate of change (RCIA). A
              bidirectional GRU with a weighted MSE loss maps the IMU signals
              of one gait cycle to the IA. The method follows Yu et al.,
              Sensors 2023;23(22):9040 (doi:10.3390/s23229040), and is
              demonstrated on the public Kuopio gait data set.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

__version__ = "1.0.0"
