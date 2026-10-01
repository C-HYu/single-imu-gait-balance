"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/model.py
Description : The bidirectional GRU network: IMU sequence (101 x 6) ->
              bi-GRU layer 1 -> bi-GRU layer 2 -> dense layer -> output layer
              of 101 x 2 values (sagittal and frontal IA).
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "linear": nn.Identity}


class BiGRU(nn.Module):
    """Two bidirectional GRU layers, a dense layer and a linear output layer.

    The whole sequence of layer-2 states feeds the dense layer, so every
    output sample can use information from the entire gait cycle.

    Parameters
    ----------
    hidden1, hidden2 : int
        Cells per direction in GRU layers 1 and 2.
    dense_units : int
        Width of the dense layer.
    activation : {"tanh", "relu", "gelu", "linear"}
        Activation after the dense layer.
    dropout : float
        Dropout after each GRU layer and after the dense layer.
    n_points, input_dim, output_dim : int
        101 samples, 6 IMU channels, 2 IA channels.
    """

    def __init__(self, hidden1: int = 256, hidden2: int = 64, dense_units: int = 202, activation: str = "tanh",
                 dropout: float = 0.0, n_points: int = 101, input_dim: int = 6, output_dim: int = 2):
        super().__init__()
        self.n_points, self.output_dim = n_points, output_dim
        self.gru1 = nn.GRU(input_dim, hidden1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(2 * hidden1, hidden2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_points * 2 * hidden2, dense_units)
        self.activation = ACTIVATIONS[activation]()
        self.output = nn.Linear(dense_units, n_points * output_dim)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        """Scaled IMU (batch, 101, 6) -> scaled IA (batch, 101, 2)."""
        states, _ = self.gru1(imu)
        states, _ = self.gru2(self.dropout(states))
        hidden = self.dropout(self.activation(self.dense(self.dropout(states).flatten(start_dim=1))))
        return self.output(hidden).view(-1, self.n_points, self.output_dim)
