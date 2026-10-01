"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/training.py
Description : Training of the bi-GRU:
              - scaling of inputs and outputs to [-1, 1] (fitted on training);
              - weighted MSE loss: mean[(IA_hat - IA)^2 + lambda (RCIA_hat - RCIA)^2],
                RCIA_hat being the finite difference of the predicted IA;
              - Adam, best epoch chosen on the validation set (mean rRMSE),
                optional early stopping;
              - saving and loading of a trained model.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from . import __version__
from .cycles import CycleSet
from .inclination import finite_difference_rcia
from .metrics import cycle_errors, mean_rrmse
from .model import BiGRU


@dataclass
class TrainConfig:
    """Hyper-parameters of one training run.

    The defaults are the settings stated in the paper: 256/64 GRU cells, a
    202-neuron dense layer, Adam with learning rate 1e-4, batch size 32, at
    most 100 epochs and lambda = 5. The paper does not state the dense
    activation, dropout or weight decay; they default to tanh, 0 and 0.
    """

    hidden1: int = 256
    hidden2: int = 64
    dense_units: int = 202
    activation: str = "tanh"
    dropout: float = 0.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    lam: float = 5.0
    max_epochs: int = 100
    patience: int | None = None  # epochs without improvement before stopping; None = never
    seed: int = 0

    @classmethod
    def from_dict(cls, values: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known - {"_comment"}
        if unknown:
            raise ValueError(f"unknown hyper-parameters: {', '.join(sorted(unknown))}")
        return cls(**{k: v for k, v in values.items() if k in known})

    def model(self) -> BiGRU:
        return BiGRU(self.hidden1, self.hidden2, self.dense_units, self.activation, self.dropout)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch; deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    """"auto" picks the GPU when one is available."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


# =============================================================================
# Scaling
# =============================================================================


class Scaler:
    """Per-channel linear scaling to [-1, 1] (training range)."""

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)

    @classmethod
    def fit(cls, data: np.ndarray) -> "Scaler":
        flat = data.reshape(-1, data.shape[-1])
        return cls(flat.min(axis=0), flat.max(axis=0))

    @property
    def gain(self) -> np.ndarray:
        """Scaled units per original unit."""
        return 2.0 / np.where(self.high > self.low, self.high - self.low, 1.0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.low) * self.gain - 1.0

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return (data + 1.0) / self.gain + self.low

    def to_dict(self) -> dict:
        return {"low": self.low.tolist(), "high": self.high.tolist()}


# =============================================================================
# Loss
# =============================================================================


def weighted_mse(ia_pred: torch.Tensor, ia_true: torch.Tensor, rcia_true: torch.Tensor,
                 lam: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted MSE in scaled units: (total, IA term, RCIA term).

    The derivative is taken with respect to the normalized cycle time
    (0 -> 1), and the target RCIA is expressed in the same units
    (gain x RCIA x cycle time), so cycles of different durations count alike.
    Because of these units, lambda is not the same number as in the paper.
    """
    ia_term = torch.mean((ia_pred - ia_true) ** 2)
    if lam == 0.0:
        return ia_term, ia_term, torch.zeros_like(ia_term)
    rcia_pred = torch.gradient(ia_pred, spacing=1.0 / (ia_pred.shape[1] - 1), dim=1)[0]
    rcia_term = torch.mean((rcia_pred - rcia_true) ** 2)
    return ia_term + lam * rcia_term, ia_term, rcia_term


# =============================================================================
# Training
# =============================================================================


@dataclass
class Tensors:
    data: CycleSet
    x: torch.Tensor   # scaled IMU
    y: torch.Tensor   # scaled IA
    dy: torch.Tensor  # scaled dIA / d(cycle fraction)


@dataclass
class Prepared:
    parts: dict[str, Tensors]
    scaler_x: Scaler
    scaler_y: Scaler
    device: torch.device


def prepare(parts: dict[str, CycleSet], device: torch.device) -> Prepared:
    """Fit the scalers on ``parts["train"]`` and move every part to ``device``."""
    scaler_x, scaler_y = Scaler.fit(parts["train"].imu), Scaler.fit(parts["train"].ia)

    def tensor(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    out = {}
    for name, data in parts.items():
        dy = data.rcia * scaler_y.gain[None, None, :] * data.cycle_time[:, None, None]
        out[name] = Tensors(data, tensor(scaler_x.transform(data.imu)), tensor(scaler_y.transform(data.ia)), tensor(dy))
    return Prepared(out, scaler_x, scaler_y, device)


@torch.no_grad()
def predict_ia(model: torch.nn.Module, x: torch.Tensor, scaler_y: Scaler, batch_size: int = 256) -> np.ndarray:
    """Predicted IA in degrees, shape (n, 101, 2)."""
    model.eval()
    out = [model(x[i:i + batch_size]) for i in range(0, x.shape[0], batch_size)]
    return scaler_y.inverse(torch.cat(out).cpu().numpy().astype(float))


def evaluate(model: torch.nn.Module, part: Tensors, scaler_y: Scaler) -> dict[str, np.ndarray]:
    ia_pred = predict_ia(model, part.x, scaler_y)
    return cycle_errors(part.data.ia, ia_pred, part.data.rcia, part.data.cycle_time)


@dataclass
class TrainResult:
    model: torch.nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_score: float
    seconds: float


def train(config: TrainConfig, prepared: Prepared, on_epoch: Callable[[int, float], None] | None = None,
          verbose: bool = False) -> TrainResult:
    """Train one network; keep the weights of the epoch with the lowest validation mean rRMSE.

    Parameters
    ----------
    prepared : Prepared
        Needs the parts "train" and "val".
    on_epoch : callable, optional
        ``on_epoch(epoch, score)`` after every epoch; it may raise to stop.
    """
    set_seed(config.seed)
    train_part, val_part = prepared.parts["train"], prepared.parts["val"]
    model = config.model().to(prepared.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999),
                                 weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    best_state, best_score, best_epoch, stale, rows = None, np.inf, 0, 0, []
    start, n = time.perf_counter(), train_part.x.shape[0]
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=generator).to(prepared.device)
        total = 0.0
        for i in range(0, n, config.batch_size):
            idx = order[i:i + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = weighted_mse(model(train_part.x[idx]), train_part.y[idx], train_part.dy[idx], config.lam)
            loss.backward()
            optimizer.step()
            total += loss.item() * idx.numel()
        score = mean_rrmse(evaluate(model, val_part, prepared.scaler_y))
        rows.append({"epoch": epoch, "loss": total / n, "validation_mean_rrmse": score})
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:4d}: loss {total / n:.5f}, validation mean rRMSE {score:.3f} %", flush=True)
        if score < best_score:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), score, epoch, 0
        else:
            stale += 1
        if on_epoch is not None:
            on_epoch(epoch, score)
        if config.patience is not None and stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return TrainResult(model, pd.DataFrame(rows), best_epoch, float(best_score), time.perf_counter() - start)


# =============================================================================
# Model files
# =============================================================================


def save_model(path: str | Path, result: TrainResult, config: TrainConfig, prepared: Prepared) -> None:
    torch.save({"state_dict": {k: v.cpu() for k, v in result.model.state_dict().items()},
                "config": asdict(config), "scaler_x": prepared.scaler_x.to_dict(),
                "scaler_y": prepared.scaler_y.to_dict(), "best_epoch": result.best_epoch,
                "software": f"single-imu-gait-balance {__version__}"}, path)


class Predictor:
    """A saved model: IMU cycles -> IA (deg) and RCIA (deg/s)."""

    def __init__(self, path: str | Path, device: str = "cpu"):
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        self.config = TrainConfig.from_dict(bundle["config"])
        self.model = self.config.model()
        self.model.load_state_dict(bundle["state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.scaler_x = Scaler(**bundle["scaler_x"])
        self.scaler_y = Scaler(**bundle["scaler_y"])

    def predict(self, imu: np.ndarray, cycle_time: np.ndarray) -> np.ndarray:
        """imu (n, 101, 6), cycle_time (n,) -> (n, 101, 4): sagittal IA, frontal IA, sagittal RCIA, frontal RCIA."""
        x = torch.as_tensor(self.scaler_x.transform(imu), dtype=torch.float32, device=self.device)
        ia = predict_ia(self.model, x, self.scaler_y)
        return np.concatenate([ia, finite_difference_rcia(ia, cycle_time)], axis=2)
