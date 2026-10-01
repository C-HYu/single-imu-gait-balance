"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/experiment.py
Description : The two jobs of the command line:
              - train_and_test: train with given hyper-parameters (one or more
                seeds), save the models and report the error of every test cycle;
              - tune: Bayesian hyper-parameter search (Optuna TPE sampler,
                median pruning) on the training and validation cycles.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .cycles import CycleSet, load_cycles
from .metrics import NAMES, PAPER, cycle_errors, mean_rrmse, summarise
from .training import Predictor, TrainConfig, get_device, prepare, save_model, train


def load_parts(train_dir: str | Path, val_dir: str | Path | None, test_dir: str | Path | None,
               val_fraction: float = 1 / 9, seed: int = 42) -> dict[str, CycleSet]:
    """Training, validation and test cycles.

    Without a validation folder, ``val_fraction`` of the training cycles are
    drawn at random (by cycle) for validation.
    """
    parts = {"train": load_cycles(train_dir)}
    if val_dir:
        parts["val"] = load_cycles(val_dir)
    else:
        idx = np.arange(len(parts["train"]))
        train_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed)
        parts["val"] = parts["train"].subset(np.sort(val_idx))
        parts["train"] = parts["train"].subset(np.sort(train_idx))
    if test_dir:
        parts["test"] = load_cycles(test_dir)
    return parts


def train_and_test(parts: dict[str, CycleSet], config: dict, out_dir: str | Path, seeds: list[int],
                   device: str = "auto") -> pd.DataFrame:
    """Train one model per seed; write models, per-cycle test errors and a summary.

    Output folder: seed<k>/model.pt, seed<k>/history.csv, seed<k>/test_errors.csv,
    summary.csv (mean of the test-set means over seeds) and report.md.
    Returns the per-seed test summaries.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    prepared = prepare(parts, get_device(device))
    print(f"train {len(parts['train'])}, validation {len(parts['val'])}, test {len(parts.get('test', []))} cycles; "
          f"device {prepared.device.type}", flush=True)
    rows = []
    for seed in seeds:
        cfg = TrainConfig.from_dict({**config, "seed": seed})
        result = train(cfg, prepared, verbose=True)
        folder = out / f"seed{seed}"
        folder.mkdir(exist_ok=True)
        save_model(folder / "model.pt", result, cfg, prepared)
        result.history.to_csv(folder / "history.csv", index=False)
        line = f"seed {seed}: best epoch {result.best_epoch}, validation mean rRMSE {result.best_score:.3f} %"
        if "test" in prepared.parts:
            errors = evaluate_model(folder / "model.pt", parts["test"], folder / "test_errors.csv")
            summary = summarise(errors)
            rows += [{"seed": seed, "variable": name, **summary.loc[name].drop("unit").to_dict()} for name in NAMES]
            line += f", test mean rRMSE {mean_rrmse(errors):.3f} %"
        print(line, flush=True)
    table = pd.DataFrame(rows)
    if not table.empty:
        _write_report(out, table, parts, config)
    return table


def evaluate_model(model_path: str | Path, data: CycleSet, csv_path: str | Path | None = None) -> dict:
    """Errors of a saved model on a set of cycles (optionally written per cycle)."""
    pred = Predictor(model_path).predict(data.imu, data.cycle_time)
    errors = cycle_errors(data.ia, pred[:, :, :2], data.rcia, data.cycle_time)
    if csv_path is not None:
        table = data.meta[["name"]].copy()
        for j, name in enumerate(NAMES):
            table[f"RMSE {name}"] = errors["rmse"][:, j]
            table[f"rRMSE {name} (%)"] = errors["rrmse"][:, j]
        table.to_csv(csv_path, index=False)
    return errors


def _write_report(out: Path, table: pd.DataFrame, parts: dict[str, CycleSet], config: dict) -> None:
    stats = table.groupby("variable", sort=False).agg(rmse=("rmse_mean", "mean"), rmse_sd_seeds=("rmse_mean", "std"),
                                                      rrmse=("rrmse_mean", "mean"),
                                                      rrmse_sd_seeds=("rrmse_mean", "std"),
                                                      rmse_sd_cycles=("rmse_sd", "mean"),
                                                      rrmse_sd_cycles=("rrmse_sd", "mean"))
    stats.to_csv(out / "summary.csv")
    units = {"sagittal_IA": "deg", "frontal_IA": "deg", "sagittal_RCIA": "deg/s", "frontal_RCIA": "deg/s"}
    lines = ["# Test error", "",
             f"Cycles: train {len(parts['train'])}, validation {len(parts['val'])}, test {len(parts['test'])}. "
             f"Seeds: {table.seed.nunique()}. Mean (SD over test cycles), averaged over seeds.", "",
             "| Variable | Unit | RMSE | rRMSE % | Paper RMSE | Paper rRMSE % |", "|---|---|---|---|---|---|"]
    for name in NAMES:
        s, p = stats.loc[name], PAPER.loc[name]
        lines.append(f"| {name} | {units[name]} | {s.rmse:.2f} ({s.rmse_sd_cycles:.2f}) | "
                     f"{s.rrmse:.2f} ({s.rrmse_sd_cycles:.2f}) | {p.rmse_mean:.2f} ({p.rmse_sd:.2f}) | "
                     f"{p.rrmse_mean:.2f} ({p.rrmse_sd:.2f}) |")
    lines += ["", f"Mean rRMSE over the four variables: {stats.rrmse.mean():.2f} % "
              f"(SD over seeds of the mean: {table.groupby('seed').rrmse_mean.mean().std():.2f}).",
              "Paper values: Yu et al. (2023), bi-GRU with weighted MSE, on the paper's own data set.", "",
              "Hyper-parameters:", "", "```json", json.dumps(config, indent=2), "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tune(parts: dict[str, CycleSet], space: dict, out_dir: str | Path, n_trials: int, max_epochs: int = 250,
         patience: int = 30, start_from: dict | None = None, device: str = "auto") -> dict:
    """Bayesian search of the hyper-parameters on the validation cycles (test cycles are not used).

    The Optuna study is stored in out_dir/study.db, so an interrupted search
    continues where it stopped. Returns the best configuration, also written
    to out_dir/best_config.json.
    """
    import optuna

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared = prepare({"train": parts["train"], "val": parts["val"]}, get_device(device))
    fixed = {"max_epochs": max_epochs, "patience": patience, "seed": 0}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name="search", load_if_exists=True,
                                storage=f"sqlite:///{(out / 'study.db').as_posix()}",
                                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True,
                                                                   n_startup_trials=min(10, max(n_trials // 4, 3))),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=40))
    if start_from:
        study.enqueue_trial({k: v for k, v in start_from.items() if k in space}, skip_if_exists=True)

    def suggest(trial) -> dict:
        params = {}
        for name, spec in space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1),
                                                 log=spec.get("log", False))
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            else:
                params[name] = trial.suggest_categorical(name, spec["options"])
        return params

    def objective(trial) -> float:
        def report(epoch: int, score: float) -> None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return train(TrainConfig.from_dict({**suggest(trial), **fixed}), prepared, on_epoch=report).best_score

    def log(study_, trial) -> None:
        value = f"{trial.value:.3f} %" if trial.value is not None else trial.state.name.lower()
        done = any(t.state == optuna.trial.TrialState.COMPLETE for t in study_.trials)
        print(f"search trial {trial.number + 1}: {value}" + (f"; best {study_.best_value:.3f} %" if done else ""),
              flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[log])
    study.trials_dataframe().to_csv(out / "search_trials.csv", index=False)
    best = {**study.best_params, **fixed}
    (out / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best
