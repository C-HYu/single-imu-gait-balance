"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/cli.py
Description : Command line (gait-balance <command>):
                datasets   list the data sets that can be downloaded
                download   fetch a data set into data/raw/<name>
                build      convert it into gait-cycle folders data/<name>/{training,validation,testing}
                train      train the bi-GRU (one or more seeds) and test it
                tune       Bayesian search of the hyper-parameters
                evaluate   test a saved model on a folder of gait cycles
                xsens-check  can Xsens .mtb files be read here (optional)
                mtb2csv    export Xsens .mtb recordings to CSV (optional)
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"


def _parts_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", help="folder with training/, validation/ and testing/ (e.g. data/kuopio)")
    p.add_argument("--train", help="training cycles (instead of --data)")
    p.add_argument("--val", help="validation cycles; without it 1/9 of the training cycles are used")
    p.add_argument("--test", help="test cycles")
    p.add_argument("--device", default="auto", help="auto, cuda or cpu")


def _folders(args) -> tuple:
    if args.data:
        root = Path(args.data)
        return root / "training", root / "validation", root / "testing"
    if not args.train:
        raise SystemExit("give --data or --train")
    return args.train, args.val, args.test


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gait-balance", description="Dynamic balance (IA, RCIA) during gait "
                                     "from one sacral IMU with a bi-GRU. Typical use: download kuopio -> build "
                                     "kuopio -> train --data data/kuopio.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("datasets", help="list the data sets that can be downloaded")
    p = sub.add_parser("download", help="fetch a data set")
    p.add_argument("name")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--method", choices=["partial", "full"], default="partial",
                   help="partial: only the files needed (default); full: whole archives")
    p.add_argument("--zip-dir", help="folder with archives downloaded before")
    p.add_argument("--with-mtb", action="store_true",
                   help="also fetch the raw Xsens .mtb recordings (used if the Xsens MT Software Suite is installed)")
    p = sub.add_parser("build", help="convert a downloaded data set into gait-cycle folders")
    p.add_argument("name")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int)
    p = sub.add_parser("train", help="train and test the bi-GRU")
    _parts_args(p)
    p.add_argument("--config", default=str(CONFIGS / "bigru_wmse.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", help="output folder (default runs/train_<time>)")
    p = sub.add_parser("tune", help="Bayesian search of the hyper-parameters")
    _parts_args(p)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--space", default=str(CONFIGS / "search_space.json"))
    p.add_argument("--start-from", default=str(CONFIGS / "bigru_wmse.json"),
                   help="configuration tried first ('' for none)")
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--out", help="output folder (default runs/tune_<time>)")
    p = sub.add_parser("evaluate", help="test a saved model")
    p.add_argument("--model", required=True)
    p.add_argument("--test", required=True, help="folder of gait cycles")
    p.add_argument("--out", help="CSV of the per-cycle errors")
    sub.add_parser("xsens-check", help="can Xsens .mtb files be read here? (and what to install if not)")
    p = sub.add_parser("mtb2csv", help="export an Xsens .mtb recording to one CSV per sensor")
    p.add_argument("mtb", nargs="+")
    p.add_argument("--out", help="output folder (default: next to each .mtb)")
    args = parser.parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.command in ("xsens-check", "mtb2csv"):
        from .xsens import check, mtb_to_csv
        if args.command == "xsens-check":
            print(check())
        else:
            for path in args.mtb:
                for written in mtb_to_csv(path, args.out):
                    print(f"written {written}")
        return

    if args.command in ("datasets", "download", "build"):
        from .datasets import DATASETS
        if args.command == "datasets":
            for name, module in DATASETS.items():
                print(f"{name}: {module.CITATION}")
            return
        if args.name not in DATASETS:
            raise SystemExit(f"unknown data set {args.name!r}; available: {', '.join(DATASETS)}")
        module, data_dir = DATASETS[args.name], Path(args.data_dir)
        if args.command == "download":
            module.download(data_dir / "raw" / args.name, args.method, args.zip_dir,
                            **({"with_mtb": True} if args.with_mtb else {}))
            print(f"downloaded to {data_dir / 'raw' / args.name}")
        else:
            module.build(data_dir / "raw" / args.name, data_dir / args.name, args.workers)
            print(f"gait cycles in {data_dir / args.name}")
        return

    from .experiment import evaluate_model, load_parts, train_and_test, tune
    from .metrics import mean_rrmse, summarise
    if args.command == "evaluate":
        from .cycles import load_cycles
        errors = evaluate_model(args.model, load_cycles(args.test), args.out)
        print(summarise(errors).round(3).to_string())
        print(f"mean rRMSE {mean_rrmse(errors):.2f} %")
        return
    train_dir, val_dir, test_dir = _folders(args)
    if args.command == "train":
        parts = load_parts(train_dir, val_dir, test_dir)
        out = Path(args.out or f"runs/train_{stamp}")
        train_and_test(parts, _read_json(args.config), out, args.seeds, args.device)
        print(f"results in {out}" + (f" (report: {out / 'report.md'})" if (out / "report.md").exists() else ""))
    else:
        parts = load_parts(train_dir, val_dir, None)
        out = Path(args.out or f"runs/tune_{stamp}")
        start = _read_json(args.start_from) if args.start_from else None
        best = tune(parts, _read_json(args.space), out, args.trials, args.max_epochs, args.patience, start, args.device)
        print(f"best hyper-parameters: {json.dumps(best)}\nsaved as {out / 'best_config.json'}; "
              f"train with: gait-balance train --data ... --config {out / 'best_config.json'}")


if __name__ == "__main__":
    main()
