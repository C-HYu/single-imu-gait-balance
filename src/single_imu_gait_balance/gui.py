"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/gui.py
Description : A simple window for the jobs of the command line: (1) data:
              download and convert a public data set, or convert your own
              trial folders (C3D + IMU) or processed gait cycles, (2) train
              and test the bi-GRU, (3) Bayesian tuning and (4) testing a
              trained model on other gait cycles. Each job runs the gait-balance command in a separate
              process, so its progress appears live and Stop ends it at once.
              Results are read from the files the job writes.
              Start it with `gait-balance gui` (or `gait-balance-gui`).
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-02
Last updated: 2026-10-02
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import __version__
from .datasets import DATASETS
from .metrics import NAMES, PAPER, VARIABLES

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
SETTINGS_FILE = Path.home() / ".single_imu_gait_balance_gui.json"
UNITS = dict(VARIABLES)
LABELS = {"sagittal_IA": "Sagittal IA", "frontal_IA": "Frontal IA", "sagittal_RCIA": "Sagittal RCIA",
          "frontal_RCIA": "Frontal RCIA"}
PRESETS = {"bi-GRU, tuned (default)": CONFIGS / "bigru_wmse.json", "As in the paper": CONFIGS / "paper.json"}
SPLITS = ("No (only convert, e.g. to test a model)", "80/10/10 by trial, random (as in the paper)",
          "From a split file")
PAD = 6


# =============================================================================
# Running a command
# =============================================================================


class Job:
    """One gait-balance command in a child process; its output lines are passed to ``on_line``."""

    def __init__(self, args: list[str], on_line, on_done):
        self.args, self.on_line, self.on_done = args, on_line, on_done
        self.process: subprocess.Popen | None = None
        self.stopped = False

    def start(self) -> None:
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":  # the window may run without a console
            python = python.with_name("python.exe")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen([str(python), "-W", "ignore", "-m", "single_imu_gait_balance", *self.args],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", env=env, creationflags=flags,
                                        start_new_session=os.name != "nt")
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self.on_line(line.rstrip("\n"))
        self.on_done(self.process.wait(), self.stopped)

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.stopped = True
        if os.name == "nt":  # end the whole process tree (conversion uses several processes)
            subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.killpg(self.process.pid, 9)


def open_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(path)  # noqa: S606
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


# =============================================================================
# Widgets
# =============================================================================


class PathRow(ttk.Frame):
    """Label + entry + Browse button."""

    def __init__(self, parent, label: str, var: tk.StringVar, kind: str = "folder", width: int = 22, filetypes=None):
        super().__init__(parent)
        ttk.Label(self, text=label, width=width).pack(side="left")
        ttk.Entry(self, textvariable=var).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(self, text="Browse…", command=lambda: self._browse(var, kind, filetypes)).pack(side="left")

    @staticmethod
    def _browse(var, kind, filetypes) -> None:
        start = var.get() or os.getcwd()
        start = start if os.path.isdir(start) else os.path.dirname(start)
        path = (filedialog.askdirectory(initialdir=start) if kind == "folder"
                else filedialog.askopenfilename(initialdir=start, filetypes=filetypes or [("All files", "*.*")]))
        if path:
            var.set(os.path.normpath(path))


class Log(ttk.Frame):
    """Read-only, scrolling text."""

    def __init__(self, parent, height: int = 10):
        super().__init__(parent)
        self.text = tk.Text(self, height=height, wrap="none", font=("Consolas", 9), state="disabled",
                            background="#fbfbfb", relief="flat", borderwidth=1)
        bar = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=bar.set)
        self.text.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")

    def add(self, line: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class ResultTable(ttk.Treeview):
    def __init__(self, parent, columns: list[tuple[str, int]], height: int = 5):
        super().__init__(parent, columns=[c for c, _ in columns], show="headings", height=height)
        for name, width in columns:
            self.heading(name, text=name)
            self.column(name, width=width, anchor="center" if width < 160 else "w", stretch=True)

    def fill(self, rows: list[list]) -> None:
        self.delete(*self.get_children())
        for row in rows:
            self.insert("", "end", values=row)


def summary_rows(table: pd.DataFrame) -> list[list[str]]:
    """Rows of variable | RMSE (SD) | rRMSE (SD) | paper RMSE | paper rRMSE from a summary with
    columns rmse, rmse_sd, rrmse, rrmse_sd (index = variable names)."""
    rows = []
    for name in NAMES:
        s, p = table.loc[name], PAPER.loc[name]
        unit = "°" if UNITS[name] == "deg" else " °/s"
        rows.append([LABELS[name], f"{s.rmse:.2f}{unit} ({s.rmse_sd:.2f})", f"{s.rrmse:.2f} ({s.rrmse_sd:.2f})",
                     f"{p.rmse_mean:.2f}{unit}", f"{p.rrmse_mean:.2f}"])
    rows.append(["Mean", "", f"{table.rrmse.mean():.2f}", "", f"{PAPER.rrmse_mean.mean():.2f}"])
    return rows


def plot_example(figure: Figure, model: Path, test_dir: Path) -> str:
    """Measured vs predicted IA and RCIA of the test cycle with the median error (2 x 2)."""
    from .cycles import load_cycles
    from .metrics import cycle_errors
    from .training import Predictor

    data = load_cycles(test_dir)
    pred = Predictor(model).predict(data.imu, data.cycle_time)
    truth = np.concatenate([data.ia, data.rcia], axis=2)
    rrmse = cycle_errors(data.ia, pred[:, :, :2], data.rcia, data.cycle_time)["rrmse"].mean(axis=1)
    pick = int(np.argsort(rrmse)[len(rrmse) // 2])
    pct = np.linspace(0, 100, 101)
    figure.clear()
    axes = figure.subplots(2, 2)
    for k, ax in enumerate(axes.ravel()):
        m, s = truth[:, :, k].mean(0), truth[:, :, k].std(0)
        ax.fill_between(pct, m - s, m + s, color="0.88", lw=0, label="all test cycles (mean ± SD)")
        ax.plot(pct, truth[pick, :, k], color="#1f4e79", lw=2, label="measured")
        ax.plot(pct, pred[pick, :, k], color="#e07b00", lw=2, ls="--", label="predicted")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(LABELS[NAMES[k]], fontsize=10, weight="bold")
        ax.set_ylabel(UNITS[NAMES[k]].replace("deg", "°").replace("/s", "/s"), fontsize=9)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=8)
        if k >= 2:
            ax.set_xlabel("Gait cycle (%)", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="upper right", frameon=False)
    figure.tight_layout()
    return f"Test cycle with the median error ({rrmse[pick]:.1f} % mean rRMSE) of {len(data)}"


def plot_history(figure: Figure, run: Path) -> str:
    figure.clear()
    ax = figure.add_subplot(111)
    for history in sorted(run.glob("seed*/history.csv")):
        h = pd.read_csv(history)
        ax.plot(h.epoch, h.validation_mean_rrmse, lw=1.5, label=history.parent.name)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Validation mean rRMSE (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    figure.tight_layout()
    return "Validation error after every training epoch"


# =============================================================================
# The window
# =============================================================================


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"Single IMU Gait Balance {__version__} — bi-GRU (Yu et al., 2023)")
        root.geometry("1180x800")
        root.minsize(980, 680)
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.settings = self._load_settings()
        self.queue: queue.Queue = queue.Queue()
        self.job: Job | None = None

        head = ttk.Frame(root, padding=(PAD * 2, PAD * 2, PAD * 2, 0))
        head.pack(fill="x")
        ttk.Label(head, text="Dynamic balance from one sacral IMU", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="   IA and RCIA during gait with a bi-directional GRU", foreground="#555").pack(side="left")
        ttk.Button(head, text="README", command=lambda: open_path(REPO / "README.md")).pack(side="right")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=PAD * 2, pady=PAD)
        self.status = tk.StringVar(value="Ready.")
        bar = ttk.Frame(root, padding=(PAD * 2, 0, PAD * 2, PAD))
        bar.pack(fill="x")
        ttk.Label(bar, textvariable=self.status).pack(side="left")
        self.progress = ttk.Progressbar(bar, length=260, maximum=1.0)
        self.progress.pack(side="right")

        self.data_dir = tk.StringVar(value=self.settings.get("data_dir", str(Path.cwd() / "data")))
        self.cycles_dir = tk.StringVar(value=self.settings.get("cycles_dir", str(Path(self.data_dir.get()) / "kuopio")))
        self._data_tab()
        self._train_tab()
        self._tune_tab()
        self._evaluate_tab()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(100, self._poll)

    # --- settings ---------------------------------------------------------------------------

    @staticmethod
    def _load_settings() -> dict:
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        values = {"data_dir": self.data_dir.get(), "cycles_dir": self.cycles_dir.get(),
                  "model": self.eval_model.get(), "eval_dir": self.eval_dir.get(), "source": self.source.get()}
        for name in ("trials_root", "trials_out", "markerset", "sensor", "imu_axes", "imu_rate", "imu_offset",
                     "trials_split", "trials_split_file", "arrays_root", "arrays_spec", "arrays_out", "arrays_split",
                     "arrays_split_file"):
            values[name] = getattr(self, name).get()
        try:
            SETTINGS_FILE.write_text(json.dumps(values, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _close(self) -> None:
        if self.job and self.job.process and self.job.process.poll() is None:
            if not messagebox.askyesno("Quit", "A job is running. Stop it and quit?"):
                return
            self.job.stop()
        self._save_settings()
        self.root.destroy()

    # --- job handling ---------------------------------------------------------------------------

    def _run(self, args: list[str], log: Log, parse, done, buttons: list[ttk.Button]) -> None:
        if self.job and self.job.process and self.job.process.poll() is None:
            messagebox.showinfo("Busy", "Another job is still running. Stop it first.")
            return
        self._save_settings()
        log.clear()
        log.add("> gait-balance " + " ".join(f'"{a}"' if " " in a else a for a in args))
        for b in buttons:
            b.state(["disabled"])
        self.progress["value"] = 0
        self.status.set("Running …")
        started = time.time()

        def on_done(code, stopped):
            self.queue.put(("done", (log, code, stopped, done, buttons, started)))

        self.job = Job(args, lambda line: self.queue.put(("line", (log, line, parse))), on_done)
        self.job.start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "line":
                    log, line, parse = payload
                    log.add(line)
                    if parse:
                        parse(line)
                else:
                    log, code, stopped, done, buttons, started = payload
                    for b in buttons:
                        b.state(["!disabled"])
                    minutes = (time.time() - started) / 60
                    if stopped:
                        self.status.set("Stopped.")
                        log.add("Stopped by the user.")
                    elif code == 0:
                        self.progress["value"] = 1.0
                        self.status.set(f"Finished in {minutes:.1f} min.")
                        if done:
                            try:
                                done()
                            except Exception as error:  # show, do not crash the window
                                log.add(f"Could not show the results: {error}")
                    else:
                        self.status.set("Failed: see the messages above.")
                        messagebox.showerror("Failed", "The job stopped with an error. The last messages in the log "
                                                       "say why.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _stop(self) -> None:
        if self.job:
            self.job.stop()

    def _set_progress(self, fraction: float, text: str) -> None:
        self.progress["value"] = max(0.0, min(1.0, fraction))
        self.status.set(text)

    # --- tab 1: data -----------------------------------------------------------------------------

    def _data_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=PAD * 2)
        self.notebook.add(tab, text="  1 · Data  ")
        self.source = tk.StringVar(value=self.settings.get("source", "public"))
        choice = ttk.Frame(tab)
        choice.pack(fill="x", pady=(0, PAD))
        ttk.Label(choice, text="Data from:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, PAD))
        for value, text in (("public", "a public data set (download)"), ("trials", "my trial folders (C3D + IMU)"),
                            ("arrays", "my processed gait cycles (.npz / .mat)")):
            ttk.Radiobutton(choice, text=text, variable=self.source, value=value,
                            command=self._show_source).pack(side="left", padx=(0, PAD * 2))
        self.source_frames = {"public": ttk.Frame(tab), "trials": ttk.Frame(tab), "arrays": ttk.Frame(tab)}
        self._public_frame(self.source_frames["public"])
        self._trials_frame(self.source_frames["trials"])
        self._arrays_frame(self.source_frames["arrays"])
        self.data_bottom = ttk.Frame(tab)
        self.data_summary = tk.StringVar(value="")
        ttk.Label(self.data_bottom, textvariable=self.data_summary, foreground="#1f4e79", wraplength=1080,
                  font=("Segoe UI", 10, "bold")).pack(fill="x")
        self.data_log = Log(self.data_bottom, height=12)
        self.data_log.pack(fill="both", expand=True, pady=(PAD, 0))
        self._show_source()
        self._show_data_summary()

    def _show_source(self) -> None:
        for frame in self.source_frames.values():
            frame.pack_forget()
        self.data_bottom.pack_forget()
        self.source_frames[self.source.get()].pack(fill="x")
        self.data_bottom.pack(fill="both", expand=True)

    def _public_frame(self, tab: ttk.Frame) -> None:
        box = ttk.LabelFrame(tab, text="Data set", padding=PAD * 2)
        box.pack(fill="x")
        self.dataset = tk.StringVar(value=next(iter(DATASETS)))
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Data set", width=22).pack(side="left")
        ttk.Combobox(row, textvariable=self.dataset, values=list(DATASETS), state="readonly", width=20).pack(side="left")
        ttk.Label(box, text=DATASETS[self.dataset.get()].CITATION, wraplength=1000, foreground="#555",
                  justify="left").pack(fill="x", pady=(4, 6))
        PathRow(box, "Data folder", self.data_dir).pack(fill="x", pady=2)
        ttk.Label(box, text="Raw files go to <data folder>\\raw\\<data set>, gait cycles to <data folder>\\<data set>. "
                            "About 9 GB for Kuopio; avoid folders synchronized by OneDrive.",
                  foreground="#555").pack(fill="x", pady=(0, 4))

        opts = ttk.LabelFrame(tab, text="Download options", padding=PAD * 2)
        opts.pack(fill="x", pady=PAD)
        self.method = tk.StringVar(value="partial")
        ttk.Radiobutton(opts, text="Only the files needed (about 7 GB)", variable=self.method,
                        value="partial").pack(anchor="w")
        ttk.Radiobutton(opts, text="Whole archives (23 GB)", variable=self.method, value="full").pack(anchor="w")
        self.with_mtb = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Also the raw Xsens .mtb recordings (0.5 GB; used if the Xsens MT Software Suite "
                                   "is installed)", variable=self.with_mtb).pack(anchor="w", pady=(4, 0))
        self.zip_dir = tk.StringVar(value="")
        PathRow(opts, "Archives already on disk", self.zip_dir).pack(fill="x", pady=(6, 0))
        ttk.Label(opts, text="(optional: a folder with the data set's zip archives, used instead of the internet)",
                  foreground="#555").pack(anchor="w")

        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", pady=PAD)
        self.b_download = ttk.Button(buttons, text="① Download", style="Big.TButton", command=self._download)
        self.b_build = ttk.Button(buttons, text="② Convert to gait cycles", style="Big.TButton", command=self._build)
        self.b_download.pack(side="left")
        self.b_build.pack(side="left", padx=PAD)
        ttk.Button(buttons, text="Stop", command=self._stop).pack(side="left")
        ttk.Button(buttons, text="Open data folder",
                   command=lambda: open_path(Path(self.data_dir.get()))).pack(side="right")

    def _split_row(self, parent, prefix: str) -> None:
        """'Divide into sets' choice + split-file row; variables self.<prefix>_split and _split_file."""
        split = tk.StringVar(value=self.settings.get(f"{prefix}_split", SPLITS[0]))
        split_file = tk.StringVar(value=self.settings.get(f"{prefix}_split_file", ""))
        setattr(self, f"{prefix}_split", split)
        setattr(self, f"{prefix}_split_file", split_file)
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(6, 2))
        ttk.Label(row, text="Divide into sets", width=22).pack(side="left")
        ttk.Combobox(row, textvariable=split, values=SPLITS, state="readonly", width=46).pack(side="left")
        PathRow(parent, "Split file (JSON)", split_file, kind="file",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")]).pack(fill="x", pady=2)

    def _split_args(self, prefix: str) -> list[str] | None:
        split, split_file = getattr(self, f"{prefix}_split").get(), getattr(self, f"{prefix}_split_file").get().strip()
        if split == SPLITS[1]:
            return ["--split", "80/10/10"]
        if split == SPLITS[2]:
            if not Path(split_file).is_file():
                messagebox.showerror("Split file", "Choose the split file (JSON with training, validation and "
                                                   "testing lists of trial names).")
                return None
            return ["--split-file", split_file]
        return []

    def _trials_frame(self, tab: ttk.Frame) -> None:
        box = ttk.LabelFrame(tab, text="Trial folders", padding=PAD * 2)
        box.pack(fill="x")
        ttk.Label(box, text="Each trial folder holds a static C3D, a walking C3D (over three or four floor force "
                            "plates) and the IMU file of the same walk (.mtb, .mat or .csv). All trial folders below "
                            "the chosen folder are converted; each becomes one gait cycle with imu_input.csv and "
                            "ground_truth.csv.", wraplength=1060, foreground="#555", justify="left").pack(fill="x")
        self.trials_root = tk.StringVar(value=self.settings.get("trials_root", ""))
        self.trials_out = tk.StringVar(value=self.settings.get("trials_out", ""))
        PathRow(box, "Trial folders in", self.trials_root).pack(fill="x", pady=(6, 2))
        PathRow(box, "Gait cycles to", self.trials_out).pack(fill="x", pady=2)
        sets = [p.stem for p in sorted((CONFIGS / "markersets").glob("*.json"))] + ["Other file…"]
        self.markerset = tk.StringVar(value=self.settings.get("markerset", "yu2023"))
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Marker set", width=22).pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.markerset, values=sets, width=40)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._markerset_changed)
        ttk.Label(row, text="  yu2023 = the paper's marker set (configs/markersets)", foreground="#555").pack(side="left")
        imu = ttk.LabelFrame(tab, text="IMU", padding=PAD * 2)
        imu.pack(fill="x", pady=PAD)
        self.sensor = tk.StringVar(value=self.settings.get("sensor", ""))
        self.imu_axes = tk.StringVar(value=self.settings.get("imu_axes", "-z,x,-y"))
        self.imu_rate = tk.StringVar(value=self.settings.get("imu_rate", "100"))
        self.imu_offset = tk.StringVar(value=self.settings.get("imu_offset", "0"))
        for label, var, note in (
                ("Sacral sensor ID", self.sensor, "only needed if the IMU files hold several sensors"),
                ("Sensor axes", self.imu_axes, "sensor axes pointing forward, up and right; -z,x,-y = Xsens MTw "
                                               "with x up and the case facing backwards"),
                ("Sampling rate (Hz)", self.imu_rate, ""),
                ("IMU start delay (s)", self.imu_offset, "0 when one trigger starts motion capture and IMU")):
            row = ttk.Frame(imu)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=22).pack(side="left")
            ttk.Entry(row, textvariable=var, width=16).pack(side="left")
            ttk.Label(row, text="  " + note, foreground="#555").pack(side="left")
        self._split_row(imu, "trials")
        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", pady=(0, PAD))
        self.b_convert = ttk.Button(buttons, text="▶ Convert trial folders", style="Big.TButton",
                                    command=self._convert_trials)
        self.b_convert.pack(side="left")
        ttk.Button(buttons, text="Stop", command=self._stop).pack(side="left", padx=PAD)
        ttk.Button(buttons, text="Open output folder",
                   command=lambda: open_path(Path(self.trials_out.get()))).pack(side="right")

    def _markerset_changed(self, _event=None) -> None:
        if self.markerset.get() != "Other file…":
            return
        path = filedialog.askopenfilename(initialdir=str(CONFIGS / "markersets"), filetypes=[("JSON", "*.json")])
        self.markerset.set(os.path.normpath(path) if path else "yu2023")

    def _arrays_frame(self, tab: ttk.Frame) -> None:
        box = ttk.LabelFrame(tab, text="Processed gait cycles", padding=PAD * 2)
        box.pack(fill="x")
        ttk.Label(box, text="One gait cycle per file (or pair of files) with the IMU input and the IA, already "
                            "processed as in the paper (body axes, filtered, right cycles mirrored). A spec file "
                            "(JSON) says where the arrays are; see docs/YOUR_DATA.md.", wraplength=1060,
                  foreground="#555", justify="left").pack(fill="x")
        self.arrays_root = tk.StringVar(value=self.settings.get("arrays_root", ""))
        self.arrays_spec = tk.StringVar(value=self.settings.get("arrays_spec", ""))
        self.arrays_out = tk.StringVar(value=self.settings.get("arrays_out", ""))
        PathRow(box, "Array files in", self.arrays_root).pack(fill="x", pady=(6, 2))
        PathRow(box, "Spec file (JSON)", self.arrays_spec, kind="file",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")]).pack(fill="x", pady=2)
        PathRow(box, "Gait cycles to", self.arrays_out).pack(fill="x", pady=2)
        self._split_row(box, "arrays")
        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", pady=PAD)
        self.b_import = ttk.Button(buttons, text="▶ Import", style="Big.TButton", command=self._import_arrays)
        self.b_import.pack(side="left")
        ttk.Button(buttons, text="Stop", command=self._stop).pack(side="left", padx=PAD)
        ttk.Button(buttons, text="Open output folder",
                   command=lambda: open_path(Path(self.arrays_out.get()))).pack(side="right")

    def _own_data_done(self, out: Path, divided: bool) -> None:
        """After converting or importing: show the summary and fill in the next tabs."""
        lines = (out / "README.txt").read_text(encoding="utf-8").splitlines() if (out / "README.txt").exists() else []
        found = [line for line in lines if "gait cycles" in line and "(" in line]
        if divided:
            self.cycles_dir.set(str(out))
            follow = "Next: '2 · Train & test' (the folder is filled in)."
        else:
            self.eval_dir.set(str(out))
            follow = "Next: test a trained model on them in '4 · Test a model' (the folder is filled in)."
        self.data_summary.set(f"{found[0] if found else 'Done.'} {follow}")

    def _convert_trials(self) -> None:
        root, out = self.trials_root.get().strip(), self.trials_out.get().strip()
        if not Path(root).is_dir() or not out:
            messagebox.showerror("Missing", "Choose the folder with the trial folders and an output folder.")
            return
        try:
            float(self.imu_rate.get()), float(self.imu_offset.get())
        except ValueError:
            messagebox.showerror("IMU", "Sampling rate and start delay must be numbers.")
            return
        split = self._split_args("trials")
        if split is None:
            return
        args = ["convert-trials", root, "--out", out, "--markerset", self.markerset.get(), "--imu-axes",
                self.imu_axes.get(), "--imu-rate", self.imu_rate.get(), "--imu-offset", self.imu_offset.get(), *split]
        if self.sensor.get().strip():
            args += ["--sensor", self.sensor.get().strip()]

        def parse(line):
            m = re.search(r"(\d+)/(\d+) converted", line)
            if m:
                self._set_progress(int(m.group(1)) / int(m.group(2)),
                                   f"Converting trial folders: {m.group(1)} of {m.group(2)}")

        self._run(args, self.data_log, parse, lambda: self._own_data_done(Path(out), bool(split)),
                  [self.b_convert, self.b_import, self.b_download, self.b_build])

    def _import_arrays(self) -> None:
        root, spec, out = self.arrays_root.get().strip(), self.arrays_spec.get().strip(), self.arrays_out.get().strip()
        if not Path(root).is_dir() or not Path(spec).is_file() or not out:
            messagebox.showerror("Missing", "Choose the folder with the array files, the spec file and an output folder.")
            return
        split = self._split_args("arrays")
        if split is None:
            return

        def parse(line):
            m = re.search(r"(\d+)/(\d+) imported", line)
            if m:
                self._set_progress(int(m.group(1)) / int(m.group(2)), f"Importing: {m.group(1)} of {m.group(2)}")

        self._run(["import-arrays", root, "--spec", spec, "--out", out, *split], self.data_log, parse,
                  lambda: self._own_data_done(Path(out), bool(split)),
                  [self.b_convert, self.b_import, self.b_download, self.b_build])

    def _show_data_summary(self) -> None:
        readme = Path(self.data_dir.get()) / self.dataset.get() / "README.txt"
        lines = readme.read_text(encoding="utf-8").splitlines() if readme.exists() else []
        found = [line for line in lines if "gait cycles from" in line]
        self.data_summary.set(f"Ready for training: {found[0]}" if found else "")

    def _download(self) -> None:
        args = ["download", self.dataset.get(), "--data-dir", self.data_dir.get(), "--method", self.method.get()]
        if self.zip_dir.get().strip():
            args += ["--zip-dir", self.zip_dir.get().strip()]
        if self.with_mtb.get():
            args.append("--with-mtb")
        archives = {"count": 0}

        def parse(line):
            if "files wanted" in line:
                archives["count"] += 1
            m = re.search(r"([\d.]+) of ([\d.]+) GB, ([\d.]+) MB/s, about (\d+) min left", line)
            if m:
                self._set_progress(float(m.group(1)) / max(float(m.group(2)), 1e-9),
                                   f"Downloading archive {archives['count']}: {m.group(1)} of {m.group(2)} GB, "
                                   f"{m.group(3)} MB/s, about {m.group(4)} min left")

        self._run(args, self.data_log, parse, None, [self.b_download, self.b_build])

    def _build(self) -> None:
        def parse(line):
            m = re.search(r"(\d+)/(\d+) converted", line)
            if m:
                self._set_progress(int(m.group(1)) / int(m.group(2)),
                                   f"Converting trials: {m.group(1)} of {m.group(2)}")

        def done():
            self._show_data_summary()
            self.cycles_dir.set(str(Path(self.data_dir.get()) / self.dataset.get()))

        self._run(["build", self.dataset.get(), "--data-dir", self.data_dir.get()], self.data_log, parse, done,
                  [self.b_download, self.b_build])

    # --- tab 2: train and test --------------------------------------------------------------------

    def _train_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=PAD * 2)
        self.notebook.add(tab, text="  2 · Train & test  ")
        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.LabelFrame(tab, text="Figure", padding=PAD)
        right.pack(side="right", fill="both", expand=True, padx=(PAD * 2, 0))

        box = ttk.LabelFrame(left, text="Settings", padding=PAD * 2)
        box.pack(fill="x")
        PathRow(box, "Gait cycles", self.cycles_dir).pack(fill="x", pady=2)
        ttk.Label(box, text="A folder with training\\, validation\\ and testing\\ (e.g. <data folder>\\kuopio); "
                            "without validation\\, 1/9 of the training cycles are used", foreground="#555"
                  ).pack(anchor="w")
        self.preset = tk.StringVar(value=next(iter(PRESETS)))
        self.config_file = tk.StringVar(value=str(PRESETS[self.preset.get()]))
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(6, 2))
        ttk.Label(row, text="Hyper-parameters", width=22).pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.preset, values=list(PRESETS) + ["Other file…"],
                             state="readonly", width=26)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._preset_changed)
        ttk.Label(box, textvariable=self.config_file, foreground="#555").pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(6, 2))
        ttk.Label(row, text="Seeds", width=22).pack(side="left")
        self.seeds = tk.StringVar(value="0")
        ttk.Entry(row, textvariable=self.seeds, width=16).pack(side="left")
        ttk.Label(row, text="  one training run per seed, e.g. 0 1 2 3 4", foreground="#555").pack(side="left")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Compute device", width=22).pack(side="left")
        self.device = tk.StringVar(value="auto")
        ttk.Combobox(row, textvariable=self.device, values=["auto", "cuda", "cpu"], state="readonly",
                     width=8).pack(side="left")
        ttk.Label(row, text="  auto uses an NVIDIA GPU when available", foreground="#555").pack(side="left")

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=PAD)
        self.b_train = ttk.Button(buttons, text="▶ Train and test", style="Big.TButton", command=self._train)
        self.b_train.pack(side="left")
        ttk.Button(buttons, text="Stop", command=self._stop).pack(side="left", padx=PAD)
        self.b_open_run = ttk.Button(buttons, text="Open results folder", state="disabled",
                                     command=lambda: open_path(self.last_run))
        self.b_open_run.pack(side="right")

        res = ttk.LabelFrame(left, text="Test error (mean over test cycles, SD in brackets)", padding=PAD)
        res.pack(fill="x")
        cols = [("Variable", 110), ("RMSE", 110), ("rRMSE (%)", 90), ("Paper RMSE", 95), ("Paper rRMSE (%)", 105)]
        self.train_table = ResultTable(res, cols, height=5)
        self.train_table.pack(fill="x")
        self.train_log = Log(left, height=9)
        self.train_log.pack(fill="both", expand=True, pady=(PAD, 0))

        self.train_plot = tk.StringVar(value="Example test cycle")
        top = ttk.Frame(right)
        top.pack(fill="x")
        plot_combo = ttk.Combobox(top, textvariable=self.train_plot, state="readonly", width=22,
                                  values=["Example test cycle", "Validation error per epoch"])
        plot_combo.pack(side="left")
        plot_combo.bind("<<ComboboxSelected>>", lambda e: self._draw_train_plot())
        self.train_caption = tk.StringVar(value="Train a model to see its results here.")
        ttk.Label(right, textvariable=self.train_caption, foreground="#555").pack(anchor="w", pady=(4, 0))
        self.train_fig = Figure(figsize=(5.2, 4.6), dpi=100)
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, master=right)
        self.train_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.last_run: Path | None = None

    def _preset_changed(self, _event=None) -> None:
        if self.preset.get() in PRESETS:
            self.config_file.set(str(PRESETS[self.preset.get()]))
            return
        path = filedialog.askopenfilename(initialdir=str(CONFIGS), filetypes=[("JSON", "*.json")])
        if path:
            self.config_file.set(os.path.normpath(path))
        else:
            self.preset.set(next(iter(PRESETS)))
            self.config_file.set(str(PRESETS[self.preset.get()]))

    def _train(self) -> None:
        seeds = self.seeds.get().replace(",", " ").split()
        if not seeds or not all(s.isdigit() for s in seeds):
            messagebox.showerror("Seeds", "Seeds must be whole numbers, e.g. 0 1 2 3 4.")
            return
        out = Path.cwd() / "runs" / time.strftime("train_%Y%m%d_%H%M%S")
        try:
            max_epochs = int(json.loads(Path(self.config_file.get()).read_text(encoding="utf-8")).get("max_epochs", 100))
        except Exception:
            max_epochs = 250
        state = {"seed": 0}

        def parse(line):
            m = re.match(r"\s*epoch\s+(\d+):", line)
            if m:
                epoch = int(m.group(1))
                self._set_progress((state["seed"] + epoch / max_epochs) / len(seeds),
                                   f"Training run {state['seed'] + 1} of {len(seeds)}: epoch {epoch} of at most "
                                   f"{max_epochs}")
            if line.startswith("seed "):
                state["seed"] += 1

        def done():
            self.last_run = out
            self.b_open_run.state(["!disabled"])
            summary = pd.read_csv(out / "summary.csv", index_col=0)
            table = summary.rename(columns={"rmse_sd_cycles": "rmse_sd", "rrmse_sd_cycles": "rrmse_sd"})
            self.train_table.fill(summary_rows(table))
            self.eval_model.set(str(out / "seed0" / "model.pt"))
            self._draw_train_plot()

        args = ["train", "--data", self.cycles_dir.get(), "--config", self.config_file.get(), "--seeds", *seeds,
                "--device", self.device.get(), "--out", str(out)]
        self._run(args, self.train_log, parse, done, [self.b_train])

    def _draw_train_plot(self) -> None:
        if self.last_run is None:
            return
        if self.train_plot.get() == "Example test cycle":
            caption = plot_example(self.train_fig, self.last_run / "seed0" / "model.pt",
                                   Path(self.cycles_dir.get()) / "testing")
        else:
            caption = plot_history(self.train_fig, self.last_run)
        self.train_caption.set(caption)
        self.train_canvas.draw()

    # --- tab 3: Bayesian tuning ------------------------------------------------------------------

    def _tune_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=PAD * 2)
        self.notebook.add(tab, text="  3 · Bayesian tuning  ")
        box = ttk.LabelFrame(tab, text="Settings", padding=PAD * 2)
        box.pack(fill="x")
        PathRow(box, "Gait cycles", self.cycles_dir).pack(fill="x", pady=2)
        ttk.Label(box, text="Only the training and validation cycles are used; the test cycles stay untouched.",
                  foreground="#555").pack(anchor="w")
        self.n_trials, self.max_epochs, self.patience = tk.StringVar(value="50"), tk.StringVar(value="250"), \
            tk.StringVar(value="30")
        for label, var, note in (("Search trials", self.n_trials, "each trial is one training run"),
                                 ("Max epochs per trial", self.max_epochs, ""),
                                 ("Early-stop patience", self.patience, "epochs without improvement")):
            row = ttk.Frame(box)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=22).pack(side="left")
            ttk.Entry(row, textvariable=var, width=8).pack(side="left")
            ttk.Label(row, text="  " + note, foreground="#555").pack(side="left")
        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", pady=PAD)
        self.b_tune = ttk.Button(buttons, text="▶ Start Bayesian tuning", style="Big.TButton", command=self._tune)
        self.b_tune.pack(side="left")
        ttk.Button(buttons, text="Stop", command=self._stop).pack(side="left", padx=PAD)
        self.b_use_best = ttk.Button(buttons, text="Use the best settings in 'Train & test'", state="disabled",
                                     command=self._use_best)
        self.b_use_best.pack(side="right")
        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True)
        trials = ttk.LabelFrame(body, text="Search trials", padding=PAD)
        trials.pack(side="left", fill="y")
        self.tune_table = ResultTable(trials, [("Trial", 50), ("Validation rRMSE (%)", 140), ("Best so far (%)", 110)],
                                      height=18)
        self.tune_table.pack(fill="y", expand=True)
        self.best_text = tk.StringVar(value="")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(PAD * 2, 0))
        ttk.Label(right, textvariable=self.best_text, foreground="#1f4e79", wraplength=640,
                  justify="left").pack(fill="x")
        self.tune_log = Log(right, height=16)
        self.tune_log.pack(fill="both", expand=True, pady=(PAD, 0))
        self.best_config: Path | None = None

    def _tune(self) -> None:
        if not all(v.get().isdigit() for v in (self.n_trials, self.max_epochs, self.patience)):
            messagebox.showerror("Settings", "Search trials, max epochs and patience must be whole numbers.")
            return
        out = Path.cwd() / "runs" / time.strftime("tune_%Y%m%d_%H%M%S")
        n = int(self.n_trials.get())
        self.tune_table.fill([])
        self.best_text.set("")

        def parse(line):
            m = re.match(r"search trial (\d+): ([^\s;]+)(?: %)?(?:; best ([\d.]+) %)?", line)
            if m:
                value = m.group(2) if m.group(2) in ("pruned", "fail", "failed") else f"{float(m.group(2)):.3f}"
                self.tune_table.insert("", "end", values=[m.group(1), value, m.group(3) or ""])
                self.tune_table.see(self.tune_table.get_children()[-1])
                self._set_progress(int(m.group(1)) / n, f"Search trial {m.group(1)} of {n}")

        def done():
            self.best_config = out / "best_config.json"
            best = json.loads(self.best_config.read_text(encoding="utf-8"))
            self.best_text.set("Best settings: " + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                                                             for k, v in best.items()))
            self.b_use_best.state(["!disabled"])

        args = ["tune", "--data", self.cycles_dir.get(), "--trials", self.n_trials.get(), "--max-epochs",
                self.max_epochs.get(), "--patience", self.patience.get(), "--out", str(out)]
        self._run(args, self.tune_log, parse, done, [self.b_tune])

    def _use_best(self) -> None:
        if self.best_config:
            self.preset.set("Other file…")
            self.config_file.set(str(self.best_config))
            self.notebook.select(1)

    # --- tab 4: evaluate ----------------------------------------------------------------------------

    def _evaluate_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=PAD * 2)
        self.notebook.add(tab, text="  4 · Test a model  ")
        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.LabelFrame(tab, text="Figure", padding=PAD)
        right.pack(side="right", fill="both", expand=True, padx=(PAD * 2, 0))
        box = ttk.LabelFrame(left, text="Model and gait cycles", padding=PAD * 2)
        box.pack(fill="x")
        self.eval_model = tk.StringVar(value=self.settings.get("model", ""))
        self.eval_dir = tk.StringVar(value=self.settings.get("eval_dir", ""))
        PathRow(box, "Trained model (model.pt)", self.eval_model, kind="file",
                filetypes=[("Model", "*.pt"), ("All files", "*.*")]).pack(fill="x", pady=2)
        PathRow(box, "Gait cycles to test", self.eval_dir).pack(fill="x", pady=2)
        ttk.Label(box, text="Any folder of gait-cycle folders, e.g. another data set's testing\\ folder",
                  foreground="#555").pack(anchor="w")
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=PAD)
        self.b_eval = ttk.Button(buttons, text="▶ Test the model", style="Big.TButton", command=self._evaluate)
        self.b_eval.pack(side="left")
        res = ttk.LabelFrame(left, text="Error (mean over the gait cycles, SD in brackets)", padding=PAD)
        res.pack(fill="x")
        cols = [("Variable", 110), ("RMSE", 110), ("rRMSE (%)", 90), ("Paper RMSE", 95), ("Paper rRMSE (%)", 105)]
        self.eval_table = ResultTable(res, cols, height=5)
        self.eval_table.pack(fill="x")
        self.eval_log = Log(left, height=10)
        self.eval_log.pack(fill="both", expand=True, pady=(PAD, 0))
        self.eval_caption = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.eval_caption, foreground="#555").pack(anchor="w")
        self.eval_fig = Figure(figsize=(5.2, 4.6), dpi=100)
        self.eval_canvas = FigureCanvasTkAgg(self.eval_fig, master=right)
        self.eval_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _evaluate(self) -> None:
        model, folder = Path(self.eval_model.get()), Path(self.eval_dir.get())
        if not model.is_file() or not folder.is_dir():
            messagebox.showerror("Missing", "Choose a model file and a folder of gait cycles.")
            return
        out = Path.cwd() / "runs" / time.strftime("evaluate_%Y%m%d_%H%M%S.csv")
        out.parent.mkdir(parents=True, exist_ok=True)

        def done():
            errors = pd.read_csv(out)
            stats = pd.DataFrame({"rmse": [errors[f"RMSE {n}"].mean() for n in NAMES],
                                  "rmse_sd": [errors[f"RMSE {n}"].std() for n in NAMES],
                                  "rrmse": [errors[f"rRMSE {n} (%)"].mean() for n in NAMES],
                                  "rrmse_sd": [errors[f"rRMSE {n} (%)"].std() for n in NAMES]}, index=NAMES)
            self.eval_table.fill(summary_rows(stats))
            self.eval_caption.set(plot_example(self.eval_fig, model, folder))
            self.eval_canvas.draw()

        self._run(["evaluate", "--model", str(model), "--test", str(folder), "--out", str(out)], self.eval_log,
                  None, done, [self.b_eval])


def main() -> None:
    if os.name == "nt":  # sharp text on high-resolution screens
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
