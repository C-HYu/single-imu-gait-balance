"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/xsens.py
Description : Optional reading of Xsens .mtb recordings (MT Manager log files).
              The format is proprietary: decoding it needs the Xsens Device
              API, which comes with the free Xsens MT Software Suite
              (https://www.xsens.com/support/software-documentation; for MTw
              Awinda use the Awinda release). Two interfaces are found
              automatically:
              - the Python module "xsensdeviceapi" of recent MT Software Suite
                versions; its wheel file in the MT SDK folder has to be
                installed into this environment (`gait-balance xsens-check`
                prints the command);
              - the COM interface that MT Software Suite 4.x registers on
                Windows (needs pywin32). Tested with version 4.6.
              The first packet of every sensor is dropped, so sample k matches
              row k of files exported with the same API (e.g. the Kuopio
              authors' extracted IMU files).
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

COM_PROGID = "xsensdeviceapi_com64.IXsensDeviceApi"
DOWNLOAD_PAGE = "https://www.xsens.com/support/software-documentation"


def python_api_available() -> bool:
    try:
        import xsensdeviceapi  # noqa: F401
    except Exception:
        return False
    return True


def com_api_available() -> bool:
    """True if MT Software Suite 4.x registered its COM interface (Windows) and pywin32 is installed."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import winreg

        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, COM_PROGID))
        import win32com.client  # noqa: F401
    except Exception:
        return False
    return True


def backend() -> str | None:
    """"python", "com" or None (no Xsens Device API found)."""
    return "python" if python_api_available() else "com" if com_api_available() else None


def find_wheels() -> list[Path]:
    """xsensdeviceapi wheel files of an installed MT Software Suite."""
    roots = [Path(p) for p in (r"C:\Program Files\Xsens", r"C:\Program Files (x86)\Xsens",
                               "/usr/local/xsens", "/opt/xsens")]
    return sorted(w for root in roots if root.exists() for w in root.rglob("xsensdeviceapi*.whl"))


def check() -> str:
    """A short report: can .mtb files be read here, and if not, what to do."""
    found = backend()
    if found == "python":
        return "Xsens Device API found (Python module xsensdeviceapi): .mtb files can be read."
    if found == "com":
        return "Xsens Device API found (COM interface of MT Software Suite 4.x): .mtb files can be read."
    lines = ["No Xsens Device API found, so .mtb files cannot be read.", "",
             f"1. Install the Xsens MT Software Suite from {DOWNLOAD_PAGE}",
             "   (for MTw Awinda recordings choose the Awinda release)."]
    wheels = find_wheels()
    version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if wheels:
        match = [w for w in wheels if version in w.name] or wheels
        lines.append(f"2. Install its Python module into this environment:\n   pip install \"{match[-1]}\"")
        if not any(version in w.name for w in wheels):
            lines.append(f"   (none of the wheels is built for this Python ({version}); create an environment "
                         "with a Python version that matches one of them)")
    else:
        lines.append("2. Then run this check again: it shows how to connect the installed software.")
    if sys.platform.startswith("win"):
        lines.append("   With MT Software Suite 4.x nothing else is needed (its COM interface is used; "
                     "pywin32 is part of environment.yml).")
    return "\n".join(lines)


# =============================================================================
# Reading
# =============================================================================


def _read_python(path: Path) -> dict[str, dict[str, np.ndarray]]:
    import xsensdeviceapi as xda

    control = xda.XsControl_construct()
    try:
        if not control.openLogFile(str(path)):
            raise RuntimeError(f"Xsens Device API could not open {path.name}")
        main = control.device(control.mainDeviceIds()[0])
        children = main.children()
        devices = [children[i] for i in range(children.size())] if children.size() else [main]
        flags = 0
        for name in ("XSO_RetainRecordingData", "XSO_RetainBufferedData", "XSO_Calibrate"):  # names vary by version
            flags |= int(getattr(xda, name, 0))
        for device in [main] + devices:
            device.setOptions(flags, xda.XSO_None)
        main.loadLogFile()
        _wait(lambda: [d.getDataPacketCount() for d in devices])
        out = {}
        for device in devices:
            rows = []
            for k in range(1, device.getDataPacketCount()):
                packet = device.getDataPacketByIndex(k)
                acc = packet.calibratedAcceleration() if packet.containsCalibratedAcceleration() else None
                gyro = packet.calibratedGyroscopeData() if packet.containsCalibratedGyroscopeData() else None
                counter = packet.packetCounter() if packet.containsPacketCounter() else len(rows)
                rows.append([counter] + [acc[i] if acc is not None else np.nan for i in range(3)]
                            + [gyro[i] if gyro is not None else np.nan for i in range(3)])
            if rows:
                out[format(device.deviceId().toInt() & 0xFFFFFFFF, "X")] = _arrays(rows)
        return out
    finally:
        control.close()


def _read_com(path: Path) -> dict[str, dict[str, np.ndarray]]:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    try:
        api = win32com.client.Dispatch(COM_PROGID)
        if not api.XsControl_openLogFile(str(path)):
            raise RuntimeError(f"Xsens Device API could not open {path.name}")
        try:
            ids = list(api.XsControl_mainDeviceIds())
            main = api.XsControl_device(ids[0])
            devices = list(api.XsDevice_children(main)) if api.XsDeviceId_isWirelessMaster(ids[0]) else [main]
            # Every sensor must keep the loaded packets; the option values are COM functions.
            enable = int(api.XsOption_XSO_RetainRecordingData()) | int(api.XsOption_XSO_Calibrate())
            for device in [main] + devices:
                try:
                    api.XsDevice_setOptions(device, enable, int(api.XsOption_XSO_None()))
                except Exception:
                    pass
            api.XsDevice_loadLogFile(main)
            _wait(lambda: [api.XsDevice_getDataPacketCount(d) for d in devices], pythoncom.PumpWaitingMessages)
            out = {}
            for device in devices:
                rows = []
                for k in range(1, api.XsDevice_getDataPacketCount(device)):
                    p = api.XsDevice_getDataPacketByIndex(device, k)
                    if not p:
                        continue
                    acc = (list(api.XsDataPacket_calibratedAcceleration(p))
                           if api.XsDataPacket_containsCalibratedAcceleration(p) else [np.nan] * 3)
                    gyro = (list(api.XsDataPacket_calibratedGyroscopeData(p))
                            if api.XsDataPacket_containsCalibratedGyroscopeData(p) else [np.nan] * 3)
                    counter = (api.XsDataPacket_packetCounter(p)
                               if api.XsDataPacket_containsPacketCounter(p) else len(rows))
                    rows.append([counter, *acc[:3], *gyro[:3]])
                if rows:
                    out[format(int(api.XsDevice_deviceId(device)) & 0xFFFFFFFF, "X")] = _arrays(rows)
            return out
        finally:
            api.XsControl_close()
    finally:
        pythoncom.CoUninitialize()


def _wait(counts, pump=None, timeout_s: float = 600.0) -> None:
    """Wait until loading the log file has finished (the packet counts stop growing)."""
    start, last, stable_since = time.time(), None, time.time()
    while time.time() - start < timeout_s:
        if pump is not None:
            pump()
        time.sleep(0.1)
        now = counts()
        if now != last:
            last, stable_since = now, time.time()
        elif sum(now) > 0 and time.time() - stable_since > 1.0:
            return
    raise RuntimeError("loading the .mtb file did not finish")


def _arrays(rows: list) -> dict[str, np.ndarray]:
    a = np.asarray(rows, dtype=float)
    return {"counter": a[:, 0], "acc": a[:, 1:4], "gyro": a[:, 4:7]}


def read_mtb(path: str | Path, timeout_s: float = 600.0) -> dict[str, dict[str, np.ndarray]]:
    """Calibrated acceleration (m/s^2) and angular velocity (rad/s) of every sensor in an .mtb file.

    Returns
    -------
    dict
        {sensor ID: {"counter": (n,), "acc": (n, 3), "gyro": (n, 3)}}, sensor axes.
    """
    path = Path(path).resolve()
    found = backend()
    if found == "python":
        return _read_python(path)
    if found == "com":
        # A separate process keeps the COM server out of this one and lets a stuck read time out.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mtb.npz"
            try:
                result = subprocess.run([sys.executable, "-m", "single_imu_gait_balance.xsens", str(path), str(target)],
                                        capture_output=True, text=True, timeout=timeout_s,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"reading {path.name} did not finish within {timeout_s:.0f} s") from None
            if result.returncode != 0 or not target.exists():
                raise RuntimeError(f"reading {path.name} failed: {result.stderr.strip()[-500:]}")
            with np.load(target) as z:
                out: dict[str, dict[str, np.ndarray]] = {}
                for key in z.files:
                    sensor, field = key.split("/")
                    out.setdefault(sensor, {})[field] = z[key]
                return out
    raise RuntimeError(f"{path.name}: .mtb files need the Xsens Device API.\n{check()}")


def mtb_to_csv(path: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """Write one CSV per sensor: packet counter, acc (m/s^2) and gyro (rad/s) in sensor axes."""
    path = Path(path)
    out = Path(out_dir) if out_dir else path.parent
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for sensor, data in read_mtb(path).items():
        table = pd.DataFrame(np.column_stack([data["counter"], data["acc"], data["gyro"]]),
                             columns=["packet_counter", "acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2",
                                      "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"])
        table["packet_counter"] = table["packet_counter"].astype(int)
        written.append(out / f"{path.stem}_{sensor}.csv")
        table.to_csv(written[-1], index=False)
    return written


if __name__ == "__main__":
    # Worker of read_mtb for the COM interface: python -m single_imu_gait_balance.xsens <in.mtb> <out.npz>
    sensors = _read_com(Path(sys.argv[1]).resolve())
    np.savez_compressed(sys.argv[2], **{f"{s}/{k}": v for s, d in sensors.items() for k, v in d.items()})
