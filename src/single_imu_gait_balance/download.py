"""
Single IMU Gait Balance
-----------------------
File        : src/single_imu_gait_balance/download.py
Description : Downloads from a public repository (e.g. Zenodo):
              - download_file: one file, resumable, checked against its MD5;
              - extract_remote: selected members of a large remote zip file,
                read through HTTP range requests, so only the needed part of
                the archive is transferred (the zip's own CRC checks each file);
              - extract_local: the same members from a zip already on disk.
Author      : Cheng-Hao Yu, PhD
Created     : 2026-10-01
Last updated: 2026-10-01
"""

from __future__ import annotations

import hashlib
import http.client
import io
import re
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import __version__

USER_AGENT = f"single-imu-gait-balance/{__version__}"
RETRY_STATUS = {429, 500, 502, 503, 504}


def _request(url: str, start: int | None = None, end: int | None = None, timeout: float = 120.0):
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-" + ("" if end is None else str(end))
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def _with_retries(action, what: str, retries: int = 8):
    """Run ``action()``, waiting and retrying on network errors and busy servers."""
    for attempt in range(retries):
        try:
            return action()
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS or attempt == retries - 1:
                raise
            retry_after = str(error.headers.get("Retry-After") or "")
            wait = float(retry_after) if retry_after.isdigit() else 2 ** attempt
        except (urllib.error.URLError, http.client.IncompleteRead, ConnectionError, TimeoutError) as error:
            if attempt == retries - 1:
                raise OSError(f"{what}: {error}") from error
            wait = 2 ** attempt
        print(f"    network problem with {what}; retrying in {wait:.0f} s", flush=True)
        time.sleep(min(wait, 120.0))
    raise OSError(f"{what}: failed after {retries} attempts")


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: str | Path, md5: str | None = None) -> Path:
    """Download ``url`` to ``target`` (resuming a partial download); verify the MD5 when given."""
    target = Path(target)
    if target.exists() and (md5 is None or md5sum(target) == md5):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")

    def fetch() -> None:
        done = partial.stat().st_size if partial.exists() else 0
        try:
            response = _request(url, done if done else None)
        except urllib.error.HTTPError as error:
            if error.code == 416:  # the partial file is already complete
                return
            raise
        with response:
            if done and response.status != 206:  # the server sent the whole file again
                done = 0
            with open(partial, "ab" if done else "wb") as f:
                shutil.copyfileobj(response, f, 1 << 20)

    _with_retries(fetch, target.name)
    if md5 is not None and md5sum(partial) != md5:
        partial.unlink()
        raise OSError(f"{target.name}: MD5 mismatch, the partial download was removed; please run again")
    partial.replace(target)
    return target


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url: str):
        super().__init__()
        self.url, self.pos, self.requests = url, 0, 0

        def size() -> int:
            with _request(url, 0, 0) as response:
                found = re.search(r"/(\d+)$", response.headers.get("Content-Range", ""))
                if response.status != 206 or not found:
                    raise OSError(f"{url} does not support range requests")
                return int(found.group(1))

        self.size = _with_retries(size, url)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        buffer[:n] = _get_range(self.url, self.pos, self.pos + n)
        self.requests += 1
        self.pos += n
        return n


def _get_range(url: str, start: int, end: int) -> bytes:
    """Bytes ``start`` to ``end - 1`` of a remote file, with retries."""
    def get() -> bytes:
        with _request(url, start, end - 1) as response:
            data = response.read()
        if len(data) != end - start:
            raise http.client.IncompleteRead(data, end - start - len(data))
        return data

    return _with_retries(get, f"bytes {start}-{end - 1}")


def _wanted(archive: zipfile.ZipFile, pattern: str, target: Path) -> tuple[list[zipfile.ZipInfo], list[zipfile.ZipInfo]]:
    """Members whose name matches ``pattern``, and those not yet in ``target`` (by size)."""
    regex = re.compile(pattern)
    members = sorted((m for m in archive.infolist() if regex.search(m.filename)), key=lambda m: m.header_offset)
    todo = [m for m in members if not ((target / m.filename).exists()
                                       and (target / m.filename).stat().st_size == m.file_size)]
    return members, todo


def _save(target: Path, member: zipfile.ZipInfo, data: bytes) -> None:
    path = target / member.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    partial.replace(path)


def _member_data(blob: bytes, at: int, member: zipfile.ZipInfo) -> bytes | None:
    """Unpack one member whose local header starts at ``blob[at]``; None if ``blob`` is too short."""
    header = blob[at:at + 30]
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        raise OSError(f"{member.filename}: no local file header at its offset")
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    start = at + 30 + name_length + extra_length
    packed = blob[start:start + member.compress_size]
    if len(packed) < member.compress_size:
        return None
    if member.compress_type == zipfile.ZIP_STORED:
        data = packed
    elif member.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(packed, -15)
    else:
        raise OSError(f"{member.filename}: compression method {member.compress_type} is not supported")
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise OSError(f"{member.filename}: CRC check failed")
    return data


def extract_remote(url: str, pattern: str, target: str | Path, connections: int = 6,
                   merge_gap: int = 1 << 20, max_request: int = 32 << 20) -> tuple[int, int]:
    """Fetch the members of a remote zip whose names match ``pattern`` into ``target``.

    Only the bytes of those members are requested: neighbouring members are
    merged into one request (gaps below ``merge_gap``), and ``connections``
    requests run at the same time. Each file is checked against the zip's CRC.
    Files already in ``target`` are skipped, so an interrupted run continues.

    Returns (files wanted, files fetched).
    """
    target = Path(target)
    raw = HttpRangeFile(url)
    with zipfile.ZipFile(io.BufferedReader(raw, buffer_size=8 << 20)) as archive:  # reads the central directory
        members, todo = _wanted(archive, pattern, target)
    slack = 1024  # room for the local header's file name and extra field

    def span(m: zipfile.ZipInfo) -> tuple[int, int]:
        return m.header_offset, m.header_offset + 30 + len(m.filename.encode()) + slack + m.compress_size

    groups: list[list[zipfile.ZipInfo]] = []
    for member in todo:
        start, end = span(member)
        if groups and start - span(groups[-1][-1])[1] < merge_gap and end - span(groups[-1][0])[0] < max_request:
            groups[-1].append(member)
        else:
            groups.append([member])
    total = sum(m.compress_size for m in todo)
    label = url.rsplit("/files/", 1)[-1].removesuffix("/content")
    print(f"  {label}: {len(members)} files wanted, {len(todo)} to fetch ({total / 1e9:.2f} GB "
          f"in {len(groups)} requests)", flush=True)

    def fetch(group: list[zipfile.ZipInfo]) -> int:
        first = span(group[0])[0]
        blob = _get_range(url, first, min(span(group[-1])[1], raw.size))
        for member in group:
            data = _member_data(blob, member.header_offset - first, member)
            if data is None:  # unusually long local header: fetch this member with the largest possible one
                end = min(member.header_offset + 30 + 2 * 65_535 + member.compress_size, raw.size)
                data = _member_data(_get_range(url, member.header_offset, end), 0, member)
            _save(target, member, data)
        return sum(m.compress_size for m in group)

    done, start = 0, time.perf_counter()
    with ThreadPoolExecutor(max_workers=connections) as pool:
        for k, size in enumerate(as_completed([pool.submit(fetch, g) for g in groups]), 1):
            done += size.result()
            if k % 50 == 0 or k == len(groups):
                rate = done / max(time.perf_counter() - start, 1e-9)
                left = (total - done) / rate if rate > 0 else 0
                print(f"    {done / 1e9:.2f} of {total / 1e9:.2f} GB, {rate / 1e6:.1f} MB/s, "
                      f"about {left / 60:.0f} min left", flush=True)
    return len(members), len(todo)


def extract_local(zip_path: str | Path, pattern: str, target: str | Path) -> tuple[int, int]:
    """The same as :func:`extract_remote` for a zip file on disk."""
    target = Path(target)
    with zipfile.ZipFile(zip_path) as archive:
        members, todo = _wanted(archive, pattern, target)
        print(f"  {Path(zip_path).name}: {len(members)} files wanted, {len(todo)} to unpack", flush=True)
        for member in todo:
            _save(target, member, archive.read(member))  # zipfile checks the CRC
    return len(members), len(todo)
