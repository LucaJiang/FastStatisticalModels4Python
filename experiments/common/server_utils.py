"""Shared helpers for Linux server CPU/A100 runners."""

from __future__ import annotations

import csv
import os
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np


RESULT_FIELDS = [
    "run_id",
    "timestamp",
    "environment_tier",
    "machine_name",
    "workload",
    "implementation",
    "backend",
    "device",
    "n",
    "p",
    "d",
    "k",
    "R",
    "batch_R",
    "seed",
    "cold_time_s",
    "warm_median_s",
    "warm_iqr_s",
    "host_peak_mem_mb",
    "gpu_peak_mem_mb",
    "validation_status",
    "notes",
]


@contextmanager
def thread_env(num_threads: int | None):
    if num_threads is None:
        yield
        return
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    for key in keys:
        os.environ[key] = str(num_threads)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / (1024 * 1024)


def iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    q1, q3 = np.percentile(np.asarray(values, dtype=float), [25, 75])
    return float(q3 - q1)


def time_call(fn, repeat: int = 3):
    t0 = time.perf_counter()
    out = fn()
    cold = time.perf_counter() - t0
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(cold), float(np.median(times)), iqr(times)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RESULT_FIELDS)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def gpu_memory_used_mb() -> float | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip().splitlines()[0])
    except Exception:
        return None


def memory_skip(required_gb: float, budget_gb: float, reason: str) -> tuple[bool, str]:
    if required_gb > budget_gb:
        return True, f"skipped: {reason}; estimated {required_gb:.1f} GiB > budget {budget_gb:.1f} GiB"
    return False, ""


def ensure_figures_dir(root: Path) -> Path:
    fig = root / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    return fig


def safe_grid(values: Iterable, quick_values: Iterable, full: bool):
    return list(values if full else quick_values)
