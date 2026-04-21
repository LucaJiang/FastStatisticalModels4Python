"""Permutation test via ``ProcessPoolExecutor``.

Every worker receives a pickled copy of ``x`` via the initializer and
uses the naive (one permutation per iter) loop. The point of the talk
is *not* that multiprocessing is bad — it's that (a) on small tasks the
process startup and data transfer dominate, and (b) each worker holds
its own copy of the array, so the total RAM footprint scales with the
worker count. ``measured_peak_children_rss_mb`` (returned as a side
channel) makes that footprint visible in the benchmarks; set
``PERMTEST_MEASURE_CHILDREN=1`` to enable it.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_DATA: np.ndarray | None = None
_N1: int = 0
LAST_CHILDREN_RSS_MB: float | None = None


def _init_worker(x: np.ndarray, n1: int) -> None:
    global _DATA, _N1
    _DATA = np.asarray(x, dtype=np.float64)
    _N1 = int(n1)


def _chunk_stats(args: tuple[int, int, int]) -> np.ndarray:
    r0, r1, subseed = args
    assert _DATA is not None
    rng = np.random.default_rng(subseed)
    n = _DATA.shape[0]
    n1 = _N1
    n2 = n - n1
    total = float(_DATA.sum())
    out = np.empty(r1 - r0, dtype=np.float64)
    for i in range(r1 - r0):
        p = rng.permutation(n)
        s1 = float(_DATA[p[:n1]].sum())
        out[i] = s1 / n1 - (total - s1) / n2
    return out


def _measure_children_rss_mb(parent_pid: int) -> float:
    try:
        import psutil
    except ImportError:
        return float("nan")
    total = 0.0
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 * 1024)


def run_permtest_multiprocessing(
    x: np.ndarray,
    n1: int,
    r: int,
    seed: int,
    *,
    max_workers: int | None = None,
    chunks: int | None = None,
) -> np.ndarray:
    global LAST_CHILDREN_RSS_MB

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)
    n_chunks = chunks if chunks is not None else max_workers
    n_chunks = max(1, min(n_chunks, r))
    edges = np.linspace(0, r, n_chunks + 1, dtype=int)
    tasks: list[tuple[int, int, int]] = []
    for j in range(n_chunks):
        r0, r1 = int(edges[j]), int(edges[j + 1])
        if r0 < r1:
            tasks.append((r0, r1, seed + 100_003 * j))

    out = np.empty(r, dtype=np.float64)
    children_peak = 0.0
    measure = os.environ.get("PERMTEST_MEASURE_CHILDREN") == "1"
    parent_pid = os.getpid()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(x, n1),
    ) as ex:
        it = ex.map(_chunk_stats, tasks)
        if measure:
            # Give workers a moment to import numpy and receive the array,
            # then sample once. Keep sampling cheap — just a single probe.
            time.sleep(0.05)
            children_peak = _measure_children_rss_mb(parent_pid)
        for (r0, r1, _), piece in zip(tasks, it):
            out[r0:r1] = piece
            if measure:
                children_peak = max(
                    children_peak, _measure_children_rss_mb(parent_pid)
                )
    LAST_CHILDREN_RSS_MB = children_peak if measure else None
    return out
