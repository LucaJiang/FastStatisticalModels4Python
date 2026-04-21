"""Permutation test via ``ThreadPoolExecutor`` (shared data).

On a **GIL** build, Python bytecode is serialized across threads and the
speedup mostly comes from NumPy C calls that release the GIL (e.g. the
``.sum()`` inside the worker). On a **free-threaded** Python 3.14 build,
these same threads actually run Python bytecode in parallel, so the
speedup against the serial NumPy baseline improves noticeably.

The code here is identical in either case — which is the whole point.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _chunk_stats_thread(
    data: np.ndarray, n1: int, r_start: int, r_end: int, subseed: int
) -> np.ndarray:
    rng = np.random.default_rng(subseed)
    n = data.shape[0]
    n2 = n - n1
    total = float(data.sum())
    out = np.empty(r_end - r_start, dtype=np.float64)
    for i in range(r_end - r_start):
        p = rng.permutation(n)
        s1 = float(data[p[:n1]].sum())
        out[i] = s1 / n1 - (total - s1) / n2
    return out


def run_permtest_threads(
    x: np.ndarray,
    n1: int,
    r: int,
    seed: int,
    *,
    max_workers: int | None = None,
    chunks: int | None = None,
) -> np.ndarray:
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)
    n_chunks = chunks if chunks is not None else max_workers
    n_chunks = max(1, min(n_chunks, r))
    edges = np.linspace(0, r, n_chunks + 1, dtype=int)
    tasks: list[tuple[int, int, int]] = []
    for j in range(n_chunks):
        r0, r1 = int(edges[j]), int(edges[j + 1])
        if r0 < r1:
            tasks.append((r0, r1, seed + 97_003 * j))

    data = np.ascontiguousarray(x, dtype=np.float64)
    out = np.empty(r, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_chunk_stats_thread, data, n1, t[0], t[1], t[2])
            for t in tasks
        ]
        for t, fut in zip(tasks, futures):
            out[t[0]:t[1]] = fut.result()
    return out
