#!/usr/bin/env python3
"""Benchmark driver for permutation-test implementations.

For every implementation we record cold (first-call) and warm (median
of ``repeat`` runs) wall time, ``tracemalloc`` peak allocation, and
process-wide peak RSS. The primary correctness check is the mean and
standard deviation of the resulting null distribution, which should
agree across implementations up to RNG noise for large R.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_gen import make_two_sample_vector  # noqa: E402
from permtest_freethreaded import run_permtest_threads  # noqa: E402
from permtest_multiprocessing import run_permtest_multiprocessing  # noqa: E402
from permtest_numpy import (  # noqa: E402
    run_permtest_numpy_batched,
    run_permtest_numpy_naive,
    run_permtest_numpy_trick,
    run_permtest_numpy_trick_batched,
)

try:
    from permtest_jax import make_permtest_jax_perm, make_permtest_jax_trick
except ImportError:
    make_permtest_jax_perm = None  # type: ignore[assignment]
    make_permtest_jax_trick = None  # type: ignore[assignment]


IMPLS_ORDER = (
    "numpy_naive",
    "numpy_batched",
    "numpy_trick",
    "numpy_trick_batched",
    "multiprocessing",
    "threads",
    "numba",
    "jax_perm",
    "jax_trick",
)


@dataclass
class BenchRow:
    impl: str
    n: int
    n1: int
    r: int
    cold_s: float
    warm_median_s: float
    warm_std_s: float
    tracemalloc_peak_mb: float
    rss_peak_mb: float
    children_rss_peak_mb: float
    stat_mean: float
    stat_std: float
    warmup: int = 0
    repeat: int = 0
    max_workers: int | None = None
    extra: dict = field(default_factory=dict)


def _rss_peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / (1024 * 1024)
    return r / 1024


def _time_runs(fn, warmup: int, repeat: int):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    arr = fn()
    cold = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for _ in range(max(0, warmup - 1)):
        fn()

    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        arr = fn()
        times.append(time.perf_counter() - t0)
    if not times:
        times = [cold]
    return (
        cold,
        float(np.median(times)),
        float(np.std(times)),
        peak / (1024 * 1024),
        arr,
    )


def _make_fn(name, x, n1, r, seed, max_workers):
    if name == "numpy_naive":
        return lambda: run_permtest_numpy_naive(x, n1, r, seed)
    if name == "numpy_batched":
        return lambda: run_permtest_numpy_batched(x, n1, r, seed)
    if name == "numpy_trick":
        return lambda: run_permtest_numpy_trick(x, n1, r, seed)
    if name == "numpy_trick_batched":
        return lambda: run_permtest_numpy_trick_batched(x, n1, r, seed)
    if name == "multiprocessing":
        return lambda: run_permtest_multiprocessing(x, n1, r, seed, max_workers=max_workers)
    if name == "threads":
        return lambda: run_permtest_threads(x, n1, r, seed, max_workers=max_workers)
    if name == "numba":
        from permtest_numba import run_permtest_numba  # noqa: F401 — import for compile

        return lambda: run_permtest_numba(x, n1, r, seed)
    if name == "jax_perm":
        if make_permtest_jax_perm is None:
            return None
        import jax
        import jax.numpy as jnp
        run = make_permtest_jax_perm()
        xj = jnp.asarray(x, dtype=jnp.float64)
        key = jax.random.PRNGKey(seed)

        def fn():
            out = run(xj, n1, r, key)
            jax.block_until_ready(out)
            return np.asarray(out)

        return fn
    if name == "jax_trick":
        if make_permtest_jax_trick is None:
            return None
        import jax
        import jax.numpy as jnp
        run = make_permtest_jax_trick()
        xj = jnp.asarray(x, dtype=jnp.float64)
        key = jax.random.PRNGKey(seed)

        def fn():
            out = run(xj, n1, r, key)
            jax.block_until_ready(out)
            return np.asarray(out)

        return fn
    raise ValueError(f"Unknown impl: {name}")


def run_impl(name, x, n1, r, seed, warmup, repeat, max_workers):
    fn = _make_fn(name, x, n1, r, seed, max_workers)
    if fn is None:
        return None
    # Enable child-RSS probe only for multiprocessing; cheap otherwise.
    os.environ["PERMTEST_MEASURE_CHILDREN"] = "1" if name == "multiprocessing" else "0"
    cold, med, std, tr_peak, arr = _time_runs(fn, warmup, repeat)
    children_peak = 0.0
    if name == "multiprocessing":
        import permtest_multiprocessing as pmp
        children_peak = float(pmp.LAST_CHILDREN_RSS_MB or 0.0)
    return BenchRow(
        impl=name,
        n=int(x.shape[0]),
        n1=int(n1),
        r=int(r),
        cold_s=cold,
        warm_median_s=med,
        warm_std_s=std,
        tracemalloc_peak_mb=tr_peak,
        rss_peak_mb=_rss_peak_mb(),
        children_rss_peak_mb=children_peak,
        stat_mean=float(np.mean(arr)),
        stat_std=float(np.std(arr)),
        warmup=warmup,
        repeat=repeat,
        max_workers=max_workers,
    )


def _env_metadata() -> dict:
    meta: dict = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    for mod in ("numpy", "numba", "jax", "sklearn"):
        try:
            m = __import__(mod)
            meta[f"{mod}_version"] = m.__version__
        except Exception:
            meta[f"{mod}_version"] = None
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n1", type=int, default=5000)
    p.add_argument("--n2", type=int, default=5000)
    p.add_argument("--r", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--impl", nargs="+", choices=IMPLS_ORDER,
                   default=["numpy_naive", "numpy_trick_batched", "numba"])
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    x, n1 = make_two_sample_vector(args.n1, args.n2, random_state=args.seed)

    rows: list[BenchRow] = []
    for impl in [i for i in IMPLS_ORDER if i in args.impl]:
        row = run_impl(impl, x, n1, args.r, args.seed, args.warmup, args.repeat, args.max_workers)
        if row is None:
            print(f"Skipping {impl} (not available)", file=sys.stderr)
            continue
        print(json.dumps(row.__dict__, indent=2, default=float))
        rows.append(row)

    if args.output_csv and rows:
        fields = list(rows[0].__dict__.keys())
        with args.output_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                d = r.__dict__.copy()
                d["extra"] = json.dumps(d.get("extra") or {})
                w.writerow(d)

    if args.output_json:
        with args.output_json.open("w") as f:
            json.dump(
                {"env": _env_metadata(), "rows": [r.__dict__ for r in rows]},
                f, indent=2, default=float,
            )


if __name__ == "__main__":
    main()
