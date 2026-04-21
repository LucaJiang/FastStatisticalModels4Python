#!/usr/bin/env python3
"""Benchmark driver for k-means implementations.

Each implementation receives the *same* initial centroids (seeded with
``numpy.random.default_rng(seed)``), so the resulting inertia values are
directly comparable up to floating-point noise. For every implementation
we record:

- ``cold_s``: wall time of the first call (compile / trace counted).
- ``warm_median_s`` / ``warm_std_s``: median of ``repeat`` runs after
  ``warmup`` discarded runs.
- ``tracemalloc_peak_mb``: Python-level peak allocation during a run.
- ``rss_peak_mb``: process-wide peak RSS (``resource.RUSAGE_SELF``).
- ``inertia``: for correctness checking.
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

from data_gen import make_cluster_data  # noqa: E402
from kmeans_loops import kmeans_loops  # noqa: E402
from kmeans_numpy import kmeans_numpy_naive, kmeans_numpy_smart  # noqa: E402


IMPLS_ORDER = ("numpy_naive", "numpy_smart", "loops", "numba", "jax")


@dataclass
class BenchRow:
    impl: str
    n_samples: int
    n_features: int
    k: int
    max_iter: int
    cold_s: float
    warm_median_s: float
    warm_std_s: float
    tracemalloc_peak_mb: float
    rss_peak_mb: float
    inertia: float
    n_iter: int
    warmup: int = 0
    repeat: int = 0
    extra: dict = field(default_factory=dict)


def _rss_peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / (1024 * 1024)  # macOS: bytes
    return r / 1024  # Linux: kilobytes


def _time_runs(fn, warmup: int, repeat: int) -> tuple[float, float, float, float]:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for _ in range(max(0, warmup - 1)):
        fn()

    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    if not times:
        times = [cold]
    return cold, float(np.median(times)), float(np.std(times)), peak / (1024 * 1024)


def _initial_centroids(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].astype(np.float64, copy=True)


def _make_runner(name: str, X: np.ndarray, init: np.ndarray, max_iter: int):
    k = init.shape[0]
    if name == "numpy_naive":
        return lambda: kmeans_numpy_naive(X, k, max_iter, init)
    if name == "numpy_smart":
        return lambda: kmeans_numpy_smart(X, k, max_iter, init)
    if name == "loops":
        return lambda: kmeans_loops(X, k, max_iter, init)
    if name == "numba":
        from kmeans_numba import kmeans_numba  # noqa: F401 — import for compile

        return lambda: kmeans_numba(X, k, max_iter, init)
    if name == "jax":
        import jax
        import jax.numpy as jnp
        from kmeans_jax import make_kmeans_jax_jitted

        jax.config.update("jax_enable_x64", True)
        jitted = make_kmeans_jax_jitted()
        X_j = jnp.asarray(X, dtype=jnp.float64)
        C_j = jnp.asarray(init, dtype=jnp.float64)

        def fn():
            out = jitted(X_j, C_j, max_iter)
            jax.block_until_ready(out)
            return out

        return fn
    raise ValueError(f"Unknown impl: {name}")


def run_impl(name: str, X: np.ndarray, init: np.ndarray, max_iter: int,
             warmup: int, repeat: int) -> BenchRow:
    fn = _make_runner(name, X, init, max_iter)
    cold, med, std, tr_peak = _time_runs(fn, warmup=warmup, repeat=repeat)
    out = fn()
    inertia = float(out[2])
    n_iter = int(out[3]) if name != "jax" else int(max_iter)
    return BenchRow(
        impl=name,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        k=int(init.shape[0]),
        max_iter=int(max_iter),
        cold_s=cold,
        warm_median_s=med,
        warm_std_s=std,
        tracemalloc_peak_mb=tr_peak,
        rss_peak_mb=_rss_peak_mb(),
        inertia=inertia,
        n_iter=n_iter,
        warmup=warmup,
        repeat=repeat,
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
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--centers", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--impl", nargs="+", choices=IMPLS_ORDER,
                   default=["numpy_naive", "numpy_smart", "numba", "jax"])
    p.add_argument("--loops-max-n", type=int, default=4000)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    X, _ = make_cluster_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        centers=args.centers,
        random_state=args.seed,
    )
    rows: list[BenchRow] = []

    for impl in [i for i in IMPLS_ORDER if i in args.impl]:
        if impl == "loops":
            n_cap = min(args.n_samples, args.loops_max_n)
            X_cur = X[:n_cap].copy()
            init = _initial_centroids(X_cur, args.k, args.seed)
            row = run_impl(
                impl, X_cur, init, args.max_iter,
                warmup=max(1, args.warmup // 2),
                repeat=max(2, args.repeat // 2),
            )
        else:
            init = _initial_centroids(X, args.k, args.seed)
            row = run_impl(impl, X, init, args.max_iter, args.warmup, args.repeat)
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
