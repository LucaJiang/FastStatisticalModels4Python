#!/usr/bin/env python3
"""Small v3 k-means runner with validation-first timing."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_generation import KMeansScenario, initial_centroids, make_gaussian_mixture
from kmeans_numpy_broadcast import kmeans_numpy_broadcast
from kmeans_numpy_matmul import kmeans_numpy_matmul


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / (1024 * 1024)


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    q = statistics.quantiles(values, n=4, method="inclusive")
    return float(q[2] - q[0])


def _time(fn, repeat: int) -> tuple[float, float, float, float]:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return cold, float(np.median(times)), _iqr(times), peak / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=Path("experiments/results/v3/kmeans_quick.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/results/v3/kmeans_quick.json"))
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--include-jax", action="store_true")
    parser.add_argument("--include-numba", action="store_true")
    args = parser.parse_args()

    scenario = KMeansScenario(n=args.n, d=args.d, k=args.k, separation=3.0, seed=7)
    x, _, _ = make_gaussian_mixture(scenario)
    init = initial_centroids(x, args.k, seed=21)
    impls = {
        "numpy_broadcast": ("cpu", "cpu", kmeans_numpy_broadcast),
        "numpy_matmul": ("cpu", "cpu", kmeans_numpy_matmul),
    }
    if args.include_numba:
        from kmeans_numba import kmeans_numba

        impls["numba"] = ("cpu", "cpu", kmeans_numba)
    if args.include_jax:
        from kmeans_jax import kmeans_jax
        import jax

        impls["jax"] = ("jax", jax.default_backend(), kmeans_jax)

    ref = kmeans_numpy_matmul(x, init, max_iter=args.max_iter)
    rows = []
    for name, (backend, device, fn) in impls.items():
        def call():
            return fn(x, init, max_iter=args.max_iter)

        cold, med, iqr, host_peak = _time(call, repeat=args.repeat)
        out = call()
        rel = abs(out[2] - ref[2]) / max(1.0, abs(ref[2]))
        rows.append({
            "workload": "kmeans",
            "implementation": name,
            "backend": backend,
            "device": device,
            "n": args.n,
            "p": "",
            "d": args.d,
            "k": args.k,
            "B": "",
            "seed": scenario.seed,
            "cold_time_s": cold,
            "warm_median_s": med,
            "warm_iqr_s": iqr,
            "host_peak_mem_mb": host_peak,
            "gpu_peak_mem_mb": "",
            "validation_status": "pass" if rel < (5e-3 if name == "jax" else 1e-6) else "check",
            "notes": f"relative_inertia_delta={rel:.3g}; rss_peak_mb={_rss_mb():.1f}",
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps({"scenario": scenario.__dict__, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
