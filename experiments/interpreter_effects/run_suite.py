"""Run CPython 3.14 GIL vs free-threaded interpreter-effects experiments.

The runner intentionally measures the current interpreter only. Run it once
from the standard CPython 3.14 environment and once from the free-threaded
CPython 3.14 environment, then use ``plot_interpreter_effects.py`` to compare
their CSV outputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from experiments.common.env_report import build_report

GIB = 1024**3
WORKERS = (1, 2, 4, 8, 16)
RAW_FIELDS = [
    "timestamp",
    "env_label",
    "python_executable",
    "python_version",
    "gil_enabled",
    "py_gil_disabled_config",
    "jit_available",
    "jit_enabled",
    "jit_claim_allowed",
    "experiment",
    "workload",
    "pool",
    "workers",
    "phase",
    "repeat_index",
    "status",
    "wall_time_sec",
    "peak_rss_gb",
    "result_value",
    "n",
    "p",
    "matrix_size",
    "resamples",
    "sample_size",
    "array_mb",
    "notes",
]

_PROCESS_ARRAY: np.ndarray | None = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return float(qs[2] - qs[0])


def _split_counts(total: int, workers: int) -> list[int]:
    base, extra = divmod(total, workers)
    return [base + (1 if idx < extra else 0) for idx in range(workers)]


def _python_info() -> dict[str, Any]:
    gil_enabled = None
    if hasattr(sys, "_is_gil_enabled"):
        try:
            gil_enabled = bool(sys._is_gil_enabled())  # type: ignore[attr-defined]
        except Exception:
            gil_enabled = None
    jit_available = False
    jit_enabled = False
    if hasattr(sys, "_jit"):
        try:
            jit_available = bool(sys._jit.is_available())  # type: ignore[attr-defined]
            jit_enabled = bool(sys._jit.is_enabled())  # type: ignore[attr-defined]
        except Exception:
            jit_available = False
            jit_enabled = False
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "gil_enabled": gil_enabled,
        "py_gil_disabled_config": sysconfig_py_gil_disabled(),
        "jit_available": jit_available,
        "jit_enabled": jit_enabled,
        "jit_claim_allowed": bool(jit_available and jit_enabled),
    }


def sysconfig_py_gil_disabled() -> Any:
    import sysconfig

    return sysconfig.get_config_var("Py_GIL_DISABLED")


def _rss_gb() -> float:
    import psutil

    return float(psutil.Process().memory_info().rss / GIB)


class PeakRSSMonitor:
    """Sample parent plus child RSS while a workload is running."""

    def __init__(self, interval_sec: float = 0.02, include_children: bool = True) -> None:
        import psutil

        self._psutil = psutil
        self._process = psutil.Process()
        self._interval_sec = interval_sec
        self._include_children = include_children
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_gb = 0.0

    def __enter__(self) -> "PeakRSSMonitor":
        self.peak_gb = self._sample()
        self._thread = threading.Thread(target=self._run, name="peak-rss-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_gb = max(self.peak_gb, self._sample())

    def _run(self) -> None:
        while not self._stop.wait(self._interval_sec):
            self.peak_gb = max(self.peak_gb, self._sample())

    def _sample(self) -> float:
        processes = [self._process]
        if self._include_children:
            try:
                processes.extend(self._process.children(recursive=True))
            except self._psutil.Error:
                pass
        total = 0
        for proc in processes:
            try:
                total += proc.memory_info().rss
            except self._psutil.Error:
                continue
        return float(total / GIB)


def _time_call(fn: Callable[[], float], include_children: bool = False) -> tuple[float, float, float]:
    with PeakRSSMonitor(include_children=include_children) as monitor:
        start = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - start
    return float(elapsed), float(monitor.peak_gb), float(value)


def pure_python_cpu_loop(n: int) -> float:
    acc = 0
    mask = (1 << 63) - 1
    for i in range(n):
        acc = (acc * 6364136223846793005 + i + 1442695040888963407) & mask
        if acc & 1:
            acc ^= i * 31
        else:
            acc += i ^ (acc >> 7)
    return float(acc & 0xFFFFFFFF)


def numpy_blas_matrix_path(matrix_size: int) -> float:
    rng = np.random.default_rng(41)
    a = rng.standard_normal((matrix_size, matrix_size), dtype=np.float64)
    b = rng.standard_normal((matrix_size, matrix_size), dtype=np.float64)
    c = a @ b
    c = c @ b.T
    return float(c[0, 0] + np.linalg.norm(c) * 1e-12)


def small_statistical_loop(n: int) -> float:
    rng = np.random.default_rng(42)
    values = rng.standard_normal(n).tolist()
    mean = 0.0
    m2 = 0.0
    m3 = 0.0
    for idx, x in enumerate(values, start=1):
        delta = x - mean
        delta_n = delta / idx
        term = delta * delta_n * (idx - 1)
        m3 += term * delta_n * (idx - 2) - 3.0 * delta_n * m2
        m2 += term
        mean += delta_n
    variance = m2 / (n - 1)
    skew = (math.sqrt(n) * m3 / (m2**1.5)) if m2 else 0.0
    return float(mean + variance + skew)


def permutation_stat_worker(args: tuple[int, int, int]) -> float:
    resamples, sample_size, seed = args
    state = seed & 0xFFFFFFFFFFFFFFFF
    exceed = 0
    total_abs = 0.0
    for _ in range(resamples):
        sum_a = 0.0
        sum_b = 0.0
        count_a = 0
        count_b = 0
        for _j in range(sample_size):
            state = (state * 2862933555777941757 + 3037000493) & 0xFFFFFFFFFFFFFFFF
            centered = ((state >> 11) & 0xFFFFF) / 524288.0 - 1.0
            if state & 1:
                sum_a += centered
                count_a += 1
            else:
                sum_b += centered
                count_b += 1
        if count_a and count_b:
            diff = abs(sum_a / count_a - sum_b / count_b)
            total_abs += diff
            exceed += int(diff > 0.035)
    return float(total_abs + exceed)


def contention_worker(args: tuple[int, int, str, Any]) -> float:
    iterations, seed, mode, shared = args
    local = 0
    state = seed & 0xFFFFFFFFFFFFFFFF
    for _ in range(iterations):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        local += int(state & 0x3FF)
        if mode == "shared_counter":
            with shared["lock"]:
                shared["counter"] += 1
        elif mode == "shared_list":
            with shared["lock"]:
                shared["values"].append(local & 0xFFFF)
                if len(shared["values"]) > 2048:
                    shared["values"].clear()
        elif mode == "shared_dict":
            key = local & 0x3F
            with shared["lock"]:
                shared["counts"][key] = shared["counts"].get(key, 0) + 1
    return float(local)


def _process_initializer(array: np.ndarray) -> None:
    global _PROCESS_ARRAY
    _PROCESS_ARRAY = array


def _memory_worker(args: tuple[int, int, int]) -> float:
    if _PROCESS_ARRAY is None:
        raise RuntimeError("process memory worker was not initialized")
    start, stop, repeats = args
    block = _PROCESS_ARRAY[start:stop]
    total = 0.0
    for _ in range(repeats):
        centered = block - block.mean(axis=0, keepdims=True)
        total += float(np.sum(centered * centered))
    return total


def _memory_thread_worker(args: tuple[np.ndarray, int, int, int]) -> float:
    array, start, stop, repeats = args
    block = array[start:stop]
    total = 0.0
    for _ in range(repeats):
        centered = block - block.mean(axis=0, keepdims=True)
        total += float(np.sum(centered * centered))
    return total


def _run_thread_scaling(total_resamples: int, sample_size: int, workers: int) -> float:
    counts = _split_counts(total_resamples, workers)
    tasks = [(count, sample_size, 10_000 + idx * 9973) for idx, count in enumerate(counts) if count > 0]
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        return float(sum(pool.map(permutation_stat_worker, tasks)))


def _run_contention(total_iterations: int, workers: int, mode: str) -> float:
    counts = _split_counts(total_iterations, workers)
    shared: dict[str, Any] = {
        "lock": threading.Lock(),
        "counter": 0,
        "values": [],
        "counts": {},
    }
    tasks = [(count, 30_000 + idx * 17, mode, shared) for idx, count in enumerate(counts) if count > 0]
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        return float(sum(pool.map(contention_worker, tasks)))


def _make_memory_array(array_mb: int, cols: int) -> np.ndarray:
    rows = max(1, int(array_mb * 1024**2 // (cols * np.dtype(np.float64).itemsize)))
    rng = np.random.default_rng(43)
    return rng.standard_normal((rows, cols), dtype=np.float64)


def _run_memory_pool(array_mb: int, cols: int, workers: int, repeats: int, pool_kind: str) -> float:
    array = _make_memory_array(array_mb, cols)
    rows = array.shape[0]
    edges = np.linspace(0, rows, workers + 1, dtype=np.int64).tolist()
    if pool_kind == "thread":
        tasks = [(array, int(edges[idx]), int(edges[idx + 1]), repeats) for idx in range(workers)]
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            return float(sum(pool.map(_memory_thread_worker, tasks)))
    if pool_kind == "process":
        tasks = [(int(edges[idx]), int(edges[idx + 1]), repeats) for idx in range(workers)]
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_process_initializer, initargs=(array,)) as pool:
            return float(sum(pool.map(_memory_worker, tasks)))
    raise ValueError(f"unknown pool kind: {pool_kind}")


def _base_row(args: argparse.Namespace, experiment: str, workload: str, pool: str, workers: int) -> dict[str, Any]:
    row = {
        "timestamp": _timestamp(),
        "env_label": args.env_label,
        "experiment": experiment,
        "workload": workload,
        "pool": pool,
        "workers": workers,
        "status": "ok",
        "notes": "",
    }
    row.update(_python_info())
    return row


def _measure_repeated(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    experiment: str,
    workload: str,
    pool: str,
    workers: int,
    fn: Callable[[], float],
    include_children: bool = False,
    params: dict[str, Any] | None = None,
) -> None:
    params = params or {}
    phases = [("warmup", idx) for idx in range(args.warmups)] + [("repeat", idx) for idx in range(args.repeats)]
    for phase, repeat_index in phases:
        row = _base_row(args, experiment, workload, pool, workers)
        row.update(params)
        row.update({"phase": phase, "repeat_index": repeat_index})
        try:
            elapsed, peak, value = _time_call(fn, include_children=include_children)
            row.update({"wall_time_sec": elapsed, "peak_rss_gb": peak, "result_value": value})
        except Exception as exc:
            row.update({"status": "error", "notes": repr(exc), "wall_time_sec": "", "peak_rss_gb": _rss_gb(), "result_value": ""})
        rows.append(row)


def run_negative_controls(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    _measure_repeated(
        args=args,
        rows=rows,
        experiment="single_thread_negative_control",
        workload="pure_python_cpu_loop",
        pool="none",
        workers=1,
        fn=lambda: pure_python_cpu_loop(args.python_loop_n),
        params={"n": args.python_loop_n},
    )
    _measure_repeated(
        args=args,
        rows=rows,
        experiment="single_thread_negative_control",
        workload="numpy_blas_matrix_path",
        pool="none",
        workers=1,
        fn=lambda: numpy_blas_matrix_path(args.matrix_size),
        params={"matrix_size": args.matrix_size},
    )
    _measure_repeated(
        args=args,
        rows=rows,
        experiment="single_thread_negative_control",
        workload="small_statistical_loop",
        pool="none",
        workers=1,
        fn=lambda: small_statistical_loop(args.stat_loop_n),
        params={"n": args.stat_loop_n},
    )


def run_thread_scaling(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    for workers in args.workers:
        _measure_repeated(
            args=args,
            rows=rows,
            experiment="thread_scaling",
            workload="python_permutation_stat",
            pool="thread",
            workers=workers,
            fn=lambda workers=workers: _run_thread_scaling(args.thread_resamples, args.thread_sample_size, workers),
            params={"resamples": args.thread_resamples, "sample_size": args.thread_sample_size},
        )


def run_memory_comparison(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    pool_kind = args.memory_pool
    if pool_kind == "auto":
        pool_kind = "thread" if args.env_label.endswith("t") else "process"
    _measure_repeated(
        args=args,
        rows=rows,
        experiment="pool_memory_runtime",
        workload="shared_large_array_reduction",
        pool=pool_kind,
        workers=args.memory_workers,
        fn=lambda: _run_memory_pool(args.memory_array_mb, args.memory_cols, args.memory_workers, args.memory_repeats, pool_kind),
        include_children=pool_kind == "process",
        params={"array_mb": args.memory_array_mb, "p": args.memory_cols},
    )


def run_contention(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    for mode in ("thread_local", "shared_counter", "shared_list", "shared_dict"):
        for workers in args.workers:
            if mode == "thread_local":
                fn = lambda workers=workers: _run_thread_scaling(args.contention_iterations, 16, workers)
                params = {"resamples": args.contention_iterations, "sample_size": 16}
            else:
                fn = lambda workers=workers, mode=mode: _run_contention(args.contention_iterations, workers, mode)
                params = {"n": args.contention_iterations}
            _measure_repeated(
                args=args,
                rows=rows,
                experiment="contention_backup",
                workload=mode,
                pool="thread",
                workers=workers,
                fn=fn,
                params=params,
            )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("phase") != "repeat":
            continue
        key = (
            row.get("env_label"),
            row.get("experiment"),
            row.get("workload"),
            row.get("pool"),
            row.get("workers"),
            row.get("n"),
            row.get("p"),
            row.get("matrix_size"),
            row.get("resamples"),
            row.get("sample_size"),
            row.get("array_mb"),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        ok = [row for row in group if row.get("status") == "ok" and row.get("wall_time_sec") != ""]
        times = [float(row["wall_time_sec"]) for row in ok]
        peaks = [float(row["peak_rss_gb"]) for row in ok if row.get("peak_rss_gb") != ""]
        first = group[0]
        summary = {
            "env_label": key[0],
            "experiment": key[1],
            "workload": key[2],
            "pool": key[3],
            "workers": key[4],
            "n": key[5],
            "p": key[6],
            "matrix_size": key[7],
            "resamples": key[8],
            "sample_size": key[9],
            "array_mb": key[10],
            "repeat_count": len(ok),
            "error_count": len(group) - len(ok),
            "median_wall_time_sec": statistics.median(times) if times else "",
            "iqr_wall_time_sec": _iqr(times) if times else "",
            "min_wall_time_sec": min(times) if times else "",
            "max_wall_time_sec": max(times) if times else "",
            "median_peak_rss_gb": statistics.median(peaks) if peaks else "",
            "max_peak_rss_gb": max(peaks) if peaks else "",
            "gil_enabled": first.get("gil_enabled"),
            "py_gil_disabled_config": first.get("py_gil_disabled_config"),
            "jit_available": first.get("jit_available"),
            "jit_enabled": first.get("jit_enabled"),
            "jit_claim_allowed": first.get("jit_claim_allowed"),
        }
        summaries.append(summary)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-label", required=True, help="Environment label, normally py314 or py314t.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/python314_interpreter_effects/latest"))
    parser.add_argument("--experiments", nargs="+", default=["negative", "thread", "memory"], choices=["negative", "thread", "memory", "contention", "all"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--workers", type=int, nargs="+", default=list(WORKERS))
    parser.add_argument("--python-loop-n", type=int, default=3_000_000)
    parser.add_argument("--matrix-size", type=int, default=650)
    parser.add_argument("--stat-loop-n", type=int, default=700_000)
    parser.add_argument("--thread-resamples", type=int, default=6_400)
    parser.add_argument("--thread-sample-size", type=int, default=384)
    parser.add_argument("--memory-pool", choices=["auto", "thread", "process"], default="auto")
    parser.add_argument("--memory-workers", type=int, default=4)
    parser.add_argument("--memory-array-mb", type=int, default=192)
    parser.add_argument("--memory-cols", type=int, default=64)
    parser.add_argument("--memory-repeats", type=int, default=4)
    parser.add_argument("--contention-iterations", type=int, default=200_000)
    parser.add_argument("--quick", action="store_true", help="Use smaller problem sizes for smoke tests.")
    return parser.parse_args()


def apply_quick_sizes(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.repeats = min(args.repeats, 2)
    args.warmups = min(args.warmups, 1)
    args.workers = [worker for worker in args.workers if worker <= 4]
    args.python_loop_n = min(args.python_loop_n, 120_000)
    args.matrix_size = min(args.matrix_size, 160)
    args.stat_loop_n = min(args.stat_loop_n, 60_000)
    args.thread_resamples = min(args.thread_resamples, 160)
    args.thread_sample_size = min(args.thread_sample_size, 96)
    args.memory_workers = min(args.memory_workers, 2)
    args.memory_array_mb = min(args.memory_array_mb, 24)
    args.memory_repeats = min(args.memory_repeats, 1)
    args.contention_iterations = min(args.contention_iterations, 2_000)


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    data = vars(args).copy()
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def main() -> None:
    args = parse_args()
    apply_quick_sizes(args)
    requested = set(args.experiments)
    if "all" in requested:
        requested = {"negative", "thread", "memory", "contention"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_report(f"python314_interpreter_effects_{args.env_label}")
    metadata["interpreter_effects_runner"] = {
        "argv": sys.argv,
        "env_label": args.env_label,
        "parameters": _jsonable_args(args),
        "blas_thread_env_pinned_in_process": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "jit_claim_allowed": _python_info()["jit_claim_allowed"],
        "notes": "Do not describe any result as a JIT speedup unless jit_claim_allowed is true.",
    }
    (args.output_dir / f"metadata_{args.env_label}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    if "negative" in requested:
        run_negative_controls(args, rows)
    if "thread" in requested:
        run_thread_scaling(args, rows)
    if "memory" in requested:
        run_memory_comparison(args, rows)
    if "contention" in requested:
        run_contention(args, rows)

    raw_path = args.output_dir / f"raw_interpreter_effects_{args.env_label}.csv"
    summary_path = args.output_dir / f"summary_interpreter_effects_{args.env_label}.csv"
    write_csv(raw_path, rows, RAW_FIELDS)
    summary_rows = summarize(rows)
    summary_fields = [
        "env_label",
        "experiment",
        "workload",
        "pool",
        "workers",
        "n",
        "p",
        "matrix_size",
        "resamples",
        "sample_size",
        "array_mb",
        "repeat_count",
        "error_count",
        "median_wall_time_sec",
        "iqr_wall_time_sec",
        "min_wall_time_sec",
        "max_wall_time_sec",
        "median_peak_rss_gb",
        "max_peak_rss_gb",
        "gil_enabled",
        "py_gil_disabled_config",
        "jit_available",
        "jit_enabled",
        "jit_claim_allowed",
    ]
    write_csv(summary_path, summary_rows, summary_fields)
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.output_dir / f'metadata_{args.env_label}.json'}")


if __name__ == "__main__":
    main()
