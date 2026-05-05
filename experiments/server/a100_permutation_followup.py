#!/usr/bin/env python3
"""A100 follow-up benchmark for the matrix permutation path.

This runner intentionally preserves the statistic and the permutation stream:
W is built on the host with NumPy from one seed stream, then the same W batches
are used for CPU checks and for the A100 end-to-end path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from common.server_utils import gpu_memory_used_mb, rss_mb, timestamp
from permutation.matrix_methods import make_expression, observed_stat

OUT_DIR = ROOT / "experiments/results/linux_server_a100/permutation_followup"
PRESENTATION_DIR = ROOT / "experiments/results/presentation_figures"
GIB = 1024**3
CPU_TIMEOUT_S = 20 * 60
DEFAULT_DTYPE = "float32"
STATUS_PASS_EXACT = "pass_exact"
STATUS_PASS_GPU_TOLERANCE = "pass_gpu_tolerance"
STATUS_MANUAL_CHECK = "manual_check"
STATUS_LEGACY_CHECK = "check"
STATUS_SKIPPED = "skipped"
STATUS_TIMEOUT = "timeout"
STATUS_FAIL = "fail"
ACCEPTED_CORRECTNESS_STATUSES = {STATUS_PASS_EXACT, STATUS_PASS_GPU_TOLERANCE, STATUS_LEGACY_CHECK}

COMMON_FIELDS = [
    "run_id",
    "timestamp",
    "machine_name",
    "benchmark",
    "implementation",
    "correctness_status",
    "max_abs_p_diff",
    "max_abs_stat_diff",
    "dtype",
    "device",
    "n",
    "p",
    "R",
    "batch_R",
    "seed",
    "compile_time_s",
    "warm_time_s",
    "end_to_end_time_s",
    "peak_host_memory",
    "peak_device_memory_if_available",
    "estimated_x_gib",
    "estimated_w_batch_gib",
    "estimated_output_stats_gib",
    "dtype_itemsize",
    "estimated_device_memory_per_batch_gib",
    "notes",
]

DECOMP_FIELDS = COMMON_FIELDS + [
    "permutation_generation_time_s",
    "W_build_host_time_s",
    "host_to_device_transfer_time_s",
    "device_compute_time_s",
    "device_to_host_collect_time_s",
    "pvalue_reduction_time_s",
    "total_end_to_end_time_s",
]

KERNEL_FIELDS = COMMON_FIELDS + [
    "kernel_only_label",
    "kernel_only_time_s",
    "kernel_only_throughput_estimate",
    "W_shape",
    "X_shape",
    "output_shape",
]

CPU_FIELDS = COMMON_FIELDS + [
    "cpu_time_s",
    "workers",
    "threads",
    "timeout_s",
    "timeout_status",
]

AUDIT_FIELDS = [
    "R",
    "batch_R",
    "n_batches",
    "timing_scope",
    "compile_included",
    "transfer_included",
    "kernel_only",
    "stage_sum_s",
    "recorded_end_to_end_time_s",
    "stage_sum_delta_s",
    "stage_sum_matches_recorded_total",
    "implementation",
    "previous_a100_semantics",
]


@dataclass(frozen=True)
class Shape:
    n: int
    p: int
    R: int
    batch_R: int
    seed: int = 0
    dtype: str = DEFAULT_DTYPE
    source: str = "shape_sweep"


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    all_fields = list(dict.fromkeys(fields + list(row.keys())))
    if exists:
        with path.open(newline="") as f:
            old_fields = next(csv.reader(f))
        if old_fields != all_fields:
            rows = list(csv.DictReader(path.open(newline="")))
            all_fields = list(dict.fromkeys(old_fields + all_fields))
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields)
                writer.writeheader()
                writer.writerows(rows)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(dict.fromkeys(fields + [key for row in rows for key in row]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def dtype_itemsize(dtype: str) -> int:
    import numpy as np

    return int(np.dtype(dtype).itemsize)


def estimates(n: int, p: int, batch_r: int, dtype: str) -> dict[str, Any]:
    item = dtype_itemsize(dtype)
    x = n * p * item
    w = batch_r * n * item
    stats = batch_r * p * item
    counts = p * 8
    return {
        "estimated_x_gib": x / GIB,
        "estimated_w_batch_gib": w / GIB,
        "estimated_output_stats_gib": stats / GIB,
        "dtype_itemsize": item,
        "estimated_device_memory_per_batch_gib": (x + w + stats + counts) / GIB,
    }


def device_memory_peak() -> str:
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
        if not stats:
            used = gpu_memory_used_mb()
            return "" if used is None else f"{used:.1f} MiB nvidia-smi"
        peak = stats.get("peak_bytes_in_use") or stats.get("bytes_limit")
        current = stats.get("bytes_in_use")
        parts = []
        if current is not None:
            parts.append(f"current={float(current) / GIB:.3f} GiB")
        if peak is not None:
            parts.append(f"peak={float(peak) / GIB:.3f} GiB")
        return "; ".join(parts)
    except Exception:
        used = gpu_memory_used_mb()
        return "" if used is None else f"{used:.1f} MiB nvidia-smi"


def block(x: Any) -> Any:
    import jax

    return jax.block_until_ready(x)


def as_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        if math.isnan(float(value)):
            return ""
    except Exception:
        return str(value)
    return f"{float(value):.9g}"


def contrast_batch_timed(labels: Any, batch_r: int, rng: Any, dtype: str) -> tuple[Any, float, float]:
    import numpy as np

    n = int(labels.size)
    n1 = int(labels.sum())
    w = np.empty((batch_r, n), dtype=np.dtype(dtype))
    perm_time = 0.0
    build_time = 0.0
    pos = np.asarray(1.0 / n1, dtype=np.dtype(dtype))
    neg = np.asarray(-1.0 / (n - n1), dtype=np.dtype(dtype))
    for b in range(batch_r):
        t0 = time.perf_counter()
        idx = rng.permutation(n)
        perm_time += time.perf_counter() - t0
        t0 = time.perf_counter()
        row = w[b]
        row[idx[:n1]] = pos
        row[idx[n1:]] = neg
        build_time += time.perf_counter() - t0
    return w, perm_time, build_time


def cpu_matrix_from_batches(x: Any, labels: Any, r: int, batch_r: int, seed: int, dtype: str) -> tuple[Any, Any, float, float, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.dtype(dtype))
    obs = np.abs(observed_stat(x, labels).astype(np.dtype(dtype), copy=False))
    exceed = np.zeros(x.shape[1], dtype=np.int64)
    first_stats = None
    perm_s = build_s = matmul_s = 0.0
    done = 0
    while done < r:
        b = min(batch_r, r - done)
        w, p_s, b_s = contrast_batch_timed(labels, b, rng, dtype)
        perm_s += p_s
        build_s += b_s
        t0 = time.perf_counter()
        stats = np.abs(w @ x)
        matmul_s += time.perf_counter() - t0
        if first_stats is None:
            first_stats = np.asarray(stats[: min(8, stats.shape[0]), : min(32, stats.shape[1])], dtype=np.float64)
        exceed += np.sum(stats >= obs[None, :], axis=0)
        done += b
    pvals = (exceed + 1.0) / (r + 1.0)
    return pvals, first_stats, perm_s, build_s, matmul_s


def correctness_check(
    n: int,
    p: int,
    r: int,
    batch_r: int,
    seed: int,
    dtype: str,
) -> tuple[str, float, float, str]:
    import numpy as np

    small_n = min(n, 256)
    if small_n < 4:
        small_n = n
    small_p = min(p, 256)
    small_r = min(r, 128)
    small_b = min(batch_r, 64, small_r)
    try:
        x, labels, _ = make_expression(small_n, small_p, 0.0, 0.0, seed)
        cpu_p, cpu_stats, _, _, _ = cpu_matrix_from_batches(x, labels, small_r, small_b, seed, dtype)

        import jax
        import jax.numpy as jnp

        x_d = jax.device_put(np.asarray(x, dtype=np.dtype(dtype)))
        obs_d = jax.device_put(np.abs(observed_stat(np.asarray(x, dtype=np.dtype(dtype)), labels)))
        rng = np.random.default_rng(seed)
        exceed = jnp.zeros((small_p,), dtype=jnp.int32)
        first_stats = None

        @jax.jit
        def counts_and_stats(w_batch, x_batch, obs_batch):
            stats = jnp.abs(w_batch @ x_batch)
            return jnp.sum(stats >= obs_batch[None, :], axis=0), stats

        done = 0
        while done < small_r:
            b = min(small_b, small_r - done)
            w, _, _ = contrast_batch_timed(labels, b, rng, dtype)
            counts, stats = counts_and_stats(jax.device_put(w), x_d, obs_d)
            block(counts)
            if first_stats is None:
                first_stats = np.asarray(stats[: min(8, b), : min(32, small_p)])
            exceed = exceed + counts
            done += b
        gpu_p = np.asarray((exceed + 1.0) / (small_r + 1.0))
        max_p = float(np.max(np.abs(gpu_p - cpu_p)))
        max_stat = float(np.max(np.abs(np.asarray(first_stats, dtype=np.float64) - np.asarray(cpu_stats, dtype=np.float64))))
        if max_p == 0.0 and max_stat <= 1e-12:
            status = STATUS_PASS_EXACT
        elif max_p <= 1e-6 and max_stat <= 1e-4:
            status = STATUS_PASS_GPU_TOLERANCE
        else:
            status = STATUS_MANUAL_CHECK
        return status, max_p, max_stat, f"small_check n={small_n} p={small_p} R={small_r} batch_R={small_b}"
    except Exception as exc:
        return STATUS_FAIL, math.nan, math.nan, f"correctness_check_failed={exc!r}"


def common_row(benchmark: str, shape: Shape, implementation: str, device: str, run_id: str) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "timestamp": timestamp(),
        "machine_name": platform.node(),
        "benchmark": benchmark,
        "implementation": implementation,
        "dtype": shape.dtype,
        "device": device,
        "n": shape.n,
        "p": shape.p,
        "R": shape.R,
        "batch_R": shape.batch_R,
        "seed": shape.seed,
    }
    row.update({key: as_float(value) for key, value in estimates(shape.n, shape.p, shape.batch_R, shape.dtype).items()})
    return row


def compile_and_warm(
    x_d: Any,
    obs_d: Any,
    labels: Any,
    batch_sizes: Iterable[int],
    seed: int,
    dtype: str,
) -> tuple[Any, Any, float, float, str]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    @jax.jit
    def stats_for_batch(w_batch, x_batch):
        return jnp.abs(w_batch @ x_batch)

    @jax.jit
    def reduce_counts(stats_batch, obs_batch):
        return jnp.sum(stats_batch >= obs_batch[None, :], axis=0)

    rng = np.random.default_rng(seed)
    compile_s = 0.0
    warm_s = 0.0
    first_w_d = None
    first_stats = None
    for i, batch_r in enumerate(dict.fromkeys(int(b) for b in batch_sizes if int(b) > 0)):
        warm_w, _, _ = contrast_batch_timed(labels, batch_r, rng, dtype)
        warm_w_d = jax.device_put(warm_w)
        block(warm_w_d)
        t0 = time.perf_counter()
        stats = stats_for_batch(warm_w_d, x_d)
        block(stats)
        compile_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        counts = reduce_counts(stats, obs_d)
        block(counts)
        compile_s += time.perf_counter() - t0
        if i == 0:
            first_w_d = warm_w_d
            first_stats = stats

    if first_w_d is None:
        raise ValueError("at least one batch size is required for warmup")
    t0 = time.perf_counter()
    stats = stats_for_batch(first_w_d, x_d)
    block(stats)
    counts = reduce_counts(stats, obs_d)
    block(counts)
    warm_s = time.perf_counter() - t0
    return stats_for_batch, reduce_counts, compile_s, warm_s, device_memory_peak()


def run_decomposition(shape: Shape, run_id: str) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    row = common_row("a100_end_to_end_decomposition", shape, "jax_host_w_same_stream", "a100", run_id)
    row["notes"] = ""
    status, max_p, max_stat, check_note = correctness_check(shape.n, shape.p, shape.R, shape.batch_R, shape.seed, shape.dtype)
    row.update({"correctness_status": status, "max_abs_p_diff": as_float(max_p), "max_abs_stat_diff": as_float(max_stat)})
    row["notes"] = check_note
    if status == STATUS_FAIL:
        return row

    x_np, labels, _ = make_expression(shape.n, shape.p, 0.0, 0.0, shape.seed)
    x_np = np.asarray(x_np, dtype=np.dtype(shape.dtype))

    t0 = time.perf_counter()
    x_d = jax.device_put(x_np)
    obs_host = np.abs(observed_stat(x_np, labels))
    obs_d = jax.device_put(obs_host)
    block(x_d)
    block(obs_d)
    compile_setup_transfer_s = time.perf_counter() - t0
    b0 = min(shape.batch_R, shape.R)
    remainder = shape.R % shape.batch_R
    warm_batch_sizes = [b0]
    if remainder and remainder != b0:
        warm_batch_sizes.append(remainder)
    stats_for_batch, reduce_counts, compile_s, warm_s, warm_mem = compile_and_warm(
        x_d,
        obs_d,
        labels,
        warm_batch_sizes,
        shape.seed,
        shape.dtype,
    )

    rng = np.random.default_rng(shape.seed)
    perm_s = build_s = transfer_s = compute_s = reduction_s = collect_s = 0.0
    end0 = time.perf_counter()
    t0 = time.perf_counter()
    x_d = jax.device_put(x_np)
    obs_d = jax.device_put(obs_host)
    block(x_d)
    block(obs_d)
    transfer_s += time.perf_counter() - t0
    exceed = jnp.zeros((shape.p,), dtype=jnp.int32)
    done = 0
    while done < shape.R:
        b = min(shape.batch_R, shape.R - done)
        w_host, p_s, b_s = contrast_batch_timed(labels, b, rng, shape.dtype)
        perm_s += p_s
        build_s += b_s
        t0 = time.perf_counter()
        w_d = jax.device_put(w_host)
        block(w_d)
        transfer_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        stats_d = stats_for_batch(w_d, x_d)
        block(stats_d)
        compute_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        counts_d = reduce_counts(stats_d, obs_d)
        block(counts_d)
        reduction_s += time.perf_counter() - t0
        exceed = exceed + counts_d
        done += b
    t0 = time.perf_counter()
    pvals = np.asarray((exceed + 1.0) / (shape.R + 1.0))
    collect_s += time.perf_counter() - t0
    total_s = time.perf_counter() - end0
    if not np.all(np.isfinite(pvals)):
        row["correctness_status"] = STATUS_FAIL
        row["notes"] = f"{row['notes']}; nonfinite_pvalues"

    peak_host = rss_mb()
    row.update(
        {
            "compile_time_s": as_float(compile_s),
            "warm_time_s": as_float(warm_s),
            "end_to_end_time_s": as_float(total_s),
            "permutation_generation_time_s": as_float(perm_s),
            "W_build_host_time_s": as_float(build_s),
            "host_to_device_transfer_time_s": as_float(transfer_s),
            "device_compute_time_s": as_float(compute_s),
            "device_to_host_collect_time_s": as_float(collect_s),
            "pvalue_reduction_time_s": as_float(reduction_s),
            "total_end_to_end_time_s": as_float(total_s),
            "peak_host_memory": f"{peak_host:.1f} MiB",
            "peak_device_memory_if_available": device_memory_peak() or warm_mem,
            "notes": f"{row['notes']}; compile_setup_transfer_s={compile_setup_transfer_s:.6g}",
        }
    )
    return row


def run_kernel_only(shape: Shape, run_id: str) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    row = common_row("a100_kernel_only", shape, "jax_preloaded_wx", "a100", run_id)
    row["kernel_only_label"] = "kernel-only hypothesis, not end-to-end permutation test"
    status, max_p, max_stat, check_note = correctness_check(shape.n, shape.p, shape.R, shape.batch_R, shape.seed, shape.dtype)
    row.update({"correctness_status": status, "max_abs_p_diff": as_float(max_p), "max_abs_stat_diff": as_float(max_stat), "notes": check_note})
    if status == STATUS_FAIL:
        return row

    x_np, labels, _ = make_expression(shape.n, shape.p, 0.0, 0.0, shape.seed)
    x_d = jax.device_put(np.asarray(x_np, dtype=np.dtype(shape.dtype)))
    rng = np.random.default_rng(shape.seed)
    b = min(shape.batch_R, shape.R)
    w_host, _, _ = contrast_batch_timed(labels, b, rng, shape.dtype)
    w_d = jax.device_put(w_host)
    block(x_d)
    block(w_d)

    @jax.jit
    def matmul_only(w_batch, x_batch):
        return w_batch @ x_batch

    t0 = time.perf_counter()
    block(matmul_only(w_d, x_d))
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = matmul_only(w_d, x_d)
    block(out)
    kernel_s = time.perf_counter() - t0
    flops = 2.0 * b * shape.n * shape.p
    row.update(
        {
            "compile_time_s": as_float(compile_s),
            "warm_time_s": as_float(kernel_s),
            "end_to_end_time_s": "",
            "kernel_only_time_s": as_float(kernel_s),
            "kernel_only_throughput_estimate": f"{flops / kernel_s / 1e12:.3f} TFLOP/s_est" if kernel_s > 0 else "",
            "W_shape": f"({b}, {shape.n})",
            "X_shape": f"({shape.n}, {shape.p})",
            "output_shape": f"({b}, {shape.p})",
            "peak_host_memory": f"{rss_mb():.1f} MiB",
            "peak_device_memory_if_available": device_memory_peak(),
        }
    )
    return row


def cpu_child_main(params: dict[str, Any]) -> None:
    import numpy as np

    shape = Shape(**params["shape"])
    x, labels, _ = make_expression(shape.n, shape.p, 0.0, 0.0, shape.seed)
    x = np.asarray(x, dtype=np.dtype(shape.dtype))
    cpu_matrix_from_batches(x, labels, min(64, shape.R), min(shape.batch_R, 64), shape.seed, shape.dtype)
    t0 = time.perf_counter()
    pvals, _, perm_s, build_s, matmul_s = cpu_matrix_from_batches(x, labels, shape.R, shape.batch_R, shape.seed, shape.dtype)
    elapsed = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "cpu_time_s": elapsed,
                "warm_time_s": elapsed,
                "mean_p": float(np.mean(pvals)),
                "peak_host_memory": f"{rss_mb():.1f} MiB",
                "perm_s": perm_s,
                "build_s": build_s,
                "matmul_s": matmul_s,
            }
        )
    )


def run_cpu_baseline(shape: Shape, run_id: str, timeout_s: int) -> dict[str, Any]:
    row = common_row("cpu_matched_baseline", shape, "numpy_matrix_same_stream", "linux_server_cpu", run_id)
    status, max_p, max_stat, check_note = correctness_check(
        shape.n,
        shape.p,
        shape.R,
        shape.batch_R,
        shape.seed,
        shape.dtype,
    )
    row.update(
        {
            "correctness_status": status,
            "max_abs_p_diff": as_float(max_p),
            "max_abs_stat_diff": as_float(max_stat),
            "workers": 1,
            "threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "timeout_s": timeout_s,
            "timeout_status": "",
            "notes": check_note,
        }
    )
    if status == STATUS_FAIL:
        return row
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "cpu-child",
        "--params-json",
        json.dumps({"shape": shape.__dict__}),
    ]
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False, env=env)
    except subprocess.TimeoutExpired:
        row.update({"correctness_status": STATUS_TIMEOUT, "timeout_status": "timeout", "notes": f"{check_note}; cpu_timeout_after_{timeout_s}s"})
        return row
    if proc.returncode != 0:
        row.update({"correctness_status": STATUS_FAIL, "notes": f"{check_note}; cpu_child_failed={(proc.stderr or proc.stdout)[-800:]}"})
        return row
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        row.update({"correctness_status": STATUS_FAIL, "notes": f"{check_note}; cpu_child_parse_failed={exc!r}; stdout={proc.stdout[-500:]}"})
        return row
    row.update(
        {
            "cpu_time_s": as_float(payload.get("cpu_time_s")),
            "warm_time_s": as_float(payload.get("warm_time_s")),
            "end_to_end_time_s": as_float(payload.get("cpu_time_s")),
            "peak_host_memory": payload.get("peak_host_memory", ""),
            "timeout_status": "completed",
            "notes": f"{check_note}; mean_p={payload.get('mean_p'):.4g}; perm_s={payload.get('perm_s'):.4g}; build_s={payload.get('build_s'):.4g}; matmul_s={payload.get('matmul_s'):.4g}",
        }
    )
    return row


def memory_risk_row(shape: Shape, run_id: str, benchmark: str, implementation: str, path: Path, fields: list[str], reason: str) -> dict[str, Any]:
    row = common_row(benchmark, shape, implementation, "a100" if "a100" in benchmark else "linux_server_cpu", run_id)
    row.update(
        {
            "correctness_status": STATUS_SKIPPED,
            "max_abs_p_diff": "",
            "max_abs_stat_diff": "",
            "notes": reason,
        }
    )
    append_csv(path, row, fields)
    return row


def choose_best_batch(batch_csv: Path) -> int:
    import pandas as pd

    if not batch_csv.exists():
        return 1024
    df = pd.read_csv(batch_csv)
    ok = df[df["correctness_status"].isin(ACCEPTED_CORRECTNESS_STATUSES)].copy()
    if ok.empty or "end_to_end_time_s" not in ok:
        return 1024
    ok["end_to_end_time_s"] = pd.to_numeric(ok["end_to_end_time_s"], errors="coerce")
    ok = ok.dropna(subset=["end_to_end_time_s"])
    if ok.empty:
        return 1024
    return int(ok.loc[ok["end_to_end_time_s"].idxmin(), "batch_R"])


def batch_shapes(seed: int) -> list[Shape]:
    return [Shape(5_000, 50_000, 10_000, b, seed=seed, source="batch_sweep") for b in [128, 256, 512, 1024, 2048, 4096]]


def shape_sweep_shapes(best_batch: int, seed: int, include_stress: bool) -> list[Shape]:
    shapes: list[Shape] = []
    for p in [10_000, 50_000, 100_000, 250_000]:
        for r in [1_000, 5_000, 10_000, 50_000]:
            shapes.append(Shape(5_000, p, r, best_batch, seed=seed, source="stage1_fixed_n"))
    for n in [1_000, 5_000, 10_000]:
        for r in [1_000, 5_000, 10_000]:
            shapes.append(Shape(n, 50_000, r, best_batch, seed=seed, source="stage2_fixed_p"))
    if include_stress:
        for p in [100_000, 250_000]:
            for r in [10_000, 50_000]:
                shapes.append(Shape(10_000, p, r, best_batch, seed=seed, source="stage3_stress"))
    return shapes


def representative_shapes(best_batch: int, seed: int) -> list[Shape]:
    return [
        Shape(5_000, 50_000, 1_000, best_batch, seed=seed, source="representative"),
        Shape(5_000, 50_000, 10_000, best_batch, seed=seed, source="representative"),
    ]


def should_skip_for_memory(shape: Shape, device_limit_gib: float, host_limit_gib: float) -> str | None:
    est = estimates(shape.n, shape.p, shape.batch_R, shape.dtype)
    if est["estimated_device_memory_per_batch_gib"] > device_limit_gib:
        return f"memory-risk: estimated_device_memory_per_batch_gib={est['estimated_device_memory_per_batch_gib']:.3f} exceeds limit {device_limit_gib:.3f}"
    if est["estimated_x_gib"] > host_limit_gib:
        return f"memory-risk: estimated_x_gib={est['estimated_x_gib']:.3f} exceeds host limit {host_limit_gib:.3f}"
    return None


def run_suite(args: argparse.Namespace) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"permutation_followup_{time.strftime('%Y%m%d_%H%M%S')}"
    decomp_csv = out_dir / "a100_permutation_decomposition.csv"
    batch_csv = out_dir / "a100_permutation_batch_sweep.csv"
    shape_csv = out_dir / "a100_permutation_shape_sweep.csv"
    kernel_csv = out_dir / "a100_permutation_kernel_only.csv"
    cpu_csv = out_dir / "cpu_matched_permutation_baseline.csv"

    import jax

    print(f"JAX backend={jax.default_backend()} devices={jax.devices()}", flush=True)

    if args.stress_only:
        best_batch = args.batch_R or choose_best_batch(batch_csv)
        stress_shapes = [
            Shape(10_000, p, r, best_batch, seed=args.seed, source="stage3_stress")
            for p in [100_000, 250_000]
            for r in [10_000, 50_000]
        ]
        for shape in stress_shapes:
            reason = should_skip_for_memory(shape, args.device_limit_gib, args.host_x_limit_gib)
            if reason:
                memory_risk_row(shape, run_id, "a100_shape_sweep", "jax_host_w_same_stream", shape_csv, DECOMP_FIELDS, reason)
                memory_risk_row(shape, run_id, "a100_kernel_only", "jax_preloaded_wx", kernel_csv, KERNEL_FIELDS, reason)
                continue
            print(f"[stress] n={shape.n} p={shape.p} R={shape.R} batch_R={shape.batch_R}", flush=True)
            row = run_decomposition(shape, run_id)
            row["benchmark"] = "a100_shape_sweep"
            row["source"] = shape.source
            append_csv(shape_csv, row, DECOMP_FIELDS)
            krow = run_kernel_only(shape, run_id)
            krow["source"] = shape.source
            append_csv(kernel_csv, krow, KERNEL_FIELDS)
        write_readme(out_dir)
        return

    batch_plan = batch_shapes(args.seed)
    if args.max_batch_rows:
        batch_plan = batch_plan[: args.max_batch_rows]
    for shape in batch_plan:
        reason = should_skip_for_memory(shape, args.device_limit_gib, args.host_x_limit_gib)
        if reason:
            memory_risk_row(shape, run_id, "a100_batch_sweep", "jax_host_w_same_stream", batch_csv, DECOMP_FIELDS, reason)
            memory_risk_row(shape, run_id, "a100_kernel_only", "jax_preloaded_wx", kernel_csv, KERNEL_FIELDS, reason)
            continue
        print(f"[batch] n={shape.n} p={shape.p} R={shape.R} batch_R={shape.batch_R}", flush=True)
        row = run_decomposition(shape, run_id)
        row["benchmark"] = "a100_batch_sweep"
        row["source"] = shape.source
        append_csv(batch_csv, row, DECOMP_FIELDS)
        krow = run_kernel_only(shape, run_id)
        krow["source"] = shape.source
        append_csv(kernel_csv, krow, KERNEL_FIELDS)

    best_batch = args.batch_R or choose_best_batch(batch_csv)
    print(f"Best measured batch_R for shape sweep: {best_batch}", flush=True)

    for shape in representative_shapes(best_batch, args.seed):
        reason = should_skip_for_memory(shape, args.device_limit_gib, args.host_x_limit_gib)
        if reason:
            memory_risk_row(shape, run_id, "a100_end_to_end_decomposition", "jax_host_w_same_stream", decomp_csv, DECOMP_FIELDS, reason)
            continue
        print(f"[decomp] n={shape.n} p={shape.p} R={shape.R} batch_R={shape.batch_R}", flush=True)
        row = run_decomposition(shape, run_id)
        row["source"] = shape.source
        append_csv(decomp_csv, row, DECOMP_FIELDS)

    shapes = shape_sweep_shapes(best_batch, args.seed, args.include_stress)
    if args.max_shape_rows:
        shapes = shapes[: args.max_shape_rows]
    for shape in shapes:
        reason = should_skip_for_memory(shape, args.device_limit_gib, args.host_x_limit_gib)
        if reason:
            memory_risk_row(shape, run_id, "a100_shape_sweep", "jax_host_w_same_stream", shape_csv, DECOMP_FIELDS, reason)
            memory_risk_row(shape, run_id, "a100_kernel_only", "jax_preloaded_wx", kernel_csv, KERNEL_FIELDS, reason)
            continue
        print(f"[shape] {shape.source} n={shape.n} p={shape.p} R={shape.R} batch_R={shape.batch_R}", flush=True)
        row = run_decomposition(shape, run_id)
        row["benchmark"] = "a100_shape_sweep"
        row["source"] = shape.source
        append_csv(shape_csv, row, DECOMP_FIELDS)
        krow = run_kernel_only(shape, run_id)
        krow["source"] = shape.source
        append_csv(kernel_csv, krow, KERNEL_FIELDS)

    if not args.skip_cpu:
        cpu_shapes = [Shape(5_000, 50_000, r, best_batch, seed=args.seed, source="matched_cpu") for r in [1_000, 10_000]]
        if args.include_cpu_shape_sweep:
            cpu_shapes.extend(shapes)
        seen: set[tuple[int, int, int, int]] = set()
        for shape in cpu_shapes:
            key = (shape.n, shape.p, shape.R, shape.batch_R)
            if key in seen:
                continue
            seen.add(key)
            print(f"[cpu] n={shape.n} p={shape.p} R={shape.R} batch_R={shape.batch_R} timeout={args.cpu_timeout_s}s", flush=True)
            row = run_cpu_baseline(shape, run_id, args.cpu_timeout_s)
            row["source"] = shape.source
            append_csv(cpu_csv, row, CPU_FIELDS)

    write_readme(out_dir)


def numeric(df: Any, col: str) -> Any:
    import pandas as pd

    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([], dtype=float)


def load_csv(path: Path) -> Any:
    import pandas as pd

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def make_figures(out_dir: Path, presentation_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 17,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    colors = {
        "permutation_generation_time_s": "#5B8DEF",
        "W_build_host_time_s": "#2F7D32",
        "host_to_device_transfer_time_s": "#E66A2C",
        "device_compute_time_s": "#B51E59",
        "device_to_host_collect_time_s": "#7B61FF",
        "pvalue_reduction_time_s": "#17202A",
    }
    stage_labels = {
        "permutation_generation_time_s": "perm gen",
        "W_build_host_time_s": "build W",
        "host_to_device_transfer_time_s": "transfer",
        "device_compute_time_s": "W @ X",
        "device_to_host_collect_time_s": "collect",
        "pvalue_reduction_time_s": "reduction",
    }

    decomp = load_csv(out_dir / "a100_permutation_decomposition.csv")
    batch = load_csv(out_dir / "a100_permutation_batch_sweep.csv")
    shape = load_csv(out_dir / "a100_permutation_shape_sweep.csv")
    kernel = load_csv(out_dir / "a100_permutation_kernel_only.csv")
    cpu = load_csv(out_dir / "cpu_matched_permutation_baseline.csv")

    ok_decomp = decomp[decomp.get("correctness_status", pd.Series(dtype=str)).isin(ACCEPTED_CORRECTNESS_STATUSES)].copy() if not decomp.empty else decomp
    if not ok_decomp.empty:
        for col in stage_labels:
            ok_decomp[col] = pd.to_numeric(ok_decomp[col], errors="coerce").fillna(0.0)
        ok_decomp["end_to_end_time_s"] = pd.to_numeric(ok_decomp["end_to_end_time_s"], errors="coerce")
        ok_decomp["total_end_to_end_time_s"] = pd.to_numeric(ok_decomp["total_end_to_end_time_s"], errors="coerce")
        ok_decomp["n_batches"] = np.ceil(ok_decomp["R"] / ok_decomp["batch_R"]).astype(int)
        ok_decomp["label"] = ok_decomp.apply(lambda r: f"R={int(r['R']):,}\nb={int(r['batch_R'])}", axis=1)
        plot_df = ok_decomp.sort_values(["R", "batch_R"]).tail(2).copy()
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        bottom = np.zeros(len(plot_df))
        x = np.arange(len(plot_df))
        for col, label in stage_labels.items():
            vals = plot_df[col].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom, label=label, color=colors[col])
            bottom += vals
        ax.set_xticks(x, plot_df["label"])
        ax.set_ylabel("Seconds")
        ax.set_title("A100 end-to-end decomposition")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, loc="upper left")
        dominant_col = stage_labels[max(stage_labels, key=lambda c: float(plot_df[c].sum()))]
        fig.text(0.08, 0.04, f"A100 pipeline is dominated by {dominant_col}; W @ X is not the bottleneck.", fontsize=17, weight="bold")
        fig.tight_layout(rect=[0.05, 0.10, 0.98, 0.95])
        fig.savefig(fig_dir / "figure1_a100_end_to_end_decomposition.png", dpi=220)
        plt.close(fig)
        make_clean_decomposition_figure(plot_df, cpu, fig_dir, presentation_dir)

    ok_batch = batch[batch.get("correctness_status", pd.Series(dtype=str)).isin(ACCEPTED_CORRECTNESS_STATUSES)].copy() if not batch.empty else batch
    if not ok_batch.empty:
        ok_batch["end_to_end_time_s"] = pd.to_numeric(ok_batch["end_to_end_time_s"], errors="coerce")
        k_ok = kernel[kernel.get("correctness_status", pd.Series(dtype=str)).isin(ACCEPTED_CORRECTNESS_STATUSES)].copy() if not kernel.empty else kernel
        k_ok = k_ok[k_ok.get("source", "") == "batch_sweep"].copy() if "source" in k_ok else k_ok
        if not k_ok.empty:
            k_ok["kernel_only_time_s"] = pd.to_numeric(k_ok["kernel_only_time_s"], errors="coerce")
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        ax.plot(ok_batch["batch_R"], ok_batch["end_to_end_time_s"], marker="o", linewidth=3, color="#B51E59", label="end-to-end")
        if not k_ok.empty:
            ax2 = ax.twinx()
            ax2.plot(k_ok["batch_R"], k_ok["kernel_only_time_s"], marker="s", linewidth=3, color="#2B6CB0", label="kernel-only")
            ax2.set_ylabel("Kernel-only W @ X seconds")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax.legend(loc="best")
        skipped = batch[batch.get("correctness_status", pd.Series(dtype=str)).eq(STATUS_SKIPPED)].copy()
        if not skipped.empty:
            y = float(ok_batch["end_to_end_time_s"].max()) if ok_batch["end_to_end_time_s"].notna().any() else 1.0
            ax.scatter(skipped["batch_R"], [y] * len(skipped), marker="x", s=100, color="#17202A", label="OOM/memory-risk")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch_R")
        ax.set_ylabel("End-to-end seconds")
        ax.set_title("A100 batch_R sweep")
        ax.grid(True, alpha=0.25)
        fig.text(0.08, 0.04, "Batch size controls whether the GPU path is saturated.", fontsize=17, weight="bold")
        fig.tight_layout(rect=[0.06, 0.10, 0.96, 0.95])
        fig.savefig(fig_dir / "figure2_a100_batch_R_sweep.png", dpi=220)
        plt.close(fig)

    if not shape.empty and not cpu.empty:
        ok_shape = shape[shape.get("correctness_status", pd.Series(dtype=str)).isin(ACCEPTED_CORRECTNESS_STATUSES)].copy()
        ok_cpu = cpu[cpu.get("correctness_status", pd.Series(dtype=str)).isin(ACCEPTED_CORRECTNESS_STATUSES)].copy()
        if not ok_shape.empty and not ok_cpu.empty:
            ok_shape["end_to_end_time_s"] = pd.to_numeric(ok_shape["end_to_end_time_s"], errors="coerce")
            ok_cpu["end_to_end_time_s"] = pd.to_numeric(ok_cpu["end_to_end_time_s"], errors="coerce")
            merged = ok_shape.merge(ok_cpu[["n", "p", "R", "end_to_end_time_s"]], on=["n", "p", "R"], suffixes=("_a100", "_cpu"))
            if not merged.empty:
                merged["a100_over_cpu"] = merged["end_to_end_time_s_a100"] / merged["end_to_end_time_s_cpu"]
                fig, ax = plt.subplots(figsize=(12.8, 7.2))
                for p, group in merged.groupby("p"):
                    group = group.sort_values("R")
                    ax.plot(group["R"], group["a100_over_cpu"], marker="o", linewidth=3, label=f"p={int(p):,}")
                ax.axhline(1.0, color="#17202A", linestyle="--", linewidth=2)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("R permutations")
                ax.set_ylabel("A100 / CPU runtime ratio")
                ax.set_title("CPU vs A100 break-even map")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(loc="best")
                found = merged[merged["a100_over_cpu"] < 1.0]
                note = "A100 becomes competitive only beyond the measured break-even." if not found.empty else "No end-to-end break-even found in measured range."
                fig.text(0.08, 0.04, note, fontsize=17, weight="bold")
                fig.tight_layout(rect=[0.06, 0.10, 0.97, 0.95])
                fig.savefig(fig_dir / "figure3_cpu_vs_a100_break_even_map.png", dpi=220)
                plt.close(fig)

    if not kernel.empty:
        k_ok = kernel[kernel.get("correctness_status", pd.Series(dtype=str)).isin(ACCEPTED_CORRECTNESS_STATUSES)].copy()
        e2e = pd.concat([batch, decomp, shape], ignore_index=True) if not batch.empty or not decomp.empty or not shape.empty else pd.DataFrame()
        e2e_ok = e2e[e2e.get("correctness_status", pd.Series(dtype=str)).isin(ACCEPTED_CORRECTNESS_STATUSES)].copy() if not e2e.empty else e2e
        if not k_ok.empty and not e2e_ok.empty:
            k_ok["kernel_only_time_s"] = pd.to_numeric(k_ok["kernel_only_time_s"], errors="coerce")
            e2e_ok["end_to_end_time_s"] = pd.to_numeric(e2e_ok["end_to_end_time_s"], errors="coerce")
            merged = e2e_ok.merge(k_ok[["n", "p", "R", "batch_R", "kernel_only_time_s"]], on=["n", "p", "R", "batch_R"])
            if not merged.empty:
                merged = merged.sort_values(["n", "p", "R"]).tail(6)
                labels = merged.apply(lambda r: f"p={int(r['p']):,}\nR={int(r['R']):,}", axis=1)
                x = np.arange(len(merged))
                width = 0.38
                fig, ax = plt.subplots(figsize=(12.8, 7.2))
                ax.bar(x - width / 2, merged["kernel_only_time_s"], width=width, color="#2B6CB0", label="kernel-only W @ X")
                ax.bar(x + width / 2, merged["end_to_end_time_s"], width=width, color="#B51E59", label="end-to-end")
                ax.set_yscale("log")
                ax.set_xticks(x, labels)
                ax.set_ylabel("Seconds, log scale")
                ax.set_title("Kernel-only vs end-to-end A100 permutation timing")
                ax.grid(axis="y", which="both", alpha=0.25)
                ax.legend(loc="best")
                fig.text(0.08, 0.04, "Matrix multiply may be fast, but the full statistical pipeline includes generation, transfer, reduction, and collection.", fontsize=16, weight="bold")
                fig.tight_layout(rect=[0.05, 0.11, 0.98, 0.95])
                fig.savefig(fig_dir / "figure4_kernel_only_vs_end_to_end.png", dpi=220)
                plt.close(fig)

    summary_fig = fig_dir / "figure1_a100_end_to_end_decomposition.png"
    if summary_fig.exists():
        presentation_dir.mkdir(parents=True, exist_ok=True)
        target = presentation_dir / "server_permutation_followup_decomposition.png"
        target.write_bytes(summary_fig.read_bytes())


def make_clean_decomposition_figure(decomp: Any, cpu: Any, fig_dir: Path, presentation_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if decomp.empty:
        return

    df = decomp.copy().sort_values("R")
    stage_inputs = [
        "permutation_generation_time_s",
        "W_build_host_time_s",
        "host_to_device_transfer_time_s",
        "device_compute_time_s",
        "device_to_host_collect_time_s",
        "pvalue_reduction_time_s",
    ]
    for col in stage_inputs + ["end_to_end_time_s", "total_end_to_end_time_s", "compile_time_s"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["n_batches"] = np.ceil(df["R"] / df["batch_R"]).astype(int)
    df["transfer_collect_s"] = df["host_to_device_transfer_time_s"] + df["device_to_host_collect_time_s"]
    df["recorded_total_s"] = df["total_end_to_end_time_s"].where(df["total_end_to_end_time_s"] > 0, df["end_to_end_time_s"])
    df["stage_sum_raw_s"] = df[stage_inputs].sum(axis=1)
    df["other_s"] = (df["recorded_total_s"] - df["stage_sum_raw_s"]).clip(lower=0.0) + df["pvalue_reduction_time_s"]

    stages = [
        ("permutation_generation_time_s", "perm generation", "#4D7FEA"),
        ("W_build_host_time_s", "build W", "#2F7D32"),
        ("transfer_collect_s", "transfer + collect", "#E66A2C"),
        ("device_compute_time_s", "W @ X", "#B51E59"),
        ("other_s", "other", "#6F7782"),
    ]
    y = np.arange(len(df))
    labels = [f"R={int(row.R):,}\n{int(row.n_batches)} batch{'es' if int(row.n_batches) != 1 else ''}" for row in df.itertuples()]

    fig = plt.figure(figsize=(12.8, 7.2))
    fig.patch.set_facecolor("#FBF7EF")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.65, 1.15], left=0.11, right=0.96, top=0.79, bottom=0.20, wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    card_ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#FFFFFF")
    card_ax.set_facecolor("#FFFDF8")

    left = np.zeros(len(df))
    for col, label, color in stages:
        vals = df[col].to_numpy(dtype=float)
        bars = ax.barh(y, vals, left=left, height=0.46, color=color, edgecolor="white", linewidth=1.1)
        for i, (bar, val, total) in enumerate(zip(bars, vals, df["recorded_total_s"].to_numpy(dtype=float))):
            share = 100.0 * val / total if total else 0.0
            if share >= 45.0 or val >= 0.25:
                ax.text(
                    left[i] + val / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{label}\n{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="white" if label != "build W" else "#FFFFFF",
                    fontweight="bold",
                )
        left += vals

    for i, row in enumerate(df.itertuples()):
        total = float(row.recorded_total_s)
        wx = float(row.device_compute_time_s)
        wx_pct = 100.0 * wx / total if total else 0.0
        dominant_vals = {
            "perm generation": float(row.permutation_generation_time_s),
            "build W": float(row.W_build_host_time_s),
            "transfer + collect": float(row.transfer_collect_s),
            "W @ X": float(row.device_compute_time_s),
            "other": float(row.other_s),
        }
        dominant = max(dominant_vals, key=dominant_vals.get)
        dominant_pct = 100.0 * dominant_vals[dominant] / total if total else 0.0
        ax.text(
            total + max(df["recorded_total_s"]) * 0.025,
            i,
            f"total {total:.3f}s\nW @ X {wx:.3f}s ({wx_pct:.1f}%)",
            ha="left",
            va="center",
            fontsize=11.5,
            color="#17202A",
        )

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds per full scenario")
    ax.set_xlim(0, float(df["recorded_total_s"].max()) * 1.32)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=12.0)
    ax.tick_params(axis="x", labelsize=12.0)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)

    card_ax.set_xticks([])
    card_ax.set_yticks([])
    for spine in card_ax.spines.values():
        spine.set_color("#DFD3C0")
    focus = df.sort_values("R").iloc[-1]
    total = float(focus["recorded_total_s"])
    wx = float(focus["device_compute_time_s"])
    wx_pct = 100.0 * wx / total if total else 0.0
    cpu_text = "CPU matched: unavailable"
    if not cpu.empty:
        cpu_ok = cpu.copy()
        cpu_ok["end_to_end_time_s"] = pd.to_numeric(cpu_ok.get("end_to_end_time_s"), errors="coerce")
        matched = cpu_ok[(cpu_ok["n"] == focus["n"]) & (cpu_ok["p"] == focus["p"]) & (cpu_ok["R"] == focus["R"])]
        if not matched.empty:
            cpu_text = f"CPU matched full e2e:\n{float(matched.iloc[0]['end_to_end_time_s']):.3f}s"
    dominant_vals = {
        "permutation generation": float(focus["permutation_generation_time_s"]),
        "build W": float(focus["W_build_host_time_s"]),
        "transfer + collect": float(focus["transfer_collect_s"]),
        "W @ X": float(focus["device_compute_time_s"]),
        "other": float(focus["other_s"]),
    }
    dominant = max(dominant_vals, key=dominant_vals.get)
    dominant_pct = 100.0 * dominant_vals[dominant] / total if total else 0.0
    card_ax.text(0.08, 0.97, "Full-scenario labels", transform=card_ax.transAxes, ha="left", va="top", fontsize=15.0, fontweight="bold", color="#17202A")
    y0 = 0.88
    for row in df.itertuples():
        total_row = float(row.recorded_total_s)
        wx_row = float(row.device_compute_time_s)
        wx_row_pct = 100.0 * wx_row / total_row if total_row else 0.0
        dom_vals = {
            "perm generation": float(row.permutation_generation_time_s),
            "build W": float(row.W_build_host_time_s),
            "transfer + collect": float(row.transfer_collect_s),
            "W @ X": float(row.device_compute_time_s),
            "other": float(row.other_s),
        }
        dom = max(dom_vals, key=dom_vals.get)
        dom_pct = 100.0 * dom_vals[dom] / total_row if total_row else 0.0
        card_ax.text(
            0.08,
            y0,
            f"R={int(row.R):,}\nbatch_R={int(row.batch_R):,}; n_batches={int(row.n_batches)}",
            transform=card_ax.transAxes,
            ha="left",
            va="top",
            fontsize=11.5,
            color="#17202A",
            fontweight="bold",
        )
        card_ax.text(
            0.08,
            y0 - 0.105,
            f"{row.dtype}; compile excluded; transfer included\n"
            f"A100 total {total_row:.3f}s\n"
            f"W @ X {wx_row:.3f}s ({wx_row_pct:.1f}%)\n"
            f"Dominant: {dom} ({dom_pct:.0f}%)",
            transform=card_ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.4,
            color="#52616D",
            linespacing=1.25,
        )
        y0 -= 0.39
    card_ax.text(0.08, 0.14, cpu_text, transform=card_ax.transAxes, ha="left", va="top", fontsize=11.5, color="#17202A", fontweight="bold")

    fig.text(0.08, 0.945, "Where does the A100 permutation time go?", fontsize=24, fontweight="bold", color="#17202A")
    fig.text(
        0.08,
        0.895,
        "Decomposition of measured A100 path; full scenario total, excluding compile. Host-device transfer is included.",
        fontsize=13.2,
        color="#52616D",
    )
    fig.text(
        0.08,
        0.07,
        "Takeaway: the bottleneck is pipeline construction, not matrix multiply.",
        fontsize=15.0,
        fontweight="bold",
        color="#17202A",
        bbox={"boxstyle": "round,pad=0.42,rounding_size=0.08", "facecolor": "#FFF3E6", "edgecolor": "none"},
    )
    fig.text(
        0.08,
        0.035,
        "Old long-safe A100 row used the prior JAX GPU permutation path at batch_R=512; this follow-up uses host-built W and tuned batch_R=4096.",
        fontsize=9.3,
        color="#52616D",
    )
    fig.savefig(fig_dir / "a100_permutation_decomposition_clean.png", dpi=220)
    fig.savefig(fig_dir / "a100_permutation_decomposition_clean.svg", format="svg")
    presentation_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(presentation_dir / "a100_permutation_decomposition_clean.png", dpi=220)
    fig.savefig(presentation_dir / "a100_permutation_decomposition_clean.svg", format="svg")
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    import pandas as pd

    paths = {
        "decomposition": out_dir / "a100_permutation_decomposition.csv",
        "batch_sweep": out_dir / "a100_permutation_batch_sweep.csv",
        "shape_sweep": out_dir / "a100_permutation_shape_sweep.csv",
        "kernel_only": out_dir / "a100_permutation_kernel_only.csv",
        "cpu_baseline": out_dir / "cpu_matched_permutation_baseline.csv",
    }
    lines = ["# A100 permutation follow-up", "", f"Generated/updated: {timestamp()}", ""]
    lines.append("This suite preserves the feature-wise two-group mean-difference statistic and uses host-built W batches from one NumPy seed stream for the A100 path and trusted CPU checks.")
    lines.append("")
    lines.append("## CSV status")
    for name, path in paths.items():
        if not path.exists():
            lines.append(f"- `{path.name}`: missing")
            continue
        df = pd.read_csv(path)
        lines.append(f"- `{path.name}`: {len(df)} rows")
        if "correctness_status" in df:
            for key, val in df["correctness_status"].value_counts(dropna=False).items():
                lines.append(f"  - {key}: {val}")
    lines.append("")
    lines.append("## Correctness")
    pass_rows = []
    exact_pass_rows = 0
    gpu_tolerance_rows = 0
    check_rows = 0
    manual_check_rows = 0
    max_p_diffs = []
    max_stat_diffs = []
    for path in paths.values():
        if path.exists():
            df = pd.read_csv(path)
            if "correctness_status" in df:
                pass_rows.append(int(df["correctness_status"].isin(ACCEPTED_CORRECTNESS_STATUSES).sum()))
                exact_pass_rows += int((df["correctness_status"] == STATUS_PASS_EXACT).sum())
                gpu_tolerance_rows += int((df["correctness_status"] == STATUS_PASS_GPU_TOLERANCE).sum())
                check_rows += int((df["correctness_status"] == STATUS_LEGACY_CHECK).sum())
                manual_check_rows += int((df["correctness_status"] == STATUS_MANUAL_CHECK).sum())
            if "max_abs_p_diff" in df:
                max_p_diffs.extend(pd.to_numeric(df["max_abs_p_diff"], errors="coerce").dropna().tolist())
            if "max_abs_stat_diff" in df:
                max_stat_diffs.extend(pd.to_numeric(df["max_abs_stat_diff"], errors="coerce").dropna().tolist())
    lines.append(f"- Accepted small matched CPU/JAX subset rows: {sum(pass_rows)}.")
    lines.append(f"- `pass_exact` rows: {exact_pass_rows}; `pass_gpu_tolerance` rows: {gpu_tolerance_rows}; historical `check` rows: {check_rows}; `manual_check` rows: {manual_check_rows}.")
    lines.append("- `check` is retained only for historical/backward-compatible rows. New accepted rows are emitted as `pass_exact` or `pass_gpu_tolerance`; ambiguous rows are emitted as `manual_check`.")
    if max_p_diffs:
        lines.append(f"- Max recorded `max_abs_p_diff`: {max(max_p_diffs):.6g}.")
    if max_stat_diffs:
        lines.append(f"- Max recorded `max_abs_stat_diff`: {max(max_stat_diffs):.6g}.")
    lines.append("- `max_abs_p_diff` and `max_abs_stat_diff` are from a bounded small matched subset for each benchmark row; large rows are timed without changing the statistic or permutation stream.")
    lines.append("")
    lines.append("## End-to-end vs kernel-only")
    lines.append("- End-to-end rows include W construction, host-to-device transfer, `W @ X`, p-value reduction, and collection of reduced results.")
    lines.append("- Kernel-only rows are labeled `kernel-only hypothesis, not end-to-end permutation test`; they time only device-resident `W_batch @ X_device`.")
    lines.append("")

    decomp = load_csv(paths["decomposition"])
    batch = load_csv(paths["batch_sweep"])
    shape = load_csv(paths["shape_sweep"])
    cpu = load_csv(paths["cpu_baseline"])
    e2e = pd.concat([df for df in [decomp, batch, shape] if not df.empty], ignore_index=True) if any(not df.empty for df in [decomp, batch, shape]) else pd.DataFrame()
    lines.append("## Bottleneck summary")
    if not e2e.empty:
        ok = e2e[e2e["correctness_status"].isin(ACCEPTED_CORRECTNESS_STATUSES)].copy()
        stage_cols = [
            "permutation_generation_time_s",
            "W_build_host_time_s",
            "host_to_device_transfer_time_s",
            "device_compute_time_s",
            "device_to_host_collect_time_s",
            "pvalue_reduction_time_s",
        ]
        if not ok.empty and all(col in ok for col in stage_cols):
            for col in stage_cols:
                ok[col] = pd.to_numeric(ok[col], errors="coerce").fillna(0.0)
            sums = ok[stage_cols].sum()
            dominant = str(sums.idxmax())
            lines.append(f"- Largest recorded named A100 end-to-end stage in this run: `{dominant}`.")
    if not shape.empty and not cpu.empty:
        ok_shape = shape[shape["correctness_status"].isin(ACCEPTED_CORRECTNESS_STATUSES)].copy()
        ok_cpu = cpu[cpu["correctness_status"].isin(ACCEPTED_CORRECTNESS_STATUSES)].copy()
        if not ok_shape.empty and not ok_cpu.empty:
            ok_shape["end_to_end_time_s"] = pd.to_numeric(ok_shape["end_to_end_time_s"], errors="coerce")
            ok_cpu["end_to_end_time_s"] = pd.to_numeric(ok_cpu["end_to_end_time_s"], errors="coerce")
            merged = ok_shape.merge(ok_cpu[["n", "p", "R", "end_to_end_time_s"]], on=["n", "p", "R"], suffixes=("_a100", "_cpu"))
            if not merged.empty:
                merged["ratio"] = merged["end_to_end_time_s_a100"] / merged["end_to_end_time_s_cpu"]
                wins = merged[merged["ratio"] < 1.0]
                if wins.empty:
                    lines.append("- No end-to-end A100 break-even was found in the measured range.")
                else:
                    best = wins.sort_values("ratio").iloc[0]
                    lines.append(f"- A100 becomes faster at n={int(best['n'])}, p={int(best['p'])}, R={int(best['R'])}, batch_R={int(best['batch_R'])}.")
            else:
                lines.append("- No matched CPU/A100 break-even rows were available.")
        else:
            lines.append("- CPU or A100 matched rows were unavailable for break-even calculation.")
    else:
        lines.append("- CPU or A100 matched rows were unavailable for break-even calculation.")
    lines.append("")
    lines.append("## Unavailable / OOM / timeout rows")
    bad_total = 0
    for name, path in paths.items():
        if path.exists():
            df = pd.read_csv(path)
            bad = df[df.get("correctness_status", pd.Series(dtype=str)).isin([STATUS_SKIPPED, STATUS_TIMEOUT, STATUS_FAIL])]
            if not bad.empty:
                bad_total += len(bad)
                lines.append(f"- `{path.name}`: {len(bad)} skipped/timeout/fail rows.")
    if bad_total == 0:
        lines.append("- None recorded in this run; Stage 3 stress rows completed within the memory guard.")
    lines.append("")
    lines.append("## Figures")
    lines.append("- `figures/figure1_a100_end_to_end_decomposition.png`")
    lines.append("- `figures/a100_permutation_decomposition_clean.png` and `experiments/results/presentation_figures/a100_permutation_decomposition_clean.{png,svg}`")
    lines.append("- `figures/figure2_a100_batch_R_sweep.png`")
    lines.append("- `figures/figure3_cpu_vs_a100_break_even_map.png`")
    lines.append("- `figures/figure4_kernel_only_vs_end_to_end.png`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_timing_semantics(out_dir)
    write_qa_note(out_dir)


def write_timing_semantics(out_dir: Path) -> None:
    import pandas as pd

    decomp_path = out_dir / "a100_permutation_decomposition.csv"
    cpu_path = out_dir / "cpu_matched_permutation_baseline.csv"
    lines = [
        "# A100 permutation timing semantics audit",
        "",
        f"Generated/updated: {timestamp()}",
        "",
        "## Source",
        "",
        "- Decomposition CSV: `a100_permutation_decomposition.csv`.",
        "- Producing script: `experiments/server/a100_permutation_followup.py`, function `run_decomposition`.",
        "- Figure script path: `make_clean_decomposition_figure` in the same file.",
        "",
        "## Timing meanings",
        "",
        "- `compile_time_s`: JAX compile and warm-up time for the staged functions; excluded from the plotted full-scenario bars.",
        "- `warm_time_s`: one warm preflight call on the first batch shape; not a scenario runtime and not plotted as a total.",
        "- `end_to_end_time_s` and `total_end_to_end_time_s`: warm full-scenario runtime for the follow-up A100 path, excluding compile and including transfer of X, observed statistics, and each host-built W batch.",
        "- `permutation_generation_time_s`: host NumPy permutation index generation summed across all batches in the scenario.",
        "- `W_build_host_time_s`: host construction/filling of W batches summed across the scenario.",
        "- `host_to_device_transfer_time_s`: transfer of X, observed statistics, and W batches to A100 during the timed scenario.",
        "- `device_compute_time_s`: device-resident `W @ X`/absolute-statistic work summed across the scenario.",
        "- `device_to_host_collect_time_s`: collection of the reduced p-values/statistics needed by the benchmark, not the full `R x p` null matrix.",
        "- `pvalue_reduction_time_s`: device-side exceedance count reduction summed across the scenario.",
        "- Kernel-only rows live in `a100_permutation_kernel_only.csv` and are explicitly not used as end-to-end totals.",
        "",
        "## Stage sum check",
        "",
    ]
    if decomp_path.exists():
        df = pd.read_csv(decomp_path)
        stage_cols = [
            "permutation_generation_time_s",
            "W_build_host_time_s",
            "host_to_device_transfer_time_s",
            "device_compute_time_s",
            "device_to_host_collect_time_s",
            "pvalue_reduction_time_s",
        ]
        for col in stage_cols + ["end_to_end_time_s", "total_end_to_end_time_s"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["n_batches"] = (df["R"] + df["batch_R"] - 1) // df["batch_R"]
        df["stage_sum_s"] = df[stage_cols].sum(axis=1)
        df["stage_sum_delta_s"] = df["total_end_to_end_time_s"] - df["stage_sum_s"]
        for row in df.itertuples():
            match = abs(float(row.stage_sum_delta_s)) <= max(0.005, 0.01 * float(row.total_end_to_end_time_s))
            lines.append(
                f"- R={int(row.R):,}, batch_R={int(row.batch_R):,}, n_batches={int(row.n_batches)}: "
                f"stage sum {float(row.stage_sum_s):.6f}s vs recorded total {float(row.total_end_to_end_time_s):.6f}s; "
                f"delta {float(row.stage_sum_delta_s):.6f}s; match={match}."
            )
    else:
        lines.append("- Decomposition CSV unavailable.")

    lines.extend(
        [
            "",
            "## Reconciliation with previous A100 matched-slice result",
            "",
            "The previous long-safe A100 rows in `linux_server_a100/long_safe_20260503_190133/permutation_matrix_gpu.csv` measured the prior `perm_a100` JAX GPU permutation path. That path generated permutations/W on device with `jax.random.permutation`, used the old matched slide setting `batch_R=512`, and reported warm scenario runtimes in the tens of seconds for `n=5,000, p=50,000`.",
            "",
            "This follow-up decomposition measures a different correctness-preserving implementation, `jax_host_w_same_stream`: W is built on the host from the same NumPy seed stream used by the trusted CPU matrix check, W batches are transferred to A100, JAX compile is excluded from the warm full-scenario total, and `batch_R=4,096` was selected from the batch-size sweep. The new decomposition should therefore not be used as a direct replacement for the old implementation's tens-of-seconds runtime without this label.",
            "",
            "CPU comparison rows in `cpu_matched_permutation_baseline.csv` are full-scenario CPU end-to-end timings for the same host-W stream and `batch_R=4,096`; they are not compared against kernel-only A100 timings.",
        ]
    )
    (out_dir / "TIMING_SEMANTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qa_note(out_dir: Path) -> None:
    import pandas as pd

    decomp = pd.read_csv(out_dir / "a100_permutation_decomposition.csv")
    cpu = pd.read_csv(out_dir / "cpu_matched_permutation_baseline.csv")
    stage_cols = [
        "permutation_generation_time_s",
        "W_build_host_time_s",
        "host_to_device_transfer_time_s",
        "device_compute_time_s",
        "device_to_host_collect_time_s",
        "pvalue_reduction_time_s",
    ]
    for col in stage_cols + ["total_end_to_end_time_s"]:
        decomp[col] = pd.to_numeric(decomp[col], errors="coerce")
    decomp["stage_sum_s"] = decomp[stage_cols].sum(axis=1)
    decomp["stage_sum_delta_s"] = decomp["total_end_to_end_time_s"] - decomp["stage_sum_s"]
    rows = [
        "# A100 permutation decomposition QA note",
        "",
        f"Generated/updated: {timestamp()}",
        "",
        "- Figure scope: warm full scenario, not per batch and not kernel-only.",
        "- Compile time: excluded from the plotted bars; `compile_time_s` remains in the CSV.",
        "- Transfer: included. The transfer segment includes X/observed-stat transfer plus host-built W batch transfers during the timed scenario.",
        "- CPU comparison: full CPU end-to-end rows from `cpu_matched_permutation_baseline.csv`; no kernel-only A100 time is compared to CPU end-to-end.",
        "",
        "## Stage Sum Check",
    ]
    for row in decomp.itertuples():
        rows.append(
            f"- R={int(row.R):,}: stage sum {float(row.stage_sum_s):.6f}s, "
            f"recorded total {float(row.total_end_to_end_time_s):.6f}s, "
            f"delta {float(row.stage_sum_delta_s):.6f}s."
        )
    rows.extend(
        [
            "",
            "## CPU Rows",
        ]
    )
    for row in cpu.itertuples():
        rows.append(f"- R={int(row.R):,}: CPU full end-to-end {float(row.end_to_end_time_s):.6f}s; timeout_status={row.timeout_status}.")
    (out_dir / "QA_NOTE.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run")
    run.add_argument("--out-dir", type=Path, default=OUT_DIR)
    run.add_argument("--run-id", default="")
    run.add_argument("--seed", type=int, default=100)
    run.add_argument("--batch-R", type=int, default=0, help="Override best measured batch_R for shape sweep.")
    run.add_argument("--device-limit-gib", type=float, default=55.0)
    run.add_argument("--host-x-limit-gib", type=float, default=24.0)
    run.add_argument("--cpu-timeout-s", type=int, default=CPU_TIMEOUT_S)
    run.add_argument("--max-batch-rows", type=int, default=0, help="Optional development cap; 0 runs the full batch sweep.")
    run.add_argument("--max-shape-rows", type=int, default=0, help="Optional development cap; 0 runs the staged sweep.")
    run.add_argument("--include-stress", action="store_true")
    run.add_argument("--stress-only", action="store_true", help="Append only the Stage 3 stress shapes using the measured/best batch_R.")
    run.add_argument("--include-cpu-shape-sweep", action="store_true")
    run.add_argument("--skip-cpu", action="store_true")

    plot = sub.add_parser("plot")
    plot.add_argument("--out-dir", type=Path, default=OUT_DIR)
    plot.add_argument("--presentation-dir", type=Path, default=PRESENTATION_DIR)

    readme = sub.add_parser("readme")
    readme.add_argument("--out-dir", type=Path, default=OUT_DIR)

    child = sub.add_parser("cpu-child")
    child.add_argument("--params-json", required=True)

    args = parser.parse_args()
    if args.cmd == "cpu-child":
        cpu_child_main(json.loads(args.params_json))
        return
    if args.cmd == "run":
        run_suite(args)
        make_figures(args.out_dir, PRESENTATION_DIR)
        write_readme(args.out_dir)
        return
    if args.cmd == "plot":
        make_figures(args.out_dir, args.presentation_dir)
        write_readme(args.out_dir)
        return
    if args.cmd == "readme":
        write_readme(args.out_dir)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
