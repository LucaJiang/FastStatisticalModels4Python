#!/usr/bin/env python3
"""36-hour safe Linux server CPU/A100 experiment orchestrator.

The parent process performs resource checks, memory estimates, resume
checks, and CSV appends. Each benchmark scenario runs in a subprocess so
host/GPU memory is released between scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "experiments", ROOT / "experiments" / "kmeans_v3"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.server_utils import RESULT_FIELDS, gpu_memory_used_mb, timestamp

PROFILE = "long_safe"
HOST_ABS_LIMIT_GIB = 300.0
HOST_AVAILABLE_FRACTION = 0.25
GPU_LIMIT_GIB = 55.0
GPU_OCCUPIED_MB = 5_000.0
SCENARIO_TIMEOUT_S = 45 * 60
GIB = 1024**3
CPU_COUNT = os.cpu_count() or 1
CPU_TARGET_UTILIZATION = 0.80
CPU_MAX_PARALLELISM = 128
CPU_MIN_PARALLELISM = 8

EXTRA_FIELDS = [
    "profile",
    "scenario_id",
    "skip_reason",
    "estimated_host_gib",
    "estimated_gpu_gib",
    "threads",
    "workers",
    "dtype",
    "transfer_included",
    "max_iter",
    "separation",
]


@dataclass
class ActiveRun:
    kind: str
    params: dict[str, Any]
    proc: subprocess.Popen[str]
    started_at: float


def host_available_gib() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available / GIB)
    except Exception:
        proc = subprocess.run(["free", "-b"], text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            lines = proc.stdout.splitlines()
            if len(lines) > 1:
                parts = lines[1].split()
                return float(parts[-1]) / GIB
    return 0.0


def load1() -> float:
    try:
        return float(os.getloadavg()[0])
    except Exception:
        return 9999.0


def cpu_cap() -> int:
    load = load1()
    target_load = max(float(CPU_MIN_PARALLELISM), CPU_COUNT * CPU_TARGET_UTILIZATION)
    headroom = max(float(CPU_MIN_PARALLELISM), target_load - load)
    cap = int(headroom // 8) * 8
    cap = max(CPU_MIN_PARALLELISM, min(CPU_MAX_PARALLELISM, cap))
    return cap


def scenario_id(kind: str, params: dict[str, Any]) -> str:
    keep = {k: params[k] for k in sorted(params) if k not in {"out_dir"}}
    return kind + "__" + "__".join(f"{k}-{keep[k]}" for k in keep)


def estimate_kmeans_host_gib(n: int, d: int, k: int) -> float:
    raw = n * d * 8 + n * k * 8 * 2 + k * d * 8
    return raw * 2.5 / GIB


def estimate_kmeans_gpu_gib(n: int, d: int, k: int) -> float:
    raw = n * d * 4 + n * k * 4 * 2 + k * d * 4
    return raw * 2.8 / GIB


def estimate_perm_gib(n: int, p: int, batch_r: int, device: str) -> float:
    elem = 4
    raw = n * p * elem + batch_r * n * elem + batch_r * p * elem + p * 12
    multiplier = 2.8 if device == "gpu" else 2.2
    return raw * multiplier / GIB


def csv_path(cpu_dir: Path, a100_dir: Path, kind: str) -> Path:
    if kind == "kmeans_cpu":
        return cpu_dir / "kmeans_cpu_scaling.csv"
    if kind == "kmeans_threads":
        return cpu_dir / "kmeans_numba_thread_sweep.csv"
    if kind == "perm_cpu":
        return cpu_dir / "permutation_cpu_scaling.csv"
    if kind == "perm_workers":
        return cpu_dir / "permutation_worker_sweep.csv"
    if kind == "perm_calibration":
        return cpu_dir / "permutation_calibration_server_subset.csv"
    if kind == "kmeans_a100":
        return a100_dir / "kmeans_jax_gpu.csv"
    if kind == "perm_a100":
        return a100_dir / "permutation_matrix_gpu.csv"
    raise ValueError(kind)


def is_gpu_kind(kind: str) -> bool:
    return kind.endswith("a100")


def is_cpu_kind(kind: str) -> bool:
    return not is_gpu_kind(kind)


def cpu_need(params: dict[str, Any]) -> int:
    return int(params.get("threads") or params.get("workers") or 1)


def completed_ids(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("scenario_id")
                if sid:
                    done.add(sid)
    return done


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(RESULT_FIELDS + EXTRA_FIELDS + list(row.keys())))
    exists = path.exists()
    if exists:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            try:
                old_fields = next(reader)
                fields = list(dict.fromkeys(old_fields + fields))
            except StopIteration:
                exists = False
    if exists and fields != old_fields:
        rows = []
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def base_row(kind: str, params: dict[str, Any], status: str, note: str) -> dict[str, Any]:
    tier = "linux_server_a100" if kind.endswith("a100") else "linux_server_cpu"
    return {
        "run_id": params.get("run_id", ""),
        "timestamp": timestamp(),
        "environment_tier": tier,
        "machine_name": platform.node(),
        "workload": "kmeans" if "kmeans" in kind else "permutation",
        "implementation": params.get("implementation", kind),
        "backend": "jax" if kind.endswith("a100") else "cpu",
        "device": "a100" if kind.endswith("a100") else "linux_server_cpu",
        "n": params.get("n", ""),
        "p": params.get("p", ""),
        "d": params.get("d", ""),
        "k": params.get("k", ""),
        "R": params.get("R", ""),
        "batch_R": params.get("batch_R", ""),
        "seed": params.get("seed", ""),
        "cold_time_s": "",
        "warm_median_s": "",
        "warm_iqr_s": "",
        "host_peak_mem_mb": "",
        "gpu_peak_mem_mb": "",
        "validation_status": status,
        "notes": note,
        "profile": PROFILE,
        "scenario_id": params["scenario_id"],
        "skip_reason": note if status in {"skipped", "timeout", "pending"} else "",
        "estimated_host_gib": params.get("estimated_host_gib", ""),
        "estimated_gpu_gib": params.get("estimated_gpu_gib", ""),
        "threads": params.get("threads", ""),
        "workers": params.get("workers", ""),
        "dtype": params.get("dtype", ""),
        "transfer_included": params.get("transfer_included", ""),
        "max_iter": params.get("max_iter", ""),
        "separation": params.get("separation", ""),
    }


def child_cmd(kind: str, params: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "child",
        "--kind",
        kind,
        "--params-json",
        json.dumps(params),
    ]


def child_env(params: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    threads = params.get("threads")
    if threads:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
            env[key] = str(threads)
    return env


def start_child(kind: str, params: dict[str, Any]) -> ActiveRun:
    proc = subprocess.Popen(
        child_cmd(kind, params),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env(params),
    )
    return ActiveRun(kind=kind, params=params, proc=proc, started_at=time.monotonic())


def collect_child(active: ActiveRun) -> dict[str, Any]:
    proc = active.proc
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        row = base_row(active.kind, active.params, "fail", (stderr or stdout)[-1500:])
        return row
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except Exception as exc:
        row = base_row(active.kind, active.params, "fail", f"could not parse child output: {exc!r}; stdout={stdout[-800:]}; stderr={stderr[-800:]}")
        return row


def time_call(fn, repeat: int) -> tuple[Any, float, float, float]:
    import numpy as np

    t0 = time.perf_counter()
    out = fn()
    cold = time.perf_counter() - t0
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    q1, q3 = np.percentile(np.asarray(times), [25, 75]) if times else (0.0, 0.0)
    return out, float(cold), float(np.median(times) if times else cold), float(q3 - q1)


def child_main(kind: str, params: dict[str, Any]) -> None:
    import numpy as np
    from common.server_utils import gpu_memory_used_mb, rss_mb

    if "kmeans" in kind:
        from kmeans_v3.data_generation import KMeansScenario, initial_centroids, make_gaussian_mixture

        scenario = KMeansScenario(
            n=int(params["n"]),
            d=int(params["d"]),
            k=int(params["k"]),
            separation=float(params.get("separation", 2.0)),
            seed=int(params["seed"]),
        )
        x, true_labels, _ = make_gaussian_mixture(scenario)
        init = initial_centroids(x, scenario.k, seed=123 + scenario.seed)

    if kind == "kmeans_cpu":
        from kmeans_v3.kmeans_numpy_matmul import kmeans_numpy_matmul

        impl = params["implementation"]
        if impl == "numba":
            import numba
            from kmeans_v3.kmeans_numba import kmeans_numba

            numba.set_num_threads(int(params.get("threads") or 1))
            fn = lambda: kmeans_numba(x, init, max_iter=int(params["max_iter"]))
        else:
            fn = lambda: kmeans_numpy_matmul(x, init, max_iter=int(params["max_iter"]))
        ref = kmeans_numpy_matmul(x, init, max_iter=int(params["max_iter"]))
        out, cold, med, warm_iqr = time_call(fn, int(params["repeat"]))
        rel = abs(out[2] - ref[2]) / max(1.0, abs(ref[2]))
        row = base_row(kind, params, "pass" if rel < 1e-6 else "check", f"inertia={out[2]:.6g}; rel_delta={rel:.3g}")
        row.update({"cold_time_s": cold, "warm_median_s": med, "warm_iqr_s": warm_iqr, "host_peak_mem_mb": rss_mb()})
        print(json.dumps(row))
        return

    if kind == "kmeans_threads":
        import numba
        from kmeans_v3.kmeans_numba import kmeans_numba

        numba.set_num_threads(int(params["threads"]))
        out, cold, med, warm_iqr = time_call(lambda: kmeans_numba(x, init, max_iter=int(params["max_iter"])), int(params["repeat"]))
        row = base_row(kind, params, "pass", f"inertia={out[2]:.6g}")
        row.update({"cold_time_s": cold, "warm_median_s": med, "warm_iqr_s": warm_iqr, "host_peak_mem_mb": rss_mb()})
        print(json.dumps(row))
        return

    if kind == "kmeans_a100":
        from kmeans_v3.kmeans_jax import kmeans_jax

        before = gpu_memory_used_mb()
        out, cold, med, warm_iqr = time_call(
            lambda: kmeans_jax(x, init, max_iter=int(params["max_iter"]), dtype=params.get("dtype", "float32")),
            int(params["repeat"]),
        )
        after = gpu_memory_used_mb()
        row = base_row(kind, params, "pass", f"inertia={out[2]:.6g}; gpu_before_mb={before}")
        row.update({"cold_time_s": cold, "warm_median_s": med, "warm_iqr_s": warm_iqr, "gpu_peak_mem_mb": after if after is not None else before})
        print(json.dumps(row))
        return

    if "perm" in kind:
        from permutation.matrix_methods import make_expression, perm_matrix_cpu, perm_matrix_threaded

        x, labels, _ = make_expression(int(params["n"]), int(params["p"]), 0.0, 0.0, int(params["seed"]))

    if kind in {"perm_cpu", "perm_calibration"}:
        out, cold, med, warm_iqr = time_call(
            lambda: perm_matrix_cpu(x, labels, int(params["R"]), int(params["batch_R"]), int(params["seed"]) + 100),
            int(params["repeat"]),
        )
        row = base_row(kind, params, "pass", f"mean_p={float(np.mean(out)):.4g}; min_p={float(np.min(out)):.4g}; max_p={float(np.max(out)):.4g}")
        row.update({"cold_time_s": cold, "warm_median_s": med, "warm_iqr_s": warm_iqr, "host_peak_mem_mb": rss_mb()})
        print(json.dumps(row))
        return

    if kind == "perm_workers":
        out, cold, med, warm_iqr = time_call(
            lambda: perm_matrix_threaded(x, labels, int(params["R"]), int(params["batch_R"]), int(params["seed"]) + 100, int(params["workers"])),
            int(params["repeat"]),
        )
        row = base_row(kind, params, "pass", f"mean_p={float(np.mean(out)):.4g}")
        row.update({"cold_time_s": cold, "warm_median_s": med, "warm_iqr_s": warm_iqr, "host_peak_mem_mb": rss_mb()})
        print(json.dumps(row))
        return

    if kind == "perm_a100":
        from permutation.matrix_methods import perm_matrix_gpu

        before = gpu_memory_used_mb()
        out, cold, med, warm_iqr = time_call(
            lambda: perm_matrix_gpu(x, labels, int(params["R"]), int(params["batch_R"]), int(params["seed"]) + 100),
            int(params["repeat"]),
        )
        after = gpu_memory_used_mb()
        row = base_row(kind, params, "pass", f"mean_p={float(np.mean(out)):.4g}; gpu_before_mb={before}")
        row.update({"cold_time_s": cold, "warm_median_s": med, "warm_iqr_s": warm_iqr, "gpu_peak_mem_mb": after if after is not None else before})
        print(json.dumps(row))
        return

    raise ValueError(kind)


def build_scenarios(run_id: str) -> list[tuple[str, dict[str, Any]]]:
    scenarios: list[tuple[str, dict[str, Any]]] = []
    seen_scenario_ids: set[str] = set()

    def add(kind: str, params: dict[str, Any]) -> None:
        params = dict(params)
        params["run_id"] = run_id
        params["profile"] = PROFILE
        params["scenario_id"] = scenario_id(kind, params)
        if params["scenario_id"] in seen_scenario_ids:
            return
        seen_scenario_ids.add(params["scenario_id"])
        scenarios.append((kind, params))

    for n in [100_000, 300_000, 1_000_000, 3_000_000, 5_000_000]:
        for d in [10, 64, 256]:
            for k in [5, 20, 50]:
                for seed in [0, 1, 2]:
                    est = estimate_kmeans_gpu_gib(n, d, k)
                    add("kmeans_a100", {"n": n, "d": d, "k": k, "seed": seed, "max_iter": 20, "repeat": 3, "dtype": "float32", "transfer_included": True, "estimated_gpu_gib": round(est, 3), "estimated_host_gib": round(n * d * 8 / GIB, 3)})

    # A100 permutation priority ordering.
    for batch_r in [128, 256, 512, 1024, 2048]:
        add("perm_a100", {"n": 5_000, "p": 50_000, "R": 5_000, "batch_R": batch_r, "seed": 0, "repeat": 2, "dtype": "float32", "transfer_included": True})
    for p in [10_000, 50_000, 100_000, 250_000]:
        add("perm_a100", {"n": 5_000, "p": p, "R": 5_000, "batch_R": 512, "seed": 0, "repeat": 2, "dtype": "float32", "transfer_included": True})
    for r in [1_000, 5_000, 10_000, 50_000]:
        add("perm_a100", {"n": 5_000, "p": 50_000, "R": r, "batch_R": 512, "seed": 0, "repeat": 2, "dtype": "float32", "transfer_included": True})
    for n in [10_000, 20_000]:
        for p in [50_000, 100_000]:
            add("perm_a100", {"n": n, "p": p, "R": 5_000, "batch_R": 512, "seed": 0, "repeat": 2, "dtype": "float32", "transfer_included": True})

    for n in [100_000, 300_000, 1_000_000, 3_000_000, 5_000_000]:
        for d in [10, 64, 256]:
            for k in [5, 20, 50]:
                for sep in [1.0, 2.0]:
                    for seed in [0, 1, 2]:
                        for impl in ["numpy_matmul", "numba"]:
                            add("kmeans_cpu", {"implementation": impl, "n": n, "d": d, "k": k, "separation": sep, "seed": seed, "max_iter": 20, "repeat": 3, "threads": min(cpu_cap(), 16) if impl == "numba" else min(cpu_cap(), 16), "estimated_host_gib": round(estimate_kmeans_host_gib(n, d, k), 3)})

    for threads in [1, 2, 4, 8, 16, 32, 64, 128]:
        add("kmeans_threads", {"implementation": "numba", "n": 1_000_000, "d": 64, "k": 20, "separation": 2.0, "seed": 0, "max_iter": 20, "repeat": 3, "threads": threads, "estimated_host_gib": round(estimate_kmeans_host_gib(1_000_000, 64, 20), 3)})

    for n in [1_000, 5_000, 10_000, 50_000]:
        for p in [1_000, 10_000, 50_000]:
            for r in [1_000, 10_000, 100_000]:
                for seed in [0, 1, 2]:
                    add("perm_cpu", {"implementation": "numpy_matrix", "n": n, "p": p, "R": r, "batch_R": 512, "seed": seed, "repeat": 3, "threads": min(cpu_cap(), 16), "estimated_host_gib": round(estimate_perm_gib(n, p, 512, "cpu"), 3)})

    for workers in [1, 2, 4, 8, 16, 32, 64, 128]:
        add("perm_workers", {"implementation": "threadpool_matrix", "n": 5_000, "p": 10_000, "R": 10_000, "batch_R": 512, "seed": 0, "repeat": 3, "workers": workers, "estimated_host_gib": round(estimate_perm_gib(5_000, 10_000, 512, "cpu"), 3)})

    for n, p, r in [(1_000, 1_000, 10_000), (5_000, 10_000, 10_000)]:
        add("perm_calibration", {"implementation": "numpy_matrix", "n": n, "p": p, "R": r, "batch_R": 512, "seed": 0, "repeat": 3, "threads": min(cpu_cap(), 16), "estimated_host_gib": round(estimate_perm_gib(n, p, 512, "cpu"), 3)})

    for kind, params in scenarios:
        if kind == "perm_a100":
            params["estimated_gpu_gib"] = round(estimate_perm_gib(int(params["n"]), int(params["p"]), int(params["batch_R"]), "gpu"), 3)
            params["estimated_host_gib"] = round(int(params["n"]) * int(params["p"]) * 4 / GIB, 3)
    return scenarios


def host_budget_gib() -> tuple[float, float]:
    avail = host_available_gib()
    host_budget = min(HOST_ABS_LIMIT_GIB, max(1.0, HOST_AVAILABLE_FRACTION * avail))
    return avail, host_budget


def classify_scenario(
    kind: str,
    params: dict[str, Any],
    active_host_gib: float,
    active_cpu_parallelism: int,
) -> tuple[str, str | None]:
    avail, host_budget = host_budget_gib()
    host_est = float(params.get("estimated_host_gib") or 0.0)
    if host_est > host_budget:
        return "skip", f"estimated host {host_est:.1f}GiB exceeds budget {host_budget:.1f}GiB from available {avail:.1f}GiB"
    if host_est + active_host_gib > host_budget:
        return "defer", f"combined host estimate {host_est + active_host_gib:.1f}GiB exceeds budget {host_budget:.1f}GiB"
    if kind.endswith("a100"):
        gpu_est = float(params.get("estimated_gpu_gib") or 0.0)
        if gpu_est > GPU_LIMIT_GIB:
            return "skip", f"estimated GPU {gpu_est:.1f}GiB exceeds budget {GPU_LIMIT_GIB:.1f}GiB"
        used = gpu_memory_used_mb()
        if used is not None and used > GPU_OCCUPIED_MB:
            return "defer", f"A100 appears occupied: {used:.0f}MiB used"
    if is_cpu_kind(kind):
        cap = cpu_cap()
        need = cpu_need(params)
        if need > cap:
            return "defer", f"dynamic CPU cap {cap} below requested {need}; load1={load1():.1f}"
        if active_cpu_parallelism + need > cap:
            return "defer", f"dynamic CPU budget {cap} below combined requested {active_cpu_parallelism + need}; load1={load1():.1f}"
    return "run", None


def try_start_next(
    pending: list[tuple[str, dict[str, Any]]],
    active_host_gib: float,
    active_cpu_parallelism: int,
) -> tuple[ActiveRun | None, list[tuple[str, dict[str, Any]]], str | None]:
    skipped_rows: list[tuple[str, dict[str, Any]]] = []
    deferred_reason: str | None = None
    idx = 0
    while idx < len(pending):
        kind, params = pending[idx]
        decision, reason = classify_scenario(kind, params, active_host_gib, active_cpu_parallelism)
        if decision == "skip":
            skipped_rows.append((kind, base_row(kind, params, "skipped", reason or "skipped")))
            pending.pop(idx)
            continue
        if decision == "run":
            pending.pop(idx)
            return start_child(kind, params), skipped_rows, None
        if deferred_reason is None and reason is not None:
            deferred_reason = reason
        idx += 1
    return None, skipped_rows, deferred_reason


def terminate_active(active: ActiveRun) -> None:
    if active.proc.poll() is not None:
        return
    active.proc.terminate()
    try:
        active.proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        active.proc.kill()
        active.proc.wait(timeout=10)


def write_env_reports(cpu_dir: Path, a100_dir: Path) -> None:
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    subprocess.run([sys.executable, str(ROOT / "experiments/common/env_report.py"), "--tier", "linux_server_cpu", "--out", str(cpu_dir / "env.json")], env=env, check=False)
    subprocess.run([sys.executable, str(ROOT / "experiments/common/env_report.py"), "--tier", "linux_server_a100", "--out", str(a100_dir / "env.json")], env=env, check=False)


def plot_and_summarize(cpu_dir: Path, a100_dir: Path) -> None:
    from server.long_safe_plots import plot_all

    try:
        plot_all(cpu_dir, a100_dir)
    except Exception as exc:
        print(f"long-safe plotting failed: {exc}", file=sys.stderr)
    write_summary(cpu_dir, "SERVER_CPU_SUMMARY.md", "Linux server CPU")
    write_summary(a100_dir, "A100_SUMMARY.md", "Linux server A100")


def write_summary(root: Path, filename: str, title: str) -> None:
    lines = [f"# {title} long-safe summary", "", f"Generated/updated: {timestamp()}", "", "## CSV status", ""]
    for path in sorted(root.glob("*.csv")):
        try:
            import pandas as pd

            df = pd.read_csv(path)
            counts = df.get("validation_status")
            lines.append(f"- `{path.name}`: {len(df)} rows")
            if counts is not None:
                for key, value in counts.value_counts(dropna=False).items():
                    lines.append(f"  - {key}: {value}")
            if "scenario_id" in df.columns:
                duplicate_count = int(df["scenario_id"].duplicated().sum())
                lines.append(f"  - duplicate scenario_id rows: {duplicate_count}")
        except Exception as exc:
            lines.append(f"- `{path.name}`: could not summarize: {exc!r}")
    lines.extend(["", "## Notes", "", "- Quick-run outputs are separate; this directory is the long-safe run surface.", "- Skipped rows are intentional memory/load guardrail outcomes."])
    (root / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parent_main(args: argparse.Namespace) -> None:
    stamp = args.stamp or time.strftime("%Y%m%d_%H%M%S")
    cpu_dir = args.cpu_dir or (ROOT / f"experiments/results/linux_server_cpu/long_safe_{stamp}")
    a100_dir = args.a100_dir or (ROOT / f"experiments/results/linux_server_a100/long_safe_{stamp}")
    cpu_dir.mkdir(parents=True, exist_ok=True)
    a100_dir.mkdir(parents=True, exist_ok=True)
    write_env_reports(cpu_dir, a100_dir)

    run_id = f"long_safe_{stamp}"
    scenarios = build_scenarios(run_id)
    all_csvs = [csv_path(cpu_dir, a100_dir, kind) for kind, _ in scenarios]
    done = completed_ids(all_csvs)
    deadline = time.monotonic() + args.hours * 3600
    completed = 0
    gpu_pending = [(kind, params) for kind, params in scenarios if is_gpu_kind(kind) and params["scenario_id"] not in done]
    cpu_pending = [(kind, params) for kind, params in scenarios if is_cpu_kind(kind) and params["scenario_id"] not in done]
    active_gpu: ActiveRun | None = None
    active_cpus: list[ActiveRun] = []
    last_idle_note = 0.0

    while True:
        if args.max_scenarios and completed >= args.max_scenarios:
            print("Reached max-scenarios budget.", flush=True)
            break
        now = time.monotonic()
        if now >= deadline:
            print("Reached wall-time budget; stopping.", flush=True)
            break

        active_host = 0.0
        if active_gpu is not None:
            active_host += float(active_gpu.params.get("estimated_host_gib") or 0.0)
        for active_cpu in active_cpus:
            active_host += float(active_cpu.params.get("estimated_host_gib") or 0.0)
        active_cpu_parallelism = sum(cpu_need(active.params) for active in active_cpus)

        if active_gpu is None and gpu_pending:
            active_gpu, skipped_rows, deferred_reason = try_start_next(gpu_pending, active_host, active_cpu_parallelism)
            for kind, row in skipped_rows:
                append_row(csv_path(cpu_dir, a100_dir, kind), row)
                done.add(row["scenario_id"])
                completed += 1
            if active_gpu is not None:
                print(f"[launch][gpu] {active_gpu.kind} {active_gpu.params['scenario_id']}", flush=True)
            elif deferred_reason and now - last_idle_note > 30:
                print(f"[defer][gpu] {deferred_reason}", flush=True)
                last_idle_note = now

        active_host = 0.0
        if active_gpu is not None:
            active_host += float(active_gpu.params.get("estimated_host_gib") or 0.0)
        for active_cpu in active_cpus:
            active_host += float(active_cpu.params.get("estimated_host_gib") or 0.0)
        while cpu_pending:
            active_cpu_parallelism = sum(cpu_need(active.params) for active in active_cpus)
            new_cpu, skipped_rows, deferred_reason = try_start_next(cpu_pending, active_host, active_cpu_parallelism)
            for kind, row in skipped_rows:
                append_row(csv_path(cpu_dir, a100_dir, kind), row)
                done.add(row["scenario_id"])
                completed += 1
            if new_cpu is None:
                if deferred_reason and now - last_idle_note > 30:
                    print(f"[defer][cpu] {deferred_reason}", flush=True)
                    last_idle_note = now
                break
            active_cpus.append(new_cpu)
            active_host += float(new_cpu.params.get("estimated_host_gib") or 0.0)
            print(f"[launch][cpu] {new_cpu.kind} {new_cpu.params['scenario_id']}", flush=True)

        progressed = False
        if active_gpu is not None:
            runtime = time.monotonic() - active_gpu.started_at
            if runtime > args.scenario_timeout_s:
                terminate_active(active_gpu)
                row = base_row(active_gpu.kind, active_gpu.params, "timeout", f"scenario exceeded {args.scenario_timeout_s}s")
            elif active_gpu.proc.poll() is not None:
                row = collect_child(active_gpu)
            else:
                row = None
            if row is not None:
                append_row(csv_path(cpu_dir, a100_dir, active_gpu.kind), row)
                done.add(active_gpu.params["scenario_id"])
                completed += 1
                progressed = True
                print(f"[done][gpu] {active_gpu.kind} {active_gpu.params['scenario_id']} -> {row['validation_status']}", flush=True)
                active_gpu = None

        remaining_cpus: list[ActiveRun] = []
        for active_cpu in active_cpus:
            runtime = time.monotonic() - active_cpu.started_at
            if runtime > args.scenario_timeout_s:
                terminate_active(active_cpu)
                row = base_row(active_cpu.kind, active_cpu.params, "timeout", f"scenario exceeded {args.scenario_timeout_s}s")
            elif active_cpu.proc.poll() is not None:
                row = collect_child(active_cpu)
            else:
                remaining_cpus.append(active_cpu)
                continue
            append_row(csv_path(cpu_dir, a100_dir, active_cpu.kind), row)
            done.add(active_cpu.params["scenario_id"])
            completed += 1
            progressed = True
            print(f"[done][cpu] {active_cpu.kind} {active_cpu.params['scenario_id']} -> {row['validation_status']}", flush=True)
        active_cpus = remaining_cpus

        if completed and completed % args.plot_every == 0 and progressed:
            plot_and_summarize(cpu_dir, a100_dir)

        if not gpu_pending and not cpu_pending and active_gpu is None and not active_cpus:
            print("All pending scenarios handled.", flush=True)
            break
        if not progressed:
            time.sleep(5)

    for active in ([active_gpu] if active_gpu is not None else []) + active_cpus:
        terminate_active(active)
    plot_and_summarize(cpu_dir, a100_dir)
    print(f"CPU results: {cpu_dir}")
    print(f"A100 results: {a100_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    child = sub.add_parser("child")
    child.add_argument("--kind", required=True)
    child.add_argument("--params-json", required=True)
    run = sub.add_parser("run")
    run.add_argument("--hours", type=float, default=36.0)
    run.add_argument("--scenario-timeout-s", type=int, default=SCENARIO_TIMEOUT_S)
    run.add_argument("--stamp", default=None)
    run.add_argument("--cpu-dir", type=Path, default=None)
    run.add_argument("--a100-dir", type=Path, default=None)
    run.add_argument("--max-scenarios", type=int, default=0)
    run.add_argument("--plot-every", type=int, default=12)
    args = parser.parse_args()
    if args.cmd == "child":
        child_main(args.kind, json.loads(args.params_json))
    elif args.cmd == "run":
        parent_main(args)
    else:
        parser.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
