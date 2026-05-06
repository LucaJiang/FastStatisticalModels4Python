#!/usr/bin/env python3
"""Targeted Linux server CPU parallelism sweep for Slide 23."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "experiments", ROOT / "experiments" / "kmeans_v3"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.server_utils import rss_mb, timestamp

OUT_DIR = ROOT / "experiments/results/linux_server_cpu/parallelism_targeted"
PRESENTATION_DIR = ROOT / "experiments/results/presentation_figures"
MAIN_COUNTS = [1, 4, 16]
ENV_KEYS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]

KMEANS_FIELDS = [
    "run_id",
    "timestamp",
    "hostname",
    "workload",
    "method",
    "thread_count",
    "n",
    "d",
    "k",
    "max_iter",
    "seed",
    "cold_time_s",
    "warm_times_s",
    "median_warm_time_s",
    "p25_warm_time_s",
    "p75_warm_time_s",
    "peak_rss_mb",
    "correctness_status",
    "final_inertia",
    "relative_inertia_diff_vs_reference",
    "iterations",
    "load_avg_before",
    "load_avg_after",
    "load_status",
    "cpu_affinity",
    "effective_cpu_count",
    "env_json",
    "notes",
]

PERM_FIELDS = [
    "run_id",
    "timestamp",
    "hostname",
    "workload",
    "method",
    "worker_count",
    "n",
    "p",
    "R",
    "batch_R",
    "seed",
    "cold_time_s",
    "warm_times_s",
    "median_warm_time_s",
    "p25_warm_time_s",
    "p75_warm_time_s",
    "peak_parent_rss_mb",
    "peak_child_rss_mb",
    "total_peak_rss_mb",
    "correctness_status",
    "max_abs_p_diff",
    "max_abs_stat_diff",
    "load_avg_before",
    "load_avg_after",
    "load_status",
    "cpu_affinity",
    "effective_cpu_count",
    "env_json",
    "notes",
]


def as_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.9g}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(dict.fromkeys(fields + [key for row in rows for key in row]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    write_csv(path, rows, fields)


def cmd_text(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=10)
        return (proc.stdout or proc.stderr).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def load_average() -> str:
    try:
        return ",".join(f"{x:.2f}" for x in os.getloadavg())
    except Exception:
        return ""


def affinity_text() -> str:
    try:
        aff = sorted(os.sched_getaffinity(0))
        if len(aff) > 18:
            return f"{aff[0]}-{aff[-1]} ({len(aff)} CPUs)"
        return ",".join(str(x) for x in aff)
    except Exception:
        return ""


def effective_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def load_status(load_text: str) -> str:
    try:
        load1 = float(load_text.split(",")[0])
    except Exception:
        return "unknown"
    return "shared_server_busy" if load1 > 0.75 * max(1, effective_cpu_count()) else "ok"


def env_snapshot() -> dict[str, str]:
    return {key: os.environ.get(key, "") for key in ENV_KEYS}


def python_package_versions() -> dict[str, str]:
    versions = {"python": sys.version.replace("\n", " ")}
    for name in ["numpy", "numba", "jax"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc!r}"
    return versions


def numpy_config_text() -> str:
    try:
        import numpy as np

        buf = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = buf
            np.show_config()
        finally:
            sys.stdout = old
        return buf.getvalue()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def environment_report() -> dict[str, Any]:
    lscpu_json = cmd_text(["lscpu", "-J"])
    lscpu = {}
    try:
        payload = json.loads(lscpu_json)
        lscpu = {item["field"].rstrip(":"): item["data"] for item in payload.get("lscpu", [])}
    except Exception:
        pass
    cgroup_paths = {}
    for rel in [
        "/proc/self/cgroup",
        "/sys/fs/cgroup/cpuset.cpus",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/memory.max",
    ]:
        path = Path(rel)
        if path.exists():
            try:
                cgroup_paths[rel] = path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                cgroup_paths[rel] = f"unavailable: {exc!r}"
    return {
        "generated_at": timestamp(),
        "hostname": platform.node(),
        "date_time": cmd_text(["date", "-Is"]),
        "cpu_model": lscpu.get("Model name", ""),
        "logical_cores": os.cpu_count(),
        "physical_cores": lscpu.get("Core(s) per socket", ""),
        "sockets": lscpu.get("Socket(s)", ""),
        "numa_node_count": lscpu.get("NUMA node(s)", ""),
        "available_memory": cmd_text(["free", "-h"]),
        "load_average": load_average(),
        "logged_in_users": cmd_text(["who"]),
        "cpu_affinity": affinity_text(),
        "effective_cpu_count": effective_cpu_count(),
        "cgroup_job_limits": cgroup_paths,
        "package_versions": python_package_versions(),
        "blas_backend": numpy_config_text(),
        "environment_variables": env_snapshot(),
    }


def row_env_for(kind: str, count: int) -> dict[str, str]:
    env = os.environ.copy()
    if kind == "numba":
        env.update(
            {
                "NUMBA_NUM_THREADS": str(count),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            }
        )
    elif kind == "blas":
        env.update(
            {
                "NUMBA_NUM_THREADS": "1",
                "OMP_NUM_THREADS": str(count),
                "MKL_NUM_THREADS": str(count),
                "OPENBLAS_NUM_THREADS": str(count),
                "NUMEXPR_NUM_THREADS": str(count),
                "VECLIB_MAXIMUM_THREADS": str(count),
            }
        )
    elif kind == "workers":
        env.update(
            {
                "NUMBA_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            }
        )
    return env


def child_command(workload: str, params: dict[str, Any]) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "child", "--workload", workload, "--params-json", json.dumps(params)]


def run_child(workload: str, params: dict[str, Any], env_kind: str, count: int, timeout_s: int = 1800) -> dict[str, Any]:
    before = load_average()
    env = row_env_for(env_kind, count)
    proc = subprocess.run(child_command(workload, params), text=True, capture_output=True, env=env, check=False, timeout=timeout_s)
    after = load_average()
    if proc.returncode != 0:
        payload = {"status": "fail", "notes": (proc.stderr or proc.stdout)[-1200:]}
    else:
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            payload = {"status": "fail", "notes": f"parse_failed={exc!r}; stdout={proc.stdout[-800:]}; stderr={proc.stderr[-800:]}"}
    payload["load_avg_before"] = before
    payload["load_avg_after"] = after
    payload["load_status"] = load_status(before)
    payload["cpu_affinity"] = affinity_text()
    payload["effective_cpu_count"] = effective_cpu_count()
    payload["env_json"] = json.dumps({key: env.get(key, "") for key in ENV_KEYS}, sort_keys=True)
    return payload


def median_iqr(times: list[float]) -> tuple[float, float, float]:
    import numpy as np

    arr = np.asarray(times, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def ru_maxrss_mb(children: bool = False) -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB.
    return float(usage.ru_maxrss) / 1024.0


def timed_repeats(fn, repeats: int = 5) -> tuple[Any, float, list[float]]:
    t0 = time.perf_counter()
    out = fn()
    cold = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, cold, times


def child_kmeans(params: dict[str, Any]) -> None:
    from kmeans_v3.data_generation import KMeansScenario, initial_centroids, make_gaussian_mixture
    from kmeans_v3.kmeans_numpy_matmul import kmeans_numpy_matmul

    scenario = KMeansScenario(
        n=int(params["n"]),
        d=int(params["d"]),
        k=int(params["k"]),
        separation=float(params.get("separation", 2.0)),
        seed=int(params["seed"]),
    )
    x, _, _ = make_gaussian_mixture(scenario)
    init = initial_centroids(x, scenario.k, seed=int(params.get("init_seed", 123)))
    max_iter = int(params["max_iter"])
    repeat = int(params["repeat"])
    method = str(params["method"])

    ref = kmeans_numpy_matmul(x, init, max_iter=max_iter)
    if method in {"numba_cpu_serial", "numba_cpu_parallel"}:
        import numba
        from kmeans_v3.kmeans_numba import kmeans_numba

        numba.set_num_threads(int(params["thread_count"]))
        fn = lambda: kmeans_numba(x, init, max_iter=max_iter)
    elif method == "numpy_blas_or_matmul":
        fn = lambda: kmeans_numpy_matmul(x, init, max_iter=max_iter)
    else:
        raise ValueError(method)

    out, cold, warm = timed_repeats(fn, repeat)
    median, p25, p75 = median_iqr(warm)
    rel = abs(float(out[2]) - float(ref[2])) / max(1.0, abs(float(ref[2])))
    status = "pass" if rel < 1e-6 else "fail"
    print(
        json.dumps(
            {
                "cold_time_s": cold,
                "warm_times_s": json.dumps(warm),
                "median_warm_time_s": median,
                "p25_warm_time_s": p25,
                "p75_warm_time_s": p75,
                "peak_rss_mb": ru_maxrss_mb(),
                "correctness_status": status,
                "final_inertia": float(out[2]),
                "relative_inertia_diff_vs_reference": rel,
                "iterations": int(out[3]),
                "notes": f"reference=numpy_matmul; same data/init/stopping; empty_clusters={int(out[4])}",
            }
        )
    )


_PERM_X = None
_PERM_OBS = None
_PERM_N = 0
_PERM_N1 = 0


def _init_perm_worker(x, obs, n: int, n1: int) -> None:
    global _PERM_X, _PERM_OBS, _PERM_N, _PERM_N1
    _PERM_X = x
    _PERM_OBS = obs
    _PERM_N = int(n)
    _PERM_N1 = int(n1)


def _contrast_row_for_perm(seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.permutation(_PERM_N)
    row = np.empty(_PERM_N, dtype=np.float32)
    row[idx[:_PERM_N1]] = np.float32(1.0 / _PERM_N1)
    row[idx[_PERM_N1:]] = np.float32(-1.0 / (_PERM_N - _PERM_N1))
    return row


def _perm_chunk(args):
    import numpy as np

    start, stop, batch_r, seed = args
    exceed = np.zeros(_PERM_X.shape[1], dtype=np.int64)
    first_stats = None
    for offset in range(start, stop, batch_r):
        seeds = range(seed + offset, seed + min(stop, offset + batch_r))
        w = np.vstack([_contrast_row_for_perm(s) for s in seeds])
        stats = np.abs(w @ _PERM_X)
        if first_stats is None:
            first_stats = stats[: min(4, stats.shape[0]), : min(32, stats.shape[1])].astype(np.float64)
        exceed += np.sum(stats >= _PERM_OBS[None, :], axis=0)
    return exceed, first_stats, ru_maxrss_mb()


def run_permutation_same_stream(n: int, p: int, r: int, batch_r: int, seed: int, workers: int) -> tuple[Any, Any, float, float]:
    import multiprocessing as mp
    import numpy as np
    from permutation.matrix_methods import make_expression, observed_stat

    x, labels, _ = make_expression(n, p, 0.0, 0.0, seed)
    x = np.asarray(x, dtype=np.float32)
    obs = np.abs(observed_stat(x, labels).astype(np.float32, copy=False))
    n1 = int(labels.sum())
    chunk_size = int(math.ceil(r / workers))
    chunks = []
    start = 0
    while start < r:
        stop = min(r, start + chunk_size)
        chunks.append((start, stop, batch_r, seed + 50_000))
        start = stop
    if workers == 1:
        _init_perm_worker(x, obs, n, n1)
        outputs = [_perm_chunk(chunks[0])]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers, initializer=_init_perm_worker, initargs=(x, obs, n, n1)) as pool:
            outputs = pool.map(_perm_chunk, chunks)
    exceed = np.sum([out[0] for out in outputs], axis=0)
    first = next((out[1] for out in outputs if out[1] is not None), None)
    child_peak = max(float(out[2]) for out in outputs) if outputs else 0.0
    pvals = (exceed + 1.0) / (r + 1.0)
    return pvals, first, child_peak, rss_mb()


def child_permutation(params: dict[str, Any]) -> None:
    import numpy as np

    n = int(params["n"])
    p = int(params["p"])
    r = int(params["R"])
    batch_r = int(params["batch_R"])
    seed = int(params["seed"])
    workers = int(params["worker_count"])
    repeat = int(params["repeat"])
    method = str(params["method"])

    ref_p, ref_stats, _, _ = run_permutation_same_stream(n, p, r, batch_r, seed, 1)
    fn = lambda: run_permutation_same_stream(n, p, r, batch_r, seed, workers)
    (pvals, stats, child_peak, parent_peak), cold, warm = timed_repeats(fn, repeat)
    median, p25, p75 = median_iqr(warm)
    max_p = float(np.max(np.abs(pvals - ref_p)))
    max_stat = float(np.max(np.abs(stats - ref_stats))) if stats is not None and ref_stats is not None else math.nan
    status = "pass" if max_p <= 1e-12 and (not math.isfinite(max_stat) or max_stat <= 1e-6) else "fail"
    print(
        json.dumps(
            {
                "cold_time_s": cold,
                "warm_times_s": json.dumps(warm),
                "median_warm_time_s": median,
                "p25_warm_time_s": p25,
                "p75_warm_time_s": p75,
                "peak_parent_rss_mb": parent_peak,
                "peak_child_rss_mb": child_peak if workers > 1 else 0.0,
                "total_peak_rss_mb": parent_peak + (child_peak * workers if workers > 1 else 0.0),
                "correctness_status": status,
                "max_abs_p_diff": max_p,
                "max_abs_stat_diff": max_stat,
                "notes": f"{method}; deterministic per-permutation seed stream; reference=worker_count_1",
            }
        )
    )


def make_kmeans_row(run_id: str, payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "timestamp": timestamp(),
        "hostname": platform.node(),
        "workload": "kmeans",
        "method": params["method"],
        "thread_count": params["thread_count"],
        "n": params["n"],
        "d": params["d"],
        "k": params["k"],
        "max_iter": params["max_iter"],
        "seed": params["seed"],
    }
    row.update(payload)
    for key in ["cold_time_s", "median_warm_time_s", "p25_warm_time_s", "p75_warm_time_s", "peak_rss_mb", "final_inertia", "relative_inertia_diff_vs_reference"]:
        row[key] = as_float(row.get(key))
    return row


def make_perm_row(run_id: str, payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "timestamp": timestamp(),
        "hostname": platform.node(),
        "workload": "permutation",
        "method": params["method"],
        "worker_count": params["worker_count"],
        "n": params["n"],
        "p": params["p"],
        "R": params["R"],
        "batch_R": params["batch_R"],
        "seed": params["seed"],
    }
    row.update(payload)
    for key in [
        "cold_time_s",
        "median_warm_time_s",
        "p25_warm_time_s",
        "p75_warm_time_s",
        "peak_parent_rss_mb",
        "peak_child_rss_mb",
        "total_peak_rss_mb",
        "max_abs_p_diff",
        "max_abs_stat_diff",
    ]:
        row[key] = as_float(row.get(key))
    return row


def plot_main(out_dir: Path, presentation_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    presentation_dir.mkdir(parents=True, exist_ok=True)
    kmeans = pd.read_csv(out_dir / "kmeans_parallelism_targeted.csv")
    perm = pd.read_csv(out_dir / "permutation_parallelism_targeted.csv")
    kplot = kmeans[
        (kmeans["method"] == "numba_cpu_parallel")
        & kmeans["thread_count"].isin(MAIN_COUNTS)
        & kmeans["correctness_status"].isin(["pass", "check"])
    ].copy()
    if kplot.empty:
        kplot = kmeans[(kmeans["method"] == "numba_cpu_serial") & kmeans["thread_count"].eq(1)].copy()
    pplot = perm[
        (perm["method"] == "process_pool_same_stream")
        & perm["worker_count"].isin(MAIN_COUNTS)
        & perm["correctness_status"].isin(["pass", "check"])
    ].copy()
    kplot = kplot.sort_values("thread_count")
    pplot = pplot.sort_values("worker_count")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    fig.patch.set_facecolor("#FBF7EF")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", alpha=0.25)
    colors = {"kmeans": "#3267B7", "perm": "#B4494B", "mem": "#6F7782"}

    ax = axes[0]
    x = kplot["thread_count"].astype(int).to_numpy()
    y = pd.to_numeric(kplot["median_warm_time_s"], errors="coerce").to_numpy(float)
    yerr = np.vstack(
        [
            y - pd.to_numeric(kplot["p25_warm_time_s"], errors="coerce").to_numpy(float),
            pd.to_numeric(kplot["p75_warm_time_s"], errors="coerce").to_numpy(float) - y,
        ]
    )
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=3, markersize=8, capsize=5, color=colors["kmeans"])
    ax.set_xticks(MAIN_COUNTS, [str(v) for v in MAIN_COUNTS])
    ax.set_xlabel("Numba threads")
    ax.set_ylabel("median warm runtime (s)")
    ax.set_title("k-means runtime vs threads", weight="bold")

    ax = axes[1]
    x = pplot["worker_count"].astype(int).to_numpy()
    y = pd.to_numeric(pplot["median_warm_time_s"], errors="coerce").to_numpy(float)
    yerr = np.vstack(
        [
            y - pd.to_numeric(pplot["p25_warm_time_s"], errors="coerce").to_numpy(float),
            pd.to_numeric(pplot["p75_warm_time_s"], errors="coerce").to_numpy(float) - y,
        ]
    )
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=3, markersize=8, capsize=5, color=colors["perm"])
    ax.set_xticks(MAIN_COUNTS, [str(v) for v in MAIN_COUNTS])
    ax.set_xlabel("process workers")
    ax.set_ylabel("median warm runtime (s)")
    ax.set_title("permutation runtime vs workers", weight="bold")
    ax.set_ylim(0, max(y) * 1.22)
    for row in pplot.itertuples():
        mem_gib = float(row.total_peak_rss_mb) / 1024.0
        ax.annotate(f"{mem_gib:.1f} GiB", (int(row.worker_count), float(row.median_warm_time_s)), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, color=colors["mem"])

    fig.tight_layout(rect=[0.03, 0.05, 0.98, 0.94])
    fig.savefig(presentation_dir / "server_cpu_parallelism_targeted.png", dpi=220)
    fig.savefig(presentation_dir / "server_cpu_parallelism_targeted.svg", format="svg")
    plt.close(fig)


def write_summary_and_readme(out_dir: Path) -> None:
    import pandas as pd

    kmeans = pd.read_csv(out_dir / "kmeans_parallelism_targeted.csv")
    perm = pd.read_csv(out_dir / "permutation_parallelism_targeted.csv")
    summary_rows = []
    for df, count_col in [(kmeans, "thread_count"), (perm, "worker_count")]:
        for row in df.to_dict("records"):
            summary_rows.append(
                {
                    "workload": row["workload"],
                    "method": row["method"],
                    "parallelism_count": row[count_col],
                    "median_warm_time_s": row["median_warm_time_s"],
                    "p25_warm_time_s": row["p25_warm_time_s"],
                    "p75_warm_time_s": row["p75_warm_time_s"],
                    "peak_memory_mb": row.get("total_peak_rss_mb") or row.get("peak_rss_mb"),
                    "correctness_status": row["correctness_status"],
                    "load_status": row["load_status"],
                }
            )
    write_csv(out_dir / "parallelism_targeted_summary.csv", summary_rows, list(summary_rows[0]))

    env = json.loads((out_dir / "environment.json").read_text(encoding="utf-8"))
    busy = sorted(set(str(x) for x in kmeans["load_status"].tolist() + perm["load_status"].tolist()))
    lines = [
        "# Linux server CPU targeted parallelism",
        "",
        f"Generated/updated: {timestamp()}",
        "",
        "## Provenance",
        "- Run tier: Linux server CPU, not MacBook.",
        f"- Hostname: `{env.get('hostname', '')}`.",
        "- Slide 23 main counts: 1, 4, and 16 workers/threads.",
        "",
        "## Workloads",
        "- k-means: `n=1,000,000`, `d=64`, `K=20`, `max_iter=20`, fixed seed/init, same stopping rule.",
        "- permutation: `n=5,000`, `p=10,000`, `R=1,000`, `batch_R=256`, fixed deterministic per-permutation seed stream.",
        "",
        "## Methods",
        "- k-means: `numba_cpu_serial`, `numba_cpu_parallel`, and `numpy_blas_or_matmul`.",
        "- permutation: `numpy_matrix_same_stream` baseline and `process_pool_same_stream` for 1/4/16 process workers.",
        "",
        "## Thread isolation",
        "- Numba rows set `NUMBA_NUM_THREADS` to the requested count and set BLAS/OpenMP thread env vars to 1.",
        "- NumPy/BLAS rows set BLAS/OpenMP env vars to the requested count and `NUMBA_NUM_THREADS=1`.",
        "- Process-worker permutation rows set BLAS/OpenMP/Numba env vars to 1 inside each worker row.",
        "",
        "## Shared-server load",
        f"- Load status values observed: `{', '.join(busy)}`.",
        "- Rows are not discarded for shared-server load; `load_status` records whether the row was run while the server appeared busy.",
        "",
        "## Safest interpretation",
        "- This is controlled Linux server CPU evidence for small realistic counts, not MacBook performance.",
        "- More workers are not automatically better; runtime and memory should be measured under controlled threading.",
        "- Very high counts such as 128 are intentionally absent from the main slide; high-count behavior on a shared many-core server can reflect memory bandwidth, NUMA placement, scheduler contention, nested threading, or other users.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"parallelism_targeted_{time.strftime('%Y%m%d_%H%M%S')}"
    (out_dir / "environment.json").write_text(json.dumps(environment_report(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    kmeans_shape = {"n": 1_000_000, "d": 64, "k": 20, "max_iter": 20, "seed": 0, "init_seed": 123, "repeat": args.repeat}
    perm_shape = {"n": 5_000, "p": 10_000, "R": 1_000, "batch_R": 256, "seed": 0, "repeat": args.repeat}

    kmeans_path = out_dir / "kmeans_parallelism_targeted.csv"
    perm_path = out_dir / "permutation_parallelism_targeted.csv"

    kmeans_jobs = [{"method": "numba_cpu_serial", "thread_count": 1, **kmeans_shape}]
    kmeans_jobs.extend({"method": "numba_cpu_parallel", "thread_count": c, **kmeans_shape} for c in MAIN_COUNTS)
    kmeans_jobs.extend({"method": "numpy_blas_or_matmul", "thread_count": c, **kmeans_shape} for c in MAIN_COUNTS)
    for params in kmeans_jobs:
        env_kind = "blas" if params["method"] == "numpy_blas_or_matmul" else "numba"
        print(f"[kmeans] {params['method']} threads={params['thread_count']}", flush=True)
        payload = run_child("kmeans", params, env_kind, int(params["thread_count"]), timeout_s=args.timeout_s)
        append_csv(kmeans_path, make_kmeans_row(run_id, payload, params), KMEANS_FIELDS)

    perm_jobs = [{"method": "numpy_matrix_same_stream", "worker_count": 1, **perm_shape}]
    perm_jobs.extend({"method": "process_pool_same_stream", "worker_count": c, **perm_shape} for c in MAIN_COUNTS)
    for params in perm_jobs:
        print(f"[permutation] {params['method']} workers={params['worker_count']}", flush=True)
        payload = run_child("permutation", params, "workers", int(params["worker_count"]), timeout_s=args.timeout_s)
        append_csv(perm_path, make_perm_row(run_id, payload, params), PERM_FIELDS)

    write_summary_and_readme(out_dir)
    plot_main(out_dir, args.presentation_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    run = sub.add_parser("run")
    run.add_argument("--out-dir", type=Path, default=OUT_DIR)
    run.add_argument("--presentation-dir", type=Path, default=PRESENTATION_DIR)
    run.add_argument("--run-id", default="")
    run.add_argument("--repeat", type=int, default=5)
    run.add_argument("--timeout-s", type=int, default=1800)
    child = sub.add_parser("child")
    child.add_argument("--workload", required=True, choices=["kmeans", "permutation"])
    child.add_argument("--params-json", required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--out-dir", type=Path, default=OUT_DIR)
    plot.add_argument("--presentation-dir", type=Path, default=PRESENTATION_DIR)
    args = parser.parse_args()
    if args.cmd == "run":
        run_suite(args)
    elif args.cmd == "child":
        params = json.loads(args.params_json)
        if args.workload == "kmeans":
            child_kmeans(params)
        else:
            child_permutation(params)
    elif args.cmd == "plot":
        write_summary_and_readme(args.out_dir)
        plot_main(args.out_dir, args.presentation_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
