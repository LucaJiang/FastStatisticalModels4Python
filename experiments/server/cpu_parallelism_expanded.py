#!/usr/bin/env python3
"""Expanded Linux server CPU parallelism sweep for Slide 23."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from common.server_utils import timestamp
from server.cpu_parallelism_targeted import (
    ENV_KEYS,
    affinity_text,
    as_float,
    cmd_text,
    effective_cpu_count,
    environment_report,
    load_average,
    load_status,
    plot_main as _unused_targeted_plot_main,
    run_child,
    write_csv,
)

OUT_DIR = ROOT / "experiments/results/linux_server_cpu/parallelism_expanded"
PRESENTATION_DIR = ROOT / "experiments/results/presentation_figures"
COUNTS = [1, 4, 16, 64, 128]

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
    "min_warm_time_s",
    "peak_rss_mb",
    "final_inertia",
    "iterations",
    "correctness_status",
    "relative_inertia_diff_vs_reference",
    "numba_threads",
    "effective_cpu_count",
    "affinity_count",
    "load_average_before",
    "load_average_after",
    "resource_isolation",
    "interpretation",
    "row_status",
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
    "min_warm_time_s",
    "parent_peak_rss_mb",
    "child_peak_rss_mb",
    "total_peak_rss_mb",
    "correctness_status",
    "max_abs_p_diff",
    "max_abs_stat_diff",
    "workers",
    "effective_cpu_count",
    "affinity_count",
    "load_average_before",
    "load_average_after",
    "resource_isolation",
    "interpretation",
    "row_status",
    "env_json",
    "notes",
]


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    write_csv(path, rows, fields)


def parse_warm_min(payload: dict[str, Any]) -> str:
    try:
        vals = json.loads(str(payload.get("warm_times_s", "[]")))
        return as_float(min(float(x) for x in vals))
    except Exception:
        return ""


def scheduler_metadata() -> dict[str, Any]:
    env = os.environ
    sched = "none"
    allocated = ""
    if env.get("SLURM_JOB_ID"):
        sched = "slurm"
        allocated = env.get("SLURM_CPUS_ON_NODE") or env.get("SLURM_CPUS_PER_TASK") or env.get("SLURM_JOB_CPUS_PER_NODE", "")
    elif env.get("PBS_JOBID"):
        sched = "pbs"
        allocated = env.get("PBS_NP", "")
    elif env.get("LSB_JOBID"):
        sched = "lsf"
        allocated = env.get("LSB_DJOB_NUMPROC", "")
    return {
        "scheduler": sched,
        "allocated_cpu_count": allocated,
        "scheduler_environment": {k: env.get(k, "") for k in sorted(env) if k.startswith(("SLURM_", "PBS_", "LSB_"))},
    }


def proc_stat_snapshot() -> list[tuple[int, int]]:
    rows = []
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            parts = line.split()
            vals = [int(v) for v in parts[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            rows.append((idle, total))
    except Exception:
        return []
    return rows


def per_core_idle_estimate() -> dict[str, Any]:
    a = proc_stat_snapshot()
    time.sleep(0.5)
    b = proc_stat_snapshot()
    if not a or len(a) != len(b):
        return {"available": False}
    shares = []
    for (idle0, total0), (idle1, total1) in zip(a, b):
        denom = max(1, total1 - total0)
        shares.append((idle1 - idle0) / denom)
    return {
        "available": True,
        "mean_idle_fraction": sum(shares) / len(shares),
        "min_idle_fraction": min(shares),
        "cores_lt_50pct_idle": sum(1 for x in shares if x < 0.5),
        "cores_sampled": len(shares),
    }


def top_cpu_processes() -> str:
    return cmd_text(["bash", "-lc", "ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -n 16"])


def resource_isolation(meta: dict[str, Any]) -> str:
    sched = meta.get("scheduler", "none")
    if sched != "none":
        return f"scheduler_{sched}"
    return "shared_server"


def high_count_interpretation(count: int, isolation: str, clean_load: bool) -> str:
    if count < 64:
        return "main controlled count"
    if isolation == "shared_server" or not clean_load:
        return "shared-server evidence; may reflect contention"
    return "clean high-count evidence"


def availability_for_count(count: int, est_mem_mb: float = 0.0) -> tuple[str, str]:
    eff = effective_cpu_count()
    if count > eff:
        return "unavailable_cpu_affinity", f"requested_count={count} exceeds effective_cpu_count={eff}"
    try:
        import psutil

        avail_mb = psutil.virtual_memory().available / (1024**2)
        if est_mem_mb and est_mem_mb > 0.70 * avail_mb:
            return "memory_risk_skip", f"estimated_memory_mb={est_mem_mb:.1f} exceeds 70% of available_memory_mb={avail_mb:.1f}"
    except Exception:
        pass
    return "run", ""


def unavailable_kmeans_row(run_id: str, params: dict[str, Any], row_status: str, note: str, isolation: str, interp: str) -> dict[str, Any]:
    return {
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
        "correctness_status": "not_run",
        "numba_threads": params["thread_count"],
        "effective_cpu_count": effective_cpu_count(),
        "affinity_count": effective_cpu_count(),
        "load_average_before": load_average(),
        "load_average_after": "",
        "resource_isolation": isolation,
        "interpretation": interp,
        "row_status": row_status,
        "env_json": json.dumps({key: "" for key in ENV_KEYS}, sort_keys=True),
        "notes": note,
    }


def unavailable_perm_row(run_id: str, params: dict[str, Any], row_status: str, note: str, isolation: str, interp: str) -> dict[str, Any]:
    return {
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
        "correctness_status": "not_run",
        "workers": params["worker_count"],
        "effective_cpu_count": effective_cpu_count(),
        "affinity_count": effective_cpu_count(),
        "load_average_before": load_average(),
        "load_average_after": "",
        "resource_isolation": isolation,
        "interpretation": interp,
        "row_status": row_status,
        "env_json": json.dumps({key: "" for key in ENV_KEYS}, sort_keys=True),
        "notes": note,
    }


def make_kmeans_row(run_id: str, payload: dict[str, Any], params: dict[str, Any], isolation: str, interp: str) -> dict[str, Any]:
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
        "numba_threads": params["thread_count"],
        "effective_cpu_count": payload.get("effective_cpu_count", effective_cpu_count()),
        "affinity_count": effective_cpu_count(),
        "load_average_before": payload.get("load_avg_before", ""),
        "load_average_after": payload.get("load_avg_after", ""),
        "resource_isolation": isolation,
        "interpretation": interp,
        "row_status": "completed" if payload.get("correctness_status") in {"pass", "check"} else payload.get("status", "completed"),
    }
    row.update(payload)
    row["min_warm_time_s"] = parse_warm_min(payload)
    row["load_average_before"] = payload.get("load_avg_before", row["load_average_before"])
    row["load_average_after"] = payload.get("load_avg_after", row["load_average_after"])
    for key in ["cold_time_s", "median_warm_time_s", "p25_warm_time_s", "p75_warm_time_s", "peak_rss_mb", "final_inertia", "relative_inertia_diff_vs_reference"]:
        row[key] = as_float(row.get(key))
    return row


def make_perm_row(run_id: str, payload: dict[str, Any], params: dict[str, Any], isolation: str, interp: str) -> dict[str, Any]:
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
        "workers": params["worker_count"],
        "effective_cpu_count": payload.get("effective_cpu_count", effective_cpu_count()),
        "affinity_count": effective_cpu_count(),
        "load_average_before": payload.get("load_avg_before", ""),
        "load_average_after": payload.get("load_avg_after", ""),
        "resource_isolation": isolation,
        "interpretation": interp,
        "row_status": "completed" if payload.get("correctness_status") in {"pass", "check"} else payload.get("status", "completed"),
    }
    row.update(payload)
    row["parent_peak_rss_mb"] = row.get("peak_parent_rss_mb", "")
    row["child_peak_rss_mb"] = row.get("peak_child_rss_mb", "")
    row["min_warm_time_s"] = parse_warm_min(payload)
    row["load_average_before"] = payload.get("load_avg_before", row["load_average_before"])
    row["load_average_after"] = payload.get("load_avg_after", row["load_average_after"])
    for key in ["cold_time_s", "median_warm_time_s", "p25_warm_time_s", "p75_warm_time_s", "parent_peak_rss_mb", "child_peak_rss_mb", "total_peak_rss_mb", "max_abs_p_diff", "max_abs_stat_diff"]:
        row[key] = as_float(row.get(key))
    return row


def clean_load() -> bool:
    try:
        return float(load_average().split(",")[0]) <= 0.75 * max(1, effective_cpu_count())
    except Exception:
        return False


def plot_expanded(out_dir: Path, presentation_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    presentation_dir.mkdir(parents=True, exist_ok=True)
    kmeans = pd.read_csv(out_dir / "kmeans_parallelism_expanded.csv")
    perm = pd.read_csv(out_dir / "permutation_parallelism_expanded.csv")
    kplot = kmeans[kmeans["row_status"].eq("completed")].sort_values("thread_count")
    pplot = perm[perm["row_status"].eq("completed")].sort_values("worker_count")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    fig.patch.set_facecolor("#FBF7EF")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", alpha=0.25)
    colors = {"kmeans": "#3267B7", "perm": "#B4494B", "warn": "#17202A", "mem": "#6F7782"}

    ax = axes[0]
    x = kplot["thread_count"].astype(int).to_numpy()
    y = pd.to_numeric(kplot["median_warm_time_s"], errors="coerce").to_numpy(float)
    yerr = np.vstack([
        y - pd.to_numeric(kplot["p25_warm_time_s"], errors="coerce").to_numpy(float),
        pd.to_numeric(kplot["p75_warm_time_s"], errors="coerce").to_numpy(float) - y,
    ])
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=3, markersize=8, capsize=5, color=colors["kmeans"])
    if 128 in set(x) and str(kplot[kplot["thread_count"].eq(128)]["interpretation"].iloc[0]).startswith("shared-server"):
        yy = float(kplot[kplot["thread_count"].eq(128)]["median_warm_time_s"].iloc[0])
        ax.scatter([128], [yy], s=150, facecolors="none", edgecolors=colors["warn"], linewidth=2.2, zorder=5)
        ax.annotate("shared /\nnot isolated", (128, yy), textcoords="offset points", xytext=(-44, 18), fontsize=9, ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(COUNTS, [str(v) for v in COUNTS])
    ax.set_xlabel("Numba threads")
    ax.set_ylabel("median warm runtime (s)")
    ax.set_title("k-means runtime vs threads", weight="bold")

    ax = axes[1]
    x = pplot["worker_count"].astype(int).to_numpy()
    y = pd.to_numeric(pplot["median_warm_time_s"], errors="coerce").to_numpy(float)
    yerr = np.vstack([
        y - pd.to_numeric(pplot["p25_warm_time_s"], errors="coerce").to_numpy(float),
        pd.to_numeric(pplot["p75_warm_time_s"], errors="coerce").to_numpy(float) - y,
    ])
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=3, markersize=8, capsize=5, color=colors["perm"])
    for row in pplot.itertuples():
        mem_gib = float(row.total_peak_rss_mb) / 1024.0
        ax.annotate(f"{mem_gib:.0f} GiB", (int(row.worker_count), float(row.median_warm_time_s)), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, color=colors["mem"])
    if 128 in set(x) and str(pplot[pplot["worker_count"].eq(128)]["interpretation"].iloc[0]).startswith("shared-server"):
        yy = float(pplot[pplot["worker_count"].eq(128)]["median_warm_time_s"].iloc[0])
        ax.scatter([128], [yy], s=150, facecolors="none", edgecolors=colors["warn"], linewidth=2.2, zorder=5)
        ax.annotate("shared /\nnot isolated", (128, yy), textcoords="offset points", xytext=(-44, 18), fontsize=9, ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(COUNTS, [str(v) for v in COUNTS])
    ax.set_xlabel("process workers")
    ax.set_ylabel("median warm runtime (s)")
    ax.set_title("permutation runtime vs workers", weight="bold")
    ax.set_ylim(0, max(y) * 1.30)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.04, 0.08, 0.98, 0.96])
    fig.savefig(fig_dir / "server_cpu_parallelism_expanded.png", dpi=220)
    fig.savefig(fig_dir / "server_cpu_parallelism_expanded.svg", format="svg")
    plt.close(fig)


def write_summary_and_readme(out_dir: Path) -> None:
    import pandas as pd

    kmeans = pd.read_csv(out_dir / "kmeans_parallelism_expanded.csv")
    perm = pd.read_csv(out_dir / "permutation_parallelism_expanded.csv")
    rows = []
    for df, count_col in [(kmeans, "thread_count"), (perm, "worker_count")]:
        for row in df.to_dict("records"):
            rows.append(
                {
                    "workload": row["workload"],
                    "method": row["method"],
                    "parallelism_count": row[count_col],
                    "median_warm_time_s": row.get("median_warm_time_s", ""),
                    "min_warm_time_s": row.get("min_warm_time_s", ""),
                    "peak_memory_mb": row.get("total_peak_rss_mb") or row.get("peak_rss_mb"),
                    "correctness_status": row.get("correctness_status", ""),
                    "row_status": row.get("row_status", ""),
                    "resource_isolation": row.get("resource_isolation", ""),
                    "interpretation": row.get("interpretation", ""),
                }
            )
    write_csv(out_dir / "parallelism_expanded_summary.csv", rows, list(rows[0]))

    env = json.loads((out_dir / "environment.json").read_text(encoding="utf-8"))
    isolation = resource_isolation(env.get("scheduler_metadata", {}))
    k_after16 = kmeans[kmeans["thread_count"].isin([16, 64, 128])][["thread_count", "median_warm_time_s", "row_status"]].to_dict("records")
    p_after16 = perm[perm["worker_count"].isin([16, 64, 128])][["worker_count", "median_warm_time_s", "total_peak_rss_mb", "row_status"]].to_dict("records")
    lines = [
        "# Linux server CPU expanded parallelism",
        "",
        f"Generated/updated: {timestamp()}",
        "",
        "## Provenance",
        "- Run tier: Linux server CPU, not MacBook.",
        f"- Hostname: `{env.get('hostname', '')}`.",
        f"- Effective CPU count from affinity: {env.get('effective_cpu_count', '')}.",
        f"- Scheduler: `{env.get('scheduler_metadata', {}).get('scheduler', 'none')}`; allocated CPUs: `{env.get('scheduler_metadata', {}).get('allocated_cpu_count', '')}`.",
        f"- Resource isolation: `{isolation}`.",
        "",
        "## Availability",
        f"- 64 available under affinity: `{effective_cpu_count() >= 64}`.",
        f"- 128 available under affinity: `{effective_cpu_count() >= 128}`.",
        "- No exclusive scheduler allocation was detected, so 64/128 are marked as shared-server evidence.",
        "",
        "## Workloads",
        "- k-means: `n=1,000,000`, `d=64`, `K=20`, `max_iter=20`, fixed data/init/stopping, Numba parallel method.",
        "- permutation: `n=5,000`, `p=10,000`, `R=1,000`, `batch_R=256`, deterministic same-stream process-pool method.",
        "",
        "## Results After 16",
        f"- k-means 16/64/128 rows: `{k_after16}`.",
        f"- permutation 16/64/128 rows: `{p_after16}`.",
        "- Permutation memory grows with worker count; see `total_peak_rss_mb`.",
        "",
        "## Safest Slide Interpretation",
        "- Because this was a shared server run without exclusive allocation, 128-worker/thread results are shared-server evidence, not clean evidence.",
        "- High worker counts on a shared server are hard to interpret without CPU affinity and load checks.",
        "- Do not claim that 128 is intrinsically bad for the algorithm; the point is to measure parallelism under explicit resource constraints.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"parallelism_expanded_{time.strftime('%Y%m%d_%H%M%S')}"
    env = environment_report()
    env["scheduler_metadata"] = scheduler_metadata()
    env["top_cpu_processes"] = top_cpu_processes()
    env["per_core_cpu_idle_estimate"] = per_core_idle_estimate()
    env["resource_isolation"] = resource_isolation(env["scheduler_metadata"])
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    isolation = str(env["resource_isolation"])
    clean = clean_load()
    kmeans_path = out_dir / "kmeans_parallelism_expanded.csv"
    perm_path = out_dir / "permutation_parallelism_expanded.csv"
    for path in [kmeans_path, perm_path]:
        if path.exists() and not args.resume:
            path.unlink()

    kmeans_shape = {"method": "numba_cpu_parallel", "n": 1_000_000, "d": 64, "k": 20, "max_iter": 20, "seed": 0, "init_seed": 123, "repeat": args.repeat}
    for count in COUNTS:
        params = {**kmeans_shape, "thread_count": count}
        interp = high_count_interpretation(count, isolation, clean)
        status, note = availability_for_count(count, est_mem_mb=2500.0)
        if status != "run":
            row = unavailable_kmeans_row(run_id, params, status, note, isolation, interp)
        else:
            print(f"[kmeans expanded] threads={count}", flush=True)
            payload = run_child("kmeans", params, "numba", count, timeout_s=args.timeout_s)
            row = make_kmeans_row(run_id, payload, params, isolation, interp)
        append_csv(kmeans_path, row, KMEANS_FIELDS)

    # Based on the targeted run, 128 workers is about 53 GiB plus parent; still check available memory.
    perm_shape = {"method": "process_pool_same_stream", "n": 5_000, "p": 10_000, "R": 1_000, "batch_R": 256, "seed": 0, "repeat": args.repeat}
    for count in COUNTS:
        params = {**perm_shape, "worker_count": count}
        interp = high_count_interpretation(count, isolation, clean)
        est_mem_mb = 900.0 + 420.0 * count
        status, note = availability_for_count(count, est_mem_mb=est_mem_mb)
        if status != "run":
            row = unavailable_perm_row(run_id, params, status, note, isolation, interp)
        else:
            print(f"[permutation expanded] workers={count}", flush=True)
            payload = run_child("permutation", params, "workers", count, timeout_s=args.timeout_s)
            row = make_perm_row(run_id, payload, params, isolation, interp)
        append_csv(perm_path, row, PERM_FIELDS)

    write_summary_and_readme(out_dir)
    plot_expanded(out_dir, args.presentation_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    run = sub.add_parser("run")
    run.add_argument("--out-dir", type=Path, default=OUT_DIR)
    run.add_argument("--presentation-dir", type=Path, default=PRESENTATION_DIR)
    run.add_argument("--run-id", default="")
    run.add_argument("--repeat", type=int, default=5)
    run.add_argument("--timeout-s", type=int, default=3600)
    run.add_argument("--resume", action="store_true")
    plot = sub.add_parser("plot")
    plot.add_argument("--out-dir", type=Path, default=OUT_DIR)
    plot.add_argument("--presentation-dir", type=Path, default=PRESENTATION_DIR)
    args = parser.parse_args()
    if args.cmd == "run":
        run_suite(args)
    elif args.cmd == "plot":
        write_summary_and_readme(args.out_dir)
        plot_expanded(args.out_dir, args.presentation_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
