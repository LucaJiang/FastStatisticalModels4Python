"""Run MacBook Air k-means validation and long evidence scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

DEFAULT_ROOT = Path("experiments/results/macbook_air_validation")
os.environ.setdefault("MPLCONFIGDIR", str((DEFAULT_ROOT / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.env_report import build_report
from experiments.common.scenario_schema import COMMON_COLUMNS, ENVIRONMENT_TIER
from experiments.kmeans.data_generation import choose_initial_centroids, make_gaussian_mixture
from experiments.kmeans.kmeans_numpy import kmeans_numpy_naive, kmeans_numpy_smart
from experiments.kmeans.validate_kmeans import summarize_kmeans_result

try:
    from experiments.kmeans.kmeans_numba import kmeans_numba, warmup as numba_warmup
except Exception:
    kmeans_numba = None
    numba_warmup = None


LONG_TIER = "macbook_air_long"


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1024**2)
    except Exception:
        return float("nan")


def _time_call(fn: Callable[[], tuple[np.ndarray, np.ndarray, float, int]], repeat: int):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    cold = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    times: list[float] = []
    last = result
    for _ in range(repeat):
        tracemalloc.start()
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
        tracemalloc.stop()
    return cold, float(statistics.median(times or [cold])), float(min(times or [cold])), float(max(times or [cold])), peak / 1024**2, _rss_mb(), last


def _scenario_id(n: int, d: int, k: int, sep: float, imbalance: str, outlier: float, seed: int, tag: str = "grid") -> str:
    return f"{tag}_N{n}_d{d}_K{k}_sep{sep:g}_imb{imbalance}_out{outlier:g}_seed{seed}"


def _rows_for_mode(mode: str) -> list[tuple[int, int, int, float, str, float, int, str]]:
    if mode == "long":
        rows: list[tuple[int, int, int, float, str, float, int, str]] = []
        for n in [1000, 5000, 10000, 50000, 100000]:
            seeds = range(10) if n <= 10000 else range(5)
            for d in [2, 10, 50, 100]:
                for k in [3, 5, 20, 50]:
                    for sep in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
                        for imb in ["balanced", "90_10"]:
                            for out in [0.0, 0.01, 0.05]:
                                for seed in seeds:
                                    rows.append((n, d, k, sep, imb, out, seed, "stat"))
        for n in [1000, 5000, 10000, 50000, 100000, 200000]:
            for seed in range(5):
                rows.append((n, 100, 50, 2.0, "balanced", 0.0, seed, "shape_K50_d100"))
        return rows

    if mode == "full":
        return [
            (n, d, k, sep, imb, out, seed, "full")
            for n in [1000, 10000, 50000]
            for d in [2, 10, 50]
            for k in [3, 5]
            for sep in [0.5, 1.0, 2.0, 4.0]
            for imb in ["balanced", "90_10"]
            for out in [0.0, 0.01]
            for seed in [0, 1, 2, 3, 4]
        ]

    base = [
        (n, d, k, sep, imb, out, seed, "quick")
        for n in [1000, 10000]
        for d in [2, 10]
        for k in [3, 5]
        for sep in [0.5, 1.0, 2.0, 4.0]
        for imb in ["balanced", "90_10"]
        for out in [0.0, 0.01]
        for seed in [0, 1]
    ]
    base.extend((50000, 10, 5, sep, "balanced", 0.0, seed, "quick_50k") for sep in [1.0, 2.0] for seed in [0, 1])
    return base


def _csv_fields(rows: list[dict]) -> list[str]:
    extras = sorted({k for row in rows for k in row.keys()} - set(COMMON_COLUMNS))
    return COMMON_COLUMNS + [x for x in extras if x not in COMMON_COLUMNS]


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _csv_fields(rows)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as f:
            existing = csv.reader(f)
            fields = next(existing)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = _csv_fields(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _completed_run_ids(path: Path) -> set[str]:
    return {row["run_id"] for row in _read_rows(path) if row.get("run_id")}


def _write_progress(root: Path, mode: str, completed: int, skipped: int, failed: int, total_scenarios: int, started_at: str) -> None:
    payload = {
        "mode": mode,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_rows": completed,
        "skipped_rows": skipped,
        "failed_rows": failed,
        "total_scenarios": total_scenarios,
    }
    (root / "kmeans_progress.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _reference_safe(n: int, d: int, k: int) -> bool:
    return n <= 10000 and (n * d * k) <= 5_000_000


def _skip_row(
    *,
    timestamp: str,
    environment_tier: str,
    machine: str,
    sid: str,
    impl_name: str,
    seed: int,
    n: int,
    d: int,
    k: int,
    sep: float,
    imb: str,
    outlier: float,
    reason: str,
) -> dict:
    return {
        "run_id": f"{timestamp}_{sid}_{impl_name}",
        "timestamp": timestamp,
        "environment_tier": environment_tier,
        "machine_name": machine,
        "workload": "kmeans",
        "implementation": impl_name,
        "scenario_id": sid,
        "seed": seed,
        "n": n,
        "p": "NA",
        "d": d,
        "k": k,
        "r": "NA",
        "batch_r": "NA",
        "workers": "NA",
        "dtype": "float64",
        "cold_time_s": "NA",
        "warm_median_s": "NA",
        "warm_min_s": "NA",
        "warm_max_s": "NA",
        "compile_time_s": "NA",
        "transfer_h2d_s": "NA",
        "transfer_d2h_s": "NA",
        "peak_python_mb": "NA",
        "peak_rss_mb": "NA",
        "peak_child_rss_mb": "NA",
        "peak_gpu_mb": "NA",
        "correctness_status": "skipped_memory_risk",
        "statistical_metric_name": "ari_true",
        "statistical_metric_value": "NA",
        "notes": reason,
        "separation": sep,
        "imbalance": imb,
        "outlier_fraction": outlier,
        "ari_true": "NA",
        "ari_vs_reference": "NA",
        "inertia": "NA",
        "inertia_abs_diff": "NA",
        "inertia_rel_diff": "NA",
        "n_iter": "NA",
        "empty_clusters": "NA",
        "numerical_failure": "NA",
    }


def _plot_ari_heatmap(rows: list[dict], fig_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty or "ari_true" not in df:
        return
    ref = df[(df["implementation"] == "reference") & (df["correctness_status"] == "pass")].copy()
    if ref.empty:
        ref = df[(df["implementation"] == "numpy_matmul") & (df["correctness_status"] == "pass")].copy()
    if ref.empty:
        return
    ref["separation"] = ref["separation"].astype(float)
    ref["d"] = ref["d"].astype(int)
    ref["ari_true"] = ref["ari_true"].astype(float)
    pivot = ref.pivot_table(index="separation", columns="d", values="ari_true", aggfunc="mean")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(pivot.values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    ax.set_xlabel("dimension d")
    ax.set_ylabel("cluster separation")
    ax.set_title("k-means recovery surface")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if math.isfinite(float(val)):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Adjusted Rand Index vs true labels")
    fig.tight_layout()
    fig.savefig(fig_dir / "kmeans_ari_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_equivalence(rows: list[dict], fig_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty or "inertia_rel_diff" not in df:
        return
    opt = df[(df["implementation"] != "reference") & (df["correctness_status"] == "pass")].copy()
    if opt.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for impl in opt["implementation"].drop_duplicates():
        values = pd.to_numeric(opt[opt["implementation"] == impl]["inertia_rel_diff"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if values.size:
            data.append(values)
            labels.append(impl)
    if not data:
        plt.close(fig)
        return
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.set_ylabel("relative inertia difference vs reference")
    ax.set_title("k-means implementation equivalence")
    ax.axhline(1e-8, color="#E66A2C", linestyle="--", linewidth=1, label="pass threshold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "kmeans_reference_equivalence.png", dpi=180)
    plt.close(fig)


def _plot_runtime(rows: list[dict], fig_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty or "warm_median_s" not in df:
        return
    df = df[df["correctness_status"] == "pass"].copy()
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["warm_median_s"] = pd.to_numeric(df["warm_median_s"], errors="coerce")
    df = df.dropna(subset=["n", "warm_median_s"])
    if df.empty:
        return
    pivot = df.groupby(["implementation", "n"])["warm_median_s"].median().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for impl, sub in pivot.groupby("implementation"):
        ax.plot(sub["n"], sub["warm_median_s"], marker="o", linewidth=2, label=impl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("median warm runtime (s)")
    ax.set_title("k-means runtime scaling")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "kmeans_runtime_scaling.png", dpi=180)
    plt.close(fig)


def _plot_memory(rows: list[dict], fig_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["peak_python_mb"] = pd.to_numeric(df["peak_python_mb"], errors="coerce")
    pass_df = df[df["correctness_status"] == "pass"].dropna(subset=["n", "peak_python_mb"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for impl, sub in pass_df.groupby("implementation"):
        mem = sub.groupby("n")["peak_python_mb"].median().reset_index()
        ax.plot(mem["n"], mem["peak_python_mb"], marker="o", linewidth=2, label=impl)
    skipped = df[df["correctness_status"].astype(str).str.startswith("skipped")]
    if not skipped.empty:
        ax.scatter(pd.to_numeric(skipped["n"], errors="coerce"), np.full(len(skipped), 1.0), marker="x", color="#C7507A", label="skipped memory risk")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Python peak allocation (MB)")
    ax.set_title("k-means memory and skip map")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "kmeans_memory_scaling.png", dpi=180)
    plt.close(fig)


def regenerate_plots(root: Path) -> None:
    rows = _read_rows(root / "kmeans_long_correctness.csv")
    if not rows:
        rows = _read_rows(root / "kmeans_correctness.csv")
    fig_dir = root / "figures"
    _plot_ari_heatmap(rows, fig_dir)
    _plot_equivalence(rows, fig_dir)
    _plot_runtime(rows, fig_dir)
    _plot_memory(rows, fig_dir)


def run(mode: str, repeat: int, max_iter: int, output_dir: Path, checkpoint_every: int) -> None:
    root = output_dir
    fig_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str((root / ".mplconfig").resolve())

    environment_tier = LONG_TIER if mode == "long" else ENVIRONMENT_TIER
    env = build_report(environment_tier)
    machine = env["machine_name"]
    timestamp = datetime.now(timezone.utc).isoformat()
    started_at = timestamp

    if numba_warmup is not None:
        numba_warmup()

    correctness_name = "kmeans_long_correctness.csv" if mode == "long" else "kmeans_correctness.csv"
    runtime_name = "kmeans_long_runtime.csv" if mode == "long" else "kmeans_quick_runtime.csv"
    correctness_path = root / correctness_name
    runtime_path = root / runtime_name
    completed_ids = _completed_run_ids(correctness_path)

    implementations: list[tuple[str, Callable | None]] = [
        ("reference", kmeans_numpy_naive),
        ("numpy_matmul", kmeans_numpy_smart),
    ]
    if kmeans_numba is not None:
        implementations.append(("numba", kmeans_numba))

    skipped_messages: list[str] = []
    buffer: list[dict] = []
    completed = 0
    skipped = 0
    failed = 0
    scenarios = _rows_for_mode(mode)

    for scenario_index, (n, d, k, sep, imb, outlier, seed, tag) in enumerate(scenarios, start=1):
        sid = _scenario_id(n, d, k, sep, imb, outlier, seed, tag)
        X = y = init = None
        reference_labels = None
        reference_inertia = None

        for impl_name, impl in implementations:
            run_id = f"{timestamp}_{sid}_{impl_name}"
            stable_id = f"{sid}_{impl_name}"
            if run_id in completed_ids or stable_id in completed_ids:
                continue

            if impl_name == "reference" and not _reference_safe(n, d, k):
                reason = f"broadcast reference skipped: N*K*d={n*k*d} exceeds local memory-risk threshold"
                row = _skip_row(timestamp=timestamp, environment_tier=environment_tier, machine=machine, sid=sid, impl_name=impl_name, seed=seed, n=n, d=d, k=k, sep=sep, imb=imb, outlier=outlier, reason=reason)
                row["run_id"] = stable_id
                buffer.append(row)
                skipped_messages.append(f"{sid}: {reason}")
                skipped += 1
                continue

            if impl_name == "numba" and impl is None:
                reason = "numba unavailable"
                row = _skip_row(timestamp=timestamp, environment_tier=environment_tier, machine=machine, sid=sid, impl_name=impl_name, seed=seed, n=n, d=d, k=k, sep=sep, imb=imb, outlier=outlier, reason=reason)
                row["correctness_status"] = "unavailable"
                row["run_id"] = stable_id
                buffer.append(row)
                skipped += 1
                continue

            try:
                if X is None or y is None or init is None:
                    X, y, _ = make_gaussian_mixture(n, d, k, sep, imb, outlier, seed)
                    init = choose_initial_centroids(X, k, seed)
                assert impl is not None
                fn = lambda impl=impl, X=X, init=init: impl(X, k, max_iter, init)
                cold, warm, warm_min, warm_max, py_mb, rss_mb, result = _time_call(fn, repeat)
                _centroids, labels, inertia, n_iter = result
                if impl_name == "reference":
                    reference_labels = labels
                    reference_inertia = inertia
                validation = summarize_kmeans_result(labels, y, inertia, k, reference_labels, reference_inertia)
                status = validation.correctness_status
                if status == "fail":
                    failed += 1
                row = {
                    "run_id": stable_id,
                    "timestamp": timestamp,
                    "environment_tier": environment_tier,
                    "machine_name": machine,
                    "workload": "kmeans",
                    "implementation": impl_name,
                    "scenario_id": sid,
                    "seed": seed,
                    "n": n,
                    "p": "NA",
                    "d": d,
                    "k": k,
                    "r": "NA",
                    "batch_r": "NA",
                    "workers": "NA",
                    "dtype": "float64",
                    "cold_time_s": cold,
                    "warm_median_s": warm,
                    "warm_min_s": warm_min,
                    "warm_max_s": warm_max,
                    "compile_time_s": "NA",
                    "transfer_h2d_s": "NA",
                    "transfer_d2h_s": "NA",
                    "peak_python_mb": py_mb,
                    "peak_rss_mb": rss_mb,
                    "peak_child_rss_mb": "NA",
                    "peak_gpu_mb": "NA",
                    "correctness_status": status,
                    "statistical_metric_name": "ari_true",
                    "statistical_metric_value": validation.ari_true,
                    "notes": f"MacBook {'long evidence' if mode == 'long' else 'validation'} tier.",
                    "separation": sep,
                    "imbalance": imb,
                    "outlier_fraction": outlier,
                    "ari_true": validation.ari_true,
                    "ari_vs_reference": validation.ari_vs_reference,
                    "inertia": inertia,
                    "inertia_abs_diff": validation.inertia_abs_diff,
                    "inertia_rel_diff": validation.inertia_rel_diff,
                    "n_iter": n_iter,
                    "empty_clusters": validation.empty_clusters,
                    "numerical_failure": validation.numerical_failure,
                }
                buffer.append(row)
                completed += 1
            except Exception as exc:
                failed += 1
                row = _skip_row(timestamp=timestamp, environment_tier=environment_tier, machine=machine, sid=sid, impl_name=impl_name, seed=seed, n=n, d=d, k=k, sep=sep, imb=imb, outlier=outlier, reason=repr(exc))
                row["run_id"] = stable_id
                row["correctness_status"] = "fail"
                buffer.append(row)

        if buffer and (scenario_index % checkpoint_every == 0):
            _append_rows(correctness_path, buffer)
            _append_rows(runtime_path, buffer)
            completed_ids.update(row["run_id"] for row in buffer)
            buffer.clear()
            _write_progress(root, mode, completed, skipped, failed, len(scenarios), started_at)

    if buffer:
        _append_rows(correctness_path, buffer)
        _append_rows(runtime_path, buffer)
        completed_ids.update(row["run_id"] for row in buffer)
        buffer.clear()

    (root / "kmeans_skipped_scenarios.txt").write_text("\n".join(skipped_messages) + ("\n" if skipped_messages else ""))
    _write_progress(root, mode, completed, skipped, failed, len(scenarios), started_at)
    regenerate_plots(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full", "long"], default="quick")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--regenerate-plots-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_ROOT
    if args.regenerate_plots_only:
        regenerate_plots(output_dir)
        return
    run(args.mode, args.repeat, args.max_iter, output_dir, max(1, args.checkpoint_every))


if __name__ == "__main__":
    main()
