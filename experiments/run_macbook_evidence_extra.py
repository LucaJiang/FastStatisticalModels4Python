"""Run targeted MacBook Air evidence sweeps beyond the validation grid."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.common.env_report import build_report
from experiments.kmeans.data_generation import choose_initial_centroids, make_gaussian_mixture
from experiments.kmeans.kmeans_numpy import kmeans_numpy_smart
from experiments.kmeans.validate_kmeans import summarize_kmeans_result
from experiments.permutation.data_generation import make_two_group_matrix
from experiments.permutation.permutation_jax_matrix import jax_available, jax_matrix_p_values
from experiments.permutation.permutation_numpy import matrix_p_values, matrix_p_values_batched
from experiments.permutation.validate_permutation import calibration_summary, power_summary, summarize_equivalence

try:
    from experiments.kmeans.kmeans_numba import kmeans_numba, warmup as numba_warmup
except Exception:
    kmeans_numba = None
    numba_warmup = None


DEFAULT_ROOT = Path("experiments/results/macbook_air_long/latest")


def _time_call(fn: Callable[[], Any], repeat: int = 1) -> tuple[Any, float, float, float, float, float]:
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
    warm_values = times or [cold]
    return last, cold, float(statistics.median(warm_values)), float(min(warm_values)), float(max(warm_values)), peak / 1024**2


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _completed(path: Path) -> set[str]:
    return {row["run_id"] for row in _read_rows(path) if row.get("run_id")}


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as f:
            fields = next(csv.reader(f))
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _base(
    *,
    run_id: str,
    workload: str,
    implementation: str,
    env: dict[str, Any],
    status: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment_tier": "macbook_air_evidence_extra",
        "machine_name": env.get("machine_name", "macbook_air"),
        "workload": workload,
        "implementation": implementation,
        "status": status,
        "notes": notes,
    }


def _run_kmeans_shape(root: Path, env: dict[str, Any], checkpoint_every: int, max_iter: int) -> None:
    out = root / "kmeans_shape_stress.csv"
    done = _completed(out)
    if numba_warmup is not None:
        numba_warmup()

    scenarios = [
        (n, d, k, sep, imb, outlier, seed)
        for n in [10_000, 50_000]
        for d in [50, 100]
        for k in [5, 20, 50]
        for sep in [0.5, 1.0, 2.0, 4.0]
        for imb in ["balanced", "90_10"]
        for outlier in [0.0, 0.05]
        for seed in [0, 1, 2]
    ]
    scenarios.extend(
        (100_000, 100, 50, sep, "balanced", 0.0, seed)
        for sep in [1.0, 2.0, 4.0]
        for seed in [0, 1]
    )

    rows: list[dict[str, Any]] = []
    for idx, (n, d, k, sep, imb, outlier, seed) in enumerate(scenarios, start=1):
        sid = f"shape_N{n}_d{d}_K{k}_sep{sep:g}_imb{imb}_out{outlier:g}_seed{seed}"
        X, y, _ = make_gaussian_mixture(n, d, k, sep, imb, outlier, seed)
        init = choose_initial_centroids(X, k, seed)

        oracle_labels = None
        oracle_inertia = None
        for impl_name, impl in [("numpy_matmul", kmeans_numpy_smart), ("numba", kmeans_numba)]:
            run_id = f"{sid}_{impl_name}"
            if run_id in done:
                continue
            if impl is None:
                row = _base(run_id=run_id, workload="kmeans_shape_stress", implementation=impl_name, env=env, status="unavailable", notes="Numba unavailable")
                rows.append(row)
                continue
            try:
                result, cold, warm, warm_min, warm_max, peak_mb = _time_call(lambda impl=impl: impl(X, k, max_iter, init), repeat=1)
                _, labels, inertia, n_iter = result
                if impl_name == "numpy_matmul":
                    oracle_labels = labels
                    oracle_inertia = inertia
                validation = summarize_kmeans_result(labels, y, inertia, k, oracle_labels, oracle_inertia, rel_tol=1e-8)
                row = _base(run_id=run_id, workload="kmeans_shape_stress", implementation=impl_name, env=env, status=validation.correctness_status)
                row.update(
                    {
                        "scenario_id": sid,
                        "n": n,
                        "d": d,
                        "k": k,
                        "separation": sep,
                        "imbalance": imb,
                        "outlier_fraction": outlier,
                        "seed": seed,
                        "max_iter": max_iter,
                        "cold_time_s": cold,
                        "warm_median_s": warm,
                        "warm_min_s": warm_min,
                        "warm_max_s": warm_max,
                        "peak_python_mb": peak_mb,
                        "ari_true": validation.ari_true,
                        "ari_vs_numpy_matmul": validation.ari_vs_reference,
                        "inertia": inertia,
                        "inertia_rel_diff_vs_numpy_matmul": validation.inertia_rel_diff,
                        "empty_clusters": validation.empty_clusters,
                        "n_iter": n_iter,
                    }
                )
                rows.append(row)
            except Exception as exc:
                row = _base(run_id=run_id, workload="kmeans_shape_stress", implementation=impl_name, env=env, status="fail", notes=repr(exc))
                row.update({"scenario_id": sid, "n": n, "d": d, "k": k, "separation": sep, "imbalance": imb, "outlier_fraction": outlier, "seed": seed})
                rows.append(row)

        if rows and idx % checkpoint_every == 0:
            _append_rows(out, rows)
            done.update(row["run_id"] for row in rows)
            rows.clear()
    if rows:
        _append_rows(out, rows)


def _run_permutation_calibration(root: Path, env: dict[str, Any], checkpoint_every: int) -> None:
    out = root / "permutation_calibration_extended.csv"
    done = _completed(out)
    rows: list[dict[str, Any]] = []
    scenarios = [
        (500, 1_000, 1_000, seed)
        for seed in range(50)
    ] + [
        (1_000, 1_000, 1_000, seed)
        for seed in range(50)
    ]

    for idx, (n, p, r, seed) in enumerate(scenarios, start=1):
        sid = f"calibration_ext_n{n}_p{p}_R{r}_seed{seed}"
        run_id = f"{sid}_numpy_matrix_batched"
        if run_id in done:
            continue
        try:
            x, labels, _ = make_two_group_matrix(n, p, delta=0.0, seed=seed + 70_000)
            (observed, p_values), cold, warm, warm_min, warm_max, peak_mb = _time_call(
                lambda: matrix_p_values_batched(x, labels, r=r, seed=seed + 71_000, batch_r=500),
                repeat=1,
            )
            row = _base(run_id=run_id, workload="permutation_calibration_extended", implementation="numpy_matrix_batched", env=env, status="pass")
            row.update(
                {
                    "scenario_id": sid,
                    "n": n,
                    "p": p,
                    "r": r,
                    "seed": seed,
                    "cold_time_s": cold,
                    "warm_median_s": warm,
                    "warm_min_s": warm_min,
                    "warm_max_s": warm_max,
                    "peak_python_mb": peak_mb,
                    "observed_mean_abs_stat": float(np.mean(np.abs(observed))),
                    **calibration_summary(p_values),
                }
            )
            rows.append(row)
        except Exception as exc:
            row = _base(run_id=run_id, workload="permutation_calibration_extended", implementation="numpy_matrix_batched", env=env, status="fail", notes=repr(exc))
            row.update({"scenario_id": sid, "n": n, "p": p, "r": r, "seed": seed})
            rows.append(row)
        if rows and idx % checkpoint_every == 0:
            _append_rows(out, rows)
            done.update(row["run_id"] for row in rows)
            rows.clear()
    if rows:
        _append_rows(out, rows)


def _run_permutation_power(root: Path, env: dict[str, Any], checkpoint_every: int) -> None:
    out = root / "permutation_power_extended.csv"
    done = _completed(out)
    rows: list[dict[str, Any]] = []
    scenarios = [
        (delta, signal_fraction, seed)
        for delta in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        for signal_fraction in [0.01, 0.05, 0.1]
        for seed in range(8)
    ]
    n, p, r = 500, 1_000, 1_000
    for idx, (delta, signal_fraction, seed) in enumerate(scenarios, start=1):
        sid = f"power_ext_n{n}_p{p}_R{r}_delta{delta:g}_sf{signal_fraction:g}_seed{seed}"
        run_id = f"{sid}_numpy_matrix_batched"
        if run_id in done:
            continue
        try:
            x, labels, signal_mask = make_two_group_matrix(n, p, delta=delta, signal_fraction=signal_fraction, seed=seed + 80_000)
            (_, p_values), cold, warm, warm_min, warm_max, peak_mb = _time_call(
                lambda: matrix_p_values_batched(x, labels, r=r, seed=seed + 81_000, batch_r=500),
                repeat=1,
            )
            row = _base(run_id=run_id, workload="permutation_power_extended", implementation="numpy_matrix_batched", env=env, status="pass")
            row.update(
                {
                    "scenario_id": sid,
                    "n": n,
                    "p": p,
                    "r": r,
                    "delta": delta,
                    "signal_fraction": signal_fraction,
                    "seed": seed,
                    "cold_time_s": cold,
                    "warm_median_s": warm,
                    "warm_min_s": warm_min,
                    "warm_max_s": warm_max,
                    "peak_python_mb": peak_mb,
                    **power_summary(p_values, signal_mask),
                }
            )
            rows.append(row)
        except Exception as exc:
            row = _base(run_id=run_id, workload="permutation_power_extended", implementation="numpy_matrix_batched", env=env, status="fail", notes=repr(exc))
            row.update({"scenario_id": sid, "n": n, "p": p, "r": r, "delta": delta, "signal_fraction": signal_fraction, "seed": seed})
            rows.append(row)
        if rows and idx % checkpoint_every == 0:
            _append_rows(out, rows)
            done.update(row["run_id"] for row in rows)
            rows.clear()
    if rows:
        _append_rows(out, rows)


def _run_permutation_runtime(root: Path, env: dict[str, Any], checkpoint_every: int) -> None:
    out = root / "permutation_runtime_scaling_extended.csv"
    done = _completed(out)
    rows: list[dict[str, Any]] = []
    jax_ok, jax_note = jax_available()
    scenarios = [(500, p, r, seed) for p in [100, 1_000, 3_000] for r in [100, 500, 1_000, 2_000, 5_000] for seed in range(3)]

    for idx, (n, p, r, seed) in enumerate(scenarios, start=1):
        x, labels, _ = make_two_group_matrix(n, p, delta=0.0, seed=seed + 90_000)
        for impl_name in ["numpy_matrix", "numpy_matrix_batched", "jax_matrix_cpu"]:
            sid = f"runtime_ext_n{n}_p{p}_R{r}_seed{seed}"
            run_id = f"{sid}_{impl_name}"
            if run_id in done:
                continue
            if impl_name == "jax_matrix_cpu" and (not jax_ok or p > 1_000 or r > 1_000):
                status = "unavailable" if not jax_ok else "skipped_memory_risk"
                note = jax_note if not jax_ok else "JAX CPU runtime sweep capped at p<=1000 and R<=1000 on MacBook"
                row = _base(run_id=run_id, workload="permutation_runtime_scaling_extended", implementation=impl_name, env=env, status=status, notes=note)
                row.update({"scenario_id": sid, "n": n, "p": p, "r": r, "seed": seed})
                rows.append(row)
                continue
            if impl_name == "numpy_matrix" and (p * r > 15_000_000):
                row = _base(run_id=run_id, workload="permutation_runtime_scaling_extended", implementation=impl_name, env=env, status="skipped_memory_risk", notes="full R x p null matrix exceeds local evidence threshold")
                row.update({"scenario_id": sid, "n": n, "p": p, "r": r, "seed": seed})
                rows.append(row)
                continue
            try:
                if impl_name == "numpy_matrix":
                    (_, null, p_values), cold, warm, warm_min, warm_max, peak_mb = _time_call(lambda: matrix_p_values(x, labels, r=r, seed=seed + 91_000), repeat=1)
                    metric = float(np.mean(p_values))
                    null_shape = "full"
                elif impl_name == "jax_matrix_cpu":
                    (_, null, p_values), cold, warm, warm_min, warm_max, peak_mb = _time_call(lambda: jax_matrix_p_values(x, labels, r=r, seed=seed + 91_000), repeat=1)
                    metric = float(np.mean(p_values))
                    null_shape = "full_jax_cpu"
                else:
                    (_, p_values), cold, warm, warm_min, warm_max, peak_mb = _time_call(lambda: matrix_p_values_batched(x, labels, r=r, seed=seed + 91_000, batch_r=min(500, r)), repeat=1)
                    metric = float(np.mean(p_values))
                    null_shape = "batched"
                row = _base(run_id=run_id, workload="permutation_runtime_scaling_extended", implementation=impl_name, env=env, status="pass")
                row.update(
                    {
                        "scenario_id": sid,
                        "n": n,
                        "p": p,
                        "r": r,
                        "seed": seed,
                        "cold_time_s": cold,
                        "warm_median_s": warm,
                        "warm_min_s": warm_min,
                        "warm_max_s": warm_max,
                        "peak_python_mb": peak_mb,
                        "mean_p": metric,
                        "null_shape": null_shape,
                    }
                )
                rows.append(row)
            except Exception as exc:
                row = _base(run_id=run_id, workload="permutation_runtime_scaling_extended", implementation=impl_name, env=env, status="fail", notes=repr(exc))
                row.update({"scenario_id": sid, "n": n, "p": p, "r": r, "seed": seed})
                rows.append(row)
        if rows and idx % checkpoint_every == 0:
            _append_rows(out, rows)
            done.update(row["run_id"] for row in rows)
            rows.clear()
    if rows:
        _append_rows(out, rows)


def write_extra_summary(root: Path) -> None:
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    for name in [
        "kmeans_shape_stress.csv",
        "permutation_calibration_extended.csv",
        "permutation_power_extended.csv",
        "permutation_runtime_scaling_extended.csv",
    ]:
        rows = _read_rows(root / name)
        status_counts: dict[str, int] = {}
        for row in rows:
            status = row.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        summary["files"][name] = {"rows": len(rows), "status_counts": status_counts}
    (root / "extra_evidence_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--skip-kmeans", action="store_true")
    parser.add_argument("--skip-permutation", action="store_true")
    args = parser.parse_args()

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    env = build_report("macbook_air_evidence_extra", machine_name="macbook_air")
    (root / "extra_env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")

    if not args.skip_kmeans:
        _run_kmeans_shape(root, env, max(1, args.checkpoint_every), args.max_iter)
    if not args.skip_permutation:
        _run_permutation_calibration(root, env, max(1, args.checkpoint_every))
        _run_permutation_power(root, env, max(1, args.checkpoint_every))
        _run_permutation_runtime(root, env, max(1, args.checkpoint_every))
    write_extra_summary(root)
    print(root)


if __name__ == "__main__":
    main()
