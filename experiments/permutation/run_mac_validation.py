"""Run MacBook Air validation and long evidence sweeps for permutation tests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_ROOT = Path("experiments/results/macbook_air_validation")
os.environ.setdefault("MPLCONFIGDIR", str((DEFAULT_ROOT / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.env_report import build_report
from experiments.common.scenario_schema import COMMON_COLUMNS, ENVIRONMENT_TIER

from .data_generation import make_two_group_matrix
from .permutation_jax_matrix import jax_available, jax_matrix_p_values
from .permutation_numpy import matrix_p_values, matrix_p_values_batched
from .permutation_reference import permutation_p_values
from .validate_permutation import calibration_summary, power_summary, summarize_equivalence


LONG_TIER = "macbook_air_long"


def _time_call(fn: Callable[[], Any]) -> tuple[Any, float, int]:
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, int(peak)


def _csv_fields(rows: list[dict[str, Any]]) -> list[str]:
    extras = sorted({key for row in rows for key in row} - set(COMMON_COLUMNS))
    return [column for column in COMMON_COLUMNS if any(column in row for row in rows)] + extras


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = _csv_fields(rows)
    if exists:
        with path.open(newline="") as f:
            fields = next(csv.reader(f))
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _completed(path: Path) -> set[str]:
    return {row["run_id"] for row in _read_rows(path) if row.get("run_id")}


def _base_row(
    *,
    run_id: str,
    environment_tier: str,
    machine_name: str,
    workload: str,
    implementation: str,
    n: int,
    p: int,
    r: int,
    delta: float,
    signal_fraction: float,
    seed: int,
    status: str,
    runtime_s: float | None = None,
    peak_bytes: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment_tier": environment_tier,
        "machine_name": machine_name,
        "workload": workload,
        "implementation": implementation,
        "scenario_id": run_id.rsplit("_", 1)[0],
        "seed": seed,
        "n": n,
        "p": p,
        "d": "NA",
        "k": "NA",
        "r": r,
        "batch_r": "NA",
        "workers": "NA",
        "dtype": "float64",
        "cold_time_s": runtime_s if runtime_s is not None else "NA",
        "warm_median_s": runtime_s if runtime_s is not None else "NA",
        "peak_python_mb": peak_bytes / 1024**2 if peak_bytes is not None else "NA",
        "correctness_status": status,
        "statistical_metric_name": "status",
        "statistical_metric_value": status,
        "status": status,
        "notes": notes,
        "delta": delta,
        "signal_fraction": signal_fraction,
        "max_abs_stat_diff": "NA",
        "max_abs_p_diff": "NA",
        "mean_abs_p_diff": "NA",
        "reference_runtime_s": "NA",
        "speedup_vs_reference": "NA",
        "observed_max_abs_diff": "NA",
        "mean_p": "NA",
        "median_p": "NA",
        "prop_below_alpha": "NA",
        "ks_uniform": "NA",
        "observed_mean_abs_stat": "NA",
        "signal_power": "NA",
        "null_false_positive_rate": "NA",
        "discoveries": "NA",
    }
    return row


def _equivalence_scenarios(mode: str) -> list[tuple[int, int, int, int]]:
    if mode == "long":
        rows: list[tuple[int, int, int, int]] = []
        for n in [100, 500, 1000, 5000]:
            for p in [10, 100, 1000, 5000]:
                for r in [100, 1000, 5000]:
                    max_seed = 10 if n <= 1000 and p <= 1000 and r <= 1000 else 5
                    for seed in range(max_seed):
                        rows.append((n, p, r, seed))
        return rows
    if mode == "full":
        return [(n, p, r, seed) for n in [100, 500, 1000] for p in [10, 100, 1000] for r in [100, 1000, 2000] for seed in range(5)]
    scenarios = [(n, p, r, seed) for n in [100, 500, 1000] for p in [10, 100] for r in [100, 1000] for seed in range(3)]
    scenarios.extend((500, 1000, 100, seed) for seed in range(2))
    return scenarios


def _reference_safe(n: int, p: int, r: int) -> bool:
    return n <= 1000 and p <= 1000 and r <= 1000


def _matrix_safe(n: int, p: int, r: int) -> bool:
    return (n * r + r * p) <= 35_000_000


def _jax_safe(n: int, p: int, r: int) -> bool:
    return n <= 1000 and p <= 1000 and r <= 1000


def _write_progress(root: Path, mode: str, completed: int, skipped: int, failed: int, started_at: str) -> None:
    payload = {
        "mode": mode,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_rows": completed,
        "skipped_rows": skipped,
        "failed_rows": failed,
    }
    (root / "permutation_progress.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_equivalence(mode: str, root: Path, environment_tier: str, machine_name: str, checkpoint_every: int) -> tuple[int, int, int]:
    out_path = root / ("permutation_long_equivalence.csv" if mode == "long" else "permutation_equivalence.csv")
    runtime_path = root / ("permutation_long_runtime.csv" if mode == "long" else "permutation_quick_runtime.csv")
    done = _completed(out_path)
    jax_ok, jax_note = jax_available()
    completed = skipped = failed = 0
    buffer: list[dict[str, Any]] = []

    for idx, (n, p, r, seed) in enumerate(_equivalence_scenarios(mode), start=1):
        sid = f"equiv_n{n}_p{p}_R{r}_seed{seed}"
        x, labels, _ = make_two_group_matrix(n, p, delta=0.0, seed=seed)

        ref_obs = ref_null = ref_p = None
        ref_time = None
        if _reference_safe(n, p, r):
            run_id = f"{sid}_reference_loop"
            if run_id not in done:
                try:
                    (ref_obs, ref_null, ref_p), ref_time, ref_peak = _time_call(lambda: permutation_p_values(x, labels, r=r, seed=seed + 1000))
                    row = _base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_equivalence", implementation="reference_loop", n=n, p=p, r=r, delta=0.0, signal_fraction=0.0, seed=seed, status="pass", runtime_s=ref_time, peak_bytes=ref_peak)
                    buffer.append(row)
                    completed += 1
                except Exception as exc:
                    buffer.append(_base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_equivalence", implementation="reference_loop", n=n, p=p, r=r, delta=0.0, signal_fraction=0.0, seed=seed, status="fail", notes=repr(exc)))
                    failed += 1
            else:
                # Existing completed rows are enough for resume; reference arrays are recomputed only when needed below.
                pass
            if ref_null is None:
                try:
                    ref_obs, ref_null, ref_p = permutation_p_values(x, labels, r=r, seed=seed + 1000)
                except Exception:
                    ref_obs = ref_null = ref_p = None

        for impl_name in ["numpy_matrix", "numpy_matrix_batched", "jax_matrix_cpu"]:
            run_id = f"{sid}_{impl_name}"
            if run_id in done:
                continue
            if impl_name == "jax_matrix_cpu" and (not jax_ok or not _jax_safe(n, p, r)):
                status = "unavailable" if not jax_ok else "skipped_memory_risk"
                note = jax_note if not jax_ok else "JAX CPU limited to safe local equivalence rows"
                buffer.append(_base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_equivalence", implementation=impl_name, n=n, p=p, r=r, delta=0.0, signal_fraction=0.0, seed=seed, status=status, notes=note))
                skipped += 1
                continue
            if impl_name == "numpy_matrix" and not _matrix_safe(n, p, r):
                buffer.append(_base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_equivalence", implementation=impl_name, n=n, p=p, r=r, delta=0.0, signal_fraction=0.0, seed=seed, status="skipped_memory_risk", notes="full R x p null matrix exceeds local threshold"))
                skipped += 1
                continue
            try:
                if impl_name == "numpy_matrix":
                    (obs, null, pvals), elapsed, peak = _time_call(lambda: matrix_p_values(x, labels, r=r, seed=seed + 1000))
                    if ref_null is not None and ref_p is not None:
                        summary = summarize_equivalence(ref_null, ref_p, null, pvals)
                        status = summary.status
                    else:
                        summary = None
                        status = "pass"
                elif impl_name == "jax_matrix_cpu":
                    (obs, null, pvals), elapsed, peak = _time_call(lambda: jax_matrix_p_values(x, labels, r=r, seed=seed + 1000))
                    if ref_null is not None and ref_p is not None:
                        summary = summarize_equivalence(ref_null, ref_p, null, pvals, abs_tol=1e-6)
                        status = summary.status
                    else:
                        summary = None
                        status = "pass"
                else:
                    (obs, pvals), elapsed, peak = _time_call(lambda: matrix_p_values_batched(x, labels, r=r, seed=seed + 1000, batch_r=min(500, r)))
                    summary = None
                    status = "pass"
                if status == "fail":
                    failed += 1
                row = _base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_equivalence", implementation=impl_name, n=n, p=p, r=r, delta=0.0, signal_fraction=0.0, seed=seed, status=status, runtime_s=elapsed, peak_bytes=peak, notes=jax_note if impl_name == "jax_matrix_cpu" else "")
                if summary is not None:
                    row.update(
                        {
                            "max_abs_stat_diff": summary.max_abs_stat_diff,
                            "max_abs_p_diff": summary.max_abs_p_diff,
                            "mean_abs_p_diff": summary.mean_abs_p_diff,
                            "reference_runtime_s": ref_time if ref_time is not None else "NA",
                            "speedup_vs_reference": ref_time / elapsed if ref_time and elapsed else "NA",
                            "observed_max_abs_diff": float(np.max(np.abs(ref_obs - obs))) if ref_obs is not None else "NA",
                        }
                    )
                buffer.append(row)
                completed += 1
            except Exception as exc:
                buffer.append(_base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_equivalence", implementation=impl_name, n=n, p=p, r=r, delta=0.0, signal_fraction=0.0, seed=seed, status="fail", notes=repr(exc)))
                failed += 1

        if buffer and idx % checkpoint_every == 0:
            _append_rows(out_path, buffer)
            _append_rows(runtime_path, buffer)
            done.update(row["run_id"] for row in buffer)
            buffer.clear()
            _write_progress(root, mode, completed, skipped, failed, datetime.now(timezone.utc).isoformat())

    if buffer:
        _append_rows(out_path, buffer)
        _append_rows(runtime_path, buffer)
    return completed, skipped, failed


def run_calibration(mode: str, root: Path, environment_tier: str, machine_name: str, checkpoint_every: int) -> tuple[int, int, int]:
    out_path = root / ("permutation_long_calibration.csv" if mode == "long" else "permutation_calibration.csv")
    runtime_path = root / ("permutation_long_runtime.csv" if mode == "long" else "permutation_quick_runtime.csv")
    done = _completed(out_path)
    rows: list[dict[str, Any]] = []
    completed = skipped = failed = 0
    reps = 100 if mode == "long" else (20 if mode == "full" else 12)
    shapes = [(500, 200, 500), (1000, 1000, 1000)] if mode == "long" else [(500, 200, 500)]
    calibration_jobs = [(n, p, r, seed) for (n, p, r) in shapes for seed in range(reps)]
    for idx, (n, p, r, seed) in enumerate(calibration_jobs, start=1):
        sid = f"calibration_n{n}_p{p}_R{r}_seed{seed}"
        run_id = f"{sid}_numpy_matrix_batched"
        if run_id in done:
            continue
        try:
            x, labels, _ = make_two_group_matrix(n, p, delta=0.0, seed=seed + 3000)
            (observed, p_values), elapsed, peak = _time_call(lambda: matrix_p_values_batched(x, labels, r=r, seed=seed + 4000, batch_r=min(500, r)))
            summary = calibration_summary(p_values)
            row = _base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_null_calibration", implementation="numpy_matrix_batched", n=n, p=p, r=r, delta=0.0, signal_fraction=0.0, seed=seed, status="pass", runtime_s=elapsed, peak_bytes=peak)
            row.update(summary)
            row["observed_mean_abs_stat"] = float(np.mean(np.abs(observed)))
            rows.append(row)
            completed += 1
        except Exception as exc:
            rows.append(_base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_null_calibration", implementation="numpy_matrix_batched", n=n, p=p, r=r, delta=0.0, signal_fraction=0.0, seed=seed, status="fail", notes=repr(exc)))
            failed += 1
        if rows and idx % checkpoint_every == 0:
            _append_rows(out_path, rows)
            _append_rows(runtime_path, rows)
            done.update(row["run_id"] for row in rows)
            rows.clear()
    if rows:
        _append_rows(out_path, rows)
        _append_rows(runtime_path, rows)
    return completed, skipped, failed


def run_power(mode: str, root: Path, environment_tier: str, machine_name: str, checkpoint_every: int) -> tuple[int, int, int]:
    out_path = root / ("permutation_long_power.csv" if mode == "long" else "permutation_power.csv")
    runtime_path = root / ("permutation_long_runtime.csv" if mode == "long" else "permutation_quick_runtime.csv")
    done = _completed(out_path)
    rows: list[dict[str, Any]] = []
    completed = skipped = failed = 0
    deltas = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0] if mode == "long" else [0.2, 0.5, 1.0]
    signal_fracs = [0.01, 0.05] if mode == "long" else [0.05]
    seeds = range(10 if mode == "long" else (5 if mode == "full" else 3))
    for idx, (delta, signal_fraction, seed) in enumerate([(d, sf, s) for d in deltas for sf in signal_fracs for s in seeds], start=1):
        n, p, r = (1000, 1000, 1000) if mode == "long" else (500, 500, 500)
        sid = f"power_n{n}_p{p}_R{r}_delta{delta:g}_sf{signal_fraction:g}_seed{seed}"
        run_id = f"{sid}_numpy_matrix_batched"
        if run_id in done:
            continue
        try:
            x, labels, signal_mask = make_two_group_matrix(n, p, delta=delta, signal_fraction=signal_fraction, seed=seed + 5000)
            (_, p_values), elapsed, peak = _time_call(lambda: matrix_p_values_batched(x, labels, r=r, seed=seed + 6000, batch_r=min(500, r)))
            summary = power_summary(p_values, signal_mask)
            row = _base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_power_quick", implementation="numpy_matrix_batched", n=n, p=p, r=r, delta=delta, signal_fraction=signal_fraction, seed=seed, status="pass", runtime_s=elapsed, peak_bytes=peak)
            row.update(summary)
            rows.append(row)
            completed += 1
        except Exception as exc:
            rows.append(_base_row(run_id=run_id, environment_tier=environment_tier, machine_name=machine_name, workload="permutation_power_quick", implementation="numpy_matrix_batched", n=n, p=p, r=r, delta=delta, signal_fraction=signal_fraction, seed=seed, status="fail", notes=repr(exc)))
            failed += 1
        if rows and idx % checkpoint_every == 0:
            _append_rows(out_path, rows)
            _append_rows(runtime_path, rows)
            done.update(row["run_id"] for row in rows)
            rows.clear()
    if rows:
        _append_rows(out_path, rows)
        _append_rows(runtime_path, rows)
    return completed, skipped, failed


def _plot_calibration(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    vals = [float(row["prop_below_alpha"]) for row in rows if row.get("status") == "pass" and row.get("prop_below_alpha") not in (None, "")]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=20, color="#1f77b4", edgecolor="white")
    ax.axvline(0.05, color="#c7507a", linestyle="--", linewidth=2, label="nominal alpha")
    ax.axvline(statistics.fmean(vals), color="#222222", linewidth=2, label=f"mean {statistics.fmean(vals):.3f}")
    ax.set_title("Permutation null calibration across replicates")
    ax.set_xlabel("proportion p <= 0.05")
    ax.set_ylabel("replicate count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "permutation_null_calibration.png", dpi=180)
    plt.close(fig)


def _plot_power(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return
    df = df[df["status"] == "pass"].copy()
    if df.empty:
        return
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df["signal_fraction"] = pd.to_numeric(df["signal_fraction"], errors="coerce")
    df["signal_power"] = pd.to_numeric(df["signal_power"], errors="coerce")
    df["null_false_positive_rate"] = pd.to_numeric(df["null_false_positive_rate"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    for sf, sub in df.groupby("signal_fraction"):
        agg = sub.groupby("delta")["signal_power"].mean().reset_index()
        ax.plot(agg["delta"], agg["signal_power"], marker="o", linewidth=2.5, label=f"signal fraction {sf:g}")
    fpr = df.groupby("delta")["null_false_positive_rate"].mean().reset_index()
    ax.plot(fpr["delta"], fpr["null_false_positive_rate"], marker="x", color="#dd9a22", linewidth=2, label="null FPR")
    ax.axhline(0.05, color="#3a3a3a", linewidth=1, linestyle=":")
    ax.set_ylim(0, 1.02)
    ax.set_title("Permutation power curve")
    ax.set_xlabel("effect size delta")
    ax.set_ylabel("proportion p <= 0.05")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "permutation_power_quick.png", dpi=180)
    plt.close(fig)


def _plot_runtime(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return
    df = df[df["status"].isin(["pass", "skipped_memory_risk"])].copy()
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df["warm_median_s"] = pd.to_numeric(df["warm_median_s"], errors="coerce")
    mat = df[(df["implementation"].isin(["numpy_matrix", "numpy_matrix_batched"])) & (df["status"] == "pass")]
    if not mat.empty:
        pivot = mat.pivot_table(index="r", columns="p", values="warm_median_s", aggfunc="median")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(int(i)) for i in pivot.index])
        ax.set_xlabel("features p")
        ax.set_ylabel("permutations R")
        ax.set_title("Permutation runtime heatmap")
        fig.colorbar(im, ax=ax, label="median runtime (s)")
        fig.tight_layout()
        fig.savefig(fig_dir / "permutation_runtime_heatmap.png", dpi=180)
        plt.close(fig)


def _plot_equivalence(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty or "max_abs_p_diff" not in df:
        return
    df = df[df["status"] == "pass"].copy()
    df["max_abs_p_diff"] = pd.to_numeric(df["max_abs_p_diff"], errors="coerce")
    df = df.dropna(subset=["max_abs_p_diff"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [sub["max_abs_p_diff"].to_numpy() for _, sub in df.groupby("implementation")]
    labels = list(df.groupby("implementation").groups)
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.set_ylabel("max absolute p-value difference")
    ax.set_title("Permutation implementation equivalence")
    fig.tight_layout()
    fig.savefig(fig_dir / "permutation_equivalence.png", dpi=180)
    plt.close(fig)


def regenerate_plots(root: Path, mode: str) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    prefix = "_long" if mode == "long" else ""
    calibration = _read_rows(root / f"permutation{prefix}_calibration.csv")
    power = _read_rows(root / f"permutation{prefix}_power.csv")
    runtime = _read_rows(root / (f"permutation{prefix}_runtime.csv" if mode == "long" else "permutation_quick_runtime.csv"))
    equivalence = _read_rows(root / f"permutation{prefix}_equivalence.csv")
    _plot_calibration(calibration, fig_dir)
    _plot_power(power, fig_dir)
    _plot_runtime(runtime, fig_dir)
    _plot_equivalence(equivalence, fig_dir)


def run(mode: str, root: Path, checkpoint_every: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str((root / ".mplconfig").resolve())
    environment_tier = LONG_TIER if mode == "long" else ENVIRONMENT_TIER
    env = build_report(environment_tier, machine_name="macbook_air")
    machine_name = env["machine_name"]
    started = datetime.now(timezone.utc).isoformat()

    completed = skipped = failed = 0
    for fn in (run_equivalence, run_calibration, run_power):
        c, s, f = fn(mode, root, environment_tier, machine_name, checkpoint_every)
        completed += c
        skipped += s
        failed += f
        _write_progress(root, mode, completed, skipped, failed, started)
    regenerate_plots(root, mode)
    skipped_note = [
        f"mode={mode}",
        "Large full-matrix rows are marked skipped_memory_risk when R*n + R*p exceeds the local threshold.",
        f"JAX status: {jax_available()[1]}",
        f"Python: {str(env.get('python_version', 'unknown')).split('|')[0].strip()}",
    ]
    (root / "permutation_skipped_scenarios.txt").write_text("\n".join(skipped_note) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full", "long"], default="quick")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--regenerate-plots-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_ROOT
    if args.regenerate_plots_only:
        regenerate_plots(output_dir, args.mode)
        return
    run(args.mode, output_dir, max(1, args.checkpoint_every))


if __name__ == "__main__":
    main()
