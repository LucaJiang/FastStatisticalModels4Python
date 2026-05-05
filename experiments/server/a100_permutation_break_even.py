#!/usr/bin/env python3
"""CPU-vs-A100 break-even benchmark for matrix permutation tests."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments/results/linux_server_a100/permutation_break_even"
PRESENTATION_DIR = ROOT / "experiments/results/presentation_figures"
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from common.server_utils import timestamp
from server.a100_permutation_followup import (  # noqa: E402
    CPU_FIELDS,
    DECOMP_FIELDS,
    KERNEL_FIELDS,
    Shape,
    append_csv,
    as_float,
    estimates,
    run_cpu_baseline,
    run_decomposition,
    run_kernel_only,
    should_skip_for_memory,
)


CSV_FIELDS = [
    "run_id",
    "timestamp",
    "stage",
    "n",
    "p",
    "R",
    "batch_R",
    "n_batches",
    "seed",
    "dtype",
    "best_cpu_implementation",
    "cpu_end_to_end_time_s",
    "cpu_status",
    "cpu_timeout_status",
    "a100_end_to_end_time_s",
    "a100_streamed_reduction_time_s",
    "a100_status",
    "kernel_only_time_s",
    "kernel_only_label",
    "speedup_cpu_over_a100",
    "winner",
    "max_abs_p_diff",
    "max_abs_stat_diff",
    "estimated_device_memory_per_batch_gib",
    "estimated_x_gib",
    "notes",
]

CORRECTNESS_FIELDS = [
    "run_id",
    "timestamp",
    "stage",
    "implementation",
    "n",
    "p",
    "R",
    "batch_R",
    "seed",
    "dtype",
    "device",
    "correctness_status",
    "max_abs_p_diff",
    "max_abs_stat_diff",
    "notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(dict.fromkeys(fields + [k for row in rows for k in row]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def n_batches(shape: Shape) -> int:
    return int(math.ceil(shape.R / shape.batch_R))


def existing_cpu_lookup(cpu_csv: Path) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    if not cpu_csv.exists():
        return {}
    df = pd.read_csv(cpu_csv)
    lookup: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in df.to_dict("records"):
        key = (int(row["n"]), int(row["p"]), int(row["R"]), int(row["batch_R"]))
        lookup[key] = row
    return lookup


def existing_summary_rows(path: Path) -> tuple[list[dict[str, Any]], set[tuple[int, int, int, int]]]:
    if not path.exists():
        return [], set()
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    keys = {(int(row["n"]), int(row["p"]), int(row["R"]), int(row["batch_R"])) for row in rows}
    return rows, keys


def correctness_rows(stage: str, row: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "timestamp": timestamp(),
            "stage": stage,
            "implementation": row.get("implementation", ""),
            "n": row.get("n", ""),
            "p": row.get("p", ""),
            "R": row.get("R", ""),
            "batch_R": row.get("batch_R", ""),
            "seed": row.get("seed", ""),
            "dtype": row.get("dtype", ""),
            "device": row.get("device", ""),
            "correctness_status": row.get("correctness_status", ""),
            "max_abs_p_diff": row.get("max_abs_p_diff", ""),
            "max_abs_stat_diff": row.get("max_abs_stat_diff", ""),
            "notes": row.get("notes", ""),
        }
    ]


def summary_row(
    *,
    stage: str,
    shape: Shape,
    run_id: str,
    cpu_row: dict[str, Any] | None,
    a100_row: dict[str, Any] | None,
    kernel_row: dict[str, Any] | None,
    notes: str = "",
) -> dict[str, Any]:
    cpu_time = None
    if cpu_row and str(cpu_row.get("correctness_status")) in {"pass", "check"}:
        cpu_time = float(cpu_row.get("end_to_end_time_s") or cpu_row.get("cpu_time_s") or "nan")
    a100_time = None
    if a100_row and str(a100_row.get("correctness_status")) in {"pass", "check"}:
        a100_time = float(a100_row.get("end_to_end_time_s") or "nan")
    speedup = ""
    winner = "unavailable"
    if cpu_time and a100_time and math.isfinite(cpu_time) and math.isfinite(a100_time) and a100_time > 0:
        speedup_val = cpu_time / a100_time
        speedup = as_float(speedup_val)
        winner = "a100" if speedup_val > 1.0 else "cpu"
    est = estimates(shape.n, shape.p, shape.batch_R, shape.dtype)
    return {
        "run_id": run_id,
        "timestamp": timestamp(),
        "stage": stage,
        "n": shape.n,
        "p": shape.p,
        "R": shape.R,
        "batch_R": shape.batch_R,
        "n_batches": n_batches(shape),
        "seed": shape.seed,
        "dtype": shape.dtype,
        "best_cpu_implementation": (cpu_row or {}).get("implementation", "numpy_matrix_same_stream"),
        "cpu_end_to_end_time_s": as_float(cpu_time),
        "cpu_status": (cpu_row or {}).get("correctness_status", ""),
        "cpu_timeout_status": (cpu_row or {}).get("timeout_status", ""),
        "a100_end_to_end_time_s": as_float(a100_time),
        "a100_streamed_reduction_time_s": as_float(a100_time),
        "a100_status": (a100_row or {}).get("correctness_status", ""),
        "kernel_only_time_s": (kernel_row or {}).get("kernel_only_time_s", ""),
        "kernel_only_label": (kernel_row or {}).get("kernel_only_label", ""),
        "speedup_cpu_over_a100": speedup,
        "winner": winner,
        "max_abs_p_diff": (a100_row or cpu_row or {}).get("max_abs_p_diff", ""),
        "max_abs_stat_diff": (a100_row or cpu_row or {}).get("max_abs_stat_diff", ""),
        "estimated_device_memory_per_batch_gib": as_float(est["estimated_device_memory_per_batch_gib"]),
        "estimated_x_gib": as_float(est["estimated_x_gib"]),
        "notes": notes,
    }


def run_shape(
    *,
    shape: Shape,
    stage: str,
    run_id: str,
    out_dir: Path,
    cpu_timeout_s: int,
    device_limit_gib: float,
    host_x_limit_gib: float,
    cpu_lookup: dict[tuple[int, int, int, int], dict[str, Any]],
    run_cpu: bool = True,
    run_kernel: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    correctness: list[dict[str, Any]] = []
    key = (shape.n, shape.p, shape.R, shape.batch_R)
    reason = should_skip_for_memory(shape, device_limit_gib, host_x_limit_gib)
    if reason:
        row = summary_row(stage=stage, shape=shape, run_id=run_id, cpu_row=None, a100_row=None, kernel_row=None, notes=reason)
        row["a100_status"] = "skipped"
        row["winner"] = "memory-risk"
        return row, None, None, None, correctness

    try:
        a100_row = run_decomposition(shape, run_id)
        a100_row["implementation"] = "a100_streamed_reduction"
        a100_row["benchmark"] = stage
        a100_row["source"] = stage
    except Exception as exc:
        note = repr(exc)
        status = "oom" if any(token in note.lower() for token in ["out of memory", "resource_exhausted", "oom"]) else "fail"
        est = estimates(shape.n, shape.p, shape.batch_R, shape.dtype)
        a100_row = {
            "run_id": run_id,
            "timestamp": timestamp(),
            "benchmark": stage,
            "implementation": "a100_streamed_reduction",
            "correctness_status": status,
            "max_abs_p_diff": "",
            "max_abs_stat_diff": "",
            "dtype": shape.dtype,
            "device": "a100",
            "n": shape.n,
            "p": shape.p,
            "R": shape.R,
            "batch_R": shape.batch_R,
            "seed": shape.seed,
            "end_to_end_time_s": "",
            "total_end_to_end_time_s": "",
            "estimated_device_memory_per_batch_gib": as_float(est["estimated_device_memory_per_batch_gib"]),
            "estimated_x_gib": as_float(est["estimated_x_gib"]),
            "notes": f"a100_streamed_reduction_{status}: {note[-700:]}",
        }
    correctness.extend(correctness_rows(stage, a100_row, run_id))

    kernel_row = None
    if run_kernel:
        try:
            kernel_row = run_kernel_only(shape, run_id)
            kernel_row["benchmark"] = f"{stage}_kernel_only"
            kernel_row["source"] = stage
        except Exception as exc:
            note = repr(exc)
            status = "oom" if any(token in note.lower() for token in ["out of memory", "resource_exhausted", "oom"]) else "fail"
            kernel_row = {
                "run_id": run_id,
                "timestamp": timestamp(),
                "benchmark": f"{stage}_kernel_only",
                "implementation": "a100_kernel_only",
                "correctness_status": status,
                "dtype": shape.dtype,
                "device": "a100",
                "n": shape.n,
                "p": shape.p,
                "R": shape.R,
                "batch_R": shape.batch_R,
                "seed": shape.seed,
                "kernel_only_time_s": "",
                "kernel_only_label": "kernel-only hypothesis, not end-to-end permutation test",
                "notes": f"a100_kernel_only_{status}: {note[-700:]}",
            }
        correctness.extend(correctness_rows(stage, kernel_row, run_id))

    cpu_row = cpu_lookup.get(key)
    if cpu_row is None and run_cpu:
        cpu_row = run_cpu_baseline(shape, run_id, cpu_timeout_s)
        cpu_row["source"] = stage
        cpu_lookup[key] = cpu_row
        append_csv(out_dir / "cpu_matched_baselines.csv", cpu_row, CPU_FIELDS)
    if cpu_row is not None:
        correctness.extend(correctness_rows(stage, cpu_row, run_id))

    row = summary_row(stage=stage, shape=shape, run_id=run_id, cpu_row=cpu_row, a100_row=a100_row, kernel_row=kernel_row)
    return row, cpu_row, a100_row, kernel_row, correctness


def choose_best_batch(batch_rows: list[dict[str, Any]]) -> int:
    ok = [
        row
        for row in batch_rows
        if row.get("a100_status") in {"pass", "check"} and row.get("a100_end_to_end_time_s") not in {"", None}
    ]
    if not ok:
        return 4096
    return int(min(ok, key=lambda r: float(r["a100_end_to_end_time_s"]))["batch_R"])


def stage2_shapes(batch_r: int, seed: int) -> list[Shape]:
    return [
        Shape(5_000, p, r, batch_r, seed=seed, source="stage2_break_even")
        for p in [10_000, 50_000, 100_000, 250_000, 500_000]
        for r in [1_000, 5_000, 10_000, 50_000]
    ]


def stage3_shapes(batch_r: int, seed: int) -> list[Shape]:
    return [
        Shape(n, p, r, batch_r, seed=seed, source="stage3_n_sensitivity")
        for n in [1_000, 5_000, 10_000]
        for p in [50_000, 100_000]
        for r in [10_000, 50_000]
    ]


def run_suite(args: argparse.Namespace) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"permutation_break_even_{time.strftime('%Y%m%d_%H%M%S')}"
    cpu_lookup = existing_cpu_lookup(out_dir / "cpu_matched_baselines.csv")

    correctness_all: list[dict[str, Any]] = []
    correctness_path = out_dir / "correctness_checks.csv"
    if correctness_path.exists():
        correctness_all = pd.read_csv(correctness_path).to_dict("records")
    batch_rows, batch_done = existing_summary_rows(out_dir / "batch_R_sweep.csv")
    fixed_cpu_shape = Shape(5_000, 50_000, 10_000, 4096, seed=args.seed, source="stage1_batch")
    fixed_cpu = cpu_lookup.get((fixed_cpu_shape.n, fixed_cpu_shape.p, fixed_cpu_shape.R, fixed_cpu_shape.batch_R))
    if fixed_cpu is None:
        fixed_cpu = run_cpu_baseline(fixed_cpu_shape, run_id, args.cpu_timeout_s)
        fixed_cpu["source"] = "stage1_batch_cpu_reference"
        cpu_lookup[(fixed_cpu_shape.n, fixed_cpu_shape.p, fixed_cpu_shape.R, fixed_cpu_shape.batch_R)] = fixed_cpu
        append_csv(out_dir / "cpu_matched_baselines.csv", fixed_cpu, CPU_FIELDS)
    correctness_all.extend(correctness_rows("stage1_batch_R_sweep", fixed_cpu, run_id))

    for batch_r in [128, 256, 512, 1024, 2048, 4096, 8192]:
        shape = Shape(5_000, 50_000, 10_000, batch_r, seed=args.seed, source="stage1_batch")
        if (shape.n, shape.p, shape.R, shape.batch_R) in batch_done:
            continue
        print(f"[stage1] batch_R={batch_r}", flush=True)
        row, _, a100_row, kernel_row, correctness = run_shape(
            shape=shape,
            stage="stage1_batch_R_sweep",
            run_id=run_id,
            out_dir=out_dir,
            cpu_timeout_s=args.cpu_timeout_s,
            device_limit_gib=args.device_limit_gib,
            host_x_limit_gib=args.host_x_limit_gib,
            cpu_lookup={(shape.n, shape.p, shape.R, shape.batch_R): fixed_cpu} if batch_r == 4096 else cpu_lookup,
            run_cpu=False,
            run_kernel=True,
        )
        row["cpu_end_to_end_time_s"] = fixed_cpu.get("end_to_end_time_s", "")
        row["cpu_status"] = fixed_cpu.get("correctness_status", "")
        if row["cpu_end_to_end_time_s"] and row["a100_end_to_end_time_s"]:
            speedup = float(row["cpu_end_to_end_time_s"]) / float(row["a100_end_to_end_time_s"])
            row["speedup_cpu_over_a100"] = as_float(speedup)
            row["winner"] = "a100" if speedup > 1 else "cpu"
        batch_rows.append(row)
        correctness_all.extend(correctness)

    write_csv(out_dir / "batch_R_sweep.csv", batch_rows, CSV_FIELDS)
    best_batch = args.batch_R or choose_best_batch(batch_rows)
    print(f"[stage1] best safe batch_R={best_batch}", flush=True)

    shape_rows, shape_done = existing_summary_rows(out_dir / "break_even_shape_sweep.csv")
    for shape in stage2_shapes(best_batch, args.seed):
        if (shape.n, shape.p, shape.R, shape.batch_R) in shape_done:
            continue
        print(f"[stage2] n={shape.n} p={shape.p} R={shape.R} batch_R={shape.batch_R}", flush=True)
        row, _, _, _, correctness = run_shape(
            shape=shape,
            stage="stage2_p_R_break_even",
            run_id=run_id,
            out_dir=out_dir,
            cpu_timeout_s=args.cpu_timeout_s,
            device_limit_gib=args.device_limit_gib,
            host_x_limit_gib=args.host_x_limit_gib,
            cpu_lookup=cpu_lookup,
            run_cpu=True,
            run_kernel=args.kernel_stage2,
        )
        shape_rows.append(row)
        correctness_all.extend(correctness)
        write_csv(out_dir / "break_even_shape_sweep.csv", shape_rows, CSV_FIELDS)
        write_csv(out_dir / "correctness_checks.csv", correctness_all, CORRECTNESS_FIELDS)

    n_rows, n_done = existing_summary_rows(out_dir / "n_sensitivity_sweep.csv")
    for shape in stage3_shapes(best_batch, args.seed):
        if (shape.n, shape.p, shape.R, shape.batch_R) in n_done:
            continue
        print(f"[stage3] n={shape.n} p={shape.p} R={shape.R} batch_R={shape.batch_R}", flush=True)
        row, _, _, _, correctness = run_shape(
            shape=shape,
            stage="stage3_n_sensitivity",
            run_id=run_id,
            out_dir=out_dir,
            cpu_timeout_s=args.cpu_timeout_s,
            device_limit_gib=args.device_limit_gib,
            host_x_limit_gib=args.host_x_limit_gib,
            cpu_lookup=cpu_lookup,
            run_cpu=True,
            run_kernel=False,
        )
        n_rows.append(row)
        correctness_all.extend(correctness)
        write_csv(out_dir / "n_sensitivity_sweep.csv", n_rows, CSV_FIELDS)
        write_csv(out_dir / "correctness_checks.csv", correctness_all, CORRECTNESS_FIELDS)

    make_representative_decomposition(out_dir, run_id, best_batch, args.seed)
    write_csv(out_dir / "correctness_checks.csv", correctness_all, CORRECTNESS_FIELDS)
    make_figures(out_dir, args.presentation_dir)
    write_readme(out_dir)


def make_representative_decomposition(out_dir: Path, run_id: str, batch_r: int, seed: int) -> None:
    shape_df = pd.read_csv(out_dir / "break_even_shape_sweep.csv")
    ok = shape_df[pd.to_numeric(shape_df["speedup_cpu_over_a100"], errors="coerce").notna()].copy()
    ok["speedup"] = pd.to_numeric(ok["speedup_cpu_over_a100"], errors="coerce")
    reps: list[Shape] = []
    if not ok.empty:
        cpu_fast = ok[ok["speedup"] < 0.8]
        near = ok[(ok["speedup"] >= 0.8) & (ok["speedup"] <= 1.25)]
        gpu_fast = ok[ok["speedup"] > 1.25]
        largest = ok.sort_values(["p", "R"]).tail(1)
        for sub in [cpu_fast.sort_values("speedup").head(1), near.iloc[(near["speedup"] - 1.0).abs().argsort()].head(1) if not near.empty else near, gpu_fast.sort_values("speedup", ascending=False).head(1), largest]:
            if not sub.empty:
                r = sub.iloc[0]
                reps.append(Shape(int(r.n), int(r.p), int(r.R), int(r.batch_R), seed=seed, source="representative_break_even"))
    unique: list[Shape] = []
    seen = set()
    for shape in reps:
        key = (shape.n, shape.p, shape.R, shape.batch_R)
        if key not in seen:
            seen.add(key)
            unique.append(shape)
    rows = []
    for shape in unique:
        print(f"[decomp] n={shape.n} p={shape.p} R={shape.R} batch_R={shape.batch_R}", flush=True)
        try:
            row = run_decomposition(shape, run_id)
        except Exception as exc:
            note = repr(exc)
            status = "oom" if any(token in note.lower() for token in ["out of memory", "resource_exhausted", "oom"]) else "fail"
            est = estimates(shape.n, shape.p, shape.batch_R, shape.dtype)
            row = {
                "run_id": run_id,
                "timestamp": timestamp(),
                "benchmark": "representative_break_even_decomposition",
                "implementation": "a100_streamed_reduction",
                "correctness_status": status,
                "dtype": shape.dtype,
                "device": "a100",
                "n": shape.n,
                "p": shape.p,
                "R": shape.R,
                "batch_R": shape.batch_R,
                "seed": shape.seed,
                "total_end_to_end_time_s": "",
                "estimated_device_memory_per_batch_gib": as_float(est["estimated_device_memory_per_batch_gib"]),
                "notes": f"representative_decomposition_{status}: {note[-700:]}",
            }
        row["benchmark"] = "representative_break_even_decomposition"
        rows.append(row)
        write_csv(out_dir / "decomposition_representative_shapes.csv", rows, DECOMP_FIELDS)


def make_figures(out_dir: Path, presentation_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    presentation_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 17, "xtick.labelsize": 11, "ytick.labelsize": 11})
    shape = pd.read_csv(out_dir / "break_even_shape_sweep.csv")
    shape["speedup"] = pd.to_numeric(shape["speedup_cpu_over_a100"], errors="coerce")
    pivot = shape.pivot_table(index="R", columns="p", values="speedup", aggfunc="first").sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.patch.set_facecolor("#FBF7EF")
    ax.set_facecolor("#FFFFFF")
    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0.25, vmax=max(2.5, float(np.nanmax(data)) if np.isfinite(data).any() else 2.5), aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), [f"{int(c):,}" for c in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), [f"{int(r):,}" for r in pivot.index])
    ax.set_xlabel("p_features")
    ax.set_ylabel("R permutations")
    ax.set_title("CPU vs A100 end-to-end break-even map", weight="bold")
    for i, r in enumerate(pivot.index):
        for j, p in enumerate(pivot.columns):
            val = pivot.loc[r, p]
            if pd.isna(val):
                txt = "timeout\n/skip"
            else:
                txt = f"{val:.1f}x"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, weight="bold", color="#17202A")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("speedup = CPU full e2e / A100 streamed full e2e")
    winners = shape[pd.to_numeric(shape["speedup_cpu_over_a100"], errors="coerce") > 1.0]
    if winners.empty:
        takeaway = "No end-to-end A100 break-even found in measured range."
    else:
        first = winners.sort_values(["p", "R"]).iloc[0]
        takeaway = f"A100 faster region appears by n={int(first.n)}, p={int(first.p):,}, R={int(first.R):,}, batch_R={int(first.batch_R):,}."
    fig.text(0.08, 0.05, takeaway, fontsize=15, weight="bold", bbox={"facecolor": "#FFF3E6", "edgecolor": "none", "boxstyle": "round,pad=0.35"})
    fig.tight_layout(rect=[0.04, 0.10, 0.98, 0.94])
    fig.savefig(presentation_dir / "cpu_vs_a100_break_even_map.png", dpi=220)
    fig.savefig(presentation_dir / "cpu_vs_a100_break_even_map.svg", format="svg")
    plt.close(fig)

    decomp = pd.read_csv(out_dir / "decomposition_representative_shapes.csv")
    decomp = decomp[decomp.get("correctness_status", pd.Series(dtype=str)).isin(["pass", "check"])].copy()
    if not decomp.empty:
        decomp["total"] = pd.to_numeric(decomp["total_end_to_end_time_s"], errors="coerce")
        decomp["perm_generation_s"] = pd.to_numeric(decomp["permutation_generation_time_s"], errors="coerce").fillna(0.0)
        decomp["build_w_s"] = pd.to_numeric(decomp["W_build_host_time_s"], errors="coerce").fillna(0.0)
        decomp["transfer_collect_s"] = (
            pd.to_numeric(decomp["host_to_device_transfer_time_s"], errors="coerce").fillna(0.0)
            + pd.to_numeric(decomp["device_to_host_collect_time_s"], errors="coerce").fillna(0.0)
        )
        decomp["wx_s"] = pd.to_numeric(decomp["device_compute_time_s"], errors="coerce").fillna(0.0)
        decomp["named_stage_sum_s"] = (
            decomp["perm_generation_s"]
            + decomp["build_w_s"]
            + decomp["transfer_collect_s"]
            + decomp["wx_s"]
            + pd.to_numeric(decomp["pvalue_reduction_time_s"], errors="coerce").fillna(0.0)
        )
        decomp["other_s"] = (decomp["total"] - decomp["named_stage_sum_s"]).clip(lower=0.0).fillna(0.0)
        stage_cols = [
            ("perm_generation_s", "perm generation", "#4D7FEA"),
            ("build_w_s", "build W", "#2F7D32"),
            ("transfer_collect_s", "transfer/collect", "#E66A2C"),
            ("wx_s", "W @ X", "#B51E59"),
            ("other_s", "other overhead", "#6F7782"),
        ]
        def compact_count(value: float) -> str:
            return f"{value / 1000:.0f}k" if value >= 1000 else str(int(value))

        labels = [
            f"p={compact_count(r.p)}, R={compact_count(r.R)}\n{int(math.ceil(r.R / r.batch_R))} batch, {float(r.total):.2f}s"
            for r in decomp.itertuples()
        ]
        fig, (ax_abs, ax_pct) = plt.subplots(
            1,
            2,
            figsize=(12.8, 7.2),
            gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.16},
        )
        fig.patch.set_facecolor("#FBF7EF")
        y = np.arange(len(decomp))
        left = np.zeros(len(decomp))
        for col, lab, color in stage_cols:
            vals = decomp[col].to_numpy(float)
            ax_abs.barh(y, vals, left=left, color=color, height=0.5, label=lab)
            left += vals
        for i, total in enumerate(decomp["total"]):
            ax_abs.text(float(total) * 1.015, i, f"{float(total):.2f}s", va="center", fontsize=10, weight="bold")
        ax_abs.set_yticks(y, labels)
        ax_abs.invert_yaxis()
        ax_abs.set_xlabel("seconds per full scenario")
        ax_abs.set_title("Measured full scenario", weight="bold", fontsize=12)
        ax_abs.grid(axis="x", alpha=0.25)
        ax_abs.set_xlim(0, float(decomp["total"].max()) * 1.18)

        pct_left = np.zeros(len(decomp))
        pct_labels = {"perm generation": "perm gen", "transfer/collect": "transfer\ncollect"}
        for col, lab, color in stage_cols:
            vals = decomp[col].to_numpy(float)
            pct = np.divide(vals, decomp["total"].to_numpy(float), out=np.zeros_like(vals), where=decomp["total"].to_numpy(float) > 0)
            ax_pct.barh(y, pct, left=pct_left, color=color, height=0.5)
            for i, share in enumerate(pct):
                if share >= 0.18:
                    display_lab = pct_labels.get(lab, lab)
                    ax_pct.text(
                        pct_left[i] + share / 2,
                        i,
                        f"{display_lab}\n{100*share:.0f}%",
                        color="white",
                        ha="center",
                        va="center",
                        fontsize=9,
                        weight="bold",
                    )
            pct_left += pct
        ax_pct.set_yticks([])
        ax_pct.set_xlim(0, 1)
        ax_pct.set_xlabel("share of full scenario")
        ax_pct.set_title("Percent of time", weight="bold", fontsize=12)
        ax_pct.grid(axis="x", alpha=0.25)
        ax_pct.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        fig.suptitle("Where does A100 time go?", weight="bold", fontsize=19, y=0.96)
        fig.text(
            0.5,
            0.91,
            "A100 streamed full scenario, compile excluded, transfer included; n=5,000, batch_R=8,192, float32",
            ha="center",
            fontsize=10,
            color="#25313C",
        )
        handles, legend_labels = ax_abs.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.875))
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.80, wspace=0.16)
        fig.savefig(presentation_dir / "a100_pipeline_decomposition_representative.png", dpi=220)
        fig.savefig(presentation_dir / "a100_pipeline_decomposition_representative.svg", format="svg")
        plt.close(fig)

    batch = pd.read_csv(out_dir / "batch_R_sweep.csv")
    batch["a100"] = pd.to_numeric(batch["a100_streamed_reduction_time_s"], errors="coerce")
    batch["kernel"] = pd.to_numeric(batch["kernel_only_time_s"], errors="coerce")
    batch["speedup"] = pd.to_numeric(batch["speedup_cpu_over_a100"], errors="coerce")
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.plot(batch["batch_R"], batch["a100"], marker="o", linewidth=3, label="A100 streamed full e2e")
    ax.plot(batch["batch_R"], batch["speedup"], marker="s", linewidth=3, label="speedup CPU/A100")
    ax.axhline(1.0, color="#17202A", linestyle="--", linewidth=1.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch_R")
    ax.set_ylabel("seconds or speedup")
    ax.set_title("Batch size controls whether the GPU path is saturated", weight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(presentation_dir / "batch_R_sweep.png", dpi=220)
    fig.savefig(presentation_dir / "batch_R_sweep.svg", format="svg")
    plt.close(fig)

    reps = shape.dropna(subset=["speedup"]).sort_values(["p", "R"]).tail(4).copy()
    if not reps.empty:
        x = np.arange(len(reps))
        width = 0.2
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        labels = [f"p={int(r.p):,}\nR={int(r.R):,}" for r in reps.itertuples()]
        ax.bar(x - 1.5 * width, pd.to_numeric(reps["kernel_only_time_s"], errors="coerce"), width, label="kernel-only W @ X\n(not full test)")
        ax.bar(x - 0.5 * width, pd.to_numeric(reps["a100_end_to_end_time_s"], errors="coerce"), width, label="A100 full e2e")
        ax.bar(x + 0.5 * width, pd.to_numeric(reps["a100_streamed_reduction_time_s"], errors="coerce"), width, label="A100 streamed reduction")
        ax.bar(x + 1.5 * width, pd.to_numeric(reps["cpu_end_to_end_time_s"], errors="coerce"), width, label="best CPU full e2e")
        ax.set_yscale("log")
        ax.set_xticks(x, labels)
        ax.set_ylabel("seconds, log scale")
        ax.set_title("A fast kernel is not the same as a fast statistical pipeline", weight="bold")
        ax.grid(axis="y", which="both", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(presentation_dir / "kernel_only_vs_end_to_end.png", dpi=220)
        fig.savefig(presentation_dir / "kernel_only_vs_end_to_end.svg", format="svg")
        plt.close(fig)


def write_readme(out_dir: Path) -> None:
    lines = ["# CPU vs A100 permutation break-even", "", f"Generated/updated: {timestamp()}", ""]
    batch = pd.read_csv(out_dir / "batch_R_sweep.csv")
    shape = pd.read_csv(out_dir / "break_even_shape_sweep.csv")
    cpu = pd.read_csv(out_dir / "cpu_matched_baselines.csv")
    correct = pd.read_csv(out_dir / "correctness_checks.csv")
    lines.append("## CPU baseline")
    lines.append("- Best trusted CPU implementation used here: `numpy_matrix_same_stream` batched matrix path.")
    lines.append(f"- CPU rows recorded: {len(cpu)}.")
    lines.append("")
    lines.append("## Break-even")
    shape["speedup"] = pd.to_numeric(shape["speedup_cpu_over_a100"], errors="coerce")
    winners = shape[shape["speedup"] > 1.0]
    if winners.empty:
        lines.append("- No end-to-end A100 break-even found in measured range.")
    else:
        first = winners.sort_values(["p", "R"]).iloc[0]
        best = winners.sort_values("speedup", ascending=False).iloc[0]
        lines.append(f"- A100 becomes faster at n={int(first.n)}, p={int(first.p)}, R={int(first.R)}, batch_R={int(first.batch_R)}.")
        lines.append(f"- Largest measured speedup: {float(best.speedup):.2f}x at n={int(best.n)}, p={int(best.p)}, R={int(best.R)}.")
    lines.append("")
    lines.append("## Streamed reduction")
    lines.append("- `a100_streamed_reduction` computes `T_null_batch = W_batch @ X_device`, accumulates exceedance counts on device, and collects final p-values/counts only.")
    lines.append("- It preserves the same statistic and same host W permutation stream used for CPU checks.")
    lines.append("- The break-even map uses this streamed full end-to-end path. A separate full-collection A100 break-even row was not used, so no speedup is claimed from streaming alone; the measured benefit is that the full `R x p` null matrix is not collected.")
    lines.append("")
    lines.append("## Kernel-only vs end-to-end")
    lines.append("- Kernel-only rows are labeled as not full permutation tests and are not used for CPU/A100 speedup decisions.")
    lines.append("")
    lines.append("## Timing semantics")
    lines.append("- CPU/A100 comparisons are full scenario end-to-end, warm timing, compile excluded, transfer included for A100.")
    lines.append("- Representative decomposition rows report named stages plus residual Python/JAX loop overhead. Figures include this residual as `other overhead` so stacked bars reconcile to `total_end_to_end_time_s`.")
    lines.append("- Kernel-only rows time only `W @ X` with device-resident inputs and are labeled as hypotheses, not full permutation tests.")
    lines.append("")
    lines.append("## Batch_R")
    best_batch = choose_best_batch(batch.to_dict("records"))
    lines.append(f"- Best safe batch_R from Stage 1: {best_batch}.")
    lines.append("")
    lines.append("## Correctness")
    lines.append(f"- Correctness check rows: {len(correct)}.")
    for key, val in correct["correctness_status"].value_counts(dropna=False).items():
        lines.append(f"  - {key}: {val}")
    lines.append("")
    lines.append("## OOM / memory-risk / timeout")
    for name in ["batch_R_sweep.csv", "break_even_shape_sweep.csv", "n_sensitivity_sweep.csv", "cpu_matched_baselines.csv"]:
        path = out_dir / name
        if path.exists():
            df = pd.read_csv(path)
            bad_cols = [c for c in ["winner", "cpu_timeout_status", "a100_status", "correctness_status"] if c in df]
            bad = df[df[bad_cols].astype(str).apply(lambda s: s.str.contains("timeout|skipped|memory-risk|fail", case=False, regex=True)).any(axis=1)] if bad_cols else pd.DataFrame()
            lines.append(f"- `{name}`: {len(bad)} timeout/skipped/memory-risk/fail rows.")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    run = sub.add_parser("run")
    run.add_argument("--out-dir", type=Path, default=OUT_DIR)
    run.add_argument("--presentation-dir", type=Path, default=PRESENTATION_DIR)
    run.add_argument("--run-id", default="")
    run.add_argument("--seed", type=int, default=100)
    run.add_argument("--batch-R", type=int, default=0)
    run.add_argument("--cpu-timeout-s", type=int, default=180)
    run.add_argument("--device-limit-gib", type=float, default=55.0)
    run.add_argument("--host-x-limit-gib", type=float, default=24.0)
    run.add_argument("--kernel-stage2", action="store_true")
    plot = sub.add_parser("plot")
    plot.add_argument("--out-dir", type=Path, default=OUT_DIR)
    plot.add_argument("--presentation-dir", type=Path, default=PRESENTATION_DIR)
    args = parser.parse_args()
    if args.cmd == "run":
        run_suite(args)
    elif args.cmd == "plot":
        make_figures(args.out_dir, args.presentation_dir)
        write_readme(args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
