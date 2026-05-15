"""Generate PNG figures and derived metrics for Python no-GIL slides.

Run from the repository root:

    conda run -n py312 python slides/assets/python314-nogil/make_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fsm4py-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/fsm4py-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "experiments/results/python314_interpreter_effects/latest"

ENV_ORDER = ("py313", "py314", "py314t")
REQUIRED_ENVS = ("py314", "py314t")
STANDARD_GIL_ENVS = ("py313", "py314")
ENV_LABELS = {
    "py313": "py313 standard GIL",
    "py314": "py314 standard GIL",
    "py314t": "py314t no-GIL",
}
ENV_PURPOSES = {
    "py313": "old standard-GIL baseline",
    "py314": "new standard-GIL baseline",
    "py314t": "execution-model change",
}

DPI = 300
PAPER = "#FBF7EF"
INK = "#17202A"
MUTED = "#52616D"
LIGHT_MUTED = "#A9B3BB"
LINE = "#D7CDC0"
BLUE = "#2368AD"
BERRY = "#B51E59"
GREEN = "#248A5A"
GOLD = "#B77418"

LEGACY_OUTPUTS = [
    "thread_scaling_runtime.png",
    "thread_scaling_speedup.png",
    "pool_runtime_memory.png",
    "pool_runtime_memory_backup.png",
    "python314_thread_scaling.svg",
]
OPTIONAL_OUTPUTS = ["version_baseline_backup.png"]


def _summary_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(RESULTS_DIR.glob("summary_interpreter_effects_*.csv")):
        env = path.stem.removeprefix("summary_interpreter_effects_")
        if env in ENV_ORDER:
            paths[env] = path
    missing = [env for env in REQUIRED_ENVS if env not in paths]
    if missing:
        raise FileNotFoundError(f"missing required summary CSVs for: {', '.join(missing)}")
    return paths


def _metadata_paths(envs: list[str]) -> dict[str, Path]:
    return {
        env: RESULTS_DIR / f"metadata_{env}.json"
        for env in envs
        if (RESULTS_DIR / f"metadata_{env}.json").exists()
    }


def _read_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path], dict[str, Path]]:
    summary_paths = _summary_paths()
    frames = []
    for env in ENV_ORDER:
        path = summary_paths.get(env)
        if path is None:
            continue
        frame = pd.read_csv(path)
        frame["source_summary_csv"] = str(path.relative_to(REPO_ROOT))
        frame["expected_env_label"] = env
        frames.append(frame)

    metadata_paths = _metadata_paths([env for env in ENV_ORDER if env in summary_paths])
    metadata = {
        env: json.loads(path.read_text(encoding="utf-8"))
        for env, path in metadata_paths.items()
    }

    df = pd.concat(frames, ignore_index=True)
    for column in ("workers", "median_wall_time_sec", "iqr_wall_time_sec", "median_peak_rss_gb", "max_peak_rss_gb"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df, metadata, summary_paths, metadata_paths


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.titlesize": 26,
            "axes.labelsize": 21,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.2,
        }
    )


def _available_envs(df: pd.DataFrame) -> list[str]:
    present = set(df["env_label"].dropna().astype(str))
    return [env for env in ENV_ORDER if env in present]


def _thread_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["experiment"].eq("thread_scaling")].copy()
    if rows.empty:
        raise ValueError("missing thread_scaling rows")
    return rows.sort_values(["env_label", "workers"])


def _pool_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["experiment"].eq("pool_memory_runtime")].copy()
    if rows.empty:
        raise ValueError("missing pool_memory_runtime rows")
    return rows.sort_values(["env_label", "pool", "workers"])


def _row(rows: pd.DataFrame, *, env: str, workers: int | None = None, pool: str | None = None) -> pd.Series:
    mask = rows["env_label"].eq(env)
    if workers is not None:
        mask &= rows["workers"].eq(workers)
    if pool is not None:
        mask &= rows["pool"].eq(pool)
    subset = rows[mask]
    if subset.empty:
        raise ValueError(f"missing row env={env!r} workers={workers!r} pool={pool!r}")
    return subset.iloc[0]


def _fmt_x(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}×"


def _fmt_sec(value: float) -> str:
    return f"{value:.3f}s"


def _fmt_pct(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}%"


def _compute_thread_metrics(thread: pd.DataFrame, env: str) -> dict[str, Any]:
    env_rows = thread[thread["env_label"].eq(env)].sort_values("workers")
    if env_rows.empty:
        raise ValueError(f"missing thread_scaling rows for {env}")
    first = env_rows.iloc[0]
    last = env_rows.iloc[-1]
    baseline = float(first["median_wall_time_sec"])
    last_time = float(last["median_wall_time_sec"])
    env_metrics: dict[str, Any] = {
        "label": ENV_LABELS.get(env, env),
        "purpose": ENV_PURPOSES.get(env, ""),
        "first_worker": int(first["workers"]),
        "last_worker": int(last["workers"]),
        "first_worker_median_wall_time_sec": baseline,
        "last_worker_median_wall_time_sec": last_time,
        "scaling_ratio_first_over_last": float(baseline / last_time),
        "last_vs_first_speedup": float(baseline / last_time),
        "last_vs_first_runtime_ratio": float(last_time / baseline),
        "workers": [],
    }
    for _, item in env_rows.iterrows():
        median = float(item["median_wall_time_sec"])
        env_metrics["workers"].append(
            {
                "workers": int(item["workers"]),
                "median_wall_time_sec": median,
                "iqr_wall_time_sec": float(item["iqr_wall_time_sec"]),
                "speedup_vs_own_1_worker": float(baseline / median),
            }
        )
    return env_metrics


def _compute_metrics(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    summary_paths: dict[str, Path],
    metadata_paths: dict[str, Path],
) -> dict[str, Any]:
    thread = _thread_rows(df)
    pool = _pool_rows(df)
    envs = _available_envs(thread)
    has_py313 = "py313" in envs
    metrics: dict[str, Any] = {
        "has_py313": has_py313,
        "available_envs": envs,
        "source_files": {
            "summary_csv": {env: str(path.relative_to(REPO_ROOT)) for env, path in summary_paths.items()},
            "metadata_json": {env: str(path.relative_to(REPO_ROOT)) for env, path in metadata_paths.items()},
        },
        "jit_claim_allowed": {
            env: bool(meta.get("interpreter_effects_runner", {}).get("jit_claim_allowed"))
            for env, meta in metadata.items()
        },
        "thread_scaling": {},
        "same_worker_py314_over_py314t_ratio": {},
        "pool_runtime_memory": {},
    }

    for env in envs:
        metrics["thread_scaling"][env] = _compute_thread_metrics(thread, env)

    worker_values = sorted(set(thread["workers"].dropna().astype(int)))
    common_py314_workers = sorted(
        set(thread[thread["env_label"].eq("py314")]["workers"].dropna().astype(int))
        & set(thread[thread["env_label"].eq("py314t")]["workers"].dropna().astype(int))
    )
    for workers in common_py314_workers:
        py314 = float(_row(thread, env="py314", workers=workers)["median_wall_time_sec"])
        py314t = float(_row(thread, env="py314t", workers=workers)["median_wall_time_sec"])
        metrics["same_worker_py314_over_py314t_ratio"][str(workers)] = float(py314 / py314t)

    if has_py313:
        first_worker = max(
            metrics["thread_scaling"]["py313"]["first_worker"],
            metrics["thread_scaling"]["py314"]["first_worker"],
        )
        last_worker = min(
            metrics["thread_scaling"]["py313"]["last_worker"],
            metrics["thread_scaling"]["py314"]["last_worker"],
        )
        py313_first = float(_row(thread, env="py313", workers=first_worker)["median_wall_time_sec"])
        py314_first = float(_row(thread, env="py314", workers=first_worker)["median_wall_time_sec"])
        py313_last = float(_row(thread, env="py313", workers=last_worker)["median_wall_time_sec"])
        py314_last = float(_row(thread, env="py314", workers=last_worker)["median_wall_time_sec"])
        metrics["version_baseline_ratios"] = {
            "first_worker": int(first_worker),
            "last_worker": int(last_worker),
            "py313_over_py314_first_worker_runtime_ratio": float(py313_first / py314_first),
            "py313_over_py314_last_worker_runtime_ratio": float(py313_last / py314_last),
        }

    if 8 in worker_values and 16 in worker_values and "py314t" in metrics["thread_scaling"]:
        py314t_8 = float(_row(thread, env="py314t", workers=8)["median_wall_time_sec"])
        py314t_16 = float(_row(thread, env="py314t", workers=16)["median_wall_time_sec"])
        metrics["py314t_8_to_16_worker_improvement"] = {
            "runtime_saved_sec": float(py314t_8 - py314t_16),
            "runtime_reduction_percent": float((py314t_8 - py314t_16) / py314t_8 * 100.0),
            "speedup_factor": float(py314t_8 / py314t_16),
        }

    process_candidates = pool[(pool["env_label"].eq("py314")) & (pool["pool"].eq("process"))]
    thread_candidates = pool[(pool["env_label"].eq("py314t")) & (pool["pool"].eq("thread"))]
    common_workers = sorted(set(process_candidates["workers"].dropna().astype(int)) & set(thread_candidates["workers"].dropna().astype(int)))
    if not common_workers:
        raise ValueError("no matching process/thread worker count found")
    pool_workers = common_workers[0]
    process_row = _row(pool, env="py314", workers=pool_workers, pool="process")
    thread_row = _row(pool, env="py314t", workers=pool_workers, pool="thread")
    process_time = float(process_row["median_wall_time_sec"])
    thread_time = float(thread_row["median_wall_time_sec"])
    process_mem = float(process_row["max_peak_rss_gb"])
    thread_mem = float(thread_row["max_peak_rss_gb"])
    runtime_ratio = process_time / thread_time
    memory_reduction = 1.0 - thread_mem / process_mem
    metrics["pool_runtime_memory"] = {
        "workers": int(pool_workers),
        "py314_process_median_wall_time_sec": process_time,
        "py314t_thread_median_wall_time_sec": thread_time,
        "py314_process_max_peak_rss_gb": process_mem,
        "py314t_thread_max_peak_rss_gb": thread_mem,
        "runtime_ratio_py314_process_over_py314t_thread": float(runtime_ratio),
        "threadpool_vs_processpool_speedup": float(runtime_ratio),
        "peak_memory_reduction_gb": float(process_mem - thread_mem),
        "peak_memory_reduction_percent": float(memory_reduction * 100.0),
    }

    standard_envs = [env for env in STANDARD_GIL_ENVS if env in metrics["thread_scaling"]]
    standard_speedups = [metrics["thread_scaling"][env]["last_vs_first_speedup"] for env in standard_envs]
    standard_gil_scaling_ratio = float(max(standard_speedups)) if standard_speedups else 1.0
    py314 = metrics["thread_scaling"]["py314"]
    py314t = metrics["thread_scaling"]["py314t"]
    pool_metrics = metrics["pool_runtime_memory"]
    backup_rows = []
    for env in [env for env in ENV_ORDER if env in metrics["thread_scaling"]]:
        item = metrics["thread_scaling"][env]
        backup_rows.append(
            {
                "env": env,
                "label": item["label"],
                "first_worker_runtime": _fmt_sec(item["first_worker_median_wall_time_sec"]),
                "last_worker_runtime": _fmt_sec(item["last_worker_median_wall_time_sec"]),
                "scaling_ratio": _fmt_x(item["last_vs_first_speedup"]),
            }
        )

    metrics["slide_callouts"] = {
        "last_worker": py314t["last_worker"],
        "last_worker_label": f"1 → {py314t['last_worker']} workers",
        "standard_gil_scaling_ratio": _fmt_x(standard_gil_scaling_ratio),
        "standard_gil_scaling_value": f"~{_fmt_x(standard_gil_scaling_ratio)}",
        "standard_gil_interpretation": "py313/py314 stay flat" if has_py313 else "py314 stays flat",
        "py314_scaling_ratio": _fmt_x(py314["last_vs_first_speedup"]),
        "py314_scaling_value": _fmt_x(py314["last_vs_first_speedup"]),
        "py314_runtime_range": f"{_fmt_sec(py314['first_worker_median_wall_time_sec'])} → {_fmt_sec(py314['last_worker_median_wall_time_sec'])}",
        "py314t_speedup": _fmt_x(py314t["last_vs_first_speedup"]),
        "py314t_speedup_1_decimal": _fmt_x(py314t["last_vs_first_speedup"], 1),
        "py314t_speedup_value": _fmt_x(py314t["last_vs_first_speedup"]),
        "py314t_runtime_range": f"{_fmt_sec(py314t['first_worker_median_wall_time_sec'])} → {_fmt_sec(py314t['last_worker_median_wall_time_sec'])}",
        "pool_worker_label": f"{pool_metrics['workers']}-worker comparison",
        "pool_runtime_ratio": _fmt_x(pool_metrics["threadpool_vs_processpool_speedup"]),
        "pool_runtime_value": f"{_fmt_x(pool_metrics['threadpool_vs_processpool_speedup'])} faster",
        "pool_memory_reduction_percent": _fmt_pct(pool_metrics["peak_memory_reduction_percent"]),
        "pool_memory_value": f"{_fmt_pct(pool_metrics['peak_memory_reduction_percent'])} less peak RSS",
        "pool_memory_backup_value": f"{_fmt_pct(pool_metrics['peak_memory_reduction_percent'])} lower",
        "pool_process_runtime_value": _fmt_sec(pool_metrics["py314_process_median_wall_time_sec"]),
        "pool_thread_runtime_value": _fmt_sec(pool_metrics["py314t_thread_median_wall_time_sec"]),
        "pool_process_memory_value": f"{pool_metrics['py314_process_max_peak_rss_gb']:.3f} GiB",
        "pool_thread_memory_value": f"{pool_metrics['py314t_thread_max_peak_rss_gb']:.3f} GiB",
        "backup_evidence_rows": backup_rows,
    }
    return metrics


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _remove_legacy_outputs() -> None:
    for name in LEGACY_OUTPUTS + OPTIONAL_OUTPUTS:
        path = OUT_DIR / name
        if path.exists():
            path.unlink()
    for path in OUT_DIR.glob("*.svg"):
        path.unlink()


def _worker_positions(thread: pd.DataFrame) -> tuple[list[int], dict[int, float]]:
    workers = sorted(set(thread["workers"].dropna().astype(int)))
    return workers, {worker: float(index) for index, worker in enumerate(workers)}


def _series(thread: pd.DataFrame, env: str, worker_to_pos: dict[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env_rows = thread[thread["env_label"].eq(env)].sort_values("workers")
    workers = env_rows["workers"].astype(int).to_numpy()
    positions = np.array([worker_to_pos[int(worker)] for worker in workers], dtype=float)
    times = env_rows["median_wall_time_sec"].to_numpy(float)
    return positions, workers, times


def _label_positions(values: list[tuple[str, float]], min_gap: float) -> dict[str, float]:
    adjusted: dict[str, float] = {}
    last_y: float | None = None
    for env, y in sorted(values, key=lambda item: item[1]):
        new_y = y if last_y is None else max(y, last_y + min_gap)
        adjusted[env] = new_y
        last_y = new_y
    return adjusted


def _line_style(env: str) -> dict[str, Any]:
    if env == "py314t":
        return {"color": BLUE, "linewidth": 5.0, "markersize": 11.0, "alpha": 1.0, "zorder": 4}
    if env == "py313":
        return {"color": LIGHT_MUTED, "linewidth": 2.4, "markersize": 7.0, "alpha": 0.95, "zorder": 2}
    return {"color": MUTED, "linewidth": 3.0, "markersize": 8.0, "alpha": 0.95, "zorder": 3}


def plot_thread_runtime_hero(df: pd.DataFrame, metrics: dict[str, Any]) -> None:
    thread = _thread_rows(df)
    envs = _available_envs(thread)
    workers, worker_to_pos = _worker_positions(thread)
    fig, ax = plt.subplots(figsize=(10.8, 5.55))
    fig.patch.set_facecolor(PAPER)

    label_inputs: list[tuple[str, float]] = []
    all_times: list[float] = []
    for env in envs:
        positions, _, times = _series(thread, env, worker_to_pos)
        style = _line_style(env)
        ax.plot(
            positions,
            times,
            marker="o",
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            color=style["color"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )
        label_inputs.append((env, float(times[-1])))
        all_times.extend(times.tolist())

    y_min, y_max = min(all_times), max(all_times)
    y_span = y_max - y_min
    label_ys = _label_positions(label_inputs, max(y_span * 0.08, 0.035))
    label_x = len(workers) - 1 + 0.34
    for env in envs:
        style = _line_style(env)
        ax.text(
            label_x,
            label_ys[env],
            ENV_LABELS[env],
            color=style["color"],
            fontsize=17 if env != "py314t" else 19,
            fontweight=900 if env == "py314t" else 800,
            ha="left",
            va="center",
            clip_on=False,
        )

    ax.text(
        0.44,
        0.39,
        f"{metrics['slide_callouts']['py314t_speedup_1_decimal']} thread scaling",
        transform=ax.transAxes,
        fontsize=24,
        fontweight=900,
        color=BLUE,
        bbox={"facecolor": "#F3F8FC", "edgecolor": LINE, "boxstyle": "round,pad=0.32"},
    )
    ax.set_xlabel("ThreadPoolExecutor workers", labelpad=10)
    ax.set_ylabel("median wall time (s)", labelpad=10)
    ax.set_xticks(np.arange(len(workers)), [str(worker) for worker in workers])
    ax.set_xlim(-0.35, len(workers) - 1 + 1.65)
    ax.set_ylim(0.0, y_max * 1.16)
    ax.margins(x=0.08)
    ax.grid(axis="y", color=LINE, alpha=0.58, linewidth=1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.78, bottom=0.18, top=0.96)
    _save(fig, "thread_scaling_runtime_hero.png")


def plot_thread_speedup_backup(df: pd.DataFrame, metrics: dict[str, Any]) -> None:
    thread = _thread_rows(df)
    envs = _available_envs(thread)
    workers, worker_to_pos = _worker_positions(thread)
    fig, ax = plt.subplots(figsize=(10.4, 5.75))
    fig.patch.set_facecolor(PAPER)
    max_speedup = 0.0
    label_inputs: list[tuple[str, float]] = []
    for env in envs:
        positions, _, times = _series(thread, env, worker_to_pos)
        speedup = times[0] / times
        max_speedup = max(max_speedup, float(speedup.max()))
        style = _line_style(env)
        ax.plot(
            positions,
            speedup,
            marker="o",
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            color=style["color"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )
        label_inputs.append((env, float(speedup[-1])))

    label_ys = _label_positions(label_inputs, max(max_speedup * 0.08, 0.20))
    label_x = len(workers) - 1 + 0.34
    for env in envs:
        style = _line_style(env)
        ax.text(
            label_x,
            label_ys[env],
            ENV_LABELS[env],
            color=style["color"],
            fontsize=17 if env != "py314t" else 19,
            fontweight=900 if env == "py314t" else 800,
            ha="left",
            va="center",
            clip_on=False,
        )
    ax.set_xlabel("ThreadPoolExecutor workers", labelpad=10)
    ax.set_ylabel("speedup vs own 1-worker baseline", labelpad=10)
    ax.set_xticks(np.arange(len(workers)), [str(worker) for worker in workers])
    ax.set_xlim(-0.35, len(workers) - 1 + 1.72)
    ax.set_ylim(0.0, max_speedup * 1.18)
    ax.grid(axis="y", color=LINE, alpha=0.58, linewidth=1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.77, bottom=0.18, top=0.96)
    _save(fig, "thread_scaling_speedup_backup.png")


def plot_version_baseline_backup(df: pd.DataFrame, metrics: dict[str, Any]) -> None:
    if not metrics["has_py313"]:
        return
    thread = _thread_rows(df)
    workers, worker_to_pos = _worker_positions(thread)
    fig, ax = plt.subplots(figsize=(10.4, 5.75))
    fig.patch.set_facecolor(PAPER)
    all_times: list[float] = []
    for env in ("py313", "py314", "py314t"):
        positions, _, times = _series(thread, env, worker_to_pos)
        style = _line_style(env)
        ax.plot(
            positions,
            times,
            marker="o",
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            color=style["color"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )
        all_times.extend(times.tolist())
    ax.set_xlabel("ThreadPoolExecutor workers", labelpad=10)
    ax.set_ylabel("median wall time (s)", labelpad=10)
    ax.set_xticks(np.arange(len(workers)), [str(worker) for worker in workers])
    ax.set_xlim(-0.35, len(workers) - 1 + 0.35)
    ax.set_ylim(0.0, max(all_times) * 1.16)
    ax.grid(axis="y", color=LINE, alpha=0.58, linewidth=1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.18, top=0.96)
    _save(fig, "version_baseline_backup.png")


def _py313_todo() -> str:
    return """## Missing py313 Control Data

py313 control data was not found. To answer the version-baseline question, rerun
the thread-scaling benchmark under standard CPython 3.13 and regenerate the
figures:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  conda run -n py313 python -m experiments.interpreter_effects.run_suite \\
  --env-label py313 \\
  --output-dir experiments/results/python314_interpreter_effects/latest \\
  --experiments thread \\
  --repeats 5 \\
  --warmups 1

conda run -n py312 python slides/assets/python314-nogil/make_figures.py
```
"""


def _write_readme(metrics: dict[str, Any]) -> None:
    py314 = metrics["thread_scaling"]["py314"]
    py314t = metrics["thread_scaling"]["py314t"]
    pool = metrics["pool_runtime_memory"]
    py313_section = "" if metrics["has_py313"] else "\n" + _py313_todo()
    source_summary_lines = "\n".join(
        f"- `{path}`" for _, path in sorted(metrics["source_files"]["summary_csv"].items())
    )
    source_metadata_lines = "\n".join(
        f"- `{path}`" for _, path in sorted(metrics["source_files"]["metadata_json"].items())
    )
    generated_files = [
        "`thread_scaling_runtime_hero.png`",
        "`thread_scaling_speedup_backup.png`",
    ]
    if metrics["has_py313"]:
        generated_files.append("`version_baseline_backup.png`")
    generated_files.extend(["`derived_metrics.json`", "`README.md`"])
    generated_file_lines = "\n".join(f"- {name}" for name in generated_files)
    standard_text = "py313 and py314 both stay flat" if metrics["has_py313"] else "py314 stays flat"
    readme = f"""# Python No-GIL Slide Figures

This directory contains PNG-only slide figures generated from local benchmark
outputs. The deck uses the generated PNG files and `derived_metrics.json`; it
does not perform live plotting.

## Data Provenance

Source summary CSV files:

{source_summary_lines}

Source metadata JSON files:

{source_metadata_lines}

Column mapping:

- `env_label`: `py313` is standard CPython 3.13 with the GIL enabled when present; `py314` is standard CPython 3.14 with the GIL enabled; `py314t` is CPython 3.14 free-threaded / no-GIL with the GIL disabled.
- `experiment`: `thread_scaling` feeds the thread-scaling figures; `pool_memory_runtime` feeds the execution-model metric cards/table.
- `pool`: `process` identifies the ProcessPoolExecutor row; `thread` identifies the ThreadPoolExecutor row.
- `workers`: worker count for each measurement row.
- `median_wall_time_sec`: repeat-only median runtime used for plotted values and speedups.
- `iqr_wall_time_sec`: repeat-only interquartile range used for runtime uncertainty bands where shown.
- `max_peak_rss_gb`: peak RSS value used for memory comparison.
- `jit_claim_allowed`: verifies whether any Python 3.14 JIT acceleration claim is allowed.

Rows used:

- `env_label == "py313"` and `experiment == "thread_scaling"`: optional standard CPython 3.13 GIL thread-scaling control.
- `env_label == "py314"` and `experiment == "thread_scaling"`: standard CPython 3.14 GIL thread-scaling series.
- `env_label == "py314t"` and `experiment == "thread_scaling"`: free-threaded / no-GIL thread-scaling series.
- `env_label == "py314"`, `experiment == "pool_memory_runtime"`, `pool == "process"`: ProcessPoolExecutor comparison row.
- `env_label == "py314t"`, `experiment == "pool_memory_runtime"`, `pool == "thread"`: ThreadPoolExecutor comparison row.

## How To Regenerate

Run from the repository root:

```bash
conda run -n py312 python slides/assets/python314-nogil/make_figures.py
```

Required Python packages: `numpy`, `pandas`, and `matplotlib`.

## Generated Files

{generated_file_lines}
{py313_section}
## Interpretation

- py313 vs py314 is a control for ordinary CPython version changes when py313 data exists locally.
- py314 vs py314t isolates the free-threaded/no-GIL execution-model change within the Python 3.14 generation.
- If py313 and py314 both stay flat across workers, the result supports the claim that standard GIL threads do not scale this CPU-bound Python workload shape.
- If py314t scales while py313/py314 do not, the result supports the execution-model claim: no-GIL makes ThreadPool viable for this workload shape.
- This does not claim Python 3.14 JIT acceleration.
- This does not claim no-GIL makes all Python code faster.

Slide callouts generated in `derived_metrics.json`:

- Standard GIL card: `{metrics["slide_callouts"]["standard_gil_scaling_value"]}`; `{standard_text}`.
- py314t no-GIL card: `{metrics["slide_callouts"]["py314t_speedup_value"]}`; `{metrics["slide_callouts"]["py314t_runtime_range"]}`.
- Execution-model payoff card: `{metrics["slide_callouts"]["pool_runtime_value"]}`; `{metrics["slide_callouts"]["pool_memory_value"]}`.

Standard py314 changes from {_fmt_sec(py314["first_worker_median_wall_time_sec"])}
at {py314["first_worker"]} worker to {_fmt_sec(py314["last_worker_median_wall_time_sec"])}
at {py314["last_worker"]} workers, a {_fmt_x(py314["last_vs_first_speedup"])}
speedup. In this local run, that is little/no thread scaling.

py314t changes from {_fmt_sec(py314t["first_worker_median_wall_time_sec"])}
at {py314t["first_worker"]} worker to {_fmt_sec(py314t["last_worker_median_wall_time_sec"])}
at {py314t["last_worker"]} workers, a {_fmt_x(py314t["last_vs_first_speedup"])}
speedup against its own 1-worker baseline.

For the {pool["workers"]}-worker execution-model comparison, py314 ProcessPool
took {_fmt_sec(pool["py314_process_median_wall_time_sec"])} and py314t
ThreadPool took {_fmt_sec(pool["py314t_thread_median_wall_time_sec"])}, so the
thread-pool row is {_fmt_x(pool["threadpool_vs_processpool_speedup"])} faster.
Peak RSS changes from {pool["py314_process_max_peak_rss_gb"]:.3f} GiB to
{pool["py314t_thread_max_peak_rss_gb"]:.3f} GiB, a
{_fmt_pct(pool["peak_memory_reduction_percent"], 1)} reduction.

## Limitations / Non-Claims

- This benchmark does not claim Python 3.14 JIT acceleration.
- This benchmark does not show that no-GIL makes all Python code faster.
- This benchmark is relevant when work can be split into reasonably independent thread tasks over shared data.
- Workloads dominated by NumPy, BLAS, GPU kernels, I/O, locks, synchronization, or shared mutable state may behave differently.
- A plateau at higher worker counts should be interpreted as overhead, memory bandwidth, scheduling, or task-granularity limits, not as a failure of free-threaded Python.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    _style()
    _remove_legacy_outputs()
    df, metadata, summary_paths, metadata_paths = _read_inputs()
    metrics = _compute_metrics(df, metadata, summary_paths, metadata_paths)
    plot_thread_runtime_hero(df, metrics)
    plot_thread_speedup_backup(df, metrics)
    plot_version_baseline_backup(df, metrics)
    (OUT_DIR / "derived_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(metrics)
    print(f"wrote PNG figures, README, and derived metrics under {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
