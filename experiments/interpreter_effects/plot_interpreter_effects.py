"""Create slide-ready plots for the CPython 3.14 interpreter-effects suite."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fsm4py-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/fsm4py-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = "#FBF7EF"
INK = "#17202A"
LINE = "#D7CDC0"
BLUE = "#2368AD"
BERRY = "#B51E59"
GREEN = "#248A5A"
ORANGE = "#E66A2C"
DPI = 220


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def _read_summaries(results_dir: Path) -> pd.DataFrame:
    paths = sorted(results_dir.glob("summary_interpreter_effects_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no summary_interpreter_effects_*.csv files found in {results_dir}")
    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    for col in ("workers", "median_wall_time_sec", "iqr_wall_time_sec", "median_peak_rss_gb", "max_peak_rss_gb"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _finish(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=DPI)
    plt.close(fig)


def plot_negative(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["experiment"].eq("single_thread_negative_control")].copy()
    if sub.empty:
        return
    order = ["pure_python_cpu_loop", "numpy_blas_matrix_path", "small_statistical_loop"]
    labels = ["Python loop", "NumPy/BLAS", "Stat loop"]
    envs = [env for env in ["py314", "py314t"] if env in set(sub["env_label"])]
    if not envs:
        envs = sorted(sub["env_label"].unique())
    x = np.arange(len(order))
    width = 0.34 if len(envs) > 1 else 0.5
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    fig.patch.set_facecolor(PAPER)
    colors = [BLUE, BERRY, GREEN]
    for idx, env in enumerate(envs):
        vals = []
        errs = []
        for workload in order:
            row = sub[(sub["env_label"].eq(env)) & (sub["workload"].eq(workload))]
            vals.append(float(row["median_wall_time_sec"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["iqr_wall_time_sec"].iloc[0]) if not row.empty else 0.0)
        offset = (idx - (len(envs) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, yerr=errs, capsize=4, color=colors[idx % len(colors)], label=env)
    ax.set_title("Single-thread negative controls", loc="left", fontweight=900)
    ax.set_ylabel("median wall time (s), IQR error bar")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color=LINE, alpha=0.55)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor=LINE)
    fig.tight_layout()
    _finish(fig, out_dir, "python314_single_thread_negative_controls")


def plot_thread_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["experiment"].eq("thread_scaling")].copy()
    if sub.empty:
        return
    envs = [env for env in ["py314", "py314t"] if env in set(sub["env_label"])]
    if not envs:
        envs = sorted(sub["env_label"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    fig.patch.set_facecolor(PAPER)
    colors = {env: color for env, color in zip(envs, [BLUE, BERRY, GREEN, ORANGE])}
    for env in envs:
        env_df = sub[sub["env_label"].eq(env)].sort_values("workers")
        workers = env_df["workers"].to_numpy(float)
        times = env_df["median_wall_time_sec"].to_numpy(float)
        iqr = env_df["iqr_wall_time_sec"].to_numpy(float)
        axes[0].plot(workers, times, marker="o", linewidth=2.4, color=colors[env], label=env)
        axes[0].fill_between(workers, times - iqr / 2.0, times + iqr / 2.0, color=colors[env], alpha=0.16, linewidth=0)
        baseline = float(times[workers == 1][0]) if np.any(workers == 1) else float(times[0])
        speedup = baseline / times
        axes[1].plot(workers, speedup, marker="o", linewidth=2.4, color=colors[env], label=env)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.grid(axis="y", color=LINE, alpha=0.55)
    axes[0].set_title("A. Runtime", loc="left", fontweight=900)
    axes[0].set_ylabel("median wall time (s)")
    axes[0].set_xlabel("ThreadPoolExecutor workers")
    axes[1].set_title("B. Speedup vs workers=1", loc="left", fontweight=900)
    axes[1].set_ylabel("speedup")
    axes[1].set_xlabel("ThreadPoolExecutor workers")
    axes[1].axhline(1.0, color=INK, linewidth=1.0, alpha=0.5)
    axes[0].legend(frameon=True, facecolor="#FFFFFF", edgecolor=LINE)
    _finish(fig, out_dir, "python314_thread_scaling")


def plot_memory(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["experiment"].eq("pool_memory_runtime")].copy()
    if sub.empty:
        return
    sub["label"] = sub["env_label"].astype(str) + " " + sub["pool"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(PAPER)
    x = np.arange(len(sub))
    labels = sub["label"].tolist()
    palette = [BLUE, BERRY, GREEN, ORANGE]
    colors = [palette[idx % len(palette)] for idx in range(len(sub))]
    axes[0].bar(x, sub["median_wall_time_sec"], color=colors)
    axes[0].set_title("A. Runtime", loc="left", fontweight=900)
    axes[0].set_ylabel("median wall time (s)")
    axes[1].bar(x, sub["max_peak_rss_gb"], color=colors)
    axes[1].set_title("B. Peak RSS", loc="left", fontweight=900)
    axes[1].set_ylabel("max peak RSS (GiB)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", color=LINE, alpha=0.55)
    _finish(fig, out_dir, "python314_pool_memory_runtime")


def plot_contention(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["experiment"].eq("contention_backup")].copy()
    if sub.empty:
        return
    envs = [env for env in ["py314", "py314t"] if env in set(sub["env_label"])]
    if not envs:
        envs = sorted(sub["env_label"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(5.4 * len(envs), 4.5), squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor(PAPER)
    for ax, env in zip(axes[0], envs):
        env_df = sub[sub["env_label"].eq(env)]
        for workload, color in zip(["thread_local", "shared_counter", "shared_list", "shared_dict"], [BLUE, BERRY, GREEN, ORANGE]):
            part = env_df[env_df["workload"].eq(workload)].sort_values("workers")
            if part.empty:
                continue
            ax.plot(part["workers"], part["median_wall_time_sec"], marker="o", linewidth=2.0, color=color, label=workload.replace("_", " "))
        ax.set_title(env, loc="left", fontweight=900)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("workers")
        ax.set_ylabel("median wall time (s)")
        ax.grid(axis="y", color=LINE, alpha=0.55)
        ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor=LINE)
    _finish(fig, out_dir, "python314_contention_backup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results/python314_interpreter_effects/latest"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    _setup_style()
    out_dir = args.out_dir or (args.results_dir / "figures")
    df = _read_summaries(args.results_dir)
    plot_negative(df, out_dir)
    plot_thread_scaling(df, out_dir)
    plot_memory(df, out_dir)
    plot_contention(df, out_dir)
    print(f"wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
