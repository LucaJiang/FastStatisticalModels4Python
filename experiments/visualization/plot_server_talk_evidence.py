"""Create 16:9 talk-ready summaries from the curated server/A100 results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fsm4py-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/fsm4py-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SLIDE_FIGSIZE = (12.8, 7.2)
DPI = 220
COLORS = {
    "numba": "#2F7D32",
    "numpy_matmul": "#2B6CB0",
    "a100": "#B51E59",
    "cpu": "#17202A",
    "thread": "#1F7A8C",
    "memory": "#E66A2C",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 19,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 2.8,
            "lines.markersize": 7,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["validation_status"].isin(["pass", "check"])].copy()


def plot_kmeans(cpu_dir: Path, a100_dir: Path, out_dir: Path) -> None:
    cpu = _clean(pd.read_csv(cpu_dir / "kmeans_cpu_scaling.csv"))
    gpu = _clean(pd.read_csv(a100_dir / "kmeans_jax_gpu.csv"))

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    fig.patch.set_facecolor("#FBF7EF")
    ax.set_facecolor("#FFFFFF")

    cpu_slice = cpu[(cpu["k"] == 20) & (cpu["separation"] == 2.0)]
    best_cpu = (
        cpu_slice.groupby(["n", "d", "k", "implementation"], as_index=False)["warm_median_s"]
        .median()
        .sort_values("warm_median_s")
        .groupby(["n", "d", "k"], as_index=False)
        .first()
        .rename(columns={"warm_median_s": "best_cpu_s"})
    )
    gpu_med = (
        gpu[gpu["k"] == 20]
        .groupby(["n", "d", "k"], as_index=False)["warm_median_s"]
        .median()
        .rename(columns={"warm_median_s": "a100_s"})
    )
    merged = best_cpu.merge(gpu_med, on=["n", "d", "k"])
    merged["cpu_over_a100"] = merged["best_cpu_s"] / merged["a100_s"]
    colors = {10: "#2B6CB0", 64: "#E66A2C", 256: "#2F7D32"}
    for d in [10, 64, 256]:
        group = merged[merged["d"] == d].sort_values("n")
        if group.empty:
            continue
        group = group.sort_values("n")
        ax.plot(
            group["n"],
            group["cpu_over_a100"],
            marker="o",
            linewidth=3.2,
            markersize=8,
            color=colors[int(d)],
        )
        last = group.iloc[-1]
        label_x = last["n"] * 1.04
        label_y = last["cpu_over_a100"]
        label = f"d={int(d)}"
        ax.text(label_x, label_y, label, color=colors[int(d)], fontsize=16, fontweight="bold", va="center")

    ax.axhline(1.0, color="#5F6B74", linestyle="--", linewidth=2.2)
    ax.text(merged["n"].min(), 1.06, "break-even", color="#5F6B74", fontsize=15, fontweight="bold", va="bottom")
    ax.axhspan(1.0, max(6.4, merged["cpu_over_a100"].max() * 1.08), color="#F7E6D9", alpha=0.45, zorder=0)
    ax.text(
        120_000,
        5.65,
        "A100 faster\nthan best CPU",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=COLORS["a100"],
    )
    ax.text(
        120_000,
        0.74,
        "CPU faster",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#5F6B74",
    )

    ax.set_xscale("log")
    ax.set_ylim(0.65, max(6.4, merged["cpu_over_a100"].max() * 1.10))
    ax.set_xlim(merged["n"].min() * 0.8, merged["n"].max() * 1.45)
    ax.set_xlabel("N samples")
    ax.set_ylabel("Best CPU runtime / A100 runtime")
    ax.set_title("A100 becomes worthwhile only for some k-means shapes", weight="bold", pad=14)
    _style_axis(ax)

    max_row = merged.loc[merged["cpu_over_a100"].idxmax()]
    fig.text(
        0.10,
        0.93,
        "Server k-means: best CPU baseline vs A100",
        fontsize=20,
        weight="bold",
        color="#17202A",
    )
    fig.text(
        0.10,
        0.87,
        f"k=20, separation=2.0; maximum observed A100 advantage {max_row['cpu_over_a100']:.1f}x.",
        fontsize=15,
        color="#5F6B74",
    )
    fig.text(
        0.10,
        0.07,
        "A100 helps only after enough regular work exists.",
        fontsize=18,
        weight="bold",
        color="#17202A",
        bbox={"boxstyle": "round,pad=0.55,rounding_size=0.15", "facecolor": "#F7E6D9", "edgecolor": "none"},
    )
    fig.tight_layout(rect=[0.08, 0.13, 0.96, 0.84])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "server_kmeans_a100_breakeven.png", dpi=DPI)
    fig.savefig(out_dir / "server_kmeans_a100_breakeven.svg", format="svg")
    fig.savefig(out_dir / "server_kmeans_cpu_a100_summary.png", dpi=DPI)
    fig.savefig(out_dir / "server_kmeans_cpu_a100_summary.svg", format="svg")
    plt.close(fig)


def plot_permutation(cpu_dir: Path, a100_dir: Path, out_dir: Path) -> None:
    cpu_raw = pd.read_csv(cpu_dir / "permutation_cpu_scaling.csv")
    gpu = _clean(pd.read_csv(a100_dir / "permutation_matrix_gpu.csv"))
    cpu = _clean(cpu_raw)

    fig = plt.figure(figsize=SLIDE_FIGSIZE)
    fig.patch.set_facecolor("#FBF7EF")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 0.55], left=0.20, right=0.90, top=0.73, bottom=0.16, hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    pipe_ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#FFFFFF")
    pipe_ax.set_facecolor("#FBF7EF")

    cpu_match = (
        cpu[(cpu["n"] == 5000) & (cpu["p"] == 50000) & (cpu["batch_R"] == 512)]
        .groupby("R", as_index=False)["warm_median_s"]
        .median()
        .rename(columns={"warm_median_s": "cpu_s"})
    )
    gpu_match = (
        gpu[(gpu["n"] == 5000) & (gpu["p"] == 50000) & (gpu["batch_R"] == 512)]
        .groupby("R", as_index=False)["warm_median_s"]
        .median()
        .rename(columns={"warm_median_s": "a100_s"})
    )
    matched = cpu_match.merge(gpu_match, on="R").sort_values("R")
    matched["a100_over_cpu"] = matched["a100_s"] / matched["cpu_s"]

    y_positions: list[float] = []
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    annotations: list[str] = []
    base = len(matched) * 2
    for i, (_, row) in enumerate(matched.iterrows()):
        y_cpu = base - i * 2
        y_gpu = y_cpu - 0.62
        y_positions.extend([y_cpu, y_gpu])
        labels.extend([f"R={int(row['R']):,}  CPU", f"R={int(row['R']):,}  A100"])
        values.extend([float(row["cpu_s"]), float(row["a100_s"])])
        colors.extend([COLORS["cpu"], COLORS["a100"]])
        annotations.extend([f"{row['cpu_s']:.1f}s", f"{row['a100_s']:.1f}s  ({row['a100_over_cpu']:.1f}x slower)"])

    ax.barh(y_positions, values, color=colors, height=0.46)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 180.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=17)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(True, axis="x", which="major", alpha=0.28)
    ax.grid(True, axis="x", which="minor", alpha=0.10)
    _style_axis(ax)
    for y, value, label, color in zip(y_positions, values, annotations, colors):
        ax.text(
            value * 1.08,
            y,
            label,
            ha="left",
            va="center",
            fontsize=15.5,
            color=color,
            fontweight="bold",
        )

    pipe_ax.axis("off")
    pipe_ax.text(
        0.00,
        0.72,
        "Measured GPU path: build W + transfer + W @ X + collect",
        ha="left",
        va="center",
        fontsize=17,
        color="#17202A",
        fontweight="bold",
        bbox=dict(facecolor="#F7E7D8", edgecolor="none", boxstyle="round,pad=0.35"),
    )
    pipe_ax.text(
        0.00,
        0.18,
        "Current CSV is end-to-end only; stage decomposition is the next bottleneck measurement.",
        ha="left",
        va="center",
        fontsize=13.5,
        color="#5F6B74",
    )

    timeouts = int((cpu_raw["validation_status"] == "timeout").sum())
    fig.text(0.06, 0.93, "Server permutation benchmark", ha="left", va="top", fontsize=27, fontweight="bold", color="#17202A")
    fig.text(
        0.06,
        0.85,
        f"Matched slice: n=5,000, p=50,000, batch_R=512; A100 float32 includes transfer. CPU timeouts: {timeouts}.",
        ha="left",
        va="bottom",
        fontsize=15,
        color="#5F6B74",
    )
    fig.text(
        0.08,
        0.035,
        "Negative GPU evidence is still useful evidence.",
        ha="left",
        va="center",
        fontsize=20,
        color="#17202A",
        fontweight="bold",
        bbox=dict(facecolor="#F7E7D8", edgecolor="none", boxstyle="round,pad=0.36"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "server_permutation_cpu_a100_summary.png", dpi=DPI)
    fig.savefig(out_dir / "server_permutation_cpu_a100_summary.svg", format="svg")
    plt.close(fig)


def plot_parallelism(cpu_dir: Path, out_dir: Path) -> None:
    threads = _clean(pd.read_csv(cpu_dir / "kmeans_numba_thread_sweep.csv")).sort_values("threads")
    workers = _clean(pd.read_csv(cpu_dir / "permutation_worker_sweep.csv")).sort_values("workers")

    fig, axes = plt.subplots(1, 2, figsize=SLIDE_FIGSIZE)
    fig.patch.set_facecolor("#FBF7EF")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")

    runtime_ax, mem_ax = axes
    runtime_ax.plot(
        threads["threads"],
        threads["warm_median_s"],
        marker="o",
        linewidth=3.2,
        markersize=9,
        color=COLORS["thread"],
    )
    runtime_ax.plot(
        workers["workers"],
        workers["warm_median_s"],
        marker="o",
        linewidth=3.2,
        markersize=9,
        color=COLORS["memory"],
    )
    runtime_ax.set_xscale("log", base=2)
    runtime_ax.set_yscale("log")
    runtime_ax.set_title("Runtime has a best region", weight="bold", fontsize=20)
    runtime_ax.set_xlabel("Threads / workers", fontsize=19)
    runtime_ax.set_ylabel("Runtime (seconds)", fontsize=19)
    runtime_ax.tick_params(labelsize=16)
    _style_axis(runtime_ax)

    mem_ax.plot(
        threads["threads"],
        threads["host_peak_mem_mb"] / 1024.0,
        marker="o",
        linewidth=3.2,
        markersize=9,
        color=COLORS["thread"],
    )
    mem_ax.plot(
        workers["workers"],
        workers["host_peak_mem_mb"] / 1024.0,
        marker="o",
        linewidth=3.2,
        markersize=9,
        color=COLORS["memory"],
    )
    mem_ax.set_xscale("log", base=2)
    mem_ax.set_title("Memory can keep rising", weight="bold", fontsize=20)
    mem_ax.set_xlabel("Threads / workers", fontsize=19)
    mem_ax.set_ylabel("Memory (GiB)", fontsize=19)
    mem_ax.tick_params(labelsize=16)
    _style_axis(mem_ax)

    best_threads = int(threads.loc[threads["warm_median_s"].idxmin(), "threads"])
    best_workers = int(workers.loc[workers["warm_median_s"].idxmin(), "workers"])
    best_thread_time = float(threads["warm_median_s"].min())
    best_worker_time = float(workers["warm_median_s"].min())
    runtime_ax.scatter([best_threads], [best_thread_time], s=210, color=COLORS["thread"], edgecolor="white", linewidth=2.0, zorder=5)
    runtime_ax.scatter([best_workers], [best_worker_time], s=210, color=COLORS["memory"], edgecolor="white", linewidth=2.0, zorder=5)
    runtime_ax.text(best_threads * 1.12, best_thread_time * 1.12, f"best: {best_threads}", fontsize=14, color=COLORS["thread"], fontweight="bold")
    runtime_ax.text(best_workers * 1.10, best_worker_time * 1.20, f"best: {best_workers}", fontsize=14, color=COLORS["memory"], fontweight="bold")
    runtime_ax.text(1.1, 30.0, "k-means Numba", color=COLORS["thread"], fontsize=15, fontweight="bold")
    runtime_ax.text(1.1, 16.0, "permutation ThreadPool", color=COLORS["memory"], fontsize=15, fontweight="bold")
    mem_ax.text(
        0.48,
        0.18,
        "worker memory rises\nafter best runtime",
        transform=mem_ax.transAxes,
        color=COLORS["memory"],
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    mem_ax.text(
        0.04,
        0.88,
        "k-means Numba",
        transform=mem_ax.transAxes,
        color=COLORS["thread"],
        fontsize=16,
        fontweight="bold",
    )
    mem_ax.text(
        0.04,
        0.79,
        "permutation ThreadPool",
        transform=mem_ax.transAxes,
        color=COLORS["memory"],
        fontsize=16,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.75, bottom=0.24, wspace=0.25)
    fig.text(0.06, 0.93, "Parallelism tradeoff", ha="left", va="top", fontsize=31, fontweight="bold", color="#17202A")
    fig.text(
        0.06,
        0.85,
        f"Measured server sweeps: best k-means at {best_threads} threads; best permutation at {best_workers} workers, not at 128.",
        ha="left",
        va="bottom",
        fontsize=17,
        color="#5F6B74",
    )
    fig.text(
        0.08,
        0.07,
        "Parallelism is a tuning parameter, not a moral victory.",
        ha="left",
        va="center",
        fontsize=23,
        color="#17202A",
        fontweight="bold",
        bbox=dict(facecolor="#F7E7D8", edgecolor="none", boxstyle="round,pad=0.36"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "server_parallelism_tradeoff.png", dpi=DPI)
    fig.savefig(out_dir / "server_parallelism_tradeoff.svg", format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, default=Path("experiments/results/linux_server_cpu/long_safe_20260503_190133"))
    parser.add_argument("--a100-dir", type=Path, default=Path("experiments/results/linux_server_a100/long_safe_20260503_190133"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/presentation_figures"))
    args = parser.parse_args()

    _apply_style()
    plot_kmeans(args.cpu_dir, args.a100_dir, args.out_dir)
    plot_permutation(args.cpu_dir, args.a100_dir, args.out_dir)
    plot_parallelism(args.cpu_dir, args.out_dir)


if __name__ == "__main__":
    main()
