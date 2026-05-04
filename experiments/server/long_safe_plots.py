#!/usr/bin/env python3
"""Plot long-safe server experiment results with explicit scenario slices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563", "#be123c"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["n", "p", "d", "k", "R", "batch_R", "threads", "workers", "warm_median_s", "cold_time_s", "host_peak_mem_mb", "estimated_host_gib"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "validation_status" in df.columns:
        df = df[df["validation_status"].isin(["pass", "check"])].copy()
    return df


def _fig_dir(root: Path) -> Path:
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _finalize(plt, path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _label_shape(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for col in cols:
        val = row[col]
        if pd.notna(val):
            parts.append(f"{col}={int(val)}")
    return ", ".join(parts)


def _power_slope(x: pd.Series, y: pd.Series) -> tuple[float, float] | None:
    valid = x.notna() & y.notna() & (x > 0) & (y > 0)
    if int(valid.sum()) < 3:
        return None
    lx = np.log10(x[valid].to_numpy(dtype=float))
    ly = np.log10(y[valid].to_numpy(dtype=float))
    slope, intercept = np.polyfit(lx, ly, deg=1)
    return float(slope), float(intercept)


def _plot_scaling_series(
    ax,
    x: pd.Series,
    y: pd.Series,
    label: str,
    color: str,
    marker: str = "o",
    fit: bool = True,
    connect: bool = True,
) -> None:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    frame = frame.dropna()
    if frame.empty:
        return
    frame = frame.groupby("x", as_index=False)["y"].median().sort_values("x")
    xs = frame["x"].to_numpy(dtype=float)
    ys = frame["y"].to_numpy(dtype=float)
    fit_params = _power_slope(pd.Series(xs), pd.Series(ys)) if fit else None
    fit_label = f"{label} (slope {fit_params[0]:.2f})" if fit_params else label
    if connect and len(xs) > 1:
        ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.55)
    ax.scatter(xs, ys, s=34, marker=marker, color=color, edgecolors="white", linewidths=0.7, alpha=0.78, label=fit_label, zorder=3)
    if fit_params:
        slope, intercept = fit_params
        fit_x = np.geomspace(xs[xs > 0].min(), xs.max(), 80)
        fit_y = 10 ** (intercept + slope * np.log10(fit_x))
        ax.plot(fit_x, fit_y, color=color, linestyle="--", linewidth=1.2, alpha=0.65)


def plot_kmeans_cpu(cpu_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = _fig_dir(cpu_dir)
    data = _read_csv(cpu_dir / "kmeans_cpu_scaling.csv")
    fixed = data[(data["k"] == 20) & (data["separation"] == 2.0)].copy()
    if fixed.empty:
        fixed = data.copy()
    agg = (
        fixed.groupby(["implementation", "d", "n"], as_index=False)
        .agg(warm_median_s=("warm_median_s", "median"), warm_iqr_s=("warm_iqr_s", "median"))
        .sort_values(["d", "implementation", "n"])
    )
    if not agg.empty:
        dims = sorted(agg["d"].dropna().unique())
        fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 3.7), sharey=True)
        if len(dims) == 1:
            axes = [axes]
        for ax, d in zip(axes, dims):
            part = agg[agg["d"] == d]
            for i, (impl, g) in enumerate(part.groupby("implementation")):
                _plot_scaling_series(ax, g["n"], g["warm_median_s"], impl, COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"d={int(d)}, K=20")
            ax.set_xlabel("N")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best", fontsize=7)
        axes[0].set_ylabel("warm median seconds")
        fig.suptitle("CPU k-means runtime by dimension")
        _finalize(plt, fig_dir / "kmeans_cpu_runtime.png")

    threads = _read_csv(cpu_dir / "kmeans_numba_thread_sweep.csv")
    if not threads.empty:
        agg = threads.groupby("threads", as_index=False)["warm_median_s"].median().sort_values("threads")
        base = float(agg.loc[agg["threads"] == agg["threads"].min(), "warm_median_s"].iloc[0])
        fig, ax1 = plt.subplots(figsize=(7, 4))
        _plot_scaling_series(ax1, agg["threads"], agg["warm_median_s"], "runtime", "#2563eb", "o", fit=False)
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax1.set_xlabel("NUMBA_NUM_THREADS")
        ax1.set_ylabel("warm median seconds", color="#2563eb")
        ax1.grid(True, which="both", alpha=0.25)
        ax2 = ax1.twinx()
        speed = base / agg["warm_median_s"]
        ax2.plot(agg["threads"], speed, color="#b45309", linewidth=1.4, alpha=0.6)
        ax2.scatter(agg["threads"], speed, s=34, marker="s", color="#b45309", edgecolors="white", linewidths=0.7, alpha=0.78, zorder=3)
        ax2.set_ylabel("speedup vs 1 thread", color="#b45309")
        ax1.set_title("Numba k-means thread sweep")
        _finalize(plt, fig_dir / "kmeans_numba_threads.png")

    mem = data[data["host_peak_mem_mb"].notna() & data["estimated_host_gib"].notna()].copy()
    if not mem.empty:
        agg = (
            mem.groupby(["implementation", "estimated_host_gib"], as_index=False)["host_peak_mem_mb"]
            .median()
            .sort_values(["implementation", "estimated_host_gib"])
        )
        plt.figure(figsize=(7, 4))
        ax = plt.gca()
        for i, (impl, g) in enumerate(agg.groupby("implementation")):
            _plot_scaling_series(
                ax,
                g["estimated_host_gib"],
                g["host_peak_mem_mb"],
                impl,
                COLORS[i % len(COLORS)],
                MARKERS[i % len(MARKERS)],
                connect=False,
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("estimated host working set GiB")
        plt.ylabel("child RSS MB at scenario end")
        plt.title("k-means host memory scaling")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend(fontsize=8)
        _finalize(plt, fig_dir / "kmeans_memory_scaling.png")


def plot_kmeans_a100(cpu_dir: Path, a100_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = _fig_dir(a100_dir)
    gpu = _read_csv(a100_dir / "kmeans_jax_gpu.csv")
    fixed = gpu[gpu["k"] == 20].copy()
    if fixed.empty:
        fixed = gpu.copy()
    agg = (
        fixed.groupby(["d", "n"], as_index=False)
        .agg(warm_median_s=("warm_median_s", "median"), cold_time_s=("cold_time_s", "median"))
        .sort_values(["d", "n"])
    )
    if not agg.empty:
        dims = sorted(agg["d"].dropna().unique())
        fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 3.7), sharey=True)
        if len(dims) == 1:
            axes = [axes]
        for ax, d in zip(axes, dims):
            g = agg[agg["d"] == d]
            _plot_scaling_series(ax, g["n"], g["cold_time_s"], "cold", COLORS[1], "s")
            _plot_scaling_series(ax, g["n"], g["warm_median_s"], "warm", COLORS[0], "o")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"d={int(d)}, K=20")
            ax.set_xlabel("N")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best", fontsize=7)
        axes[0].set_ylabel("seconds")
        fig.suptitle("A100 JAX k-means cold vs warm")
        _finalize(plt, fig_dir / "kmeans_jax_cold_vs_warm.png")

    cpu_path = cpu_dir / "kmeans_cpu_scaling.csv"
    if cpu_path.exists() and not gpu.empty:
        cpu = _read_csv(cpu_path)
        cpu = cpu[(cpu["k"] == 20) & (cpu["separation"] == 2.0)].copy()
        gpu_match = gpu[gpu["k"] == 20].copy()
        cpu_best = (
            cpu.groupby(["n", "d", "implementation"], as_index=False)["warm_median_s"].median()
            .sort_values("warm_median_s")
            .groupby(["n", "d"], as_index=False)
            .first()
        )
        gpu_med = gpu_match.groupby(["n", "d"], as_index=False)["warm_median_s"].median()
        merged = cpu_best.merge(gpu_med, on=["n", "d"], suffixes=("_cpu", "_gpu"))
        if not merged.empty:
            merged["speedup"] = merged["warm_median_s_cpu"] / merged["warm_median_s_gpu"]
            dims = sorted(merged["d"].dropna().unique())
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for i, d in enumerate(dims):
                g = merged[merged["d"] == d].sort_values("n")
                color = COLORS[i % len(COLORS)]
                _plot_scaling_series(axes[0], g["n"], g["warm_median_s_cpu"], f"CPU d={int(d)}", color, "o")
                _plot_scaling_series(axes[0], g["n"], g["warm_median_s_gpu"], f"A100 d={int(d)}", color, "s")
                _plot_scaling_series(axes[1], g["n"], g["speedup"], f"d={int(d)}", color, "o", fit=False)
            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("N")
            axes[0].set_ylabel("warm median seconds")
            axes[0].set_title("best CPU vs A100")
            axes[1].set_xscale("log")
            axes[1].axhline(1.0, color="black", linewidth=1, alpha=0.4)
            axes[1].set_xlabel("N")
            axes[1].set_ylabel("CPU / A100 warm time")
            axes[1].set_title("break-even ratio")
            for ax in axes:
                ax.grid(True, which="both", alpha=0.25)
            axes[0].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
            axes[1].legend(fontsize=7, loc="best")
            fig.suptitle("k-means CPU/GPU break-even, K=20")
            _finalize(plt, fig_dir / "kmeans_cpu_gpu_break_even.png")


def plot_permutation_cpu(cpu_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = _fig_dir(cpu_dir)
    data = _read_csv(cpu_dir / "permutation_cpu_scaling.csv")
    ok = data[(data["batch_R"] == 512)].copy()
    if not ok.empty:
        selected = ok[ok["p"].isin([1000, 10000, 50000])]
        agg = (
            selected.groupby(["n", "p", "R"], as_index=False)["warm_median_s"]
            .median()
            .sort_values(["n", "p", "R"])
        )
        labels = agg[["n", "p"]].drop_duplicates().sort_values(["p", "n"]).tail(8)
        plt.figure(figsize=(8, 4.5))
        ax = plt.gca()
        for i, (_, shape) in enumerate(labels.iterrows()):
            g = agg[(agg["n"] == shape["n"]) & (agg["p"] == shape["p"])]
            _plot_scaling_series(ax, g["R"], g["warm_median_s"], _label_shape(shape, ["n", "p"]), COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("R permutations")
        plt.ylabel("warm median seconds")
        plt.title("CPU permutation runtime by shape")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        _finalize(plt, fig_dir / "permutation_cpu_runtime.png")

    workers = _read_csv(cpu_dir / "permutation_worker_sweep.csv")
    if not workers.empty:
        agg = workers.groupby("workers", as_index=False).agg(warm_median_s=("warm_median_s", "median"), host_peak_mem_mb=("host_peak_mem_mb", "median")).sort_values("workers")
        base = float(agg.loc[agg["workers"] == agg["workers"].min(), "warm_median_s"].iloc[0])
        plt.figure(figsize=(7, 4))
        _plot_scaling_series(plt.gca(), agg["workers"], agg["warm_median_s"], "runtime", "#2563eb", "o", fit=False)
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("ThreadPool workers")
        plt.ylabel("warm median seconds")
        plt.title("Permutation worker sweep")
        plt.grid(True, which="both", alpha=0.25)
        _finalize(plt, fig_dir / "permutation_worker_sweep.png")

        fig, ax1 = plt.subplots(figsize=(7, 4))
        _plot_scaling_series(ax1, agg["workers"], agg["host_peak_mem_mb"], "RSS", "#2563eb", "o", fit=False)
        ax1.set_xscale("log", base=2)
        ax1.set_xlabel("ThreadPool workers")
        ax1.set_ylabel("child RSS MB", color="#2563eb")
        ax2 = ax1.twinx()
        speed = base / agg["warm_median_s"]
        ax2.plot(agg["workers"], speed, color="#b45309", linewidth=1.4, alpha=0.6)
        ax2.scatter(agg["workers"], speed, s=34, marker="s", color="#b45309", edgecolors="white", linewidths=0.7, alpha=0.78, zorder=3)
        ax2.set_ylabel("speedup vs 1 worker", color="#b45309")
        ax1.grid(True, which="both", alpha=0.25)
        ax1.set_title("Worker memory and speedup")
        _finalize(plt, fig_dir / "process_vs_thread_memory.png")


def plot_permutation_a100(cpu_dir: Path, a100_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = _fig_dir(a100_dir)
    gpu = _read_csv(a100_dir / "permutation_matrix_gpu.csv")
    if not gpu.empty:
        scaling_p = (
            gpu[(gpu["n"] == 5000) & (gpu["R"] == 5000) & (gpu["batch_R"] == 512)]
            .groupby("p", as_index=False)["warm_median_s"]
            .median()
            .sort_values("p")
        )
        if not scaling_p.empty:
            plt.figure(figsize=(7, 4))
            _plot_scaling_series(plt.gca(), scaling_p["p"], scaling_p["warm_median_s"], "n=5000, R=5000, batch=512", "#2563eb", "o")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("p features")
            plt.ylabel("warm median seconds")
            plt.title("A100 permutation scaling over p")
            plt.grid(True, which="both", alpha=0.25)
            plt.legend(fontsize=8)
            _finalize(plt, fig_dir / "permutation_gpu_runtime.png")

        batch = (
            gpu[(gpu["n"] == 5000) & (gpu["p"] == 50000) & (gpu["R"] == 5000)]
            .groupby("batch_R", as_index=False)["warm_median_s"]
            .median()
            .sort_values("batch_R")
        )
        if not batch.empty:
            plt.figure(figsize=(7, 4))
            _plot_scaling_series(plt.gca(), batch["batch_R"], batch["warm_median_s"], "batch sweep", "#2563eb", "o", fit=False)
            plt.xscale("log", base=2)
            plt.yscale("log")
            plt.xlabel("batch_R")
            plt.ylabel("warm median seconds")
            plt.title("A100 permutation batch sweep")
            plt.grid(True, which="both", alpha=0.25)
            _finalize(plt, fig_dir / "permutation_gpu_batch_sweep.png")

    cpu_path = cpu_dir / "permutation_cpu_scaling.csv"
    if cpu_path.exists() and not gpu.empty:
        cpu = _read_csv(cpu_path)
        cpu_cmp = cpu[(cpu["n"] == 5000) & (cpu["p"] == 50000) & (cpu["batch_R"] == 512)].copy()
        gpu_cmp = gpu[(gpu["n"] == 5000) & (gpu["p"] == 50000) & (gpu["batch_R"] == 512)].copy()
        cpu_agg = cpu_cmp.groupby("R", as_index=False)["warm_median_s"].median()
        gpu_agg = gpu_cmp.groupby("R", as_index=False)["warm_median_s"].median()
        merged = cpu_agg.merge(gpu_agg, on="R", suffixes=("_cpu", "_gpu")).sort_values("R")
        if not merged.empty:
            merged["speedup"] = merged["warm_median_s_cpu"] / merged["warm_median_s_gpu"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            _plot_scaling_series(axes[0], merged["R"], merged["warm_median_s_cpu"], "CPU", COLORS[0], "o")
            _plot_scaling_series(axes[0], merged["R"], merged["warm_median_s_gpu"], "A100", COLORS[1], "s")
            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("R permutations")
            axes[0].set_ylabel("warm median seconds")
            axes[0].set_title("n=5000, p=50000")
            _plot_scaling_series(axes[1], merged["R"], merged["speedup"], "CPU/A100", COLORS[2], "o", fit=False)
            axes[1].axhline(1.0, color="black", linewidth=1, alpha=0.4)
            axes[1].set_xscale("log")
            axes[1].set_xlabel("R permutations")
            axes[1].set_ylabel("CPU / A100 warm time")
            axes[1].set_title("break-even ratio")
            for ax in axes:
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=8) if ax is axes[0] else None
            fig.suptitle("Permutation CPU/GPU comparison")
            _finalize(plt, fig_dir / "permutation_cpu_gpu_break_even.png")

    plt.figure(figsize=(7, 3.5))
    plt.axis("off")
    plt.text(0.02, 0.68, "Permutation test as streamed matrix products", fontsize=17, weight="bold")
    plt.text(0.02, 0.48, "For each batch: W_batch @ X -> statistics over p features", fontsize=12)
    plt.text(0.02, 0.32, "Only exceedance counts are accumulated; full R x p output is never materialized.", fontsize=12)
    plt.text(0.02, 0.16, "This is the computational reason GPU batching matters for large p and R.", fontsize=12)
    _finalize(plt, fig_dir / "permutation_matrix_reformulation.png")


def plot_all(cpu_dir: Path, a100_dir: Path) -> None:
    plot_kmeans_cpu(cpu_dir)
    plot_kmeans_a100(cpu_dir, a100_dir)
    plot_permutation_cpu(cpu_dir)
    plot_permutation_a100(cpu_dir, a100_dir)
