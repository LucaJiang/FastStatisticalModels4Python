"""Create the simplified slide-specific Linux server CPU parallelism figure."""

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


COUNTS = [1, 4, 16, 64, 128]
DPI = 220

INK = "#17202A"
PAPER = "#FBF7EF"
LINE = "#D7CDC0"
BLUE = "#2368AD"
BERRY = "#B51E59"
ORANGE = "#E66A2C"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _completed(df: pd.DataFrame, count_col: str) -> pd.DataFrame:
    return df[df["row_status"].eq("completed")].copy().sort_values(count_col)


def _format_log2_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.78, 160)
    ax.set_xticks(COUNTS, [str(value) for value in COUNTS])
    ax.tick_params(axis="x", pad=4)


def _saved_percent_of_baseline(times: np.ndarray) -> np.ndarray:
    """Runtime saved by each count jump, expressed as percent of count=1 time."""

    return (times[:-1] - times[1:]) / times[0] * 100.0


def plot_cpu_parallelism_slide(out_dir: Path, presentation_dir: Path) -> None:
    """Write slide-optimized PNG/SVG artifacts from the expanded sweep CSVs."""

    presentation_dir.mkdir(parents=True, exist_ok=True)
    kmeans = _completed(pd.read_csv(out_dir / "kmeans_parallelism_expanded.csv"), "thread_count")
    perm = _completed(pd.read_csv(out_dir / "permutation_parallelism_expanded.csv"), "worker_count")

    k_x = kmeans["thread_count"].astype(int).to_numpy()
    k_y = _numeric(kmeans["median_warm_time_s"]).to_numpy(float)
    p_x = perm["worker_count"].astype(int).to_numpy()
    p_y = _numeric(perm["median_warm_time_s"]).to_numpy(float)
    p_mem_gib = _numeric(perm["total_peak_rss_mb"]).to_numpy(float) / 1000.0
    p_mem_128 = float(p_mem_gib[p_x == 128][0])

    jump_labels = ["1→4", "4→16", "16→64", "64→128"]
    k_saved = _saved_percent_of_baseline(k_y)
    p_saved = _saved_percent_of_baseline(p_y)

    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )

    fig = plt.figure(figsize=(12.8, 4.05), constrained_layout=False)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.18)
    ax_time = fig.add_subplot(gs[0, 0])
    ax_gain = fig.add_subplot(gs[0, 1])

    for ax in [ax_time, ax_gain]:
        ax.grid(axis="y", color="#D9D0C4", alpha=0.55, linewidth=0.9)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(LINE)

    ax_time.plot(k_x, k_y, marker="o", linewidth=3.4, markersize=8.5, color=BLUE, label="k-means Numba threads")
    ax_time.plot(p_x, p_y, marker="s", linewidth=3.4, markersize=8.0, color=BERRY, label="permutation process workers")
    _format_log2_axis(ax_time)
    ax_time.set_yscale("log")
    ax_time.set_ylim(0.65, 26.0)
    ax_time.set_title("A. Total time", loc="left", fontweight=900, pad=10)
    ax_time.set_xlabel("parallelism count")
    ax_time.set_ylabel("runtime (s, log)")
    ax_time.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor=LINE, framealpha=0.96)
    ax_time.text(
        0.04,
        0.08,
        f"128 permutation workers\nused ~{p_mem_128:.0f} GiB peak RSS",
        transform=ax_time.transAxes,
        color=ORANGE,
        fontsize=12.0,
        fontweight=900,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF4EC", "edgecolor": ORANGE, "linewidth": 1.0},
    )

    x = np.arange(len(jump_labels))
    width = 0.34
    ax_gain.axhline(0, color=INK, linewidth=1.1, alpha=0.7)
    ax_gain.bar(x - width / 2, k_saved, width=width, color=BLUE, label="k-means")
    ax_gain.bar(x + width / 2, p_saved, width=width, color=BERRY, label="permutation")
    ax_gain.set_title("B. Diminishing returns", loc="left", fontweight=900, pad=10)
    ax_gain.set_xticks(x, jump_labels)
    ax_gain.set_ylabel("runtime saved by jump\n(% of 1-count time)")
    ax_gain.set_ylim(-12, 66)
    ax_gain.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor=LINE, framealpha=0.96)

    for values, dx, color in [(k_saved, -width / 2, BLUE), (p_saved, width / 2, BERRY)]:
        for xi, value in zip(x, values):
            va = "bottom" if value >= 0 else "top"
            offset = 1.2 if value >= 0 else -1.6
            ax_gain.text(
                xi + dx,
                value + offset,
                f"{value:+.0f}%",
                ha="center",
                va=va,
                fontsize=11.5,
                color=color,
                fontweight=850,
            )

    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.18)
    fig.savefig(presentation_dir / "server_cpu_parallelism_simple.png", dpi=DPI)
    fig.savefig(presentation_dir / "server_cpu_parallelism_simple.svg", format="svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/results/linux_server_cpu/parallelism_expanded"),
        help="Directory containing the expanded parallelism CSVs.",
    )
    parser.add_argument(
        "--presentation-dir",
        type=Path,
        default=Path("experiments/results/presentation_figures"),
        help="Directory for slide-ready PNG/SVG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_cpu_parallelism_slide(args.out_dir, args.presentation_dir)


if __name__ == "__main__":
    main()
