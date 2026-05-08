"""Create deck-ready MacBook Air evidence figures from validation CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fsm4py-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/fsm4py-cache")

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from textwrap import fill

DEFAULT_ROOT = Path("experiments/results/macbook_air_long/latest")
SLIDE_FIGSIZE = (12.8, 7.2)
DPI = 220

PY_BLUE = "#2B6CB0"
PY_GOLD = "#D99A1E"
NUMBA_GREEN = "#2F7D32"
JAX_BERRY = "#B51E59"
THREAD_TEAL = "#1F7A8C"
PROCESS_ORANGE = "#E66A2C"
LOOP_PURPLE = "#7046A1"
INK = "#17202A"
MUTED = "#5F6B74"
PAPER = "#FBF7EF"

COLORS = {
    "reference": LOOP_PURPLE,
    "reference_loop": LOOP_PURPLE,
    "numpy_matmul": PY_BLUE,
    "numpy_matrix": PY_BLUE,
    "numpy_matrix_batched": THREAD_TEAL,
    "numba": NUMBA_GREEN,
    "jax_matrix_cpu": JAX_BERRY,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7CDC0",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#D7CDC0",
            "grid.alpha": 0.35,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 2.8,
            "lines.markersize": 7,
        }
    )


def strip_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)


def compact_label(value: float | int) -> str:
    value = float(value)
    if value >= 1000 and value.is_integer():
        return f"{int(value / 1000)}k"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def add_panel_caption(ax: plt.Axes, text: str, *, y: float = -0.33) -> None:
    ax.text(
        0,
        y,
        fill(text, 54),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        color=MUTED,
        linespacing=1.12,
    )


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _status_column(df: pd.DataFrame) -> str:
    if "status" in df:
        return "status"
    return "correctness_status"


def _save(fig, path: Path, manifest: list[dict[str, str]], source: str, purpose: str, *, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    manifest.append({"figure": path.name, "source": source, "purpose": purpose})


def _kmeans_recovery_grid(root: Path) -> tuple[list[float], list[tuple[str, dict[str, object]]], np.ndarray]:
    df = _read(root / "kmeans_correctness.csv")
    if df.empty:
        return [], [], np.array([])
    df = _numeric(df, ["d", "separation", "ari_true", "outlier_fraction"])
    status = _status_column(df)
    df = df[(df["implementation"] == "numpy_matmul") & (df[status] == "pass")].copy()
    if df.empty:
        return [], [], np.array([])

    scenario_specs: list[tuple[str, dict[str, object]]] = [
        ("Clean\nlow-d", {"imbalance": "balanced", "outlier_fraction": 0.0, "d": 2}),
        ("Clean\nhigh-d", {"imbalance": "balanced", "outlier_fraction": 0.0, "d": 50}),
        ("Outliers\n1%", {"imbalance": "balanced", "outlier_fraction": 0.01, "d": 10}),
        ("Imbalance\n90/10", {"imbalance": "90_10", "outlier_fraction": 0.0, "d": 10}),
        ("Imbalance\n+ outliers", {"imbalance": "90_10", "outlier_fraction": 0.01, "d": 10}),
    ]
    separations = [0.5, 1.0, 2.0, 4.0]
    median = np.full((len(separations), len(scenario_specs)), np.nan)

    for col, (_, filters) in enumerate(scenario_specs):
        sub = df.copy()
        for key, value in filters.items():
            if isinstance(value, float):
                sub = sub[np.isclose(sub[key], value)]
            else:
                sub = sub[sub[key] == value]
        for row, separation in enumerate(separations):
            values = sub[np.isclose(sub["separation"], separation)]["ari_true"].dropna()
            if values.empty:
                continue
            median[row, col] = values.median()

    return separations, scenario_specs, median


def plot_kmeans_recovery(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    separations, scenario_specs, median = _kmeans_recovery_grid(root)
    if median.size == 0:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=SLIDE_FIGSIZE)
    ax = fig.add_axes([0.09, 0.28, 0.75, 0.56])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#F1ECE4")
    im = ax.imshow(median, vmin=0, vmax=1, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(scenario_specs)), [label for label, _ in scenario_specs])
    ax.set_yticks(range(len(separations)), [f"{x:g}" for x in separations])
    ax.set_ylabel("Cluster separation\n(larger = less overlap)")
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        1.0,
        1.0,
        "easy",
        ha="right",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        3.48,
        0.25,
        "hard under\nimbalance",
        ha="left",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#FFFFFF",
        linespacing=0.95,
    )

    cax = fig.add_axes([0.87, 0.34, 0.025, 0.42])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Median ARI", fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    cbar.outline.set_visible(False)

    fig.suptitle("K-means recovery by scenario difficulty", x=0.09, y=0.94, ha="left", fontsize=22, fontweight="bold", color=INK)
    fig.text(
        0.09,
        0.88,
        "ARI = adjusted rand index; 1 is perfect recovery, around 0 is random-like.",
        ha="left",
        va="top",
        fontsize=14,
        color=MUTED,
    )
    fig.text(
        0.09,
        0.06,
        "Low ARI is a statistical recovery result;\nspeed is not a success claim there.",
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color=INK,
        bbox={"boxstyle": "round,pad=0.55,rounding_size=0.15", "facecolor": "#F7E6D9", "edgecolor": "none"},
    )
    fig.savefig(fig_dir / "kmeans_recovery_difficulty_map.png", dpi=DPI)
    fig.savefig(fig_dir / "kmeans_recovery_difficulty_map.svg", format="svg")
    fig.savefig(fig_dir / "kmeans_recovery_scenario_facets.png", dpi=DPI)
    manifest.extend(
        [
            {
                "figure": "kmeans_recovery_difficulty_map.png",
                "source": "kmeans_correctness.csv",
                "purpose": "Slide-ready ARI heatmap showing k-means scenario difficulty across separation and stressors.",
            },
            {
                "figure": "kmeans_recovery_difficulty_map.svg",
                "source": "kmeans_correctness.csv",
                "purpose": "Vector version of the slide-ready k-means recovery difficulty map.",
            },
            {
                "figure": "kmeans_recovery_scenario_facets.png",
                "source": "kmeans_correctness.csv",
                "purpose": "Compatibility copy of the redrawn k-means recovery difficulty map.",
            },
        ]
    )
    plt.close(fig)


def plot_kmeans_recovery_slide_heatmap(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    separations, scenario_specs, median = _kmeans_recovery_grid(root)
    if median.size == 0:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.8, 4.6))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    fig.subplots_adjust(left=0.11, right=0.9, bottom=0.25, top=0.94)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#F1ECE4")
    im = ax.imshow(median, vmin=0, vmax=1, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(scenario_specs)), [label for label, _ in scenario_specs])
    ax.set_yticks(range(len(separations)), [f"{x:g}" for x in separations])
    ax.set_xlabel("Scenario stressor", fontsize=20, color=INK, labelpad=16, fontweight="bold")
    ax.set_ylabel("Cluster separation", fontsize=20, color=INK, labelpad=18, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=10, labelsize=19, colors=INK)
    ax.tick_params(axis="y", length=0, pad=8, labelsize=20, colors=INK)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(scenario_specs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(separations), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(median.shape[0]):
        for col in range(median.shape[1]):
            value = median[row, col]
            if np.isnan(value):
                continue
            text_color = "#FFFFFF" if value < 0.62 else INK
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color=text_color,
            )

    hard_outline = Rectangle((2.5, -0.5), 2.0, 1.0, fill=False, edgecolor=PROCESS_ORANGE, linewidth=4)
    easy_outline = Rectangle((-0.5, 2.5), 2.0, 1.0, fill=False, edgecolor=NUMBA_GREEN, linewidth=4)
    ax.add_patch(hard_outline)
    ax.add_patch(easy_outline)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("Median ARI", fontsize=19, fontweight="bold", color=INK, labelpad=12)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    cbar.ax.tick_params(labelsize=18, colors=INK, length=0, pad=6)
    cbar.outline.set_visible(False)

    png_path = fig_dir / "kmeans_recovery_slide_heatmap.png"
    svg_path = fig_dir / "kmeans_recovery_slide_heatmap.svg"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    manifest.extend(
        [
            {
                "figure": png_path.name,
                "source": "kmeans_correctness.csv",
                "purpose": "Slide-specific large ARI heatmap for k-means recovery before timing.",
            },
            {
                "figure": svg_path.name,
                "source": "kmeans_correctness.csv",
                "purpose": "Vector version of the slide-specific k-means recovery heatmap.",
            },
        ]
    )


def plot_kmeans_shape_runtime(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    df = _read(root / "kmeans_shape_stress.csv")
    if df.empty:
        return
    df = _numeric(df, ["n", "d", "k", "separation", "outlier_fraction", "warm_median_s", "ari_true", "peak_python_mb"])
    status = _status_column(df)
    df = df[df[status] == "pass"].copy()
    focus = df[
        (df["d"] == 100)
        & (df["imbalance"] == "balanced")
        & np.isclose(df["outlier_fraction"], 0.0)
        & np.isclose(df["separation"], 2.0)
        & (df["n"].isin([10_000, 50_000, 100_000]))
    ]
    if focus.empty:
        return

    fig = plt.figure(figsize=SLIDE_FIGSIZE)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.45, 1.0],
        left=0.07,
        right=0.98,
        top=0.82,
        bottom=0.11,
        hspace=0.42,
        wspace=0.28,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, n in zip(axes[:2], [10_000, 50_000]):
        sub = focus[focus["n"] == n]
        if sub.empty:
            ax.set_axis_off()
            continue
        for impl, impl_df in sub.groupby("implementation"):
            agg = impl_df.groupby("k")["warm_median_s"].median().reset_index().sort_values("k")
            ax.plot(agg["k"], agg["warm_median_s"], marker="o", linewidth=2.8, markersize=7, color=COLORS.get(impl, PY_GOLD), label=impl)
        ax.set_title(f"N={int(n):,}, d=100")
        ax.set_xlabel("clusters K")
        ax.set_yscale("log")
        ax.grid(True, axis="y")
        strip_spines(ax)

    ax = axes[2]
    sub100 = focus[focus["n"] == 100_000]
    if not sub100.empty:
        med100 = sub100.groupby("implementation", as_index=False)["warm_median_s"].median()
        ax.bar(
            med100["implementation"],
            med100["warm_median_s"],
            color=[COLORS.get(x, PY_GOLD) for x in med100["implementation"]],
            width=0.58,
        )
        ax.set_title("N=100,000, d=100, K=50")
        ax.set_ylabel("median warm runtime (s)")
        ax.tick_params(axis="x", rotation=12)
        ax.grid(True, axis="y")
        strip_spines(ax)
    else:
        ax.set_axis_off()

    axes[0].set_ylabel("median warm runtime (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.91))

    ratio_ax = fig.add_subplot(gs[1, :2])
    wide = (
        focus.groupby(["n", "k", "implementation"], as_index=False)["warm_median_s"]
        .median()
        .pivot_table(index=["n", "k"], columns="implementation", values="warm_median_s")
        .dropna()
        .reset_index()
    )
    if not wide.empty and {"numba", "numpy_matmul"}.issubset(wide.columns):
        wide["numba_speedup"] = wide["numpy_matmul"] / wide["numba"]
        wide["shape"] = wide.apply(lambda r: f"N={int(r.n/1000)}k\nK={int(r.k)}", axis=1)
        colors = [NUMBA_GREEN if v >= 1 else PROCESS_ORANGE for v in wide["numba_speedup"]]
        ratio_ax.bar(wide["shape"], wide["numba_speedup"], color=colors)
        ratio_ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
        ratio_ax.set_ylabel("NumPy matmul time / Numba time")
        ratio_ax.set_title("Positive bars mean Numba is faster")
        ratio_ax.grid(True, axis="y")
        strip_spines(ratio_ax)
    else:
        ratio_ax.set_axis_off()

    card_ax = fig.add_subplot(gs[1, 2])
    card_ax.axis("off")
    max_n = int(df["n"].max())
    best = wide.loc[wide["numba_speedup"].idxmax()] if not wide.empty and "numba_speedup" in wide else None
    card_text = (
        f"max local N {max_n:,}\n"
        f"d=100 stress slice\n"
        "shape controls bottleneck\n"
    )
    if best is not None:
        card_text += f"largest Numba edge {best['numba_speedup']:.1f}x"
    card_ax.text(
        0.02,
        0.78,
        card_text,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=INK,
        linespacing=1.55,
    )

    fig.suptitle("k-means shape stress: speed depends on N, K, and d", y=0.97)
    _save(
        fig,
        fig_dir / "kmeans_shape_stress_runtime.png",
        manifest,
        "kmeans_shape_stress.csv",
        "Targeted K/d/N stress runtime for NumPy matmul and Numba.",
        tight=False,
    )


def plot_kmeans_equivalence(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    df = _read(root / "kmeans_correctness.csv")
    if df.empty or "inertia_rel_diff" not in df:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = _numeric(df, ["inertia_rel_diff", "ari_vs_reference", "n"])
    status = _status_column(df)
    df = df[(df[status] == "pass") & (df["implementation"] != "reference") & df["inertia_rel_diff"].notna()].copy()
    if df.empty:
        return
    fig = plt.figure(figsize=SLIDE_FIGSIZE)
    ax = fig.add_axes([0.21, 0.31, 0.63, 0.43])

    labels = ["NumPy\nmatmul", "Numba"]
    impl_order = ["numpy_matmul", "numba"]
    y_positions = np.arange(len(impl_order))
    tolerance = 1e-8
    xmin, xmax = 1e-16, 1e-7
    ax.axvspan(xmin, tolerance, color=NUMBA_GREEN, alpha=0.10, zorder=0)
    ax.axvline(tolerance, color=PROCESS_ORANGE, linewidth=3.0, linestyle="--", zorder=1)

    for ypos, impl in zip(y_positions, impl_order):
        values = np.maximum(df.loc[df["implementation"] == impl, "inertia_rel_diff"].to_numpy(), xmin)
        if values.size == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, values.size)
        if values.size > 1:
            rng = np.random.default_rng(14 + ypos)
            jitter = rng.uniform(-0.13, 0.13, values.size)
        ax.scatter(
            values,
            np.full(values.size, ypos) + jitter,
            s=14,
            color=COLORS.get(impl, PY_GOLD),
            alpha=0.18,
            linewidths=0,
            zorder=2,
        )
        max_value = max(float(values.max()), xmin)
        ax.scatter(
            max_value,
            ypos,
            s=170,
            color=COLORS.get(impl, PY_GOLD),
            edgecolor="#FFFFFF",
            linewidth=2.4,
            zorder=3,
        )
        ax.text(
            max_value * 1.55,
            ypos,
            f"max {max_value:.1e}",
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )

    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.55, len(impl_order) - 0.45)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Relative final-inertia error vs reference")
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")
    strip_spines(ax)
    ax.tick_params(axis="y", labelsize=16, length=0)
    ax.tick_params(axis="x", labelsize=14)
    ax.text(
        tolerance,
        len(impl_order) - 0.42,
        "tolerance 1e-8",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=PROCESS_ORANGE,
    )
    ax.text(
        2e-15,
        len(impl_order) - 0.42,
        "within tolerance",
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=NUMBA_GREEN,
    )

    max_rel = df["inertia_rel_diff"].max()
    fig.suptitle("K-means equivalence to reference", x=0.12, y=0.92, ha="left", fontsize=23, fontweight="bold", color=INK)
    fig.text(
        0.12,
        0.84,
        f"All optimized rows are inside the 1e-8 tolerance band. Largest observed relative error: {max_rel:.1e}.",
        ha="left",
        va="top",
        fontsize=15,
        color=MUTED,
    )
    fig.text(
        0.12,
        0.08,
        "Optimized paths preserve the reference solution within tolerance.",
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color=INK,
        bbox={"boxstyle": "round,pad=0.55,rounding_size=0.15", "facecolor": "#F7E6D9", "edgecolor": "none"},
    )
    fig.text(
        0.12,
        0.16,
        "Expected memory-risk rows are documented in the result README.",
        ha="left",
        va="bottom",
        fontsize=14,
        color=MUTED,
    )
    fig.savefig(fig_dir / "kmeans_reference_equivalence.png", dpi=DPI)
    fig.savefig(fig_dir / "kmeans_reference_equivalence.svg", format="svg")
    fig.savefig(fig_dir / "kmeans_equivalence_tolerance.png", dpi=DPI)
    fig.savefig(fig_dir / "kmeans_equivalence_tolerance.svg", format="svg")
    manifest.extend(
        [
            {
                "figure": "kmeans_reference_equivalence.png",
                "source": "kmeans_correctness.csv",
                "purpose": "Compatibility copy of the k-means optimized-vs-reference tolerance plot.",
            },
            {
                "figure": "kmeans_reference_equivalence.svg",
                "source": "kmeans_correctness.csv",
                "purpose": "Vector compatibility copy of the k-means optimized-vs-reference tolerance plot.",
            },
            {
                "figure": "kmeans_equivalence_tolerance.png",
                "source": "kmeans_correctness.csv",
                "purpose": "Slide-ready difference plot showing optimized k-means paths preserve reference inertia within tolerance.",
            },
            {
                "figure": "kmeans_equivalence_tolerance.svg",
                "source": "kmeans_correctness.csv",
                "purpose": "Vector version of the slide-ready k-means equivalence tolerance plot.",
            },
        ]
    )
    plt.close(fig)


def plot_kmeans_tradeoff(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    df = _read(root / "kmeans_shape_stress.csv")
    if df.empty:
        return
    df = _numeric(df, ["n", "d", "k", "warm_median_s", "ari_true", "peak_python_mb"])
    status = _status_column(df)
    df = df[df[status] == "pass"].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=SLIDE_FIGSIZE, gridspec_kw={"width_ratios": [1.05, 1.1]})
    base = df[df["implementation"] == "numpy_matmul"].copy()
    base["recovery_band"] = pd.cut(
        base["ari_true"],
        bins=[-0.01, 0.5, 0.8, 1.01],
        labels=["low", "partial", "high"],
    )
    band = (
        base.groupby(["imbalance", "outlier_fraction", "recovery_band"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    band["share"] = band["count"] / band.groupby(["imbalance", "outlier_fraction"])["count"].transform("sum")
    band["setting"] = band.apply(lambda r: f"{r.imbalance}\noutliers {float(r.outlier_fraction):g}", axis=1)
    pivot = band.pivot_table(index="setting", columns="recovery_band", values="share", fill_value=0, observed=True)
    left = np.zeros(len(pivot))
    band_colors = {"low": "#9AA3AD", "partial": PY_GOLD, "high": NUMBA_GREEN}
    for label in ["low", "partial", "high"]:
        values = pivot[label].to_numpy() if label in pivot else np.zeros(len(pivot))
        axes[0].barh(pivot.index, values, left=left, color=band_colors[label], label=label)
        left += values
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("share of shape-stress scenarios")
    axes[0].set_title("Recovery has to be visible first")
    axes[0].legend(frameon=False, ncol=3, loc="lower right")
    strip_spines(axes[0])

    good = df[df["ari_true"] >= 0.8].copy()
    if not good.empty:
        for impl, sub in good.groupby("implementation"):
            agg = sub.groupby("n", as_index=False)["warm_median_s"].median().sort_values("n")
            axes[1].plot(agg["n"], agg["warm_median_s"], marker="o", linewidth=2.8, markersize=7, color=COLORS.get(impl, PY_GOLD), label=impl)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("N samples, high-recovery scenarios only")
        axes[1].set_ylabel("median warm runtime (s)")
        axes[1].set_title("Then compare runtime")
        axes[1].grid(True, axis="both")
        strip_spines(axes[1])
        axes[1].legend(frameon=False)
    else:
        axes[1].set_axis_off()
    fig.suptitle("Local timing is useful only after the statistic recovers the signal", y=0.97)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.14, wspace=0.28)
    _save(
        fig,
        fig_dir / "kmeans_runtime_recovery_tradeoff.png",
        manifest,
        "kmeans_shape_stress.csv",
        "Runtime versus statistical recovery across the targeted shape-stress scenarios.",
    )


def plot_permutation_power(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    df = _read(root / "permutation_power_extended.csv")
    if df.empty:
        return
    df = _numeric(df, ["delta", "signal_fraction", "signal_power", "null_false_positive_rate"])
    df = df[df[_status_column(df)] == "pass"].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    for sf, sub in df.groupby("signal_fraction"):
        agg = sub.groupby("delta").agg(
            mean_power=("signal_power", "mean"),
            lo=("signal_power", lambda x: np.quantile(x, 0.1)),
            hi=("signal_power", lambda x: np.quantile(x, 0.9)),
        ).reset_index()
        color = {0.01: PY_BLUE, 0.05: THREAD_TEAL, 0.1: PROCESS_ORANGE}.get(float(sf), PY_GOLD)
        ax.plot(agg["delta"], agg["mean_power"], marker="o", linewidth=2.8, markersize=7, color=color, label=f"signal fraction {sf:g}")
        ax.fill_between(agg["delta"], agg["lo"], agg["hi"], color=color, alpha=0.13, linewidth=0)
    fpr = df.groupby("delta")["null_false_positive_rate"].mean().reset_index()
    ax.plot(fpr["delta"], fpr["null_false_positive_rate"], color="#333333", linestyle="--", marker="x", label="null FPR")
    ax.axhline(0.05, color="#333333", linewidth=1, linestyle=":")
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("effect size delta")
    ax.set_ylabel("proportion p <= 0.05")
    ax.set_title("Permutation power: weak effects need denser evidence")
    ax.grid(True, axis="y")
    strip_spines(ax)
    ax.legend(frameon=False, ncol=2)
    _save(
        fig,
        fig_dir / "permutation_power_extended.png",
        manifest,
        "permutation_power_extended.csv",
        "Power curve across more deltas and signal fractions.",
    )


def plot_permutation_calibration(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    df = _read(root / "permutation_calibration_extended.csv")
    if df.empty:
        return
    df = _numeric(df, ["n", "p", "r", "prop_below_alpha", "ks_uniform", "mean_p"])
    df = df[df[_status_column(df)] == "pass"].copy()
    if df.empty:
        return
    values = df["prop_below_alpha"].dropna().sort_values().to_numpy()
    if values.size == 0:
        return
    n_features = int(df["p"].dropna().iloc[0]) if "p" in df else 1000
    nominal = 0.05
    se = np.sqrt(nominal * (1.0 - nominal) / n_features)
    band_low = nominal - 1.96 * se
    band_high = nominal + 1.96 * se
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    y = np.arange(values.size) + 1

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    fig.subplots_adjust(left=0.14, right=0.92, top=0.74, bottom=0.30)
    ax.axvspan(band_low, band_high, color="#DCEEDB", alpha=0.9, zorder=0)
    ax.axvline(nominal, color=INK, linestyle="--", linewidth=3.0, zorder=1)
    ax.scatter(values, y, s=46, color=THREAD_TEAL, alpha=0.42, edgecolors="none", zorder=2)
    ax.errorbar(
        median_value,
        values.size + 8,
        xerr=[[median_value - q25], [q75 - median_value]],
        fmt="o",
        markersize=16,
        color=PY_BLUE,
        ecolor=PY_BLUE,
        elinewidth=7,
        capsize=0,
        zorder=4,
    )
    ax.scatter([mean_value], [values.size + 15], s=260, color=JAX_BERRY, edgecolor="white", linewidth=2.0, zorder=5)
    ax.text(mean_value + 0.002, values.size + 15, f"mean {mean_value:.3f}", va="center", ha="left", fontsize=18, color=INK, fontweight="bold")
    ax.text(median_value + 0.002, values.size + 8, f"median {median_value:.3f}", va="center", ha="left", fontsize=17, color=INK, fontweight="bold")
    ax.text(nominal + 0.002, 8, "nominal 0.05", va="center", ha="left", fontsize=18, color=INK, fontweight="bold")
    ax.text(
        band_high - 0.001,
        values.size - 6,
        "feature-level binomial band",
        va="center",
        ha="right",
        fontsize=16,
        color=NUMBA_GREEN,
        fontweight="bold",
    )
    ax.set_xlim(0.025, 0.075)
    ax.set_ylim(0, values.size + 24)
    ax.set_xlabel("Type-I error estimate (p <= 0.05)", fontsize=20, labelpad=10)
    ax.set_ylabel("Null replicate", fontsize=20, labelpad=10)
    ax.set_yticks([1, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=17)
    ax.grid(True, axis="x")
    strip_spines(ax)
    fig.text(0.06, 0.93, "Permutation null calibration", ha="left", va="top", fontsize=30, fontweight="bold", color=INK)
    fig.text(
        0.06,
        0.85,
        f"Optimized NumPy batched matrix path under the null; observed mean {mean_value:.3f}. "
        f"Band uses feature-level binomial variation for p={n_features:,}.",
        ha="left",
        va="bottom",
        fontsize=18,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.06,
        "Fast code is useful only if null calibration remains correct.",
        ha="left",
        va="center",
        fontsize=20,
        color=INK,
        fontweight="bold",
        bbox=dict(facecolor="#F7E7D8", edgecolor="none", boxstyle="round,pad=0.36"),
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "permutation_calibration_extended.png"
    svg_path = fig_dir / "permutation_calibration_extended.svg"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    purpose = "Type-I error summary across existing null calibration replicates for the optimized batched matrix path."
    manifest.append({"figure": png_path.name, "source": "permutation_calibration_extended.csv", "purpose": purpose})
    manifest.append({"figure": svg_path.name, "source": "permutation_calibration_extended.csv", "purpose": purpose})


def plot_permutation_runtime(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    df = _read(root / "permutation_runtime_scaling_extended.csv")
    if df.empty:
        return
    df = _numeric(df, ["n", "p", "r", "warm_median_s", "peak_python_mb"])
    df = df[df[_status_column(df)] == "pass"].copy()
    if df.empty:
        return
    order = ["numpy_matrix", "numpy_matrix_batched", "jax_matrix_cpu"]
    labels = {
        "numpy_matrix": "NumPy matrix",
        "numpy_matrix_batched": "Batched NumPy",
        "jax_matrix_cpu": "JAX CPU",
    }
    p_values = sorted(df["p"].dropna().unique())
    p_colors = {
        p: color
        for p, color in zip(p_values, [PY_BLUE, THREAD_TEAL, PROCESS_ORANGE, JAX_BERRY, NUMBA_GREEN])
    }

    fig, axes = plt.subplots(1, 3, figsize=SLIDE_FIGSIZE, sharey=True)
    for ax, implementation in zip(axes, order):
        sub = df[df["implementation"] == implementation]
        if sub.empty:
            ax.set_axis_off()
            continue
        for p_value, p_df in sub.groupby("p"):
            agg = p_df.groupby("r", as_index=False)["warm_median_s"].median().sort_values("r")
            ax.plot(
                agg["r"],
                agg["warm_median_s"],
                marker="o",
                markersize=5.5,
                linewidth=2.3,
                color=p_colors.get(p_value, PY_GOLD),
                label=f"p={int(p_value):,}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(labels.get(implementation, implementation), fontsize=19)
        ax.set_xlabel("permutations R")
        ax.grid(True, axis="both", which="major", alpha=0.30)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: compact_label(x)))
        strip_spines(ax)

    axes[0].set_ylabel("warm median runtime (s)")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.56, 0.88))

    max_p = int(max(p_values))
    max_r = int(df["r"].max())
    fig.text(0.06, 0.94, "Permutation runtime scaling on the local validation tier", ha="left", va="top", fontsize=27, fontweight="bold", color=INK)
    fig.text(
        0.06,
        0.83,
        f"MacBook Air sweep at n=500; p up to {max_p:,}, R up to {max_r:,}. Memory-risk shapes are documented, not ranked here.",
        ha="left",
        va="bottom",
        fontsize=15.5,
        color=MUTED,
    )
    fig.text(
        0.06,
        0.06,
        "Local runtime evidence finds the bottleneck shape before server/GPU scaling.",
        ha="left",
        va="center",
        fontsize=19,
        color=INK,
        fontweight="bold",
        bbox=dict(facecolor="#F7E7D8", edgecolor="none", boxstyle="round,pad=0.36"),
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.76, bottom=0.19, wspace=0.22)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "permutation_runtime_scaling_extended.png"
    grid_png_path = fig_dir / "permutation_runtime_scaling_grid.png"
    grid_svg_path = fig_dir / "permutation_runtime_scaling_grid.svg"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(grid_png_path, dpi=DPI)
    fig.savefig(grid_svg_path, format="svg")
    plt.close(fig)
    purpose = "Small-multiple runtime scaling sweep by implementation, p, and R on the MacBook validation tier."
    manifest.append({"figure": png_path.name, "source": "permutation_runtime_scaling_extended.csv", "purpose": purpose})
    manifest.append({"figure": grid_png_path.name, "source": "permutation_runtime_scaling_extended.csv", "purpose": purpose})
    manifest.append({"figure": grid_svg_path.name, "source": "permutation_runtime_scaling_extended.csv", "purpose": purpose})


def plot_permutation_local_runtime_signal(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    """Create the main-path local runtime signal figure after validation passes."""
    df = _read(root / "permutation_runtime_scaling_extended.csv")
    if df.empty:
        return
    df = _numeric(df, ["n", "p", "r", "warm_median_s"])
    df = df[df[_status_column(df)] == "pass"].copy()
    df = df[(df["implementation"] == "numpy_matrix_batched") & df["warm_median_s"].notna()].copy()
    if df.empty:
        return

    p_values = sorted(df["p"].dropna().unique())
    p_colors = {
        p: color
        for p, color in zip(p_values, [PY_BLUE, THREAD_TEAL, PROCESS_ORANGE, JAX_BERRY, NUMBA_GREEN])
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.05))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    fig.subplots_adjust(left=0.13, right=0.96, top=0.92, bottom=0.25)

    for p_value, sub in df.groupby("p"):
        agg = sub.groupby("r", as_index=False)["warm_median_s"].median().sort_values("r")
        ax.plot(
            agg["r"],
            agg["warm_median_s"],
            marker="o",
            markersize=8.5,
            linewidth=3.3,
            color=p_colors.get(p_value, PY_GOLD),
            label=f"p = {int(p_value):,}",
        )
        if not agg.empty:
            last = agg.iloc[-1]
            ax.annotate(
                f"p={int(p_value):,}",
                xy=(last["r"], last["warm_median_s"]),
                xytext=(10, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=p_colors.get(p_value, PY_GOLD),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    r_min = float(df["r"].min())
    r_max = float(df["r"].max())
    ax.set_xlim(r_min / 1.2, r_max * 1.9)
    ax.set_xlabel("Permutations R", fontsize=21, fontweight="bold", labelpad=10)
    ax.set_ylabel("Warm median runtime (s)", fontsize=21, fontweight="bold", labelpad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: compact_label(x)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.tick_params(axis="both", labelsize=18, pad=6)
    ax.grid(True, axis="both", which="major", alpha=0.34)
    ax.text(
        0.035,
        0.955,
        "MacBook Air · NumPy batched matrix · n=500",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.14", "facecolor": "#FFFFFF", "edgecolor": "#D7CDC0", "alpha": 0.94},
    )
    fig.text(
        0.13,
        0.042,
        "batch_R = min(500, R); warm median over seeds",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
    )
    strip_spines(ax)

    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "permutation_local_runtime_signal.png"
    svg_path = fig_dir / "permutation_local_runtime_signal.svg"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    purpose = (
        "Main-path local permutation runtime signal for the validated batched NumPy path, "
        "showing warm runtime growth as p and R increase."
    )
    manifest.append({"figure": png_path.name, "source": "permutation_runtime_scaling_extended.csv", "purpose": purpose})
    manifest.append({"figure": svg_path.name, "source": "permutation_runtime_scaling_extended.csv", "purpose": purpose})


def plot_permutation_equivalence(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    df = _read(root / "permutation_equivalence.csv")
    if df.empty or "max_abs_p_diff" not in df:
        return
    df = _numeric(df, ["max_abs_p_diff", "max_abs_stat_diff", "p", "r"])
    status = _status_column(df)
    df = df[(df[status] == "pass") & df["max_abs_p_diff"].notna()].copy()
    if df.empty:
        return
    summary = (
        df.groupby("implementation", as_index=False)
        .agg(
            max_abs_p_diff=("max_abs_p_diff", "max"),
            max_abs_stat_diff=("max_abs_stat_diff", "max"),
            rows=("implementation", "size"),
        )
        .sort_values("implementation")
    )
    row_lookup = {row["implementation"]: row for _, row in summary.iterrows()}
    rows = [
        ("NumPy matrix\np-values", "numpy_matrix", "max_abs_p_diff"),
        ("JAX CPU/x64 matrix\np-values", "jax_matrix_cpu", "max_abs_p_diff"),
        ("NumPy matrix\nstatistics", "numpy_matrix", "max_abs_stat_diff"),
        ("JAX CPU/x64 matrix\nstatistics", "jax_matrix_cpu", "max_abs_stat_diff"),
    ]
    rows = [row for row in rows if row[1] in row_lookup]
    floor = 1e-18
    tolerance = 1e-6
    max_p_diff = float(df["max_abs_p_diff"].max())
    max_stat_diff = float(df["max_abs_stat_diff"].max())
    matched_workloads = df[["n", "p", "r"]].drop_duplicates().shape[0]
    matched_seeds = df["seed"].nunique() if "seed" in df else 0

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    fig.subplots_adjust(left=0.26, right=0.93, top=0.72, bottom=0.25)
    ax.axvspan(floor, tolerance, color="#DCEEDB", alpha=0.8, zorder=0)
    ax.axvline(tolerance, color=NUMBA_GREEN, linewidth=3.0, linestyle="--", zorder=1)
    y = np.arange(len(rows))[::-1]
    for yi, (label, implementation, metric) in zip(y, rows):
        value = float(row_lookup[implementation][metric])
        plotted_value = max(value, floor)
        color = COLORS.get(implementation, PY_GOLD)
        ax.scatter(plotted_value, yi, s=190, color=color, edgecolor="white", linewidth=2.0, zorder=3)
        ax.hlines(yi, floor, plotted_value, color=color, linewidth=5, alpha=0.55, zorder=2)
        text = "0.0 recorded" if value == 0 else f"{value:.1e}"
        ax.annotate(
            text,
            xy=(plotted_value, yi),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=18,
            color=INK,
            fontweight="bold",
        )

    ax.text(
        tolerance * 1.15,
        y.max() + 0.28,
        "tolerance 1e-6",
        ha="left",
        va="center",
        fontsize=17,
        color=NUMBA_GREEN,
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xlim(floor, 1e-4)
    ax.set_ylim(-1.1, len(rows) - 0.45)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows], fontsize=19, color=INK)
    ax.set_xlabel("Maximum absolute difference vs reference", fontsize=23, labelpad=12)
    fig.text(
        0.06,
        0.93,
        "Permutation equivalence",
        ha="left",
        va="top",
        fontsize=25,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.86,
        (
            f"Same permutation stream; {matched_workloads} workloads x {matched_seeds} seeds. "
            f"Max p diff {max_p_diff:.1f}; max stat diff {max_stat_diff:.1e}."
        ),
        ha="left",
        va="bottom",
        fontsize=16,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.08,
        "Same statistic, different computational shape.",
        ha="left",
        va="center",
        fontsize=24,
        color=INK,
        fontweight="bold",
        bbox=dict(facecolor="#F7E7D8", edgecolor="none", boxstyle="round,pad=0.42"),
    )
    ax.grid(True, axis="x", which="major")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelsize=17)
    strip_spines(ax)

    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "permutation_equivalence_detail.png"
    svg_path = fig_dir / "permutation_equivalence_detail.svg"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    purpose = (
        "Absolute-difference tolerance plot for matched permutation reference, NumPy matrix, "
        "and JAX CPU/x64 matrix rows."
    )
    manifest.append({"figure": png_path.name, "source": "permutation_equivalence.csv", "purpose": purpose})
    manifest.append({"figure": svg_path.name, "source": "permutation_equivalence.csv", "purpose": purpose})


def plot_permutation_local_gate_slide(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    """Create the backup permutation validation inventory figure."""
    equiv = _read(root / "permutation_equivalence.csv")
    calibration = _read(root / "permutation_calibration_extended.csv")
    runtime = _read(root / "permutation_runtime_scaling_extended.csv")
    if any(df.empty for df in [equiv, calibration, runtime]):
        return

    equiv = _numeric(equiv, ["max_abs_p_diff", "max_abs_stat_diff"])
    calibration = _numeric(calibration, ["p", "prop_below_alpha"])
    runtime = _numeric(runtime, ["p", "r", "warm_median_s"])

    status = _status_column(equiv)
    equiv = equiv[(equiv[status] == "pass") & equiv["max_abs_p_diff"].notna()].copy()
    calibration = calibration[(calibration[_status_column(calibration)] == "pass") & calibration["prop_below_alpha"].notna()].copy()
    runtime = runtime[(runtime[_status_column(runtime)] == "pass") & runtime["warm_median_s"].notna()].copy()
    if equiv.empty or calibration.empty or runtime.empty:
        return

    eq_summary = equiv.groupby("implementation", as_index=False).agg(
        max_abs_p_diff=("max_abs_p_diff", "max"),
        max_abs_stat_diff=("max_abs_stat_diff", "max"),
    )
    row_lookup = {row["implementation"]: row for _, row in eq_summary.iterrows()}
    eq_rows = [
        ("NumPy p", "numpy_matrix", "max_abs_p_diff"),
        ("JAX p", "jax_matrix_cpu", "max_abs_p_diff"),
        ("NumPy stat", "numpy_matrix", "max_abs_stat_diff"),
        ("JAX stat", "jax_matrix_cpu", "max_abs_stat_diff"),
    ]
    eq_rows = [row for row in eq_rows if row[1] in row_lookup]

    nominal = 0.05
    n_features = int(calibration["p"].dropna().iloc[0]) if calibration["p"].notna().any() else 1000
    se = np.sqrt(nominal * (1.0 - nominal) / n_features)
    band_low = nominal - 1.96 * se
    band_high = nominal + 1.96 * se
    cal_values = calibration["prop_below_alpha"].dropna().to_numpy()
    cal_mean = float(np.mean(cal_values))
    cal_q25, cal_q75 = np.quantile(cal_values, [0.25, 0.75])
    cal_min = float(np.min(cal_values))
    cal_max = float(np.max(cal_values))

    rt = runtime[runtime["implementation"] == "numpy_matrix_batched"].copy()
    if rt.empty:
        rt = runtime[runtime["implementation"] == "numpy_matrix"].copy()
    if rt.empty:
        return
    p_values = sorted(rt["p"].dropna().unique())
    p_colors = {
        p: color
        for p, color in zip(p_values, [PY_BLUE, THREAD_TEAL, PROCESS_ORANGE, JAX_BERRY, NUMBA_GREEN])
    }

    fig, axes = plt.subplots(1, 3, figsize=SLIDE_FIGSIZE)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.18, wspace=0.46)

    # A. Equivalence tolerance
    ax = axes[0]
    floor = 1e-18
    tolerance = 1e-6
    ax.axvspan(floor, tolerance, color="#DCEEDB", alpha=0.9, zorder=0)
    ax.axvline(tolerance, color=NUMBA_GREEN, linewidth=2.4, linestyle="--", zorder=1)
    y_pos = np.arange(len(eq_rows))[::-1]
    for yi, (label, implementation, metric) in zip(y_pos, eq_rows):
        value = float(row_lookup[implementation][metric])
        plotted = max(value, floor)
        color = COLORS.get(implementation, PY_GOLD)
        ax.hlines(yi, floor, plotted, color=color, linewidth=5.0, alpha=0.6, zorder=2)
        ax.scatter(plotted, yi, s=150, color=color, edgecolor="white", linewidth=1.8, zorder=3)
        value_label = "0" if value == 0 else f"{value:.1e}"
        ax.annotate(
            value_label,
            xy=(plotted, yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=16,
            color=INK,
            fontweight="bold",
        )
    ax.text(
        tolerance * 1.2,
        len(eq_rows) - 0.9,
        "tol. 1e-6",
        ha="left",
        va="center",
        fontsize=15,
        color=NUMBA_GREEN,
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xlim(floor, 1e-4)
    ax.set_ylim(-0.7, len(eq_rows) - 0.25)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([row[0] for row in eq_rows], fontsize=16, color=INK)
    ax.set_title("A. Equivalence", loc="left", fontsize=21, pad=12)
    ax.set_xlabel("max absolute diff", fontsize=18, labelpad=7)
    ax.text(
        0.03,
        0.96,
        "same stream + p-value rule",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.5,
        color=MUTED,
        fontweight="bold",
        bbox=dict(facecolor="#FFFFFF", edgecolor="none", alpha=0.82, pad=2.5),
    )
    ax.grid(True, axis="x", which="major", alpha=0.28)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelsize=16)
    strip_spines(ax)

    # B. Null calibration
    ax = axes[1]
    ax.axvspan(band_low, band_high, ymin=0.32, ymax=0.68, color="#DCEEDB", alpha=0.95, zorder=0)
    ax.axvline(nominal, color=INK, linewidth=2.8, linestyle="--", zorder=1)
    ax.hlines(0.5, cal_min, cal_max, color="#C9B9A5", linewidth=8, alpha=0.75, zorder=2)
    ax.hlines(0.5, cal_q25, cal_q75, color=THREAD_TEAL, linewidth=14, alpha=0.9, zorder=3)
    ax.scatter([cal_mean], [0.5], s=260, color=JAX_BERRY, edgecolor="white", linewidth=2.0, zorder=4)
    ax.text(cal_mean + 0.002, 0.5, f"mean {cal_mean:.3f}", ha="left", va="center", fontsize=18, color=INK, fontweight="bold")
    ax.text(nominal + 0.002, 0.78, "alpha 0.050", ha="left", va="center", fontsize=17, color=INK, fontweight="bold")
    ax.text(
        band_high,
        0.23,
        f"binomial band\n{band_low:.3f}-{band_high:.3f}",
        ha="right",
        va="top",
        fontsize=14.5,
        color=NUMBA_GREEN,
        fontweight="bold",
        linespacing=1.08,
    )
    ax.set_xlim(0.025, 0.075)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("B. Null calibration", loc="left", fontsize=21, pad=12)
    ax.set_xlabel("type-I error estimate", fontsize=18, labelpad=7)
    ax.text(
        0.03,
        0.96,
        "p <= 0.05 stays near alpha",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.5,
        color=MUTED,
        fontweight="bold",
        bbox=dict(facecolor="#FFFFFF", edgecolor="none", alpha=0.82, pad=2.5),
    )
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(True, axis="x", alpha=0.28)
    strip_spines(ax)

    # C. Runtime shape
    ax = axes[2]
    for p_value, sub in rt.groupby("p"):
        agg = sub.groupby("r", as_index=False)["warm_median_s"].median().sort_values("r")
        ax.plot(
            agg["r"],
            agg["warm_median_s"],
            marker="o",
            markersize=7,
            linewidth=3.0,
            color=p_colors.get(p_value, PY_GOLD),
            label=f"p={int(p_value):,}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("C. Runtime shape", loc="left", fontsize=21, pad=12)
    ax.set_xlabel("permutations R", fontsize=18, labelpad=7)
    ax.set_ylabel("runtime (s)", fontsize=18, labelpad=8)
    ax.text(
        0.03,
        0.96,
        "local scaling by p and R",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.5,
        color=MUTED,
        fontweight="bold",
        bbox=dict(facecolor="#FFFFFF", edgecolor="none", alpha=0.82, pad=2.5),
    )
    ax.legend(frameon=False, fontsize=14, loc="upper left", bbox_to_anchor=(0.02, 0.88), handlelength=1.7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: compact_label(x)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, axis="both", which="major", alpha=0.28)
    strip_spines(ax)

    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "permutation_local_gate_slide.png"
    svg_path = fig_dir / "permutation_local_gate_slide.svg"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    source = "; ".join(
        [
            "permutation_equivalence.csv",
            "permutation_calibration_extended.csv",
            "permutation_runtime_scaling_extended.csv",
        ]
    )
    purpose = "Backup permutation local validation inventory: equivalence, null calibration, and local runtime-shape evidence."
    manifest.append({"figure": png_path.name, "source": source, "purpose": purpose})
    manifest.append({"figure": svg_path.name, "source": source, "purpose": purpose})


def plot_local_validation_suite_overview(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    """Create a compact 2x3 overview for the local-validation slide."""
    kmeans = _read(root / "kmeans_correctness.csv")
    shape = _read(root / "kmeans_shape_stress.csv")
    perm_eq = _read(root / "permutation_equivalence.csv")
    calibration = _read(root / "permutation_calibration_extended.csv")
    power = _read(root / "permutation_power_extended.csv")
    runtime = _read(root / "permutation_runtime_scaling_extended.csv")
    if any(df.empty for df in [kmeans, shape, perm_eq, calibration, power, runtime]):
        return

    fig, axes = plt.subplots(2, 3, figsize=SLIDE_FIGSIZE)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.13, wspace=0.31, hspace=0.72)
    fig.text(0.055, 0.95, "Local validation suite overview", ha="left", va="top", fontsize=24, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.895,
        "Correctness and statistical behavior come before acceleration claims.",
        ha="left",
        va="top",
        fontsize=13.5,
        color=MUTED,
    )

    # A. K-means recovery difficulty
    ax = axes[0, 0]
    km = _numeric(kmeans, ["d", "separation", "ari_true", "outlier_fraction"])
    km = km[(km["implementation"] == "numpy_matmul") & (km[_status_column(km)] == "pass")].copy()
    scenario_specs = [
        ("clean\nlow-d", {"imbalance": "balanced", "outlier_fraction": 0.0, "d": 2}),
        ("clean\nhigh-d", {"imbalance": "balanced", "outlier_fraction": 0.0, "d": 50}),
        ("outliers", {"imbalance": "balanced", "outlier_fraction": 0.01, "d": 10}),
        ("90/10", {"imbalance": "90_10", "outlier_fraction": 0.0, "d": 10}),
        ("90/10+\noutliers", {"imbalance": "90_10", "outlier_fraction": 0.01, "d": 10}),
    ]
    separations = sorted(km["separation"].dropna().unique())
    recovery = np.full((len(separations), len(scenario_specs)), np.nan)
    for col, (_, filters) in enumerate(scenario_specs):
        sub = km.copy()
        for key, value in filters.items():
            if isinstance(value, float):
                sub = sub[np.isclose(sub[key], value)]
            else:
                sub = sub[sub[key] == value]
        for row, separation in enumerate(separations):
            values = sub[np.isclose(sub["separation"], separation)]["ari_true"].dropna()
            if not values.empty:
                recovery[row, col] = values.median()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#F1ECE4")
    ax.imshow(recovery, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_title("A. K-means recovery difficulty", loc="left", fontsize=12.5, pad=7)
    ax.set_xticks(range(len(scenario_specs)), [label for label, _ in scenario_specs], fontsize=7.7)
    ax.set_yticks(range(len(separations)), [compact_label(x) for x in separations], fontsize=8.5)
    ax.set_ylabel("separation", fontsize=9.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_caption(ax, "Some simulated shapes are statistically hard; low ARI is not a speed success.", y=-0.36)

    # B. K-means equivalence
    ax = axes[0, 1]
    km_eq = _numeric(kmeans, ["inertia_rel_diff"])
    km_eq = km_eq[
        (km_eq[_status_column(km_eq)] == "pass")
        & (km_eq["implementation"] != "reference")
        & km_eq["inertia_rel_diff"].notna()
    ].copy()
    impl_order = ["numpy_matmul", "numba"]
    labels = {"numpy_matmul": "NumPy", "numba": "Numba"}
    y_pos = np.arange(len(impl_order))[::-1]
    floor = 1e-16
    tolerance = 1e-8
    ax.axvspan(floor, tolerance, color="#DCEEDB", alpha=0.9, zorder=0)
    ax.axvline(tolerance, color=PROCESS_ORANGE, linewidth=1.6, linestyle="--")
    for yi, impl in zip(y_pos, impl_order):
        values = km_eq.loc[km_eq["implementation"] == impl, "inertia_rel_diff"].dropna()
        if values.empty:
            continue
        value = max(float(values.max()), floor)
        ax.hlines(yi, floor, value, color=COLORS.get(impl, PY_GOLD), linewidth=4, alpha=0.55)
        ax.scatter(value, yi, s=70, color=COLORS.get(impl, PY_GOLD), edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(value * 1.7, yi, f"{value:.0e}", ha="left", va="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(floor, 1e-7)
    ax.set_ylim(-0.7, len(impl_order) - 0.3)
    ax.set_yticks(y_pos, [labels[x] for x in impl_order], fontsize=9.2)
    ax.set_title("B. K-means equivalence", loc="left", fontsize=12.5, pad=7)
    ax.set_xlabel("relative inertia error", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(True, axis="x", alpha=0.25)
    strip_spines(ax)
    add_panel_caption(ax, "Optimized paths preserve reference inertia within tolerance.", y=-0.36)

    # C. K-means shape stress
    ax = axes[0, 2]
    sh = _numeric(shape, ["n", "d", "k", "separation", "outlier_fraction", "warm_median_s"])
    sh = sh[sh[_status_column(sh)] == "pass"].copy()
    focus = sh[
        (sh["d"] == 100)
        & (sh["imbalance"] == "balanced")
        & np.isclose(sh["outlier_fraction"], 0.0)
        & np.isclose(sh["separation"], 2.0)
        & (sh["n"].isin([10_000, 50_000]))
    ].copy()
    for (implementation, n_value), sub in focus.groupby(["implementation", "n"]):
        if implementation not in {"numpy_matmul", "numba"}:
            continue
        agg = sub.groupby("k", as_index=False)["warm_median_s"].median().sort_values("k")
        linestyle = "-" if int(n_value) == 10_000 else "--"
        label = f"{labels.get(implementation, implementation)} N={int(n_value/1000)}k"
        ax.plot(agg["k"], agg["warm_median_s"], marker="o", markersize=4.5, linewidth=1.9, linestyle=linestyle, color=COLORS.get(implementation, PY_GOLD), label=label)
    ax.set_title("C. K-means shape stress", loc="left", fontsize=12.5, pad=7)
    ax.set_xlabel("clusters K", fontsize=9.5)
    ax.set_ylabel("runtime (s)", fontsize=9.5)
    ax.set_yscale("log")
    ax.tick_params(labelsize=8.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7.6, ncol=1, loc="upper left")
    strip_spines(ax)
    add_panel_caption(ax, "N, d, and K change the local bottleneck.", y=-0.36)

    # D. Permutation equivalence
    ax = axes[1, 0]
    pe = _numeric(perm_eq, ["max_abs_p_diff", "max_abs_stat_diff"])
    pe = pe[(pe[_status_column(pe)] == "pass") & pe["max_abs_p_diff"].notna()].copy()
    eq_summary = pe.groupby("implementation", as_index=False).agg(
        max_abs_p_diff=("max_abs_p_diff", "max"),
        max_abs_stat_diff=("max_abs_stat_diff", "max"),
    )
    row_lookup = {row["implementation"]: row for _, row in eq_summary.iterrows()}
    rows = [("NumPy p", "numpy_matrix", "max_abs_p_diff"), ("JAX p", "jax_matrix_cpu", "max_abs_p_diff"), ("NumPy stat", "numpy_matrix", "max_abs_stat_diff"), ("JAX stat", "jax_matrix_cpu", "max_abs_stat_diff")]
    rows = [row for row in rows if row[1] in row_lookup]
    floor = 1e-18
    tolerance = 1e-6
    ax.axvspan(floor, tolerance, color="#DCEEDB", alpha=0.9, zorder=0)
    ax.axvline(tolerance, color=NUMBA_GREEN, linewidth=1.6, linestyle="--")
    y = np.arange(len(rows))[::-1]
    for yi, (label, implementation, metric) in zip(y, rows):
        value = float(row_lookup[implementation][metric])
        plotted = max(value, floor)
        ax.hlines(yi, floor, plotted, color=COLORS.get(implementation, PY_GOLD), linewidth=3.2, alpha=0.55)
        ax.scatter(plotted, yi, s=58, color=COLORS.get(implementation, PY_GOLD), edgecolor="white", linewidth=1.1, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(floor, 1e-4)
    ax.set_yticks(y, [row[0] for row in rows], fontsize=8.8)
    ax.set_title("D. Permutation equivalence", loc="left", fontsize=12.5, pad=7)
    ax.set_xlabel("max absolute difference", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(True, axis="x", alpha=0.25)
    strip_spines(ax)
    add_panel_caption(ax, "Same permutation stream; same p-value definition.", y=-0.36)

    # E. Null calibration and power
    ax = axes[1, 1]
    cal = _numeric(calibration, ["prop_below_alpha", "p"])
    cal = cal[cal[_status_column(cal)] == "pass"].copy()
    values = cal["prop_below_alpha"].dropna().to_numpy()
    nominal = 0.05
    n_features = int(cal["p"].dropna().iloc[0]) if "p" in cal and cal["p"].notna().any() else 1000
    se = np.sqrt(nominal * (1.0 - nominal) / n_features)
    band_low = nominal - 1.96 * se
    band_high = nominal + 1.96 * se
    mean_value = float(np.mean(values)) if values.size else nominal
    ax.axvspan(band_low, band_high, ymin=0.70, ymax=0.92, color="#DCEEDB", alpha=0.95)
    ax.axvline(nominal, ymin=0.66, ymax=0.96, color=INK, linestyle="--", linewidth=1.4)
    ax.scatter([mean_value], [0.81], s=90, color=JAX_BERRY, edgecolor="white", linewidth=1.1, zorder=4)
    pw = _numeric(power, ["delta", "signal_fraction", "signal_power"])
    pw = pw[pw[_status_column(pw)] == "pass"].copy()
    for sf, sub in pw.groupby("signal_fraction"):
        if float(sf) not in {0.01, 0.05, 0.1}:
            continue
        agg = sub.groupby("delta", as_index=False)["signal_power"].mean().sort_values("delta")
        color = {0.01: PY_BLUE, 0.05: THREAD_TEAL, 0.1: PROCESS_ORANGE}.get(float(sf), PY_GOLD)
        ax.plot(agg["delta"], agg["signal_power"], marker="o", markersize=3.8, linewidth=1.8, color=color, label=f"{float(sf):g}")
    ax.text(0.012, 0.86, "null", ha="left", va="center", fontsize=8.5, color=MUTED, fontweight="bold")
    ax.set_xlim(0, max(1.0, float(pw["delta"].max())))
    ax.set_ylim(0, 1.04)
    ax.set_title("E. Null calibration / power", loc="left", fontsize=12.5, pad=7)
    ax.set_xlabel("effect size delta", fontsize=9.5)
    ax.set_ylabel("p <= 0.05", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.legend(title="signal frac.", title_fontsize=7.6, frameon=False, fontsize=7.6, loc="lower right")
    ax.grid(True, axis="y", alpha=0.25)
    strip_spines(ax)
    add_panel_caption(ax, "Under the null, type-I error stays near alpha; signal power increases with effect size.", y=-0.36)

    # F. Permutation runtime scaling
    ax = axes[1, 2]
    rt = _numeric(runtime, ["p", "r", "warm_median_s"])
    rt = rt[(rt[_status_column(rt)] == "pass") & (rt["implementation"] == "numpy_matrix_batched")].copy()
    p_values = sorted(rt["p"].dropna().unique())
    p_colors = {
        p: color
        for p, color in zip(p_values, [PY_BLUE, THREAD_TEAL, PROCESS_ORANGE, JAX_BERRY, NUMBA_GREEN])
    }
    for p_value, sub in rt.groupby("p"):
        agg = sub.groupby("r", as_index=False)["warm_median_s"].median().sort_values("r")
        ax.plot(agg["r"], agg["warm_median_s"], marker="o", markersize=4.2, linewidth=1.9, color=p_colors.get(p_value, PY_GOLD), label=f"p={int(p_value):,}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("F. Permutation runtime scaling", loc="left", fontsize=12.5, pad=7)
    ax.set_xlabel("permutations R", fontsize=9.5)
    ax.set_ylabel("runtime (s)", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: compact_label(x)))
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.grid(True, axis="both", alpha=0.25)
    strip_spines(ax)
    add_panel_caption(ax, "Runtime scales with p and R; local evidence finds the bottleneck shape.", y=-0.36)

    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "local_validation_suite_overview.png"
    svg_path = fig_dir / "local_validation_suite_overview.svg"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    source = "; ".join(
        [
            "kmeans_correctness.csv",
            "kmeans_shape_stress.csv",
            "permutation_equivalence.csv",
            "permutation_calibration_extended.csv",
            "permutation_power_extended.csv",
            "permutation_runtime_scaling_extended.csv",
        ]
    )
    purpose = "Composite 2x3 local-validation overview for correctness, equivalence, calibration, power, and runtime-shape evidence."
    manifest.append({"figure": png_path.name, "source": source, "purpose": purpose})
    manifest.append({"figure": svg_path.name, "source": source, "purpose": purpose})


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["figure", "source", "purpose"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    apply_style()
    root = args.results_dir
    fig_dir = args.output_dir or (root / "figures")
    manifest: list[dict[str, str]] = []
    plot_kmeans_recovery(root, fig_dir, manifest)
    plot_kmeans_recovery_slide_heatmap(root, fig_dir, manifest)
    plot_kmeans_shape_runtime(root, fig_dir, manifest)
    plot_kmeans_equivalence(root, fig_dir, manifest)
    plot_kmeans_tradeoff(root, fig_dir, manifest)
    plot_permutation_power(root, fig_dir, manifest)
    plot_permutation_calibration(root, fig_dir, manifest)
    plot_permutation_runtime(root, fig_dir, manifest)
    plot_permutation_local_runtime_signal(root, fig_dir, manifest)
    plot_permutation_equivalence(root, fig_dir, manifest)
    plot_permutation_local_gate_slide(root, fig_dir, manifest)
    plot_local_validation_suite_overview(root, fig_dir, manifest)
    write_manifest(root / "figure_manifest.csv", manifest)
    print(fig_dir)


if __name__ == "__main__":
    main()
