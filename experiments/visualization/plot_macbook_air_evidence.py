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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

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


def plot_kmeans_recovery(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    df = _read(root / "kmeans_correctness.csv")
    if df.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = _numeric(df, ["d", "separation", "ari_true", "outlier_fraction"])
    status = _status_column(df)
    df = df[(df["implementation"] == "numpy_matmul") & (df[status] == "pass")].copy()
    if df.empty:
        return

    scenario_specs = [
        ("Clean\nlow-d", {"imbalance": "balanced", "outlier_fraction": 0.0, "d": 2}),
        ("Clean\nhigh-d", {"imbalance": "balanced", "outlier_fraction": 0.0, "d": 50}),
        ("Outliers\n1%", {"imbalance": "balanced", "outlier_fraction": 0.01, "d": 10}),
        ("Imbalance\n90/10", {"imbalance": "90_10", "outlier_fraction": 0.0, "d": 10}),
        ("Imbalance\n+ outliers", {"imbalance": "90_10", "outlier_fraction": 0.01, "d": 10}),
    ]
    separations = sorted(df["separation"].dropna().unique())
    median = np.full((len(separations), len(scenario_specs)), np.nan)
    iqr = np.full_like(median, np.nan)

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
            iqr[row, col] = values.quantile(0.75) - values.quantile(0.25)

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
        "Recovery depends on data geometry;\nruntime matters only after recovery is visible.",
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
    pass_rows = len(df)
    max_n = int(df["n"].max())
    best = wide.loc[wide["numba_speedup"].idxmax()] if not wide.empty and "numba_speedup" in wide else None
    card_text = (
        f"{pass_rows:,} pass rows\n"
        f"max local N {max_n:,}\n"
        f"d=100 stress slice\n"
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
    all_rows = df.copy()
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

    counts = all_rows[status].value_counts()
    pass_rows = int(counts.get("pass", 0))
    skips = int(counts.get("skipped_memory_risk", 0))
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
    fig.text(0.12, 0.16, f"{pass_rows:,} pass rows; {skips:,} expected memory-risk skips; no failed optimized rows.", ha="left", va="bottom", fontsize=14, color=MUTED)
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
        "expected band",
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
        f"Optimized NumPy batched matrix path under the null; {values.size} seeds, n=500, p={n_features:,}, R=1,000.",
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
    df = _numeric(df, ["p", "r", "warm_median_s", "peak_python_mb"])
    df = df[df[_status_column(df)] == "pass"].copy()
    if df.empty:
        return
    scenario = df[(df["n"] == 500) & (df["p"] == 1000) & (df["r"] == 1000)].copy()
    if scenario.empty:
        scenario = df[(df["p"] == 1000) & (df["r"] == 1000)].copy()
    order = ["numpy_matrix", "numpy_matrix_batched", "jax_matrix_cpu"]
    summary = (
        scenario.groupby("implementation", as_index=False)["warm_median_s"]
        .median()
        .set_index("implementation")
        .reindex([name for name in order if name in scenario["implementation"].unique()])
        .dropna()
        .reset_index()
    )
    if summary.empty:
        return

    labels = {
        "numpy_matrix": "NumPy matrix",
        "numpy_matrix_batched": "Batched NumPy matrix",
        "jax_matrix_cpu": "JAX CPU",
    }
    colors = [COLORS.get(name, PY_GOLD) for name in summary["implementation"]]
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    fig.patch.set_facecolor("#FBF7EF")
    ax.set_facecolor("#FFFFFF")
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["warm_median_s"], color=colors, width=0.58)
    ax.set_xticks(x, [labels.get(name, name) for name in summary["implementation"]])
    ax.set_ylabel("warm median runtime (seconds)")
    ax.set_ylim(0, max(summary["warm_median_s"]) * 1.45)
    ax.grid(True, axis="y", alpha=0.28)
    strip_spines(ax)
    for bar, value in zip(bars, summary["warm_median_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.04, f"{value:.3f}s", ha="center", va="bottom", fontsize=18, fontweight="bold", color=INK)

    n_val = int(scenario["n"].dropna().iloc[0])
    p_val = int(scenario["p"].dropna().iloc[0])
    r_val = int(scenario["r"].dropna().iloc[0])
    fig.text(0.07, 0.93, "Local validation scale: methods are close enough", ha="left", va="top", fontsize=30, fontweight="bold", color=INK)
    fig.text(
        0.07,
        0.85,
        f"MacBook CPU validation tier; n={n_val:,}, p={p_val:,}, R={r_val:,}; same validated p-values.",
        ha="left",
        va="bottom",
        fontsize=18,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.07,
        "Choose the clearest correct implementation until the bottleneck is real.",
        ha="left",
        va="center",
        fontsize=20,
        color=INK,
        fontweight="bold",
        bbox=dict(facecolor="#F7E7D8", edgecolor="none", boxstyle="round,pad=0.36"),
    )
    fig.subplots_adjust(left=0.13, right=0.96, top=0.75, bottom=0.22)
    _save(
        fig,
        fig_dir / "permutation_runtime_scaling_extended.png",
        manifest,
        "permutation_runtime_scaling_extended.csv",
        "Single validation-scale runtime comparison for matrix, batched matrix, and JAX CPU implementations.",
        tight=False,
    )


def plot_permutation_equivalence(root: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    df = _read(root / "permutation_equivalence.csv")
    if df.empty or "max_abs_p_diff" not in df:
        return
    df = _numeric(df, ["max_abs_p_diff", "max_abs_stat_diff", "p", "r"])
    status = _status_column(df)
    all_rows = df.copy()
    df = df[(df[status] == "pass") & df["max_abs_p_diff"].notna()].copy()
    if df.empty:
        return
    counts = all_rows[status].value_counts()
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
    plot_kmeans_shape_runtime(root, fig_dir, manifest)
    plot_kmeans_equivalence(root, fig_dir, manifest)
    plot_kmeans_tradeoff(root, fig_dir, manifest)
    plot_permutation_power(root, fig_dir, manifest)
    plot_permutation_calibration(root, fig_dir, manifest)
    plot_permutation_runtime(root, fig_dir, manifest)
    plot_permutation_equivalence(root, fig_dir, manifest)
    write_manifest(root / "figure_manifest.csv", manifest)
    print(fig_dir)


if __name__ == "__main__":
    main()
