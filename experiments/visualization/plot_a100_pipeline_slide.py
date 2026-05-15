"""Create slide-specific A100 pipeline figures from committed evidence."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fsm4py-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/fsm4py-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BREAK_EVEN_DIR = ROOT / "experiments/results/linux_server_a100/permutation_break_even"
PRESENTATION_DIR = ROOT / "experiments/results/presentation_figures"

DPI = 220
PAPER = "#FBF7EF"
PANEL = "#FFFFFF"
INK = "#17202A"
MUTED = "#52616D"
LINE = "#DFD3C0"
BLUE = "#2368AD"
GOLD = "#D7961C"
GREEN = "#2F7D32"
BERRY = "#B51E59"
ORANGE = "#E66A2C"
PURPLE = "#7046A1"
GRAY = "#6F7782"

REQUIRED_STAGE_COLUMNS = [
    "permutation_generation_time_s",
    "W_build_host_time_s",
    "host_to_device_transfer_time_s",
    "device_compute_time_s",
    "pvalue_reduction_time_s",
    "device_to_host_collect_time_s",
]


def compact_count(value: float) -> str:
    return f"{int(value / 1000)}k" if value >= 1000 else str(int(value))


def read_best_safe_batch_r(readme_path: Path) -> int | None:
    if not readme_path.exists():
        return None
    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Best safe batch_R from Stage 1:\s*([0-9,]+)", text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def normalize_batch_csv(path: Path, source_label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "batch_R" not in df:
        return None
    time_col = next(
        (
            col
            for col in [
                "a100_streamed_reduction_time_s",
                "a100_streamed_full_end_to_end_time_s",
                "a100_end_to_end_time_s",
                "end_to_end_time_s",
                "total_end_to_end_time_s",
            ]
            if col in df
        ),
        None,
    )
    if time_col is None:
        return None
    out = pd.DataFrame(
        {
            "batch_R": pd.to_numeric(df["batch_R"], errors="coerce"),
            "a100_end_to_end_time_s": pd.to_numeric(df[time_col], errors="coerce"),
            "source_label": source_label,
        }
    )
    if "speedup_cpu_over_a100" in df:
        out["speedup_cpu_over_a100"] = pd.to_numeric(df["speedup_cpu_over_a100"], errors="coerce")
    elif "cpu_end_to_end_time_s" in df:
        cpu = pd.to_numeric(df["cpu_end_to_end_time_s"], errors="coerce")
        out["speedup_cpu_over_a100"] = cpu / out["a100_end_to_end_time_s"]
    else:
        out["speedup_cpu_over_a100"] = np.nan
    out = out.dropna(subset=["batch_R", "a100_end_to_end_time_s"]).sort_values("batch_R")
    return out if not out.empty else None


def load_batch_data() -> pd.DataFrame:
    candidates = [
        (BREAK_EVEN_DIR / "batch_R_sweep.csv", "canonical break-even batch_R_sweep.csv"),
        (BREAK_EVEN_DIR / "batch_R_sweep_summary.csv", "committed lightweight batch_R sweep summary"),
    ]
    for path, label in candidates:
        df = normalize_batch_csv(path, label)
        if df is not None:
            return df
    raise FileNotFoundError("No usable canonical batch_R CSV found.")


def plot_batch_tuning(df: pd.DataFrame, out_dir: Path, best_safe_batch_r: int | None) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    fig, (ax_time, ax_speed) = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]})
    fig.patch.set_facecolor(PAPER)
    for ax in [ax_time, ax_speed]:
        ax.set_facecolor(PANEL)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_alpha(0.35)

    x = df["batch_R"].to_numpy(dtype=float)
    ax_time.plot(x, df["a100_end_to_end_time_s"], marker="o", linewidth=3.2, markersize=8, color=BERRY)
    ax_time.set_xscale("log", base=2)
    ax_time.set_ylabel("A100 full end-to-end time (s)")
    ax_time.set_title("A100 batch_R sweep: n=5k, p=50k, R=10k", weight="bold", loc="left")

    speedup = pd.to_numeric(df["speedup_cpu_over_a100"], errors="coerce")
    if speedup.notna().any():
        ax_speed.plot(x, speedup, marker="s", linewidth=3.0, markersize=7, color=BLUE)
        ax_speed.axhline(1.0, color=INK, linestyle="--", linewidth=1.6)
        ax_speed.set_ylabel("CPU/A100 speedup")
    else:
        ax_speed.text(0.5, 0.5, "speedup unavailable in committed fallback", transform=ax_speed.transAxes, ha="center", va="center", fontsize=18, color=MUTED)
        ax_speed.set_ylabel("speedup")
    ax_speed.set_xlabel("batch_R (log2 scale)")
    ax_speed.set_xticks(x, [compact_count(v) for v in x])

    if best_safe_batch_r:
        target_x = float(best_safe_batch_r)
        for ax in [ax_time, ax_speed]:
            ax.axvline(target_x, color=GOLD, linewidth=2.2, linestyle=":")
        ax_time.text(
            target_x,
            ax_time.get_ylim()[1],
            f" chosen measured setting: {int(target_x)}",
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=INK,
            bbox={"facecolor": "#FFF3E6", "edgecolor": "none", "boxstyle": "round,pad=0.22"},
        )

    source = str(df["source_label"].iloc[0])
    fig.text(0.055, 0.025, source, fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=[0.04, 0.045, 0.99, 0.98])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "a100_batch_R_tuning.png", dpi=DPI)
    fig.savefig(out_dir / "a100_batch_R_tuning.svg", format="svg")
    plt.close(fig)


def load_decomposition_from_csv(path: Path, source_label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    total_col = "total_end_to_end_time_s" if "total_end_to_end_time_s" in df else "end_to_end_time_s"
    if total_col not in df:
        return None
    out = df.copy()
    for col in [total_col, "speedup_cpu_over_a100", "w_matmul_share"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["total_end_to_end_time_s"] = out[total_col]
    if not all(col in out for col in REQUIRED_STAGE_COLUMNS):
        if "w_matmul_share" not in out:
            return None
        for col in REQUIRED_STAGE_COLUMNS:
            out[col] = 0.0
        out["device_compute_time_s"] = out["total_end_to_end_time_s"] * out["w_matmul_share"]
    for col in REQUIRED_STAGE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["other_overhead_s"] = (out["total_end_to_end_time_s"] - out[REQUIRED_STAGE_COLUMNS].sum(axis=1)).clip(lower=0.0)
    out["source_label"] = source_label
    return out.dropna(subset=["total_end_to_end_time_s"])


def load_decomposition_data() -> tuple[pd.DataFrame, bool]:
    candidates = [
        (BREAK_EVEN_DIR / "decomposition_representative_shapes.csv", "canonical break-even decomposition_representative_shapes.csv"),
        (BREAK_EVEN_DIR / "decomposition_representative_shapes_summary.csv", "canonical break-even decomposition_representative_shapes_summary.csv"),
    ]
    for path, label in candidates:
        df = load_decomposition_from_csv(path, label)
        if df is not None and not df.empty:
            return df, True
    raise FileNotFoundError("No usable canonical break-even decomposition CSV found.")


def plot_decomposition(df: pd.DataFrame, has_required_stage_csv: bool, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 13,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "legend.fontsize": 9.5,
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    df = df.sort_values(["p", "R"]).copy()
    if has_required_stage_csv:
        stages = [
            ("permutation_generation_time_s", "perm gen", BLUE),
            ("W_build_host_time_s", "build W", GREEN),
            ("host_to_device_transfer_time_s", "H2D transfer", ORANGE),
            ("device_compute_time_s", "W @ X", BERRY),
            ("pvalue_reduction_time_s", "reduction", PURPLE),
            ("device_to_host_collect_time_s", "collect", GOLD),
            ("other_overhead_s", "other", GRAY),
        ]
    else:
        stages = [
            ("permutation_generation_time_s", "perm gen", BLUE),
            ("W_build_host_time_s", "build W", GREEN),
            ("transfer_collect_time_s", "transfer + collect", ORANGE),
            ("device_compute_time_s", "W @ X", BERRY),
            ("reduction_and_other_time_s", "reduction + other", GRAY),
        ]

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PANEL)
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for col, label, color in stages:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.55, label=label, color=color, edgecolor="white", linewidth=1.1)
        left += vals

    labels = [f"p={compact_count(row.p)} R={compact_count(row.R)}" for row in df.itertuples()]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_title("Full-scenario A100 stages", weight="bold", loc="left")
    ax.grid(axis="x", alpha=0.25)
    max_total = float(df["total_end_to_end_time_s"].max())
    ax.set_xlim(0, max_total * 1.55)
    for i, row in enumerate(df.itertuples()):
        total = float(row.total_end_to_end_time_s)
        wx = float(row.device_compute_time_s)
        wx_share = 100.0 * wx / total if total else 0.0
        speedup = getattr(row, "speedup_cpu_over_a100", np.nan)
        speedup_text = f"\n{float(speedup):.1f}x" if np.isfinite(speedup) else ""
        ax.text(total + max_total * 0.035, i, f"{total:.3f}s\nW @ X {wx_share:.1f}%{speedup_text}", va="center", ha="left", fontsize=11, color=INK, fontweight="bold")

    ax.legend(loc="upper right", ncol=1, frameon=True, facecolor="#FFFDF8", edgecolor=LINE, framealpha=0.92)
    source = str(df["source_label"].iloc[0])
    fig.text(0.08, 0.035, source, fontsize=9.7, color=MUTED)
    fig.tight_layout(rect=[0.02, 0.07, 0.99, 0.98])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "a100_pipeline_decomposition_representative.png", dpi=DPI)
    fig.savefig(out_dir / "a100_pipeline_decomposition_representative.svg", format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation-dir", type=Path, default=PRESENTATION_DIR)
    args = parser.parse_args()

    batch = load_batch_data()
    plot_batch_tuning(batch, args.presentation_dir, read_best_safe_batch_r(BREAK_EVEN_DIR / "README.md"))
    decomp, has_required_stage_csv = load_decomposition_data()
    plot_decomposition(decomp, has_required_stage_csv, args.presentation_dir)


if __name__ == "__main__":
    main()
