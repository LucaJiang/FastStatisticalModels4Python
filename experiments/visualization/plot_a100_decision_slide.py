"""Create the slide-specific A100 permutation decision map."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fsm4py-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/fsm4py-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd


P_FEATURES = [10_000, 50_000, 100_000, 250_000, 500_000]
R_PERMUTATIONS = [1_000, 5_000, 10_000, 50_000]
DPI = 220

INK = "#17202A"
MUTED = "#5F6B74"
PAPER = "#FBF7EF"
LINE = "#D7CDC0"
COLORS = {
    "cpu": "#E8895B",
    "near": "#F8E6A1",
    "a100": "#89C987",
    "strong": "#1F7A3A",
    "unavailable": "#C7CCD1",
}


def compact_count(value: int) -> str:
    return f"{int(value / 1000)}k" if value >= 1000 else str(value)


def speedup_label(value: float) -> str:
    if value >= 8:
        return f"{value:.2f}×"
    return f"{value:.1f}×"


def is_unavailable(row: pd.Series | None) -> bool:
    if row is None:
        return True
    status_text = " ".join(
        str(row.get(column, "")).lower()
        for column in ["winner", "a100_status", "correctness_status", "timing_note", "notes"]
    )
    if any(token in status_text for token in ["oom", "unavailable", "timeout_skip", "memory-risk"]):
        return True
    return pd.isna(pd.to_numeric(row.get("speedup_cpu_over_a100"), errors="coerce"))


def bin_for_speedup(value: float) -> str:
    if value < 1.0:
        return "cpu"
    if value < 2.0:
        return "near"
    if value < 5.0:
        return "a100"
    return "strong"


def plot_decision_map_slide(summary_csv: Path, presentation_dir: Path) -> None:
    """Write a slide-optimized decision map PNG/SVG from the committed summary CSV."""

    df = pd.read_csv(summary_csv)
    df["speedup_cpu_over_a100"] = pd.to_numeric(df["speedup_cpu_over_a100"], errors="coerce")
    keyed = {(int(row.p), int(row.R)): row for row in df.itertuples()}

    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 14,
        }
    )

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor("#FFFFFF")

    for y, r_value in enumerate(R_PERMUTATIONS):
        for x, p_value in enumerate(P_FEATURES):
            tuple_row = keyed.get((p_value, r_value))
            row = pd.Series(tuple_row._asdict()) if tuple_row is not None else None
            unavailable = is_unavailable(row)
            if unavailable:
                color = COLORS["unavailable"]
                hatch = "///"
                label = "A100\nOOM"
                text_color = INK
            else:
                speedup = float(row["speedup_cpu_over_a100"])
                bin_name = bin_for_speedup(speedup)
                color = COLORS[bin_name]
                hatch = None
                label = f"{speedup_label(speedup)}\n{'A100' if speedup > 1.0 else 'CPU'}"
                text_color = "#FFFFFF" if bin_name == "strong" else INK

            rect = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="#FFFFFF",
                linewidth=4.0,
                hatch=hatch,
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=22,
                fontweight="900",
                color=text_color,
                linespacing=1.05,
            )

    first_win = (P_FEATURES.index(10_000), R_PERMUTATIONS.index(5_000))
    ax.add_patch(
        Rectangle(
            (first_win[0] - 0.5, first_win[1] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#1A365D",
            linewidth=5.5,
        )
    )
    ax.text(
        first_win[0] - 0.40,
        first_win[1] + 0.39,
        "first win",
        ha="left",
        va="center",
        fontsize=13,
        fontweight="900",
        color="#1A365D",
        bbox={"boxstyle": "round,pad=0.12,rounding_size=0.08", "facecolor": "#FFFFFF", "edgecolor": "#1A365D"},
    )

    max_win = (P_FEATURES.index(500_000), R_PERMUTATIONS.index(5_000))
    ax.add_patch(
        Rectangle(
            (max_win[0] - 0.5, max_win[1] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#0B3D20",
            linewidth=5.5,
        )
    )
    ax.text(max_win[0] + 0.31, max_win[1] + 0.32, "★", ha="center", va="center", fontsize=27, color="#F6C445")

    ax.set_xlim(-0.5, len(P_FEATURES) - 0.5)
    ax.set_ylim(-0.5, len(R_PERMUTATIONS) - 0.5)
    ax.set_xticks(range(len(P_FEATURES)), [compact_count(p) for p in P_FEATURES])
    ax.set_yticks(range(len(R_PERMUTATIONS)), [compact_count(r) for r in R_PERMUTATIONS])
    ax.set_xlabel("p features", labelpad=12, fontweight="800")
    ax.set_ylabel("R permutations", labelpad=12, fontweight="800")
    ax.tick_params(axis="both", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=COLORS["cpu"], label="CPU <1×"),
        Patch(facecolor=COLORS["near"], label="1-2×"),
        Patch(facecolor=COLORS["a100"], label="A100 2-5×"),
        Patch(facecolor=COLORS["strong"], label="A100 >5×"),
        Patch(facecolor=COLORS["unavailable"], hatch="///", label="A100 OOM"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.52, 0.995),
        handlelength=1.35,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.20)

    presentation_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(presentation_dir / "a100_decision_map_slide.png", dpi=DPI)
    fig.savefig(presentation_dir / "a100_decision_map_slide.svg", format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("experiments/results/linux_server_a100/permutation_break_even/break_even_shape_sweep_summary.csv"),
    )
    parser.add_argument(
        "--presentation-dir",
        type=Path,
        default=Path("experiments/results/presentation_figures"),
    )
    args = parser.parse_args()
    plot_decision_map_slide(args.summary_csv, args.presentation_dir)


if __name__ == "__main__":
    main()
