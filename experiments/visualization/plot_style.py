"""Shared matplotlib style for the PyCon slides and poster figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

INK = "#17202A"
MUTED = "#5F6B74"
PAPER = "#FBF7EF"
PANEL = "#FFFFFF"
GRID = "#D7D0C4"
PY_BLUE = "#2B6CB0"
PY_GOLD = "#D99A1E"
NUMBA_GREEN = "#2F7D32"
JAX_BERRY = "#B51E59"
PROCESS_ORANGE = "#E66A2C"
THREAD_TEAL = "#1F7A8C"
LOOP_PURPLE = "#7046A1"
GRAY = "#60717A"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#AFA79B",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.8,
            "savefig.facecolor": PAPER,
            "savefig.edgecolor": PAPER,
        }
    )


def strip_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AFA79B")
    ax.spines["bottom"].set_color("#AFA79B")
