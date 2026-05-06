"""Generate poster-native visuals for poster_v4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.datasets import load_iris


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

INK = "#17202A"
MUTED = "#5F6B74"
LINE = "#E2D6C8"
CITYU_RED = "#B01117"
CITYU_MAROON = "#981B49"
CITYU_ORANGE = "#E07541"
CPU_BLUE = "#1F7A8C"
VALID_GREEN = "#2F7D32"
A100_BERRY = "#B51E59"
PAPER = "#FFF8F2"


def save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(
            ASSETS / f"{name}.{ext}",
            dpi=280,
            bbox_inches="tight",
            facecolor=PAPER,
            edgecolor="none",
        )
    plt.close(fig)


def arrow(
    ax: plt.Axes,
    xy1: tuple[float, float],
    xy2: tuple[float, float],
    color: str,
    lw: float = 4.0,
    scale: float = 28.0,
    zorder: int = 8,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def kmeans_path(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids = np.array([[1.45, 0.18], [3.90, 1.18], [5.75, 2.15]], dtype=float)
    paths = [centroids.copy()]
    labels = np.zeros(len(x), dtype=int)
    for _ in range(6):
        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        centroids = np.vstack([x[labels == k].mean(axis=0) for k in range(3)])
        paths.append(centroids.copy())
    return labels, np.stack(paths)


def poster_v4_kmeans_iris() -> None:
    iris = load_iris()
    x = iris.data[:, [2, 3]]
    labels, paths = kmeans_path(x)
    colors = np.array([CITYU_RED, CPU_BLUE, CITYU_ORANGE])

    fig, ax = plt.subplots(figsize=(11.0, 9.2))
    ax.set_facecolor(PAPER)
    ax.grid(True, color="#E9DFD3", linewidth=1.45)

    for k, color in enumerate(colors):
        pts = x[labels == k]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=160,
            color=color,
            alpha=0.74,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
        )

    for k, color in enumerate(colors):
        path = paths[:, k, :]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=3.4, alpha=0.88, zorder=7)
        arrow(ax, tuple(path[-2]), tuple(path[-1]), color=color, lw=4.2, scale=30)
        ax.scatter(path[0, 0], path[0, 1], marker="x", s=380, color=color, linewidth=5.2, zorder=9)
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            marker="*",
            s=900,
            color=PAPER,
            edgecolor=color,
            linewidth=3.8,
            zorder=10,
        )

    ax.text(
        0.035,
        0.95,
        "species labels are for explanation;\nk-means sees only measurements",
        transform=ax.transAxes,
        color=INK,
        fontsize=17,
        fontweight="bold",
        va="top",
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.04", "facecolor": "#FFF1E9", "edgecolor": "none"},
    )
    ax.text(
        0.97,
        0.055,
        "K = 3 centroids",
        transform=ax.transAxes,
        color=INK,
        fontsize=23,
        fontweight="bold",
        ha="right",
    )

    ax.set_xlabel("petal length", fontsize=27, fontweight="bold", labelpad=14, color=INK)
    ax.set_ylabel("petal width", fontsize=27, fontweight="bold", labelpad=14, color=INK)
    ax.tick_params(axis="both", labelsize=18, colors=MUTED, width=1.6, length=6)
    for spine in ax.spines.values():
        spine.set_color(LINE)
        spine.set_linewidth(2.0)
    ax.set_xlim(0.8, 7.15)
    ax.set_ylim(-0.05, 2.72)
    save(fig, "poster_v4_kmeans_iris")


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    face: str,
    edge: str,
    lw: float = 3.0,
    radius: float = 0.14,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=f"round,pad=0.03,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
        )
    )


def matrix(ax: plt.Axes, x0: float, y0: float, rows: int, cols: int, cw: float, ch: float) -> None:
    for r in range(rows):
        for c in range(cols):
            fill = "#F0F6FB" if (r + c) % 2 else PAPER
            ax.add_patch(
                Rectangle(
                    (x0 + c * cw, y0 + (rows - 1 - r) * ch),
                    cw,
                    ch,
                    facecolor=fill,
                    edgecolor=LINE,
                    linewidth=1.6,
                )
            )
    ax.add_patch(Rectangle((x0, y0), cols * cw, rows * ch, fill=False, edgecolor=INK, linewidth=3.3))


def poster_v4_permutation_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 9.7))
    ax.set_facecolor(PAPER)
    ax.set_xlim(0.0, 10.8)
    ax.set_ylim(0.0, 8.9)
    ax.axis("off")

    matrix(ax, 0.75, 2.34, 6, 5, 0.43, 0.72)
    ax.text(1.82, 7.20, "X matrix", ha="center", fontsize=29, fontweight="bold", color=INK)
    ax.text(1.82, 1.70, "rows = samples", ha="center", fontsize=17, fontweight="bold", color=MUTED)
    ax.text(1.82, 1.30, "columns = features", ha="center", fontsize=17, fontweight="bold", color=MUTED)
    for i, lab in enumerate(["A", "A", "A", "B", "B", "B"]):
        color = CPU_BLUE if lab == "A" else CITYU_ORANGE
        ax.add_patch(Rectangle((0.22, 2.34 + (5 - i) * 0.72), 0.38, 0.72, facecolor=color, edgecolor=PAPER, linewidth=1.3))
        ax.text(0.41, 2.70 + (5 - i) * 0.72, lab, ha="center", va="center", fontsize=20, fontweight="bold", color="white")

    arrow(ax, (3.15, 4.50), (3.72, 4.50), MUTED, lw=4.8, scale=30)
    rounded_box(ax, (3.90, 3.06), 1.18, 2.88, "#FFF0DE", CITYU_ORANGE, lw=3.2)
    ax.text(4.49, 6.42, "shuffle\nlabels", ha="center", va="center", fontsize=22, fontweight="bold", color=CITYU_ORANGE, linespacing=0.92)
    for i, color in enumerate([CITYU_ORANGE, CPU_BLUE, CITYU_ORANGE, CPU_BLUE, CITYU_ORANGE, CPU_BLUE]):
        ax.add_patch(Rectangle((4.23, 5.42 - i * 0.38), 0.52, 0.26, facecolor=color, edgecolor=PAPER, linewidth=1.0))

    arrow(ax, (5.20, 4.50), (5.78, 4.50), MUTED, lw=4.8, scale=30)
    rounded_box(ax, (5.98, 3.20), 1.55, 2.42, PAPER, CPU_BLUE, lw=3.3)
    ax.text(6.76, 4.98, "one\nnull draw", ha="center", va="center", fontsize=23, fontweight="bold", color=CPU_BLUE, linespacing=0.86)
    ax.text(6.76, 3.78, "feature-wise\nstatistics", ha="center", va="center", fontsize=15.5, fontweight="bold", color=MUTED, linespacing=0.92)
    for x, h in zip([6.16, 6.42, 6.68, 6.94, 7.20], [0.32, 0.52, 0.86, 0.62, 0.40], strict=True):
        ax.add_patch(Rectangle((x, 2.14), 0.16, h, facecolor=CPU_BLUE, edgecolor=CPU_BLUE, alpha=0.74))

    arrow(ax, (7.72, 4.50), (8.34, 5.72), MUTED, lw=4.4, scale=28)
    ax.text(7.84, 1.28, "repeat many times", ha="center", fontsize=18, fontweight="bold", color=MUTED)

    rounded_box(ax, (8.48, 5.92), 1.54, 1.64, "#F8E7EF", A100_BERRY, lw=3.4)
    ax.text(9.25, 6.80, "W @ X", ha="center", va="center", fontsize=29, fontweight="bold", color=A100_BERRY)
    ax.text(9.25, 6.22, "batched", ha="center", va="center", fontsize=16.5, fontweight="bold", color=A100_BERRY)

    arrow(ax, (9.25, 5.78), (9.25, 4.84), A100_BERRY, lw=4.4, scale=28)
    rounded_box(ax, (8.82, 2.48), 0.88, 2.16, PAPER, A100_BERRY, lw=3.3)
    for i, h in enumerate([0.30, 0.72, 1.10, 0.88]):
        ax.add_patch(Rectangle((9.04, 2.74 + i * 0.36), 0.31, h * 0.32, facecolor=A100_BERRY, edgecolor=A100_BERRY, alpha=0.84))
    ax.text(9.94, 3.30, "streamed\nexceedance\ncounts", ha="left", va="center", fontsize=18.5, fontweight="bold", color=INK, linespacing=0.9)

    save(fig, "poster_v4_permutation_workflow")


def main() -> None:
    poster_v4_kmeans_iris()
    poster_v4_permutation_workflow()


if __name__ == "__main__":
    main()
