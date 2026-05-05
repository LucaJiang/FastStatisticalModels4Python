"""Generate short method animations for the Reveal deck."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "slides" / "assets" / "videos"
POSTER_DIR = ROOT / "slides" / "assets" / "posters"
FPS = 24
W = 1280
H = 720
DPI = 100

PAPER = "#fbf6ec"
PANEL = "#fffdf8"
INK = "#15202b"
MUTED = "#52616d"
BLUE = "#2368ad"
GOLD = "#d7961c"
GREEN = "#2f7d32"
BERRY = "#b51e59"
ORANGE = "#e66a2c"
TEAL = "#1f7a8c"
LINE = "#dfd3c0"
COLORS = np.array([BLUE, ORANGE, GREEN])


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _encode(frame_dir: Path, stem: str) -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    mp4 = VIDEO_DIR / f"{stem}.mp4"
    webm = VIDEO_DIR / f"{stem}.webm"
    pattern = str(frame_dir / "frame_%04d.png")
    _run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            pattern,
            "-vf",
            "format=yuv420p",
            "-movflags",
            "+faststart",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            pattern,
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            "32",
            "-pix_fmt",
            "yuv420p",
            str(webm),
        ]
    )


def _new_fig() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _save_frame(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, facecolor=PAPER)
    plt.close(fig)


def _soft_panel(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, radius: float = 0.02) -> None:
    del radius
    ax.add_patch(Rectangle(xy, w, h, facecolor=PANEL, edgecolor=LINE, linewidth=1.8))


def _draw_header(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.text(0.055, 0.93, title, color=INK, fontsize=34, weight="bold", ha="left", va="top")
    ax.text(0.055, 0.87, subtitle, color=MUTED, fontsize=20, ha="left", va="top")


def generate_kmeans() -> None:
    rng = np.random.default_rng(12)
    centers = np.array([[-2.2, -0.7], [0.2, 1.7], [2.4, -0.5]])
    data = np.vstack([rng.normal(loc=c, scale=0.42, size=(52, 2)) for c in centers])
    centroids = [np.array([[-2.8, 1.5], [1.9, 1.4], [1.2, -1.8]])]
    labels = []
    for _ in range(6):
        d = ((data[:, None, :] - centroids[-1][None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        labels.append(lab)
        new = centroids[-1].copy()
        for k in range(3):
            if np.any(lab == k):
                new[k] = data[lab == k].mean(axis=0)
        centroids.append(new)

    x0, x1 = -3.4, 3.4
    y0, y1 = -2.4, 2.5

    def sx(x: np.ndarray) -> np.ndarray:
        return 0.10 + (x - x0) / (x1 - x0) * 0.78

    def sy(y: np.ndarray) -> np.ndarray:
        return 0.12 + (y - y0) / (y1 - y0) * 0.72

    total = 240
    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp)
        for frame in range(total):
            t = frame / total
            cycle = min(int(t * 6), 5)
            local = (t * 6) - cycle
            assign_phase = local < 0.45
            move_alpha = 0 if assign_phase else min((local - 0.45) / 0.45, 1)
            lab = labels[cycle]
            current = centroids[cycle] * (1 - move_alpha) + centroids[cycle + 1] * move_alpha

            fig, ax = _new_fig()
            _soft_panel(ax, (0.055, 0.065), 0.89, 0.84)
            ax.text(0.085, 0.855, f"Iteration {cycle + 1}: " + ("assign points" if assign_phase else "move centroids"), fontsize=30, color=INK, weight="bold")
            ax.text(0.085, 0.805, "Centroids move only after assignments are known.", fontsize=22, color=MUTED)

            point_colors = COLORS[lab]
            if frame < 10:
                point_colors = np.array(["#aeb7bf"] * len(data))
            ax.scatter(sx(data[:, 0]), sy(data[:, 1]), s=72, c=point_colors, edgecolors="white", linewidths=0.8, alpha=0.92)

            for k in range(3):
                path = np.array([c[k] for c in centroids[: cycle + 1]])
                if len(path) > 1:
                    segments = np.stack([np.column_stack([sx(path[:-1, 0]), sy(path[:-1, 1])]), np.column_stack([sx(path[1:, 0]), sy(path[1:, 1])])], axis=1)
                    ax.add_collection(LineCollection(segments, colors=[COLORS[k]], linewidths=3.2, alpha=0.35))
                if not assign_phase:
                    prev = centroids[cycle][k]
                    ax.plot([sx(prev[0]), sx(current[k, 0])], [sy(prev[1]), sy(current[k, 1])], color=COLORS[k], linewidth=4.0, alpha=0.55)
                ax.scatter(sx(current[k, 0]), sy(current[k, 1]), marker="X", s=700, c=[COLORS[k]], edgecolors=INK, linewidths=3.0, zorder=8)

            _save_frame(fig, frame_dir / f"frame_{frame:04d}.png")

        shutil.copy(frame_dir / "frame_0168.png", POSTER_DIR / "kmeans_animation_poster.png")
        _encode(frame_dir, "kmeans_animation")


def generate_permutation() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(8, 10))
    base_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    shuffles = [rng.permutation(base_labels) for _ in range(10)]
    stats = np.array([rng.normal(loc=0.0, scale=0.55, size=10) for _ in shuffles])

    total = 240
    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp)
        for frame in range(total):
            t = frame / total
            shuffle_idx = min(int(t * 10), 9)
            local = (t * 10) - shuffle_idx
            labels = shuffles[shuffle_idx]
            fig, ax = _new_fig()
            _soft_panel(ax, (0.055, 0.08), 0.89, 0.82)
            ax.text(0.085, 0.855, "Shuffle labels -> statistic -> repeat", fontsize=34, color=INK, weight="bold")

            left, bottom = 0.18, 0.24
            cell_w, cell_h = 0.038, 0.052
            for r in range(8):
                lab_color = BLUE if labels[r] == 0 else ORANGE
                ax.add_patch(Rectangle((0.095, bottom + (7 - r) * cell_h), 0.060, cell_h * 0.82, facecolor=lab_color, edgecolor="white", linewidth=1.2))
                ax.text(0.125, bottom + (7 - r) * cell_h + cell_h * 0.41, "A" if labels[r] == 0 else "B", color="white", ha="center", va="center", fontsize=17, weight="bold")
                for c in range(10):
                    val = x[r, c]
                    col = plt.cm.RdBu_r((val + 2.4) / 4.8)
                    ax.add_patch(Rectangle((left + c * cell_w, bottom + (7 - r) * cell_h), cell_w * 0.9, cell_h * 0.82, facecolor=col, edgecolor="white", linewidth=0.8))
            ax.text(0.095, 0.69, "group labels", fontsize=18, color=MUTED)
            ax.text(0.27, 0.69, "features in X", fontsize=18, color=MUTED)

            arrow = FancyArrowPatch((0.58, 0.45), (0.66, 0.45), arrowstyle="->", mutation_scale=28, linewidth=3.0, color=MUTED)
            ax.add_patch(arrow)
            ax.text(0.585, 0.50, "shuffle", fontsize=20, color=MUTED)

            _soft_panel(ax, (0.67, 0.21), 0.11, 0.50)
            ax.text(0.692, 0.665, "one null\nstatistic", fontsize=19, color=INK, weight="bold", va="top")
            completed = min(shuffle_idx + (1 if local > 0.55 else 0), len(shuffles))
            for i in range(completed):
                y = 0.26 + i * 0.038
                ax.plot([0.705, 0.765], [y, y], color=TEAL, linewidth=2.8, alpha=0.8)
                ax.scatter([0.735 + stats[i].mean() * 0.025], [y], s=38, color=BERRY, zorder=3)

            if t > 0.58:
                _soft_panel(ax, (0.815, 0.23), 0.105, 0.45)
                ax.text(0.835, 0.64, "W", fontsize=31, color=INK, weight="bold")
                rows = min(int((t - 0.58) / 0.035) + 1, 8)
                for r in range(rows):
                    for c in range(8):
                        color = BLUE if shuffles[r][c] == 0 else ORANGE
                        ax.add_patch(Rectangle((0.835 + c * 0.011, 0.565 - r * 0.032), 0.009, 0.022, facecolor=color, edgecolor="none", alpha=0.8))
                ax.text(0.846, 0.292, "@ X", fontsize=26, color=INK, weight="bold")

            _save_frame(fig, frame_dir / f"frame_{frame:04d}.png")

        shutil.copy(frame_dir / "frame_0180.png", POSTER_DIR / "permutation_animation_poster.png")
        _encode(frame_dir, "permutation_animation")


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    generate_kmeans()
    generate_permutation()


if __name__ == "__main__":
    main()
