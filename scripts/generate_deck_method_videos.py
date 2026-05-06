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
    try:
        from sklearn.datasets import load_iris

        iris = load_iris()
        raw = iris.data[:, [2, 3]].astype(float)
    except Exception:
        rng = np.random.default_rng(12)
        centers = np.array([[1.45, 0.25], [4.35, 1.35], [5.45, 2.00]])
        raw = np.vstack([rng.normal(loc=c, scale=[0.32, 0.16], size=(50, 2)) for c in centers])

    data = raw
    centroids = [
        np.array(
            [
                [5.171757483762562, 0.4162410699562752],
                [0.9606008259748919, 0.392536523038085],
                [2.0764111475764966, 2.1738646186361663],
            ]
        )
    ]
    labels = []
    changed = []
    for _ in range(9):
        d = ((data[:, None, :] - centroids[-1][None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        labels.append(lab)
        changed.append(len(data) if len(labels) == 1 else int(np.sum(labels[-1] != labels[-2])))
        new = centroids[-1].copy()
        for k in range(3):
            if np.any(lab == k):
                new[k] = data[lab == k].mean(axis=0)
        centroids.append(new)

    x0, x1 = 0.7, 7.3
    y0, y1 = 0.0, 2.7
    plot_left, plot_right = 0.085, 0.690
    plot_bottom, plot_top = 0.155, 0.745

    def sx(x: np.ndarray) -> np.ndarray:
        return plot_left + (x - x0) / (x1 - x0) * (plot_right - plot_left)

    def sy(y: np.ndarray) -> np.ndarray:
        return plot_bottom + (y - y0) / (y1 - y0) * (plot_top - plot_bottom)

    total = 240
    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp)
        for frame in range(total):
            t = frame / total
            cycle = min(int(t * 9), 8)
            local = (t * 9) - cycle
            assign_phase = local < 0.45
            move_alpha = 0 if assign_phase else min((local - 0.45) / 0.45, 1)
            lab = labels[cycle]
            current = centroids[cycle] * (1 - move_alpha) + centroids[cycle + 1] * move_alpha

            fig, ax = _new_fig()
            _soft_panel(ax, (0.055, 0.065), 0.89, 0.84)
            ax.text(0.085, 0.855, "Iris petal measurements", fontsize=30, color=INK, weight="bold")
            ax.text(0.085, 0.807, "each point = one flower; K = 3 centroids", fontsize=20, color=MUTED)
            phase = "assignment phase" if assign_phase else "update phase"
            phase_color = BLUE if assign_phase else ORANGE
            ax.text(0.725, 0.850, f"Iteration {cycle + 1} / 9", fontsize=24, color=INK, weight="bold", ha="left")
            ax.text(0.725, 0.810, phase, fontsize=21, color=phase_color, weight="bold", ha="left")
            ax.text(0.725, 0.772, f"{changed[cycle]} flowers changed cluster", fontsize=14, color=MUTED, ha="left")

            ax.add_patch(Rectangle((plot_left, plot_bottom), plot_right - plot_left, plot_top - plot_bottom, facecolor="#fffaf2", edgecolor=LINE, linewidth=1.6))
            for xt in np.arange(1, 8, 1):
                ax.plot([sx(xt), sx(xt)], [plot_bottom, plot_top], color=LINE, linewidth=0.8, alpha=0.55)
                ax.text(sx(xt), plot_bottom - 0.035, f"{xt:g}", fontsize=12, color=MUTED, ha="center", va="top")
            for yt in np.arange(0.5, 2.6, 0.5):
                ax.plot([plot_left, plot_right], [sy(yt), sy(yt)], color=LINE, linewidth=0.8, alpha=0.55)
                ax.text(plot_left - 0.020, sy(yt), f"{yt:g}", fontsize=12, color=MUTED, ha="right", va="center")
            ax.text((plot_left + plot_right) / 2, 0.075, "petal length", fontsize=18, color=INK, weight="bold", ha="center")
            ax.text(0.030, (plot_bottom + plot_top) / 2, "petal width", fontsize=18, color=INK, weight="bold", rotation=90, ha="center", va="center")

            point_colors = COLORS[lab]
            ax.scatter(sx(data[:, 0]), sy(data[:, 1]), s=58, c=point_colors, edgecolors="white", linewidths=0.8, alpha=0.88)

            for k in range(3):
                path = np.array([c[k] for c in centroids[: cycle + 1]])
                if len(path) > 1:
                    segments = np.stack([np.column_stack([sx(path[:-1, 0]), sy(path[:-1, 1])]), np.column_stack([sx(path[1:, 0]), sy(path[1:, 1])])], axis=1)
                    ax.add_collection(LineCollection(segments, colors=[COLORS[k]], linewidths=3.8, alpha=0.46, zorder=6))
                if not assign_phase:
                    prev = centroids[cycle][k]
                    ax.plot([sx(prev[0]), sx(current[k, 0])], [sy(prev[1]), sy(current[k, 1])], color=COLORS[k], linewidth=5.0, alpha=0.62, zorder=7)
                ax.scatter(sx(current[k, 0]), sy(current[k, 1]), marker="X", s=780, c=[COLORS[k]], edgecolors=INK, linewidths=3.2, zorder=9)

            _soft_panel(ax, (0.725, 0.455), 0.215, 0.270)
            ax.text(0.745, 0.686, "Iris data", fontsize=16, color=INK, weight="bold")
            ax.text(
                0.745,
                0.646,
                "three species measured\nby sepal and petal dimensions",
                fontsize=11.0,
                color=MUTED,
                linespacing=1.25,
            )
            ax.text(0.745, 0.572, "shown here:", fontsize=11.0, color=INK, weight="bold")
            ax.text(0.745, 0.546, "petal length + petal width", fontsize=11.0, color=MUTED)
            ax.text(0.745, 0.505, "species names are context;\nclustering uses only the axes", fontsize=10.8, color=MUTED, linespacing=1.18)
            ax.text(0.725, 0.390, "Colors are k-means assignments.", fontsize=13, color=INK, weight="bold")

            for k, label in enumerate(["cluster A", "cluster B", "cluster C"]):
                y = 0.282 - k * 0.047
                ax.scatter(0.745, y, s=86, color=COLORS[k], edgecolor="white", linewidth=0.9)
                ax.scatter(0.795, y, marker="X", s=150, color=COLORS[k], edgecolor=INK, linewidth=1.4)
                ax.text(0.825, y, label, fontsize=11.5, color=MUTED, va="center")

            _save_frame(fig, frame_dir / f"frame_{frame:04d}.png")

        shutil.copy(frame_dir / "frame_0144.png", POSTER_DIR / "kmeans_animation_poster.png")
        _encode(frame_dir, "kmeans_animation")


def generate_permutation() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(8, 10))
    base_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    shuffles = [rng.permutation(base_labels) for _ in range(12)]

    def feature_diff(labels: np.ndarray) -> np.ndarray:
        return x[labels == 1].mean(axis=0) - x[labels == 0].mean(axis=0)

    stats = np.array([feature_diff(labels) for labels in shuffles])
    observed = feature_diff(base_labels)
    exceed = np.cumsum(np.abs(stats[:, :5]) >= np.abs(observed[:5]), axis=0)

    total = 240
    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp)
        for frame in range(total):
            t = frame / total
            shuffle_idx = min(int(t * 8), 7)
            local = (t * 8) - shuffle_idx
            labels = shuffles[shuffle_idx]
            fig, ax = _new_fig()
            _soft_panel(ax, (0.055, 0.065), 0.89, 0.84)

            stage_w = 0.265
            stage_y = 0.135
            stage_h = 0.69
            x1, x2, x3 = 0.075, 0.382, 0.674
            for x0, title in [(x1, "1. observed data"), (x2, "2. one permutation"), (x3, "3. many permutations")]:
                _soft_panel(ax, (x0, stage_y), stage_w, stage_h)
                ax.text(x0 + 0.018, stage_y + stage_h - 0.050, title, fontsize=17, color=INK, weight="bold")

            mat_left, mat_bottom = x1 + 0.078, 0.315
            cell_w, cell_h = 0.0175, 0.034
            ax.text(x1 + 0.135, 0.700, "X matrix", fontsize=18, color=INK, weight="bold", ha="center")
            ax.text(x1 + 0.135, 0.668, "rows = samples   columns = features", fontsize=10.5, color=MUTED, ha="center")
            for r in range(8):
                lab_color = BLUE if base_labels[r] == 0 else ORANGE
                y = mat_bottom + (7 - r) * cell_h
                ax.add_patch(Rectangle((x1 + 0.026, y), 0.038, cell_h * 0.84, facecolor=lab_color, edgecolor="white", linewidth=1.0))
                ax.text(x1 + 0.045, y + cell_h * 0.42, "A" if base_labels[r] == 0 else "B", color="white", ha="center", va="center", fontsize=10.5, weight="bold")
                for c in range(10):
                    val = x[r, c]
                    col = plt.cm.RdBu_r((val + 2.4) / 4.8)
                    ax.add_patch(Rectangle((mat_left + c * cell_w, y), cell_w * 0.9, cell_h * 0.84, facecolor=col, edgecolor="white", linewidth=0.6))
            ax.text(x1 + 0.045, 0.610, "A/B labels", fontsize=10.5, color=MUTED, ha="center")
            ax.text(x1 + 0.165, 0.610, "features", fontsize=10.5, color=MUTED, ha="center")
            ax.text(x1 + 0.026, 0.245, "observed statistic T_obs", fontsize=13, color=INK, weight="bold")

            ax.add_patch(FancyArrowPatch((x1 + stage_w + 0.014, 0.48), (x2 - 0.014, 0.48), arrowstyle="->", mutation_scale=24, linewidth=2.8, color=MUTED))

            perm_y = 0.690
            for r in range(8):
                color = BLUE if labels[r] == 0 else ORANGE
                x_pos = x2 + 0.028 + r * 0.025
                ax.add_patch(Rectangle((x_pos, perm_y), 0.020, 0.040, facecolor=color, edgecolor="white", linewidth=0.8))
                ax.text(x_pos + 0.010, perm_y + 0.020, "A" if labels[r] == 0 else "B", fontsize=8.8, color="white", ha="center", va="center", weight="bold")
            ax.text(x2 + 0.026, 0.610, "compute group\ndifference for\nevery feature", fontsize=13.0, color=INK, weight="bold", linespacing=1.08)
            stat = stats[shuffle_idx]
            xs = np.linspace(x2 + 0.035, x2 + 0.225, len(stat))
            y_mid = 0.440
            for i, val in enumerate(stat):
                h = min(abs(val) / max(np.max(np.abs(stat)), 1e-9), 1) * 0.090
                y0 = y_mid if val >= 0 else y_mid - h
                ax.add_patch(Rectangle((xs[i] - 0.007, y0), 0.014, h, facecolor=BERRY if val >= 0 else TEAL, edgecolor="none", alpha=0.92))
            ax.plot([x2 + 0.026, x2 + 0.236], [y_mid, y_mid], color=LINE, linewidth=1.6)
            ax.text(x2 + 0.026, 0.258, "one null draw = one vector\nof feature statistics", fontsize=13.3, color=MUTED, linespacing=1.18)

            ax.add_patch(FancyArrowPatch((x2 + stage_w + 0.004, 0.48), (x3 - 0.004, 0.48), arrowstyle="->", mutation_scale=24, linewidth=2.8, color=MUTED))

            completed = min(shuffle_idx + (1 if local > 0.45 else 0), 8)
            stack_left = x3 + 0.035
            stack_top = 0.665
            for i in range(completed):
                y = stack_top - i * 0.030
                vals = stats[i, :8]
                for c, val in enumerate(vals):
                    color = BERRY if val >= 0 else TEAL
                    ax.add_patch(Rectangle((stack_left + c * 0.020, y), 0.016, 0.019, facecolor=color, edgecolor="white", linewidth=0.25, alpha=0.84))
            if completed:
                counts = exceed[completed - 1]
                ax.text(x3 + 0.035, 0.385, "null-statistics collection", fontsize=12.5, color=INK, weight="bold")
                ax.text(x3 + 0.035, 0.350, "streamed exceedance counts: " + " ".join(str(int(c)) for c in counts[:4]), fontsize=11.0, color=MUTED)
            ax.add_patch(FancyArrowPatch((x3 + 0.132, 0.338), (x3 + 0.132, 0.296), arrowstyle="->", mutation_scale=18, linewidth=2.0, color=MUTED))
            ax.add_patch(Rectangle((x3 + 0.036, 0.228), 0.076, 0.048, facecolor="#fff7e9", edgecolor=LINE, linewidth=1.0))
            ax.text(x3 + 0.074, 0.252, "contrast\nrows W", fontsize=8.8, color=INK, weight="bold", ha="center", va="center")
            ax.text(x3 + 0.128, 0.252, "→", fontsize=18, color=MUTED, weight="bold", va="center")
            ax.text(x3 + 0.155, 0.252, "W @ X", fontsize=19, color=BERRY, weight="bold", va="center")
            ax.text(x3 + 0.035, 0.198, "matrix path appears after repeat", fontsize=10.8, color=MUTED)

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
