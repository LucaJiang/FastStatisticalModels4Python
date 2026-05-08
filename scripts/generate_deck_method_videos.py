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
IRIS_IMAGE = ROOT / "slides" / "assets" / "iris.png"
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
    try:
        from sklearn.datasets import load_iris

        iris = load_iris()
        raw = iris.data[:, [2, 3]].astype(float)
    except Exception:
        centers = np.array([[1.5, 0.2], [4.3, 1.3], [5.6, 2.0]])
        raw = np.vstack([rng.normal(loc=c, scale=[0.28, 0.18], size=(50, 2)) for c in centers])

    data = raw
    init_centroids = np.array(
        [
            [1.1, 2.35],
            [3.4, 0.35],
            [6.7, 0.75],
        ],
        dtype=float,
    )
    centroids = [init_centroids.copy()]
    iris_img = plt.imread(IRIS_IMAGE)
    labels = []
    changed = []
    n_iter = 8
    for _ in range(n_iter):
        d = ((data[:, None, :] - centroids[-1][None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        labels.append(lab)
        changed.append(len(data) if len(labels) == 1 else int(np.sum(labels[-1] != labels[-2])))
        new = centroids[-1].copy()
        for k in range(3):
            if np.any(lab == k):
                new[k] = data[lab == k].mean(axis=0)
        centroids.append(new)
        if len(labels) > 1 and changed[-1] == 0:
            break

    n_iter = len(labels)
    x0, x1 = 0.8, 7.1
    y0, y1 = 0.0, 2.7

    assign_frames = 26
    update_frames = 26
    init_frames = 42
    converged_frames = 42
    total = init_frames + n_iter * (assign_frames + update_frames) + converged_frames
    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp)
        for frame in range(total):
            if frame < init_frames:
                phase = "init"
                cycle = 0
                move_alpha = 0.0
                lab = None
                current = centroids[0]
            else:
                phase_frame = frame - init_frames
                per_iter = assign_frames + update_frames
                if phase_frame >= n_iter * per_iter:
                    phase = "converged"
                    cycle = n_iter - 1
                    move_alpha = 1.0
                    lab = labels[-1]
                    current = centroids[-1]
                else:
                    cycle = min(phase_frame // per_iter, n_iter - 1)
                    local_frame = phase_frame % per_iter
                    phase = "assign" if local_frame < assign_frames else "update"
                    move_alpha = 0.0 if phase == "assign" else min((local_frame - assign_frames) / max(update_frames - 1, 1), 1.0)
                    lab = labels[cycle]
                    current = centroids[cycle] * (1 - move_alpha) + centroids[cycle + 1] * move_alpha

            fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=PAPER)
            title_ax = fig.add_axes([0, 0, 1, 1])
            title_ax.set_xlim(0, 1)
            title_ax.set_ylim(0, 1)
            title_ax.axis("off")
            title_ax.text(0.055, 0.930, "Iris petal measurements", fontsize=32, color=INK, weight="bold", va="top")
            if phase == "init":
                status = "Iteration 0 / random starting centroids"
                status_color = BERRY
                status_detail = "points start neutral; assignments come next"
            elif phase == "assign":
                status = f"Iteration {cycle + 1} / assign points"
                status_color = BLUE
                status_detail = f"{changed[cycle]} flowers changed assignment"
            elif phase == "converged":
                status = "Converged / assignments stable"
                status_color = GREEN
                status_detail = "no flower changes assignment"
            else:
                status = f"Iteration {cycle + 1} / update centroids"
                status_color = ORANGE
                status_detail = "centroids move to assigned-flower means"
            title_ax.text(0.945, 0.865, status, fontsize=21, color=status_color, weight="bold", ha="right", va="top")
            title_ax.text(0.945, 0.822, status_detail, fontsize=14, color=MUTED, ha="right", va="top")

            ax = fig.add_axes([0.085, 0.115, 0.845, 0.665], facecolor="#fffaf2")
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.grid(color=LINE, linewidth=0.8, alpha=0.55)
            for spine in ax.spines.values():
                spine.set_color(LINE)
                spine.set_linewidth(1.2)
            ax.tick_params(colors=MUTED, labelsize=13)
            ax.set_xlabel("petal length", fontsize=17, color=INK, labelpad=10)
            ax.set_ylabel("petal width", fontsize=17, color=INK, labelpad=10)
            ax.text(
                0.015,
                0.985,
                "colors = k-means assignments; X = centroids",
                transform=ax.transAxes,
                fontsize=11.5,
                color=MUTED,
                weight="semibold",
                ha="left",
                va="top",
                bbox={"facecolor": "#fffaf2", "edgecolor": "none", "alpha": 0.66, "pad": 2.0},
                zorder=10,
            )

            iax = ax.inset_axes([0.73, -0.06, 0.46, 0.58], transform=ax.transAxes)
            iax.imshow(iris_img, alpha=0.90)
            iax.set_xticks([])
            iax.set_yticks([])
            for spine in iax.spines.values():
                spine.set_edgecolor("#fffaf2")
                spine.set_linewidth(1.2)
            iax.set_facecolor("#fffaf2")
            iax.set_zorder(4)

            if lab is None:
                point_colors = np.array(["#a8b0b6"] * len(data))
                point_alpha = 0.68
            else:
                point_colors = COLORS[lab]
                point_alpha = 0.88
            ax.scatter(data[:, 0], data[:, 1], s=78, c=point_colors, edgecolors="white", linewidths=0.9, alpha=point_alpha, zorder=3)

            for k in range(3):
                path = np.array([c[k] for c in centroids[: cycle + 1]])
                if len(path) > 1:
                    ax.plot(path[:, 0], path[:, 1], color=COLORS[k], linewidth=2.6, alpha=0.46, zorder=5)
                if phase == "update":
                    prev = centroids[cycle][k]
                    ax.scatter(
                        prev[0],
                        prev[1],
                        marker="X",
                        s=360,
                        c=["#fffaf2"],
                        edgecolors=COLORS[k],
                        linewidths=2.2,
                        alpha=0.72,
                        zorder=6,
                    )
                    ax.add_patch(
                        FancyArrowPatch(
                            (prev[0], prev[1]),
                            (current[k, 0], current[k, 1]),
                            arrowstyle="-|>",
                            mutation_scale=18,
                            linewidth=3.2,
                            color=COLORS[k],
                            alpha=0.60,
                            zorder=6,
                        )
                    )
                    ax.plot([prev[0], current[k, 0]], [prev[1], current[k, 1]], color=COLORS[k], linewidth=3.6, alpha=0.68, zorder=6)
                ax.scatter(
                    current[k, 0],
                    current[k, 1],
                    marker="X",
                    s=520,
                    c=[COLORS[k]],
                    edgecolors=INK,
                    linewidths=2.8,
                    zorder=8,
                )

            _save_frame(fig, frame_dir / f"frame_{frame:04d}.png")

        poster_frame = init_frames + assign_frames + update_frames - 1
        shutil.copy(frame_dir / f"frame_{poster_frame:04d}.png", POSTER_DIR / "kmeans_animation_poster.png")
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
