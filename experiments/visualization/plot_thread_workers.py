#!/usr/bin/env python3
"""Plot ThreadPool scaling under GIL and no-GIL Python builds."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import PY_BLUE, THREAD_TEAL, apply_style, strip_spines


LABELS = {
    "py312-gil": "Python 3.12 GIL build",
    "py314t-nogil": "Python 3.14t free-threaded",
}
COLORS = {
    "py312-gil": PY_BLUE,
    "py314t-nogil": THREAD_TEAL,
}


def _load(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    apply_style()
    rows = _load(args.input)
    if not rows:
        raise SystemExit("No rows found")

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        label = row["env_label"]
        series[label].append((int(row["max_workers"]), float(row["warm_median_s"])))
    for vals in series.values():
        vals.sort()

    n = int(rows[0]["n"])
    r = int(rows[0]["r"])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for label in ("py312-gil", "py314t-nogil"):
        if label not in series:
            continue
        workers = [w for w, _ in series[label]]
        times = [t for _, t in series[label]]
        ax.plot(
            workers,
            times,
            marker="o",
            lw=2.6,
            label=LABELS.get(label, label),
            color=COLORS.get(label),
        )
        for idx, (w, t) in enumerate(zip(workers, times)):
            offset = 0.035 if label == "py312-gil" else -0.045
            ax.text(w, t + offset, f"{t:.2f}s", ha="center", fontsize=9)
    ax.set_xlabel("ThreadPool workers")
    ax.set_ylabel("warm median time (seconds)")
    ax.set_title(f"Permutation thread scaling (n={n:,}, R={r:,})")
    ax.set_xticks(sorted({int(row["max_workers"]) for row in rows}))
    ax.margins(y=0.18)
    ax.grid(True, axis="y", alpha=0.3)
    strip_spines(ax)
    ax.legend(loc="best", framealpha=0.92, facecolor="white", edgecolor="#D8D1C4")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
