#!/usr/bin/env python3
"""Plot high-dimensional k-means shape experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plot_kmeans import IMPL_COLORS, IMPL_LABEL, IMPL_ORDER
from plot_style import apply_style, strip_spines


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
        series[row["impl"]].append(
            (int(row["n_samples"]), float(row["warm_median_s"]))
        )
    for vals in series.values():
        vals.sort()

    first = rows[0]
    k = int(first["k"])
    d = int(first["n_features"])
    max_iter = int(first["max_iter"])

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for impl in IMPL_ORDER:
        if impl not in series:
            continue
        ns = [n for n, _ in series[impl]]
        ts = [t for _, t in series[impl]]
        ax.plot(
            ns,
            ts,
            marker="o",
            lw=2.4,
            label=IMPL_LABEL[impl],
            color=IMPL_COLORS[impl],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (data points)")
    ax.set_ylabel("warm median time (seconds, log)")
    ax.set_title(f"k-means shape stress: K={k}, d={d}, max_iter={max_iter}")
    ax.grid(True, which="both", alpha=0.3)
    strip_spines(ax)
    ax.legend(loc="best", framealpha=0.92, facecolor="white", edgecolor="#D8D1C4")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
