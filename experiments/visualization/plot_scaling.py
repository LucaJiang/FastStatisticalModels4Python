#!/usr/bin/env python3
"""Line plot: expects CSV with columns like impl, max_workers, median_s (custom benchmarks)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", type=str, default="Scaling (median_s vs workers)")
    args = p.parse_args()

    rows: list[dict[str, str]] = []
    with args.input.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        impl = row.get("impl", "impl")
        w = float(row.get("max_workers", row.get("workers", "0")))
        t = float(row["median_s"])
        series[impl].append((w, t))

    fig, ax = plt.subplots(figsize=(8, 4))
    for impl, pts in series.items():
        pts = sorted(pts, key=lambda z: z[0])
        ws = [p[0] for p in pts]
        ts = [p[1] for p in pts]
        ax.plot(ws, ts, marker="o", label=impl)
    ax.set_xlabel("workers")
    ax.set_ylabel("median_s")
    ax.set_title(args.title)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
