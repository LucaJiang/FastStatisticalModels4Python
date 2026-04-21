#!/usr/bin/env python3
"""Bar plot of median runtime from bench CSV (expects 'impl' and 'median_s' columns)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", type=str, default="Median runtime (s)")
    args = p.parse_args()

    rows: list[dict[str, str]] = []
    with args.input.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    labels = [row.get("impl", "?") for row in rows]
    vals = [float(row["median_s"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color="#4C72B0")
    ax.set_ylabel("seconds")
    ax.set_title(args.title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
