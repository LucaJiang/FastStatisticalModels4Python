#!/usr/bin/env python3
"""Bar plot of RSS after benchmark (optional column rss_mb_after)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", type=str, default="Process RSS after run (MB)")
    args = p.parse_args()

    rows: list[dict[str, str]] = []
    with args.input.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    labels = [row.get("impl", "?") for row in rows]
    vals = []
    for row in rows:
        v = row.get("rss_mb_after") or row.get("rss_mb")
        vals.append(float(v) if v not in (None, "") else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color="#55A868")
    ax.set_ylabel("MB (best-effort)")
    ax.set_title(args.title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
