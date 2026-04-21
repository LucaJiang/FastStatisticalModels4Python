#!/usr/bin/env python3
"""Rough LOC metrics for talk narrative (not a quality score)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _count_lines(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    non_empty = sum(1 for ln in lines if ln.strip() and not ln.lstrip().startswith("#"))
    return len(lines), non_empty


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    files = [
        root / "kmeans" / "kmeans_numpy.py",
        root / "kmeans" / "kmeans_loops.py",
        root / "kmeans" / "kmeans_numba.py",
        root / "kmeans" / "kmeans_jax.py",
        root / "permutation_test" / "permtest_numpy.py",
        root / "permutation_test" / "permtest_multiprocessing.py",
        root / "permutation_test" / "permtest_freethreaded.py",
        root / "permutation_test" / "permtest_numba.py",
        root / "permutation_test" / "permtest_jax.py",
    ]

    rows: list[dict[str, object]] = []
    for f in files:
        if not f.exists():
            continue
        total, non_empty = _count_lines(f)
        rel = f.relative_to(root)
        rows.append(
            {
                "path": str(rel),
                "lines_total": total,
                "lines_non_empty_non_comment": non_empty,
            }
        )

    for r in rows:
        print(r)

    if args.output and rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
