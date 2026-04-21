#!/usr/bin/env python3
"""Radar plot of the evaluation matrix (runtime / memory / debug / effort).

Scores are 0–1 (higher = better). ``runtime`` and ``memory`` are
derived from the sweep CSVs: the fastest / lightest implementation
scores 1.0, slower ones get proportionally lower scores on a log scale
so that a 10× slowdown maps to ~0.3. ``debuggability`` and ``effort``
are speaker judgement calls based on our experience preparing the talk.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Speaker-assigned subjective scores (0 = painful, 1 = painless).
# These reflect our experience porting real statistical code, not micro-benchmarks.
SUBJECTIVE = {
    # impl: (debug, effort)
    "NumPy (matmul)": (1.00, 0.95),  # regular Python, pdb just works
    "Numba @njit": (0.65, 0.80),     # nopython errors are cryptic until you get used to them
    "JAX scan+jit": (0.55, 0.55),    # functional/tracing mental model, shape errors, vmap axes
    "Multiprocessing": (0.75, 0.55), # picklability errors, worker crashes, but stdlib
    "Numba prange":  (0.60, 0.70),
    "JAX vmap":      (0.50, 0.55),
}

# Where each subjective bucket gets its runtime/memory numbers from.
KMEANS_BY_LABEL = {
    "NumPy (matmul)": "numpy_smart",
    "Numba @njit": "numba",
    "JAX scan+jit": "jax",
}
PERM_BY_LABEL = {
    "Multiprocessing": "multiprocessing",
    "Numba prange": "numba",
    "JAX vmap": "jax_trick",
}


def _load(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _logscore(value: float, best: float, worst: float) -> float:
    if value <= 0 or best <= 0:
        return 0.0
    lv = math.log10(value)
    lb = math.log10(best)
    lw = math.log10(worst)
    if lw <= lb:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (lv - lb) / (lw - lb)))


def _scores_from_sweep(
    rows: list[dict],
    label_to_impl: dict[str, str],
    size_key: str,
    size_val: int,
) -> tuple[dict[str, float], dict[str, float]]:
    sub = [r for r in rows if int(r[size_key]) == size_val]
    t = {r["impl"]: float(r["warm_median_s"]) for r in sub}
    m = {r["impl"]: float(r["tracemalloc_peak_mb"]) for r in sub}
    labels = list(label_to_impl)
    ts = {lab: t[label_to_impl[lab]] for lab in labels if label_to_impl[lab] in t}
    ms = {lab: m[label_to_impl[lab]] for lab in labels if label_to_impl[lab] in m}
    best_t, worst_t = min(ts.values()), max(ts.values())
    best_m, worst_m = min(ms.values()), max(ms.values())
    runtime = {lab: _logscore(ts[lab], best_t, worst_t) for lab in ts}
    memory = {lab: _logscore(ms[lab], best_m, worst_m) for lab in ms}
    return runtime, memory


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kmeans", type=Path, required=True)
    p.add_argument("--permtest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-kmeans", type=int, default=500_000)
    p.add_argument("--r-perm", type=int, default=10_000)
    args = p.parse_args()

    km = _load(args.kmeans)
    pm = _load(args.permtest)

    rt_k, mem_k = _scores_from_sweep(km, KMEANS_BY_LABEL, "n_samples", args.n_kmeans)
    rt_p, mem_p = _scores_from_sweep(pm, PERM_BY_LABEL, "r", args.r_perm)

    scores: dict[str, tuple[float, float, float, float]] = {}
    for label in list(rt_k):
        debug, effort = SUBJECTIVE[label]
        scores[label] = (rt_k[label], mem_k.get(label, 0.0), debug, effort)
    for label in list(rt_p):
        debug, effort = SUBJECTIVE[label]
        scores[label] = (rt_p[label], mem_p.get(label, 0.0), debug, effort)

    categories = ["runtime", "memory", "debuggability", "dev effort"]
    angles = np.linspace(0, 2 * math.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    palette = [
        "#1976D2", "#2E7D32", "#AD1457",
        "#EF6C00", "#6A1B9A", "#455A64",
    ]
    for i, (label, vals) in enumerate(scores.items()):
        row = list(vals) + [vals[0]]
        color = palette[i % len(palette)]
        ax.plot(angles, row, lw=2, label=label, color=color)
        ax.fill(angles, row, alpha=0.10, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title("Trade-off radar (0 = worst, 1 = best)\n"
                 f"runtime/memory from sweeps: kmeans N={args.n_kmeans:,}  perm R={args.r_perm:,}",
                 fontsize=11, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
