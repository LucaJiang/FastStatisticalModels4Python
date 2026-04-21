#!/usr/bin/env python3
"""Plot k-means results from a sweep CSV.

Produces four figures:

- ``kmeans_scaling.png``: log-log runtime vs N per implementation.
- ``kmeans_speedup.png``: speedup vs ``numpy_naive`` per N, grouped bars.
- ``kmeans_cold_vs_warm.png``: cold (first call) vs warm median, per
  implementation at the largest N — makes JAX / Numba compile cost
  explicit.
- ``kmeans_memory.png``: ``tracemalloc`` peak per (impl, N).

Expects columns produced by ``sweep_kmeans.py``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IMPL_COLORS = {
    "numpy_naive": "#E65100",
    "numpy_smart": "#1976D2",
    "loops": "#6A1B9A",
    "numba": "#2E7D32",
    "jax": "#AD1457",
}
IMPL_ORDER = ["numpy_naive", "numpy_smart", "loops", "numba", "jax"]
IMPL_LABEL = {
    "numpy_naive": "NumPy (naive)",
    "numpy_smart": "NumPy (matmul)",
    "loops": "Python loops",
    "numba": "Numba @njit",
    "jax": "JAX scan+jit",
}


def _load(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _by_impl(rows, key="n_samples"):
    s = defaultdict(list)
    for row in rows:
        s[row["impl"]].append((int(row[key]), row))
    for v in s.values():
        v.sort(key=lambda z: z[0])
    return s


def plot_scaling(rows, output: Path) -> None:
    series = _by_impl(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for impl in IMPL_ORDER:
        if impl not in series:
            continue
        ns = [p[0] for p in series[impl]]
        ts = [float(p[1]["warm_median_s"]) for p in series[impl]]
        ax.plot(ns, ts, marker="o", lw=2, label=IMPL_LABEL[impl],
                color=IMPL_COLORS[impl])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (data points)")
    ax.set_ylabel("warm median time (s, log)")
    ax.set_title("k-means runtime vs N (Lloyd, k=5, d=10, max_iter=30)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def plot_speedup(rows, output: Path) -> None:
    by_n: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["impl"] == "loops":
            continue
        by_n[int(row["n_samples"])][row["impl"]] = float(row["warm_median_s"])

    ns = sorted(by_n)
    impls = [i for i in IMPL_ORDER if i != "loops" and any(i in by_n[n] for n in ns)]
    width = 0.8 / max(1, len(impls))
    x = np.arange(len(ns))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, impl in enumerate(impls):
        vals = []
        for n in ns:
            baseline = by_n[n].get("numpy_naive")
            here = by_n[n].get(impl)
            vals.append(baseline / here if (baseline and here) else 0.0)
        ax.bar(x + i * width, vals, width, color=IMPL_COLORS[impl],
               label=IMPL_LABEL[impl])
    ax.axhline(1.0, color="black", lw=0.6, ls=":", alpha=0.6)
    ax.set_xticks(x + (len(impls) - 1) * width / 2)
    ax.set_xticklabels([f"N={n:,}" for n in ns])
    ax.set_ylabel("speedup vs NumPy-naive  (higher = faster)")
    ax.set_title("k-means speedup over the textbook NumPy baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def plot_cold_vs_warm(rows, output: Path) -> None:
    """Show cold vs warm at the **smallest** main-sweep N.

    Compile/trace cost is a fixed one-time hit; at small N the hit
    dominates relative to the bulk compute, which is exactly where the
    story lands. Use N=1e6 and the bar heights would be almost equal.
    """
    by_n: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        by_n[int(row["n_samples"])][row["impl"]] = {
            "cold": float(row["cold_s"]),
            "warm": float(row["warm_median_s"]),
        }
    candidate_ns = sorted(
        [n for n in by_n if any(i != "loops" for i in by_n[n])]
    )
    target_n = candidate_ns[0]  # smallest
    impls = [i for i in IMPL_ORDER if i in by_n[target_n] and i != "loops"]
    x = np.arange(len(impls))
    width = 0.38
    cold = [by_n[target_n][i]["cold"] for i in impls]
    warm = [by_n[target_n][i]["warm"] for i in impls]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - width / 2, cold, width, color="#D32F2F", label="cold (1st call)")
    ax.bar(x + width / 2, warm, width, color="#388E3C", label="warm median")
    ax.set_xticks(x)
    ax.set_xticklabels([IMPL_LABEL[i] for i in impls], rotation=15, ha="right")
    ax.set_ylabel("seconds (log)")
    ax.set_yscale("log")
    ax.set_title(f"k-means cold vs warm (N={target_n:,}) — JIT tax is visible at small N")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()
    for xi, (c, w_) in enumerate(zip(cold, warm)):
        ax.text(xi - width / 2, c * 1.08, f"{c:.3f}s", ha="center", fontsize=8)
        ax.text(xi + width / 2, w_ * 1.08, f"{w_:.3f}s", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def plot_memory(rows, output: Path) -> None:
    series = _by_impl(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for impl in IMPL_ORDER:
        if impl not in series:
            continue
        ns = [p[0] for p in series[impl]]
        ms = [float(p[1]["tracemalloc_peak_mb"]) for p in series[impl]]
        ax.plot(ns, ms, marker="s", lw=2, label=IMPL_LABEL[impl],
                color=IMPL_COLORS[impl])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (data points)")
    ax.set_ylabel("tracemalloc peak (MiB, log)")
    ax.set_title("k-means Python-level peak memory vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    rows = _load(args.input)
    plot_scaling(rows, args.output_dir / "kmeans_scaling.png")
    plot_speedup(rows, args.output_dir / "kmeans_speedup.png")
    plot_cold_vs_warm(rows, args.output_dir / "kmeans_cold_vs_warm.png")
    plot_memory(rows, args.output_dir / "kmeans_memory.png")


if __name__ == "__main__":
    main()
