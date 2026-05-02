#!/usr/bin/env python3
"""Plot permutation test results from a sweep CSV.

Produces:

- ``perm_scaling.png``: warm runtime vs R per implementation.
- ``perm_speedup.png``: speedup vs serial NumPy-naive at largest R.
- ``perm_memory.png``: tracemalloc peak vs implementation at largest R.
- ``perm_cold_vs_warm.png``: cold vs warm at largest R.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import (
    GRAY,
    JAX_BERRY,
    LOOP_PURPLE,
    NUMBA_GREEN,
    PROCESS_ORANGE,
    PY_BLUE,
    PY_GOLD,
    THREAD_TEAL,
    apply_style,
    strip_spines,
)

IMPL_COLORS = {
    "numpy_naive": PY_GOLD,
    "numpy_batched": PROCESS_ORANGE,
    "numpy_trick": PY_BLUE,
    "numpy_trick_batched": THREAD_TEAL,
    "multiprocessing": GRAY,
    "threads": "#2C6F7E",
    "numba": NUMBA_GREEN,
    "jax_perm": JAX_BERRY,
    "jax_trick": LOOP_PURPLE,
}
IMPL_ORDER = [
    "numpy_naive",
    "numpy_batched",
    "numpy_trick",
    "numpy_trick_batched",
    "multiprocessing",
    "threads",
    "numba",
    "jax_perm",
    "jax_trick",
]
IMPL_LABEL = {
    "numpy_naive": "NumPy loop",
    "numpy_batched": "NumPy batched",
    "numpy_trick": "NumPy trick",
    "numpy_trick_batched": "NumPy trick+batch",
    "multiprocessing": "Multiprocessing",
    "threads": "ThreadPool",
    "numba": "Numba prange",
    "jax_perm": "JAX vmap (perm)",
    "jax_trick": "JAX vmap (trick)",
}


def _load(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _by_impl(rows, key="r"):
    s = defaultdict(list)
    for row in rows:
        s[row["impl"]].append((int(row[key]), row))
    for v in s.values():
        v.sort(key=lambda z: z[0])
    return s


def plot_scaling(rows, output: Path) -> None:
    series = _by_impl(rows)
    fig, ax = plt.subplots(figsize=(11, 5.9))
    for impl in IMPL_ORDER:
        if impl not in series:
            continue
        rs = [p[0] for p in series[impl]]
        ts = [float(p[1]["warm_median_s"]) for p in series[impl]]
        ax.plot(rs, ts, marker="o", lw=2, label=IMPL_LABEL[impl],
                color=IMPL_COLORS[impl])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("R (number of permutations)")
    ax.set_ylabel("warm median time (seconds, log)")
    ax.set_title("Permutation test: runtime vs R (n=10,000, n1=n2=5000)")
    ax.grid(True, which="both", alpha=0.3)
    strip_spines(ax)
    ax.legend(loc="best", fontsize=10, framealpha=0.92, facecolor="white", edgecolor="#D8D1C4")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")


def plot_speedup(rows, output: Path) -> None:
    biggest_r = max(int(r["r"]) for r in rows)
    sub = [r for r in rows if int(r["r"]) == biggest_r]
    impl_t = {r["impl"]: float(r["warm_median_s"]) for r in sub}
    base = impl_t.get("numpy_naive")
    if base is None:
        base = min(impl_t.values())
    impls = [i for i in IMPL_ORDER if i in impl_t]
    x = np.arange(len(impls))
    speed = [base / impl_t[i] for i in impls]

    fig, ax = plt.subplots(figsize=(11, 5.9))
    bars = ax.bar(x, speed, color=[IMPL_COLORS[i] for i in impls])
    ax.axhline(1.0, color="black", lw=0.7, ls=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([IMPL_LABEL[i] for i in impls], rotation=25, ha="right")
    ax.set_ylabel("speedup vs NumPy naive loop (higher is faster)")
    ax.set_title(f"Permutation speedup at R={biggest_r:,} (n=10k)")
    ax.grid(True, axis="y", alpha=0.3)
    strip_spines(ax)
    for b, s in zip(bars, speed):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                f"{s:.1f}×", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")


def plot_memory(rows, output: Path) -> None:
    """Peak memory per implementation.

    Two bars per impl:
    - ``tracemalloc_peak_mb``: Python-level allocation in the running
      interpreter. Captures big broadcast / batch allocations.
    - ``children_rss_peak_mb``: RSS summed across worker processes
      (recorded only for multiprocessing), which is exactly the story
      of *copies vs shared data*.

    Child RSS is invisible to both tracemalloc and the parent's
    ru_maxrss; we sample it via psutil during the multiprocessing run.
    """
    biggest_r = max(int(r["r"]) for r in rows)
    sub = [r for r in rows if int(r["r"]) == biggest_r]
    impl_m = {r["impl"]: float(r["tracemalloc_peak_mb"]) for r in sub}
    impl_ch = {}
    for r in sub:
        v = r.get("children_rss_peak_mb", "")
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            impl_ch[r["impl"]] = v

    impls = [i for i in IMPL_ORDER if i in impl_m]
    x = np.arange(len(impls))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.9))
    ax.bar(x - width / 2, [impl_m[i] for i in impls], width,
           color=PY_BLUE, label="tracemalloc peak (parent process)")
    child_vals = [impl_ch.get(i, 0.0) for i in impls]
    if any(child_vals):
        ax.bar(x + width / 2, child_vals, width,
               color=PROCESS_ORANGE, label="child processes RSS (psutil)")
    ax.set_xticks(x)
    ax.set_xticklabels([IMPL_LABEL[i] for i in impls], rotation=25, ha="right")
    ax.set_ylabel("memory (MiB, log)")
    ax.set_yscale("log")
    ax.set_title(f"Permutation peak memory at R={biggest_r:,} (n=10k)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    for xi, v in enumerate([impl_m[i] for i in impls]):
        ax.text(xi - width / 2, max(v, 0.1) * 1.15, f"{v:.1f}",
                ha="center", fontsize=8)
    for xi, v in enumerate(child_vals):
        if v > 0:
            ax.text(xi + width / 2, v * 1.15, f"{v:.0f}",
                    ha="center", fontsize=8)
    strip_spines(ax)
    ax.legend(loc="upper right", framealpha=0.92, facecolor="white", edgecolor="#D8D1C4")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")


def plot_cold_vs_warm(rows, output: Path) -> None:
    biggest_r = max(int(r["r"]) for r in rows)
    sub = [r for r in rows if int(r["r"]) == biggest_r]
    impls = [i for i in IMPL_ORDER if any(r["impl"] == i for r in sub)]
    by_impl = {r["impl"]: r for r in sub}
    x = np.arange(len(impls))
    width = 0.38
    cold = [float(by_impl[i]["cold_s"]) for i in impls]
    warm = [float(by_impl[i]["warm_median_s"]) for i in impls]

    fig, ax = plt.subplots(figsize=(11, 5.9))
    ax.bar(x - width / 2, cold, width, color="#D32F2F", label="cold (1st call)")
    ax.bar(x + width / 2, warm, width, color="#388E3C", label="warm median")
    ax.set_xticks(x)
    ax.set_xticklabels([IMPL_LABEL[i] for i in impls], rotation=25, ha="right")
    ax.set_ylabel("seconds")
    ax.set_yscale("log")
    ax.set_title(f"Permutation cold vs warm at R={biggest_r:,}")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    strip_spines(ax)
    ax.legend(framealpha=0.92, facecolor="white", edgecolor="#D8D1C4")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    apply_style()
    rows = _load(args.input)
    plot_scaling(rows, args.output_dir / "perm_scaling.png")
    plot_speedup(rows, args.output_dir / "perm_speedup.png")
    plot_memory(rows, args.output_dir / "perm_memory.png")
    plot_cold_vs_warm(rows, args.output_dir / "perm_cold_vs_warm.png")


if __name__ == "__main__":
    main()
