"""Permutation test: the four NumPy variants.

Four flavors, deliberately. They make the talk's most important point:
an algorithmic rethink usually beats any parallelism.

- ``run_permtest_numpy_naive``:
      baseline loop – one ``rng.permutation(n)`` + fancy indexing per run.
      This is how people write it first.

- ``run_permtest_numpy_batched``:
      vectorized across R permutations with ``np.argsort(rng.random((R, n)))``.
      One BLAS-ish bulk shuffle instead of R small shuffles.

- ``run_permtest_numpy_trick``:
      algorithmic insight. ``mean(A) - mean(B)`` only depends on
      ``S1 = sum(x[perm][:n1])`` because ``sum(x)`` is fixed. Sampling
      ``n1`` indices without replacement and summing that subset is
      equivalent and O(n1) per draw, not O(n).

- ``run_permtest_numpy_trick_batched``:
      the same trick, vectorized across R using ``np.argpartition`` on a
      random matrix (draw the n1 smallest keys → the "left" partition).
      Combines algorithmic and vectorized wins.
"""

from __future__ import annotations

import numpy as np


def run_permtest_numpy_naive(x: np.ndarray, n1: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    n2 = n - n1
    out = np.empty(r, dtype=np.float64)
    total = float(x.sum())
    for i in range(r):
        p = rng.permutation(n)
        s1 = float(x[p[:n1]].sum())
        out[i] = s1 / n1 - (total - s1) / n2
    return out


def run_permtest_numpy_batched(x: np.ndarray, n1: int, r: int, seed: int) -> np.ndarray:
    """Bulk-permute R × n keys, argsort columns, take first n1 of each row."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    n2 = n - n1
    keys = rng.random((r, n))
    perms = np.argsort(keys, axis=1)  # (R, n)
    # Gather values in batch, then split.
    x_batched = x[perms]  # (R, n)
    total = x.sum()
    s1 = x_batched[:, :n1].sum(axis=1)
    return s1 / n1 - (total - s1) / n2


def run_permtest_numpy_trick(x: np.ndarray, n1: int, r: int, seed: int) -> np.ndarray:
    """Algorithmic win: only the subset sum matters."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    n2 = n - n1
    total = float(x.sum())
    out = np.empty(r, dtype=np.float64)
    for i in range(r):
        idx = rng.choice(n, size=n1, replace=False)
        s1 = float(x[idx].sum())
        out[i] = s1 / n1 - (total - s1) / n2
    return out


def run_permtest_numpy_trick_batched(
    x: np.ndarray, n1: int, r: int, seed: int
) -> np.ndarray:
    """Batched subset-sum: `argpartition` gives the n1 smallest keys per row
    in O(n) each; summing along those indices gives S1 per row.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    n2 = n - n1
    keys = rng.random((r, n))
    idx = np.argpartition(keys, kth=n1 - 1, axis=1)[:, :n1]  # (R, n1)
    s1 = x[idx].sum(axis=1)  # (R,)
    total = float(x.sum())
    return s1 / n1 - (total - s1) / n2


# Back-compat name used by old bench script.
run_permtest_numpy = run_permtest_numpy_naive
