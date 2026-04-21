"""k-means Lloyd written with explicit Python loops.

This file exists only to benchmark what the CPython interpreter (with or
without the 3.14 experimental JIT) can do on tight Python-level loops.
It is *intentionally* orders of magnitude slower than the NumPy or Numba
paths; cap `n_samples` at a few thousand when benchmarking.
"""

from __future__ import annotations

import numpy as np


def _assign_labels_py(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> None:
    n, d = X.shape
    k = centroids.shape[0]
    for i in range(n):
        best_j = 0
        best = 1e300
        xi = X[i]
        for j in range(k):
            cj = centroids[j]
            s = 0.0
            for t in range(d):
                u = xi[t] - cj[t]
                s += u * u
            if s < best:
                best = s
                best_j = j
        labels[i] = best_j


def _update_centroids_py(
    X: np.ndarray, labels: np.ndarray, k: int, out: np.ndarray
) -> np.ndarray:
    n, d = X.shape
    sums = np.zeros((k, d), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for i in range(n):
        li = int(labels[i])
        counts[li] += 1
        xi = X[i]
        for t in range(d):
            sums[li, t] += xi[t]
    for j in range(k):
        if counts[j] > 0:
            for t in range(d):
                out[j, t] = sums[j, t] / counts[j]
        # else: leave `out[j]` as-is (previous centroid), matching smart impl.
    return out


def kmeans_loops(
    X: np.ndarray,
    k: int,
    max_iter: int,
    init_centroids: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    centroids = init_centroids.astype(np.float64, copy=True)
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int64)
    new_c = centroids.copy()

    for it in range(max_iter):
        _assign_labels_py(X, centroids, labels)
        new_c[:] = centroids
        _update_centroids_py(X, labels, k, new_c)

        shift = 0.0
        for j in range(k):
            for t in range(centroids.shape[1]):
                u = centroids[j, t] - new_c[j, t]
                shift += u * u
        centroids = new_c.copy()
        if shift < tol:
            break

    _assign_labels_py(X, centroids, labels)
    inertia = 0.0
    for i in range(n):
        li = int(labels[i])
        for t in range(X.shape[1]):
            u = X[i, t] - centroids[li, t]
            inertia += u * u
    return centroids, labels, float(inertia), it + 1
