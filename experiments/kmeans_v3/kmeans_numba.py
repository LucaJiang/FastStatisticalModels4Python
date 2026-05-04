"""Numba k-means implementation for v3."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None
    prange = range


if njit is not None:

    @njit(parallel=True)
    def _assign(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        n, d = x.shape
        k = centroids.shape[0]
        labels = np.empty(n, dtype=np.int64)
        for i in prange(n):
            best = 0
            best_dist = 1.0e308
            for j in range(k):
                dist = 0.0
                for m in range(d):
                    diff = x[i, m] - centroids[j, m]
                    dist += diff * diff
                if dist < best_dist:
                    best = j
                    best_dist = dist
            labels[i] = best
        return labels


def kmeans_numba(
    x: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    if njit is None:
        raise RuntimeError("numba is not available")
    centroids = init_centroids.astype(np.float64, copy=True)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    empty = 0
    for it in range(1, max_iter + 1):
        labels = _assign(x, centroids)
        new_centroids = centroids.copy()
        empty = 0
        for j in range(centroids.shape[0]):
            mask = labels == j
            if np.any(mask):
                new_centroids[j] = x[mask].mean(axis=0)
            else:
                empty += 1
        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if shift <= tol:
            break
    inertia = float(np.sum((x - centroids[labels]) ** 2))
    return centroids, labels, inertia, it, empty
