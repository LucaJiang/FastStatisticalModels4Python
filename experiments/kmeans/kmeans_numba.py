"""k-means Lloyd compiled with `@njit`.

The structure mirrors ``kmeans_loops.py`` so attendees can visually
map `for` loops that are slow in pure Python to the same loops that
Numba turns into tight machine code.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _assign_labels(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> None:
    n, d = X.shape
    k = centroids.shape[0]
    for i in range(n):
        best_j = 0
        best = 1e300
        for j in range(k):
            s = 0.0
            for t in range(d):
                u = X[i, t] - centroids[j, t]
                s += u * u
            if s < best:
                best = s
                best_j = j
        labels[i] = best_j


@njit(cache=True, fastmath=True)
def _update_centroids(
    X: np.ndarray, labels: np.ndarray, k: int, centroids: np.ndarray
) -> None:
    n, d = X.shape
    sums = np.zeros((k, d), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for i in range(n):
        li = labels[i]
        counts[li] += 1
        for t in range(d):
            sums[li, t] += X[i, t]
    for j in range(k):
        if counts[j] > 0:
            for t in range(d):
                centroids[j, t] = sums[j, t] / counts[j]


@njit(cache=True, fastmath=True)
def _shift_sq(a: np.ndarray, b: np.ndarray) -> float:
    s = 0.0
    for j in range(a.shape[0]):
        for t in range(a.shape[1]):
            u = a[j, t] - b[j, t]
            s += u * u
    return s


@njit(cache=True, fastmath=True)
def _inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    s = 0.0
    for i in range(X.shape[0]):
        li = labels[i]
        for t in range(X.shape[1]):
            u = X[i, t] - centroids[li, t]
            s += u * u
    return s


@njit(cache=True, fastmath=True)
def _run(X: np.ndarray, centroids: np.ndarray, max_iter: int, tol: float):
    k = centroids.shape[0]
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int64)
    prev = np.empty_like(centroids)
    it = 0
    for it in range(max_iter):
        _assign_labels(X, centroids, labels)
        prev[:] = centroids
        _update_centroids(X, labels, k, centroids)
        if _shift_sq(prev, centroids) < tol:
            break
    _assign_labels(X, centroids, labels)
    inertia = _inertia(X, labels, centroids)
    return centroids, labels, inertia, it + 1


def kmeans_numba(
    X: np.ndarray,
    k: int,
    max_iter: int,
    init_centroids: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    centroids = init_centroids.astype(np.float64, copy=True)
    return _run(X, centroids, max_iter, tol)


def warmup() -> None:
    """Eager-compile the kernels on tiny arrays so we don't count compile time."""
    rng = np.random.default_rng(0)
    X_tiny = rng.standard_normal((16, 2))
    c_tiny = X_tiny[:2].copy()
    _run(X_tiny, c_tiny, 2, 1e-12)
