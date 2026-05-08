"""k-means Lloyd compiled with `@njit`.

The implementation keeps the assignment and update loops explicit so attendees
can map the statistical definition to the machine-code path Numba generates.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


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


@njit(cache=True, fastmath=True, parallel=True)
def _assign_parallel(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
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


def _server_numba(
    x: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    centroids = init_centroids.astype(np.float64, copy=True)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    empty = 0
    for it in range(1, max_iter + 1):
        labels = _assign_parallel(x, centroids)
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


def kmeans_numba(
    X: np.ndarray,
    k: int | np.ndarray,
    max_iter: int = 30,
    init_centroids: np.ndarray | None = None,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    if isinstance(k, np.ndarray):
        return _server_numba(X, k, max_iter=max_iter, tol=max(tol, 1e-6))
    if init_centroids is None:
        raise TypeError("init_centroids is required for the MacBook kmeans_numba signature")
    centroids = init_centroids.astype(np.float64, copy=True)
    return _run(X, centroids, max_iter, tol)


def warmup() -> None:
    """Eager-compile the kernels on tiny arrays so we don't count compile time."""
    rng = np.random.default_rng(0)
    X_tiny = rng.standard_normal((16, 2))
    c_tiny = X_tiny[:2].copy()
    _run(X_tiny, c_tiny, 2, 1e-12)
    _server_numba(X_tiny, c_tiny, max_iter=2)
