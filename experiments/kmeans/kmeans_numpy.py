"""k-means: NumPy Lloyd implementations.

Two baselines are provided on purpose:

- `kmeans_numpy_naive`: the textbook broadcast pattern
  (`X[:, None, :] - C[None, :, :]`). Allocates an (N, K, d) tensor
  every iteration, so memory + bandwidth dominate at large N.

- `kmeans_numpy_smart`: the matmul identity
  `||x - c||^2 = ||x||^2 + ||c||^2 - 2 x @ c.T`. No (N, K, d) tensor;
  relies on BLAS for the dot product.

Both accept an explicit `init_centroids` so every implementation in the
talk shares the same starting point (→ inertia is directly comparable).
"""

from __future__ import annotations

import numpy as np


def _init_from_rng(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].astype(np.float64, copy=True)


def kmeans_numpy_naive(
    X: np.ndarray,
    k: int,
    max_iter: int,
    init_centroids: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Broadcast-based Lloyd. Returns (centroids, labels, inertia, n_iter)."""
    centroids = init_centroids.astype(np.float64, copy=True)
    labels = np.empty(X.shape[0], dtype=np.int64)

    for it in range(max_iter):
        diff = X[:, None, :] - centroids[None, :, :]
        dists_sq = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        labels = np.argmin(dists_sq, axis=1)

        new_centroids = centroids.copy()
        for kk in range(k):
            mask = labels == kk
            if mask.any():
                new_centroids[kk] = X[mask].mean(axis=0)

        shift = np.sum((centroids - new_centroids) ** 2)
        centroids = new_centroids
        if shift < tol:
            break

    assigned = centroids[labels]
    inertia = float(np.sum((X - assigned) ** 2))
    return centroids, labels, inertia, it + 1


def kmeans_numpy_smart(
    X: np.ndarray,
    k: int,
    max_iter: int,
    init_centroids: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Matmul-based distance. Same algorithm, far better constants.

    Update step keeps the small-K masked reduction (``X[mask].mean``)
    because ``np.add.at`` with fancy indexing is substantially slower on
    CPython than K ≤ a few dozen masked means.
    """
    centroids = init_centroids.astype(np.float64, copy=True)
    X_sq = np.einsum("ij,ij->i", X, X)[:, None]  # (N, 1)
    labels = np.empty(X.shape[0], dtype=np.int64)

    for it in range(max_iter):
        C_sq = np.einsum("ij,ij->i", centroids, centroids)[None, :]  # (1, K)
        dists_sq = X_sq + C_sq - 2.0 * (X @ centroids.T)
        labels = np.argmin(dists_sq, axis=1)

        new_centroids = centroids.copy()
        for kk in range(k):
            mask = labels == kk
            if mask.any():
                new_centroids[kk] = X[mask].mean(axis=0)

        shift = np.sum((centroids - new_centroids) ** 2)
        centroids = new_centroids
        if shift < tol:
            break

    C_sq = np.einsum("ij,ij->i", centroids, centroids)[None, :]
    dists_sq = X_sq + C_sq - 2.0 * (X @ centroids.T)
    labels = np.argmin(dists_sq, axis=1)
    assigned = centroids[labels]
    inertia = float(np.sum((X - assigned) ** 2))
    return centroids, labels, inertia, it + 1


def kmeans_numpy(
    X: np.ndarray,
    k: int,
    max_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Back-compat wrapper: naive path, rng-driven init."""
    init = _init_from_rng(X, k, rng)
    centroids, labels, inertia, _ = kmeans_numpy_naive(X, k, max_iter, init)
    return centroids, labels, inertia
