"""NumPy matmul-distance k-means implementation."""

from __future__ import annotations

import numpy as np


def kmeans_numpy_matmul(
    x: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    centroids = init_centroids.astype(np.float64, copy=True)
    x_norm = np.sum(x * x, axis=1)[:, None]
    labels = np.zeros(x.shape[0], dtype=np.int64)
    empty = 0
    for it in range(1, max_iter + 1):
        c_norm = np.sum(centroids * centroids, axis=1)[None, :]
        distances = x_norm + c_norm - 2.0 * (x @ centroids.T)
        labels = np.argmin(distances, axis=1)
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
