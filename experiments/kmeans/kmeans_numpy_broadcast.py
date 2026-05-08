"""Broadcast-distance k-means implementation."""

from __future__ import annotations

import numpy as np

from .kmeans_numpy import kmeans_numpy_naive


def _server_broadcast(
    x: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    centroids = init_centroids.astype(np.float64, copy=True)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    empty = 0
    for it in range(1, max_iter + 1):
        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
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


def kmeans_numpy_broadcast(*args, **kwargs):
    if len(args) >= 2 and isinstance(args[1], np.ndarray):
        x = args[0]
        init = args[1]
        return _server_broadcast(x, init, max_iter=kwargs.get("max_iter", 30), tol=kwargs.get("tol", 1e-6))
    return kmeans_numpy_naive(*args, **kwargs)
