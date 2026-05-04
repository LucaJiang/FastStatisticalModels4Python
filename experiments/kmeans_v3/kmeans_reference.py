"""Readable k-means reference implementation for small v3 cases."""

from __future__ import annotations

import numpy as np


def assign_labels_loop(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    labels = np.empty(x.shape[0], dtype=np.int64)
    for i in range(x.shape[0]):
        best = 0
        best_dist = float("inf")
        for j in range(centroids.shape[0]):
            dist = float(np.sum((x[i] - centroids[j]) ** 2))
            if dist < best_dist:
                best = j
                best_dist = dist
        labels[i] = best
    return labels


def kmeans_reference(
    x: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    centroids = init_centroids.astype(np.float64, copy=True)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for it in range(1, max_iter + 1):
        labels = assign_labels_loop(x, centroids)
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
    diff = x - centroids[labels]
    inertia = float(np.sum(diff * diff))
    return centroids, labels, inertia, it, empty
