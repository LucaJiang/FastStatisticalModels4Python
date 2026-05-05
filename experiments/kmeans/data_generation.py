"""Scenario-based Gaussian-mixture data for k-means validation."""

from __future__ import annotations

import numpy as np


def _cluster_sizes(n: int, k: int, imbalance: str) -> np.ndarray:
    if imbalance == "balanced":
        sizes = np.full(k, n // k, dtype=int)
        sizes[: n % k] += 1
        return sizes
    if imbalance == "90_10":
        if k == 1:
            return np.array([n], dtype=int)
        major = int(round(0.9 * n))
        rest = n - major
        sizes = np.full(k, max(1, rest // (k - 1)), dtype=int)
        sizes[0] = major
        while sizes.sum() > n:
            for j in range(k - 1, 0, -1):
                if sizes.sum() <= n:
                    break
                if sizes[j] > 1:
                    sizes[j] -= 1
        while sizes.sum() < n:
            sizes[-1] += 1
        return sizes
    raise ValueError(f"Unknown imbalance: {imbalance}")


def make_gaussian_mixture(
    n: int,
    d: int,
    k: int,
    separation: float,
    imbalance: str,
    outlier_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, labels, true_centers)`` for a controlled k-means scenario."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.maximum(norms, 1e-12) * separation * np.sqrt(d)
    sizes = _cluster_sizes(n, k, imbalance)

    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for idx, size in enumerate(sizes):
        chunks.append(centers[idx] + rng.normal(scale=1.0, size=(int(size), d)))
        labels.append(np.full(int(size), idx, dtype=np.int64))
    X = np.vstack(chunks)
    y = np.concatenate(labels)

    order = rng.permutation(n)
    X = X[order]
    y = y[order]

    n_out = int(round(outlier_fraction * n))
    if n_out > 0:
        out_idx = rng.choice(n, size=n_out, replace=False)
        span = max(6.0, separation * 4.0)
        X[out_idx] = rng.uniform(-span, span, size=(n_out, d))

    return X.astype(np.float64, copy=False), y.astype(np.int64, copy=False), centers.astype(np.float64)


def choose_initial_centroids(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_007)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].astype(np.float64, copy=True)
