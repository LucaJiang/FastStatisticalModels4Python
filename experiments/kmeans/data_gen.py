"""Synthetic clustered data for k-means benchmarks."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs


def make_cluster_data(
    n_samples: int,
    n_features: int,
    centers: int,
    cluster_std: float = 1.0,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with X shape (n_samples, n_features), float64."""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return X.astype(np.float64, copy=False), y.astype(np.int64, copy=False)
