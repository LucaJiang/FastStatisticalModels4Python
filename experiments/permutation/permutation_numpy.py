"""Matrix formulation for feature-wise permutation tests."""

from __future__ import annotations

import numpy as np

from .permutation_reference import contrast_from_labels, observed_statistics, p_values_from_null


def contrast_matrix(labels: np.ndarray, *, r: int, seed: int) -> np.ndarray:
    """Return an ``R x n`` matrix of permuted mean-difference contrasts."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    w = np.empty((r, labels.shape[0]), dtype=np.float64)
    for i in range(r):
        w[i] = contrast_from_labels(rng.permutation(labels))
    return w


def matrix_null_statistics(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    r: int,
    seed: int,
) -> np.ndarray:
    return contrast_matrix(labels, r=r, seed=seed) @ np.asarray(x)


def matrix_p_values(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    r: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = observed_statistics(x, labels)
    null = matrix_null_statistics(x, labels, r=r, seed=seed)
    return observed, null, p_values_from_null(observed, null)


def matrix_p_values_batched(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    r: int,
    seed: int,
    batch_r: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Streaming p-values without materializing the full ``R x p`` null array."""
    x = np.asarray(x)
    observed = observed_statistics(x, labels)
    counts = np.zeros(x.shape[1], dtype=np.int64)
    done = 0
    while done < r:
        this_r = min(batch_r, r - done)
        null = matrix_null_statistics(x, labels, r=this_r, seed=seed + done)
        counts += np.sum(np.abs(null) >= np.abs(observed), axis=0)
        done += this_r
    return observed, (counts + 1.0) / (r + 1.0)
