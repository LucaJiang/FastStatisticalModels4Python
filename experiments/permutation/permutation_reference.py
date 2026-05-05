"""Readable reference implementation for feature-wise permutation tests."""

from __future__ import annotations

import numpy as np


def contrast_from_labels(labels: np.ndarray) -> np.ndarray:
    """Build a mean-difference contrast for binary labels."""
    labels = np.asarray(labels)
    group1 = labels == 1
    group0 = ~group1
    n1 = int(group1.sum())
    n0 = int(group0.sum())
    if n0 == 0 or n1 == 0:
        raise ValueError("labels must contain both groups")
    w = np.empty(labels.shape[0], dtype=np.float64)
    w[group1] = 1.0 / n1
    w[group0] = -1.0 / n0
    return w


def observed_statistics(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return contrast_from_labels(labels) @ np.asarray(x)


def permutation_null_statistics(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    r: int,
    seed: int,
) -> np.ndarray:
    """Loop over permutations; this is intentionally close to the math."""
    x = np.asarray(x)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    null = np.empty((r, x.shape[1]), dtype=np.float64)
    for i in range(r):
        permuted = rng.permutation(labels)
        null[i] = contrast_from_labels(permuted) @ x
    return null


def p_values_from_null(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed)
    null = np.asarray(null)
    return (np.sum(np.abs(null) >= np.abs(observed), axis=0) + 1.0) / (null.shape[0] + 1.0)


def permutation_p_values(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    r: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = observed_statistics(x, labels)
    null = permutation_null_statistics(x, labels, r=r, seed=seed)
    return observed, null, p_values_from_null(observed, null)
