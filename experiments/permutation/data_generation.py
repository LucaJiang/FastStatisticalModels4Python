"""Controlled data generators for permutation-test validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PermutationScenario:
    n: int
    p: int
    delta: float
    signal_fraction: float
    seed: int


def make_two_group_matrix(
    n: int,
    p: int,
    *,
    delta: float = 0.0,
    signal_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``X, labels, signal_mask`` for a two-sample feature-wise test."""
    if n < 4:
        raise ValueError("n must be at least 4")
    if p < 1:
        raise ValueError("p must be positive")
    if not 0 <= signal_fraction <= 1:
        raise ValueError("signal_fraction must be in [0, 1]")

    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.int8)
    labels[n // 2 :] = 1
    rng.shuffle(labels)

    x = rng.normal(0.0, 1.0, size=(n, p)).astype(np.float64, copy=False)
    signal_count = int(round(p * signal_fraction)) if delta else 0
    signal_count = min(p, max(0, signal_count))
    signal_mask = np.zeros(p, dtype=bool)
    if signal_count:
        signal_mask[:signal_count] = True
        x[np.ix_(labels == 1, signal_mask)] += delta
    return x, labels, signal_mask
