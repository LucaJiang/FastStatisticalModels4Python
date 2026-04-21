"""Two-sample 1-D data merged into a single vector for permutation benchmarks."""

from __future__ import annotations

import numpy as np


def make_two_sample_vector(
    n1: int,
    n2: int,
    mean1: float = 0.0,
    mean2: float = 0.5,
    std: float = 1.0,
    random_state: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Return (x, n1) where x concatenates group A (length n1) and group B (length n2).
    """
    rng = np.random.default_rng(random_state)
    a = rng.normal(mean1, std, size=n1)
    b = rng.normal(mean2, std, size=n2)
    x = np.concatenate([a, b]).astype(np.float64, copy=False)
    return x, n1
