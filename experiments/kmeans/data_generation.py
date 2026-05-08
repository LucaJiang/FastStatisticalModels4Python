"""Scenario-based Gaussian-mixture data for k-means validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KMeansScenario:
    """Server-scale k-means scenario with explicit simulation knobs."""

    n: int = 10_000
    d: int = 10
    k: int = 5
    separation: float = 2.0
    imbalance: str = "balanced"
    covariance: str = "spherical"
    outlier_fraction: float = 0.0
    seed: int = 0


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


def _cluster_weights(k: int, imbalance: str) -> np.ndarray:
    if imbalance == "balanced":
        weights = np.ones(k)
    elif imbalance in {"90/10", "90_10"}:
        weights = np.ones(k) * (0.10 / max(1, k - 1))
        weights[0] = 0.90
    elif imbalance == "long-tail":
        weights = 1.0 / np.arange(1, k + 1)
    else:
        raise ValueError(f"Unknown imbalance: {imbalance}")
    return weights / weights.sum()


def _covariance_transform(d: int, covariance: str) -> np.ndarray:
    if covariance == "spherical":
        return np.eye(d)
    if covariance == "anisotropic":
        return np.diag(np.linspace(0.5, 2.0, d))
    if covariance == "correlated":
        rho = 0.35
        idx = np.arange(d)
        cov = rho ** np.abs(idx[:, None] - idx[None, :])
        return np.linalg.cholesky(cov)
    raise ValueError(f"Unknown covariance: {covariance}")


def _make_scenario_gaussian_mixture(scenario: KMeansScenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(scenario.seed)
    weights = _cluster_weights(scenario.k, scenario.imbalance)
    labels = rng.choice(scenario.k, size=scenario.n, p=weights)
    centers = rng.normal(size=(scenario.k, scenario.d)).astype(np.float64)
    centers *= scenario.separation / np.sqrt(max(1, scenario.d))
    transform = _covariance_transform(scenario.d, scenario.covariance)
    x = rng.normal(size=(scenario.n, scenario.d)) @ transform.T
    x += centers[labels]
    if scenario.outlier_fraction > 0:
        n_out = int(round(scenario.n * scenario.outlier_fraction))
        if n_out:
            idx = rng.choice(scenario.n, size=n_out, replace=False)
            x[idx] += rng.normal(scale=8.0, size=(n_out, scenario.d))
    return x.astype(np.float64, copy=False), labels.astype(np.int64), centers


def make_gaussian_mixture(
    n: int | KMeansScenario,
    d: int | None = None,
    k: int | None = None,
    separation: float | None = None,
    imbalance: str | None = None,
    outlier_fraction: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, labels, true_centers)`` for a controlled k-means scenario."""
    if isinstance(n, KMeansScenario):
        return _make_scenario_gaussian_mixture(n)
    if d is None or k is None or separation is None or imbalance is None or outlier_fraction is None or seed is None:
        raise TypeError("make_gaussian_mixture requires either KMeansScenario or n, d, k, separation, imbalance, outlier_fraction, seed")
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


def initial_centroids(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Choose fixed initial centroids without the MacBook offset convention."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=k, replace=False)
    return x[idx].astype(np.float64, copy=True)
