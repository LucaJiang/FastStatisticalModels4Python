"""Simulation data for k-means v3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KMeansScenario:
    n: int = 10_000
    d: int = 10
    k: int = 5
    separation: float = 2.0
    imbalance: str = "balanced"
    covariance: str = "spherical"
    outlier_fraction: float = 0.0
    seed: int = 0


def cluster_weights(k: int, imbalance: str) -> np.ndarray:
    if imbalance == "balanced":
        weights = np.ones(k)
    elif imbalance == "90/10":
        weights = np.ones(k) * (0.10 / max(1, k - 1))
        weights[0] = 0.90
    elif imbalance == "long-tail":
        weights = 1.0 / np.arange(1, k + 1)
    else:
        raise ValueError(f"unknown imbalance: {imbalance}")
    return weights / weights.sum()


def _covariance_transform(rng: np.random.Generator, d: int, covariance: str) -> np.ndarray:
    if covariance == "spherical":
        return np.eye(d)
    if covariance == "anisotropic":
        return np.diag(np.linspace(0.5, 2.0, d))
    if covariance == "correlated":
        rho = 0.35
        idx = np.arange(d)
        cov = rho ** np.abs(idx[:, None] - idx[None, :])
        return np.linalg.cholesky(cov)
    raise ValueError(f"unknown covariance: {covariance}")


def make_gaussian_mixture(scenario: KMeansScenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(scenario.seed)
    weights = cluster_weights(scenario.k, scenario.imbalance)
    labels = rng.choice(scenario.k, size=scenario.n, p=weights)
    centers = rng.normal(size=(scenario.k, scenario.d)).astype(np.float64)
    centers *= scenario.separation / np.sqrt(max(1, scenario.d))
    transform = _covariance_transform(rng, scenario.d, scenario.covariance)
    x = rng.normal(size=(scenario.n, scenario.d)) @ transform.T
    x += centers[labels]
    if scenario.outlier_fraction > 0:
        n_out = int(round(scenario.n * scenario.outlier_fraction))
        if n_out:
            idx = rng.choice(scenario.n, size=n_out, replace=False)
            x[idx] += rng.normal(scale=8.0, size=(n_out, scenario.d))
    return x.astype(np.float64, copy=False), labels.astype(np.int64), centers


def initial_centroids(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=k, replace=False)
    return x[idx].astype(np.float64, copy=True)
