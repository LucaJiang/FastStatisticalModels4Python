"""JAX k-means implementation for v3."""

from __future__ import annotations

import numpy as np


def kmeans_jax(
    x: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int = 30,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    import jax
    import jax.numpy as jnp

    use_dtype = jnp.float32 if dtype == "float32" else jnp.float64
    xj = jnp.asarray(x, dtype=use_dtype)
    cj = jnp.asarray(init_centroids, dtype=use_dtype)
    k = init_centroids.shape[0]

    def step(centroids, _):
        x_norm = jnp.sum(xj * xj, axis=1, keepdims=True)
        c_norm = jnp.sum(centroids * centroids, axis=1, keepdims=True).T
        labels = jnp.argmin(x_norm + c_norm - 2.0 * (xj @ centroids.T), axis=1)
        sums = jnp.zeros_like(centroids).at[labels].add(xj)
        counts = jnp.bincount(labels, length=k).reshape((k, 1))
        new_centroids = jnp.where(counts > 0, sums / jnp.maximum(counts, 1), centroids)
        empty = jnp.sum(counts[:, 0] == 0)
        return new_centroids, (labels, empty)

    run = jax.jit(lambda c: jax.lax.scan(step, c, None, length=max_iter))
    centroids, (labels_hist, empty_hist) = run(cj)
    labels = labels_hist[-1]
    diff = xj - centroids[labels]
    inertia = jnp.sum(diff * diff)
    jax.block_until_ready(inertia)
    return (
        np.asarray(centroids),
        np.asarray(labels),
        float(inertia),
        int(max_iter),
        int(np.asarray(empty_hist[-1])),
    )
