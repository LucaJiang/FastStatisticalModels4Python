"""k-means Lloyd in JAX: `jax.lax.scan` + `jax.jit`.

The Lloyd step is expressed functionally so the whole iteration fuses
into a single XLA program. We use the matmul-distance trick so the
computation stays BLAS-heavy rather than broadcasting a big tensor.
The starting centroids are passed in explicitly (no RNG inside the jit)
so JAX, NumPy, and Numba all start from the same point.
"""

from __future__ import annotations

from functools import partial

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:  # pragma: no cover
    jax = None
    jnp = None
    _ERR: Exception | None = e
else:
    _ERR = None


def _require_jax() -> None:
    if jax is None:
        raise ImportError("JAX required; see experiments/setup/requirements-jax.txt") from _ERR


def make_kmeans_jax_jitted():
    """Return a jitted (X, init_centroids, max_iter) -> (centroids, labels, inertia) callable."""
    _require_jax()
    jax.config.update("jax_enable_x64", True)

    @partial(jax.jit, static_argnames=("max_iter",))
    def run(X: "jnp.ndarray", init_centroids: "jnp.ndarray", max_iter: int):
        k = init_centroids.shape[0]
        X_sq = jnp.sum(X * X, axis=1, keepdims=True)  # (N, 1)

        def step(centroids, _):
            C_sq = jnp.sum(centroids * centroids, axis=1, keepdims=True).T  # (1, K)
            dists_sq = X_sq + C_sq - 2.0 * (X @ centroids.T)
            labels = jnp.argmin(dists_sq, axis=1)
            onehot = jax.nn.one_hot(labels, k, dtype=X.dtype)  # (N, K)
            counts = jnp.sum(onehot, axis=0)  # (K,)
            sums = onehot.T @ X  # (K, d)
            new_c = jnp.where(
                counts[:, None] > 0, sums / jnp.maximum(counts[:, None], 1.0), centroids
            )
            return new_c, None

        final_centroids, _ = jax.lax.scan(step, init_centroids, None, length=max_iter)
        C_sq = jnp.sum(final_centroids * final_centroids, axis=1, keepdims=True).T
        dists_sq = X_sq + C_sq - 2.0 * (X @ final_centroids.T)
        labels = jnp.argmin(dists_sq, axis=1)
        assigned = final_centroids[labels]
        inertia = jnp.sum((X - assigned) ** 2)
        return final_centroids, labels, inertia

    return run


def kmeans_jax(
    X: np.ndarray,
    k: int,
    max_iter: int,
    init_centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    _require_jax()
    jax.config.update("jax_enable_x64", True)
    run = make_kmeans_jax_jitted()
    X_j = jnp.asarray(X, dtype=jnp.float64)
    C_j = jnp.asarray(init_centroids, dtype=jnp.float64)
    c, lab, inertia = run(X_j, C_j, max_iter)
    jax.block_until_ready((c, lab, inertia))
    return np.asarray(c), np.asarray(lab), float(inertia), max_iter
