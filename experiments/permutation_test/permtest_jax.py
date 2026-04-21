"""Permutation test in JAX: ``vmap`` across the R permutation seeds.

Two compiled variants are exposed so the talk can show how ``vmap``
decouples the math from the batch axis:

- ``make_permtest_jax_perm`` does a full ``jax.random.permutation``
  per seed (the JAX mirror of the NumPy naive code).

- ``make_permtest_jax_trick`` uses the same algorithmic observation as
  ``permtest_numpy.run_permtest_numpy_trick``: only ``S1`` matters, so we
  can ``jax.random.choice(..., replace=False)`` and sum a subset.
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
        raise ImportError("JAX required") from _ERR


def make_permtest_jax_perm():
    _require_jax()
    jax.config.update("jax_enable_x64", True)

    @partial(jax.jit, static_argnames=("n1", "r"))
    def run(x: "jnp.ndarray", n1: int, r: int, key: "jax.Array") -> "jnp.ndarray":
        keys = jax.random.split(key, r)
        n = x.shape[0]
        n2 = n - n1
        total = jnp.sum(x)

        def one(k):
            perm = jax.random.permutation(k, n)
            s1 = jnp.sum(x[perm[:n1]])
            return s1 / n1 - (total - s1) / n2

        return jax.vmap(one)(keys)

    return run


def make_permtest_jax_trick():
    _require_jax()
    jax.config.update("jax_enable_x64", True)

    @partial(jax.jit, static_argnames=("n1", "r"))
    def run(x: "jnp.ndarray", n1: int, r: int, key: "jax.Array") -> "jnp.ndarray":
        keys = jax.random.split(key, r)
        n = x.shape[0]
        n2 = n - n1
        total = jnp.sum(x)

        def one(k):
            idx = jax.random.choice(k, n, shape=(n1,), replace=False)
            s1 = jnp.sum(x[idx])
            return s1 / n1 - (total - s1) / n2

        return jax.vmap(one)(keys)

    return run


def run_permtest_jax(x: np.ndarray, n1: int, r: int, seed: int) -> np.ndarray:
    _require_jax()
    run = make_permtest_jax_perm()
    key = jax.random.PRNGKey(seed)
    xj = jnp.asarray(x, dtype=jnp.float64)
    out = run(xj, n1, r, key)
    jax.block_until_ready(out)
    return np.asarray(out)


def run_permtest_jax_trick(x: np.ndarray, n1: int, r: int, seed: int) -> np.ndarray:
    _require_jax()
    run = make_permtest_jax_trick()
    key = jax.random.PRNGKey(seed)
    xj = jnp.asarray(x, dtype=jnp.float64)
    out = run(xj, n1, r, key)
    jax.block_until_ready(out)
    return np.asarray(out)
