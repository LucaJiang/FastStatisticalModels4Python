"""Matrix-form permutation-test kernels used by the long-safe server runner."""

from __future__ import annotations

import concurrent.futures

import numpy as np


def make_expression(n: int, p: int, effect_size: float, affected_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.int8)
    labels[n // 2 :] = 1
    x = rng.normal(size=(n, p)).astype(np.float32)
    affected = np.zeros(p, dtype=bool)
    m = int(round(p * affected_fraction))
    if m:
        affected[:m] = True
        x[labels == 1, :m] += np.float32(effect_size)
    return x, labels, affected


def observed_stat(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return x[labels == 1].mean(axis=0) - x[labels == 0].mean(axis=0)


def contrast_batch(labels: np.ndarray, batch_r: int, rng: np.random.Generator) -> np.ndarray:
    n = labels.size
    n1 = int(labels.sum())
    w = np.empty((batch_r, n), dtype=np.float32)
    for b in range(batch_r):
        idx = rng.permutation(n)
        row = np.empty(n, dtype=np.float32)
        row[idx[:n1]] = 1.0 / n1
        row[idx[n1:]] = -1.0 / (n - n1)
        w[b] = row
    return w


def perm_matrix_cpu(x: np.ndarray, labels: np.ndarray, r: int, batch_r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs = np.abs(observed_stat(x, labels))
    exceed = np.zeros(x.shape[1], dtype=np.int64)
    done = 0
    while done < r:
        b = min(batch_r, r - done)
        w = contrast_batch(labels, b, rng)
        stats = np.abs(w @ x)
        exceed += np.sum(stats >= obs[None, :], axis=0)
        done += b
    return (exceed + 1) / (r + 1)


def perm_matrix_threaded(x: np.ndarray, labels: np.ndarray, r: int, batch_r: int, seed: int, workers: int) -> np.ndarray:
    chunks = [batch_r] * (r // batch_r)
    if r % batch_r:
        chunks.append(r % batch_r)

    def one(i_b):
        i, b = i_b
        return perm_matrix_cpu(x, labels, b, b, seed + 10_000 * i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pvals = list(pool.map(one, enumerate(chunks)))
    # This combines chunk-level p-values only for timing/shape diagnostics.
    return np.mean(np.vstack(pvals), axis=0)


def perm_matrix_gpu(x_np: np.ndarray, labels_np: np.ndarray, r: int, batch_r: int, seed: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(x_np, dtype=jnp.float32)
    labels = np.asarray(labels_np)
    n = labels.size
    n1 = int(labels.sum())
    obs = jnp.abs(jnp.mean(x[labels == 1], axis=0) - jnp.mean(x[labels == 0], axis=0))

    @jax.jit
    def batch_counts(keys):
        def one(k):
            perm = jax.random.permutation(k, n)
            row = jnp.empty((n,), dtype=jnp.float32)
            row = row.at[perm[:n1]].set(1.0 / n1)
            row = row.at[perm[n1:]].set(-1.0 / (n - n1))
            return row

        w = jax.vmap(one)(keys)
        stats = jnp.abs(w @ x)
        return jnp.sum(stats >= obs[None, :], axis=0)

    exceed = jnp.zeros((x_np.shape[1],), dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    done = 0
    while done < r:
        b = min(batch_r, r - done)
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, b)
        exceed = exceed + batch_counts(keys)
        done += b
    pvals = (exceed + 1) / (r + 1)
    jax.block_until_ready(pvals)
    return np.asarray(pvals)

