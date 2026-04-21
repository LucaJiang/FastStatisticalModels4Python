"""Permutation test in Numba with ``prange`` parallelism.

One process, one copy of ``x`` shared read-only across threads, and the
outer loop over permutations distributed by Numba's thread pool. The
RNG is a deterministic LCG per iteration (seeded by ``base_seed + i``)
so the parallel runs are reproducible.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _splitmix64(z: np.uint64) -> np.uint64:
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))


@njit(cache=True, inline="always")
def _xorshift64(s: np.uint64) -> np.uint64:
    s ^= (s << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    s ^= s >> np.uint64(7)
    s ^= (s << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return s


@njit(cache=True, parallel=True, fastmath=True)
def run_permtest_numba(x: np.ndarray, n1: int, r: int, base_seed: int) -> np.ndarray:
    """Parallel Fisher–Yates per resample, deterministic per (base_seed, i).

    Seeding: we SplitMix64-hash ``(base_seed, i)`` so every iteration
    gets an independent high-quality stream, and then step the stream
    with xorshift64 inside the shuffle. Without the SplitMix step,
    nearby ``i`` values produce correlated permutations and the null
    distribution drifts — we got bitten by this in an earlier version.
    """
    n = x.shape[0]
    n2 = n - n1
    total = 0.0
    for j in range(n):
        total += x[j]
    out = np.empty(r, dtype=np.float64)

    for i in prange(r):
        seed_mix = _splitmix64(np.uint64(base_seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(i))
        state = seed_mix if seed_mix != np.uint64(0) else np.uint64(1)
        perm = np.empty(n, dtype=np.int64)
        for j in range(n):
            perm[j] = j
        # Fisher–Yates
        for j in range(n - 1, 0, -1):
            state = _xorshift64(state)
            kk = int(state % np.uint64(j + 1))
            tmp = perm[j]
            perm[j] = perm[kk]
            perm[kk] = tmp
        s1 = 0.0
        for j in range(n1):
            s1 += x[perm[j]]
        out[i] = s1 / n1 - (total - s1) / n2
    return out


def warmup() -> None:
    x = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    run_permtest_numba(x, 16, 4, 0)
