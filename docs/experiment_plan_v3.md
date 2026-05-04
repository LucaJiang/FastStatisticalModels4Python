# Experiment plan v3

The v3 experiments use two workloads to show the same workflow under different computational pressure.

## Workload 1: k-means

K-means represents iterative simulation pressure: convergence, initialization sensitivity, memory temporaries, and CPU/GPU break-even behavior.

Simulation data is a Gaussian mixture with known labels. The full design varies `N`, `d`, `K`, separation, imbalance, covariance shape, outlier fraction, and seed. Small and medium cases validate implementations before large timing runs.

Implementations:

- Reference Lloyd implementation for small data.
- NumPy broadcast distance computation.
- NumPy matmul distance identity.
- Numba CPU loops.
- JAX `jit` / `lax.scan` on CPU or GPU.

Validation records final inertia, iteration count, empty clusters, and adjusted Rand index when scikit-learn is available.

Planned figures:

- `kmeans_simulation_grid.png`
- `kmeans_runtime_cpu_gpu.png`
- `kmeans_memory_scaling.png`
- `kmeans_gpu_break_even.png`

## Workload 2: high-dimensional permutation test

Permutation tests represent resampling pressure: exact checks, null calibration, power, many features, many permutations, RNG policy, and memory planning.

The simulated data matrix is `X: n_samples x p_features` with two groups, optional sparse affected features, and optional block correlation.

The key rewrite is a contrast matrix:

```text
T_null = W @ X
```

where each row of `W` encodes one permuted two-sample contrast. This is the main GPU-friendly path. Naive `vmap(jax.random.permutation)` is an anti-pattern unless explicitly labeled as such.

Implementations:

- Exact/reference enumeration for tiny `n`.
- NumPy loop.
- NumPy batched/chunked matrix multiplication.
- Numba prange loop when available.
- ThreadPool version for free-threaded Python.
- JAX chunked matrix multiplication.

Validation records exact small-n agreement, null calibration, KS diagnostic, power, and affected/unaffected feature behavior.

Planned figures:

- `perm_null_calibration.png`
- `perm_power_curve.png`
- `perm_runtime_cpu_gpu.png`
- `perm_memory_chunking.png`
- `perm_gpu_break_even.png`
