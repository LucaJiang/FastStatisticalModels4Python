# Guideline of this project

Use `conda` to manage Python environments for this repository.

## Environments on this machine

```bash
conda activate py312   # baseline: NumPy / Numba / JAX / plotting
conda activate py314   # standard CPython 3.14 (GIL build)
conda activate py314t  # CPython 3.14 free-threaded build
```

## What each env is for

- `py312`: main development environment. Use this for NumPy, Numba, JAX, plotting, and most benchmark runs.
- `py314`: standard CPython 3.14 environment for `kmeans_loops` and other pure-Python 3.14 checks.
- `py314t`: free-threaded CPython 3.14 environment for the thread-based permutation benchmark.

## Important notes

- The current `conda-forge` `py314` build exposes `sys._jit`, but `sys._jit.is_available()` is still `False` on this machine. Treat it as a standard 3.14 environment, not a confirmed JIT-enabled build.
- `py314t` is confirmed free-threaded: `sys._is_gil_enabled()` returns `False`.
- The benchmark drivers now lazy-import optional backends, so `py314` does not need `numba`, and `py314t` does not need `numba` or `jax` just to run `loops` or `threads`.

## Verified commands

```bash
conda run -n py312 python experiments/kmeans/bench_kmeans.py \
  --impl numpy_smart numba jax --n-samples 200 --n-features 4 --k 3 --centers 3 \
  --max-iter 5 --warmup 1 --repeat 1

conda run -n py314 python experiments/kmeans/bench_kmeans.py \
  --impl loops --n-samples 200 --n-features 4 --k 3 --centers 3 \
  --max-iter 5 --loops-max-n 200 --warmup 1 --repeat 1

conda run -n py314t python experiments/permutation_test/bench_permtest.py \
  --impl numpy_trick threads --n1 40 --n2 40 --r 64 --warmup 1 --repeat 1 --max-workers 2
```
