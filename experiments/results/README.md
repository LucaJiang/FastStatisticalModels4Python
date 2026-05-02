# 本机基准结果（v2 refresh）

## 环境

- `py312`: Python 3.12.2, NumPy 1.26.4, Numba 0.59.1, JAX 0.4.25 (CPU), scikit-learn 1.7.1
- `py314`: Python 3.14.4 GIL build; `sys._jit.is_available()` is `False`
- `py314t`: Python 3.14.0 free-threading build; `sys._is_gil_enabled()` is `False`
- Hardware: Apple Silicon, 8 logical cores, macOS ARM64
- JAX timings call `jax.block_until_ready`.
- Each implementation/size runs in a fresh subprocess so large allocations do not contaminate later measurements.

## 图表（演讲直接引用）

| 文件 | 内容 |
|------|------|
| [`v2/kmeans_scaling.png`](v2/kmeans_scaling.png) | k-means runtime vs N |
| [`v2/kmeans_speedup.png`](v2/kmeans_speedup.png) | k-means speedup over NumPy naive |
| [`v2/kmeans_cold_vs_warm.png`](v2/kmeans_cold_vs_warm.png) | cold first call vs warm median |
| [`v2/kmeans_memory.png`](v2/kmeans_memory.png) | k-means Python-level peak memory |
| [`v2/kmeans_shape_k50_d100.png`](v2/kmeans_shape_k50_d100.png) | K=50, d=100 k-means shape stress |
| [`v2/perm_scaling.png`](v2/perm_scaling.png) | permutation runtime vs R |
| [`v2/perm_speedup.png`](v2/perm_speedup.png) | permutation speedup over NumPy loop |
| [`v2/perm_memory.png`](v2/perm_memory.png) | permutation peak memory, including child RSS |
| [`v2/perm_cold_vs_warm.png`](v2/perm_cold_vs_warm.png) | permutation cold vs warm |
| [`v2/perm_threads_py312_py314t.png`](v2/perm_threads_py312_py314t.png) | GIL vs free-threaded ThreadPool scaling |
| [`v2/tradeoff_radar.png`](v2/tradeoff_radar.png) | runtime/memory/debug/effort trade-off map |

## k-means summary

Warm median, `k=5`, `d=10`, `max_iter=30`, same initial centroids:

| N | NumPy naive | NumPy matmul | Numba | JAX |
|---|------------:|-------------:|------:|----:|
| 10k | 0.043 s | 0.049 s | **0.005 s** | 0.008 s |
| 100k | 0.050 s | 0.134 s | **0.006 s** | 0.047 s |
| 500k | 0.371 s | 0.328 s | **0.039 s** | 0.281 s |
| 1M | 5.22 s | 4.96 s | **0.482 s** | 0.499 s |

Observations:

- At `N=1M`, Numba is 10.8x faster than NumPy naive and uses 22 MiB of Python-visible peak memory vs 810 MiB.
- JAX is very close to Numba at `N=1M` on this CPU once compiled.
- Pure Python loops at `N=2k` take 0.414 s warm, which is the interpreter-overhead story for CPython JIT discussion.
- A fixed-init sklearn sanity check passes with relative inertia difference around `1.5e-16`.

Shape stress, `K=50`, `d=100`, `max_iter=10`:

| N | NumPy naive | NumPy matmul | Numba | JAX |
|---|------------:|-------------:|------:|----:|
| 5k | 0.300 s | 0.054 s | 0.044 s | **0.022 s** |
| 20k | skipped | 0.222 s | 0.175 s | **0.104 s** |
| 50k | skipped | 0.860 s | 0.424 s | **0.203 s** |

The naive broadcast path is capped at `N=5k` because it allocates an `O(N*K*d)` temporary tensor.

## Permutation summary

Warm median, `n=10k`, `n1=n2=5000`, `R=10k`, 8 workers where applicable:

| Implementation | warm | vs NumPy loop | peak memory |
|----------------|-----:|--------------:|------------:|
| NumPy loop | 0.856 s | 1.0x | 0.2 MiB |
| NumPy trick | 0.831 s | 1.0x | 0.2 MiB |
| ThreadPool (`py312` GIL) | 0.392 s | 2.2x | 1.4 MiB |
| ThreadPool (`py314t` no-GIL) | **0.173 s** | **4.9x** | 2.1 MiB |
| `multiprocessing` (8) | 1.71 s | 0.5x | parent 0.9 MiB + children 833 MiB |
| Numba `prange` | **0.064 s** | **13.4x** | 19.3 MiB |
| JAX `vmap` (CPU) | 37.4 s | 0.02x | parent 0.9 MiB + XLA memory |

Observations:

- Numba `prange` is the best local CPU option for this kernel.
- Free-threaded Python improves the same thread implementation from 0.32 s to 0.17 s at 8 workers in the dedicated worker sweep.
- `multiprocessing` has visible child-process RSS; the parent process alone hides most of the memory story.
- JAX CPU `vmap` over random permutations is an anti-pattern here, despite being attractive for accelerator-oriented batching.

## Re-run commands

```bash
# k-means main sweep
/Users/lucajiang/anaconda3/envs/py312/bin/python experiments/kmeans/sweep_kmeans.py \
  --n-list 10000 100000 500000 1000000 \
  --loops-n 2000 --max-iter 30 --warmup 1 --repeat 3 \
  --output-csv experiments/results/v2/kmeans_sweep.csv \
  --output-json experiments/results/v2/kmeans_sweep.json

# k-means K=50, d=100 shape stress
/Users/lucajiang/anaconda3/envs/py312/bin/python experiments/kmeans/sweep_kmeans.py \
  --n-list 5000 20000 50000 --n-features 100 --k 50 \
  --max-iter 10 --warmup 1 --repeat 2 --skip-loops \
  --max-numpy-naive-n 5000 \
  --output-csv experiments/results/v2/kmeans_shape_k50_d100.csv \
  --output-json experiments/results/v2/kmeans_shape_k50_d100.json

# permutation main sweep
/Users/lucajiang/anaconda3/envs/py312/bin/python experiments/permutation_test/sweep_permtest.py \
  --n1 5000 --n2 5000 --r-list 500 2000 10000 \
  --max-workers 8 --warmup 1 --repeat 3 \
  --output-csv experiments/results/v2/perm_sweep.csv \
  --output-json experiments/results/v2/perm_sweep.json

# GIL vs no-GIL ThreadPool worker sweep
/Users/lucajiang/anaconda3/envs/py312/bin/python experiments/permutation_test/sweep_thread_workers.py \
  --py312-python /Users/lucajiang/anaconda3/envs/py312/bin/python \
  --py314t-python /Users/lucajiang/anaconda3/envs/py314t/bin/python \
  --workers 1 2 4 8 --n1 5000 --n2 5000 --r 10000 \
  --warmup 1 --repeat 3 \
  --output-csv experiments/results/v2/perm_threads_workers.csv \
  --output-json experiments/results/v2/perm_threads_workers.json

# figures
/Users/lucajiang/anaconda3/envs/py312/bin/python experiments/visualization/plot_kmeans.py \
  --input experiments/results/v2/kmeans_sweep.csv --output-dir experiments/results/v2/
/Users/lucajiang/anaconda3/envs/py312/bin/python experiments/visualization/plot_kmeans_shape.py \
  --input experiments/results/v2/kmeans_shape_k50_d100.csv \
  --output experiments/results/v2/kmeans_shape_k50_d100.png
/Users/lucajiang/anaconda3/envs/py312/bin/python experiments/visualization/plot_permtest.py \
  --input experiments/results/v2/perm_sweep.csv --output-dir experiments/results/v2/
/Users/lucajiang/anaconda3/envs/py312/bin/python experiments/visualization/plot_thread_workers.py \
  --input experiments/results/v2/perm_threads_workers.csv \
  --output experiments/results/v2/perm_threads_py312_py314t.png
/Users/lucajiang/anaconda3/envs/py312/bin/python experiments/visualization/plot_tradeoff.py \
  --kmeans experiments/results/v2/kmeans_sweep.csv \
  --permtest experiments/results/v2/perm_sweep.csv \
  --output experiments/results/v2/tradeoff_radar.png \
  --n-kmeans 1000000 --r-perm 10000
```
