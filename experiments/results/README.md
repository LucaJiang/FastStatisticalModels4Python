# 本机基准结果（v2 — 实验全部重跑）

## 环境

- Python 3.11.6（Anaconda, macOS arm64）
- NumPy 1.24.4 · Numba 0.57.1 · JAX 0.4.33（CPU）· scikit-learn 1.3.0 · psutil 6.0.0
- Apple Silicon 8 核心 (ARM64)
- **JAX 计时**：已在 `bench_kmeans.py` / `bench_permtest.py` 中对返回值调用 `jax.block_until_ready`，避免异步执行导致时间偏短。
- **子进程隔离**：[`sweep_kmeans.py`](../kmeans/sweep_kmeans.py) 和 [`sweep_permtest.py`](../permutation_test/sweep_permtest.py) 把每个 (impl, N) 放到独立子进程——这让 `numpy_batched` 的 2.3 GB 分配不会污染后续 impl 的内存与时间测量。

## 目录

### 顶层 CSV（绘图输入）

| 文件 | 内容 |
|------|------|
| [`kmeans_sweep.csv`](kmeans_sweep.csv) | k-means：NumPy naive/matmul, loops, Numba, JAX × N ∈ {10k, 100k, 500k, 1M} |
| [`kmeans_sweep.json`](kmeans_sweep.json) | 同上 + 环境元数据 (`env`) |
| [`perm_sweep.csv`](perm_sweep.csv) | Permutation：9 种实现 × R ∈ {500, 2k, 10k}, n=10k |
| [`perm_sweep.json`](perm_sweep.json) | 同上 + 环境元数据 |

### 每个 (impl, size) 的原始子进程输出

- [`_subprocess_km/`](_subprocess_km/) — k-means，每个 (impl, N) 一份 JSON
- [`_subprocess/`](_subprocess/) — permutation，每个 (impl, R) 一份 JSON

### 图表（演讲直接引用）

| 文件 | 内容 |
|------|------|
| [`kmeans_scaling.png`](kmeans_scaling.png) | k-means runtime vs N 对数图，5 实现 |
| [`kmeans_speedup.png`](kmeans_speedup.png) | k-means 相对朴素 NumPy 的加速比，按 N 分组 |
| [`kmeans_cold_vs_warm.png`](kmeans_cold_vs_warm.png) | 最小 N 下冷启动 vs 稳态——JIT 税的可视化 |
| [`kmeans_memory.png`](kmeans_memory.png) | tracemalloc 峰值 vs N |
| [`perm_scaling.png`](perm_scaling.png) | permutation runtime vs R 对数图 |
| [`perm_speedup.png`](perm_speedup.png) | permutation 相对朴素 NumPy 的加速比（Numba 13×） |
| [`perm_memory.png`](perm_memory.png) | permutation peak memory，父进程 + 子进程 RSS |
| [`perm_cold_vs_warm.png`](perm_cold_vs_warm.png) | permutation 冷 vs 稳态 |
| [`tradeoff_radar.png`](tradeoff_radar.png) | 四轴雷达：runtime/memory/debug/effort，前两轴由 CSV 对数打分 |

## 简要解读

### k-means（warm median, max_iter=30, d=10, k=5）

| N | NumPy 朴素 | NumPy matmul | Numba | JAX |
|---|-----------:|-------------:|------:|----:|
| 10k | 0.092 s | 0.543 s | **0.015 s** | 0.015 s |
| 100k | 0.088 s | 0.414 s | **0.010 s** | 0.169 s |
| 500k | 0.813 s | 1.92 s | **0.081 s** | 0.715 s |
| 1M | 12.6 s | 23.0 s | **1.21 s** | 1.35 s |

观察：
- Numba 在所有 N 都是最快；到 N=1M 时 JAX 追到 1.35 s，差距缩小到 12%。
- 朴素 NumPy 在 N=1M 时 tracemalloc 峰值 **810 MB**；Numba **22 MB**（见 [`kmeans_memory.png`](kmeans_memory.png)）。
- Matmul-distance 在 K=5、d=10 下反而比朴素广播慢——距离矩阵计算太"瘦"，BLAS 的收益吃不完两次 norm 计算。**在 K≳50、d≳100 时这张票开始能赚钱**。

### Permutation（warm median, n=10k, n1=n2=5000, R=10000）

| 实现 | warm (s) | 相对 NumPy loop | 峰值内存 |
|------|---------:|--------------:|---------|
| NumPy naive loop | 2.04 | 1.0× | 0.2 MiB |
| NumPy subset-sum trick | 2.13 | 0.96× | 0.2 MiB |
| NumPy trick + batched | 4.67 | 0.44× | 1 907 MiB ⚠︎ |
| ThreadPool (GIL 构建) | 1.46 | **1.4×** | 1.4 MiB |
| ProcessPoolExecutor (8) | 3.34 | 0.6× | parent 0.9 MiB + **children 757 MiB** |
| Numba `prange` | **0.158** | **13.0×** | 20.8 MiB |
| JAX vmap (CPU) | 75.5 | 0.03× | 1 MiB parent + XLA 隐藏 |

观察：
- **Numba `prange` 大胜**（13× over naive）。
- 在 GIL 构建上线程已经能拿到 **1.4×**（因为 NumPy 在 C 层释放 GIL）。无 GIL 构建应能继续扩大此差距。
- `multiprocessing` 的 ~3 秒固定启动成本让它在小 R 下吃亏；但 757 MB 的"看不见的内存"是演讲里讲"copies vs shared"的最佳素材。
- **JAX CPU 反例**：`jax.random.permutation` 在 XLA 上编译成慢速 Fisher–Yates，`vmap` 后不仅没更快，反而慢 37×。

## 重新跑这些数字

```bash
# k-means sweep (~5 分钟)
python experiments/kmeans/sweep_kmeans.py \
  --n-list 10000 100000 500000 1000000 \
  --loops-n 2000 --max-iter 30 --warmup 1 --repeat 3 \
  --output-csv experiments/results/v2/kmeans_sweep.csv \
  --output-json experiments/results/v2/kmeans_sweep.json

# permutation sweep (~15 分钟, 因为 JAX 慢)
python experiments/permutation_test/sweep_permtest.py \
  --n1 5000 --n2 5000 --r-list 500 2000 10000 --max-workers 8 \
  --warmup 1 --repeat 3 \
  --output-csv experiments/results/v2/perm_sweep.csv \
  --output-json experiments/results/v2/perm_sweep.json

# 出图
python experiments/visualization/plot_kmeans.py \
  --input experiments/results/v2/kmeans_sweep.csv \
  --output-dir experiments/results/v2/
python experiments/visualization/plot_permtest.py \
  --input experiments/results/v2/perm_sweep.csv \
  --output-dir experiments/results/v2/
python experiments/visualization/plot_tradeoff.py \
  --kmeans experiments/results/v2/kmeans_sweep.csv \
  --permtest experiments/results/v2/perm_sweep.csv \
  --output experiments/results/v2/tradeoff_radar.png \
  --n-kmeans 1000000 --r-perm 10000
```
