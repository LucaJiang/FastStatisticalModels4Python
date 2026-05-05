# Numba 数值计算速查

本文档对应演讲中的 k-means 与 permutation workloads。旧版脚本和图片已经
移除；请使用当前 `experiments/kmeans/`,
`experiments/permutation/`, `experiments/server/` 与
`experiments/results/*/README.md`。

## 适用场景

Numba 最适合把“统计定义已经很清楚、但 Python 正在执行热循环”的代码
编译成机器码：

- k-means assignment/update 的内层距离循环。
- 置换、bootstrap、simulation replicate 这类独立重复任务。
- 小型固定结构的归约、计数和状态更新。

不适合的场景是任意 Python 对象、字符串、字典、动态 shape 控制流，或已经
完全由高质量 BLAS/GPU kernel 主导的代码。

## 当前仓库里的 Numba 角色

`experiments/kmeans/kmeans_numba.py` 保留了显式 loop 结构，因此可以和
reference/NumPy 版本逐步对照。它的意义不是“Numba 魔法”，而是：

- 不生成 `(N, K, d)` 广播临时张量。
- 将 labels、centroids 和 inertia 的循环留在一个 compiled path。
- 保持和 reference 相同的初始化与 stopping rule，便于检查 inertia。

Server CPU 的 `kmeans_cpu_scaling.csv` 说明 Numba 在低维 k-means 上很强；
但当 `d` 变大、矩阵乘法更饱满时，NumPy matmul 会变得更有竞争力。

## 并行与线程数

`@njit(parallel=True)` 与 `prange` 适合独立外层循环，但线程数仍然要调。
本次 server CPU `kmeans_numba_thread_sweep.csv` 的结论是：

- 32--64 threads 是本次 k-means Numba sweep 的高效区间。
- 128 threads 不是更好；调度、缓存和内存带宽会反噬。

置换检验也有同样模式：worker count 不是越多越好。当前 slide/poster 中把
这一点放在“tune, do not maximize”的主线上。

## 开发建议

1. 先写 reference，并用小规模 simulation 锁定统计等价性。
2. 再把最热的循环抽成小函数，添加 `@njit(cache=True)`。
3. 单线程版本通过后，再考虑 `parallel=True`。
4. 对随机数使用可复现、彼此独立的 seed 或 key；不要让并行顺序影响结果。
5. 记录 cold/warm 时间，讲清首次编译成本和稳态吞吐是两回事。

## 当前可引用材料

- k-means local figures:
  `experiments/results/macbook_air_long/latest/figures/kmeans_reference_equivalence.png`
  and `kmeans_shape_stress_runtime.png`
- server CPU figures:
  `experiments/results/linux_server_cpu/long_safe_20260503_190133/figures/kmeans_cpu_runtime.png`
  and `kmeans_numba_threads.png`
- talk-ready figures:
  `experiments/results/presentation_figures/server_kmeans_cpu_a100_summary.png`
  and `server_parallelism_tradeoff.png`
