# k-means 实现与内存形状

本文档对应演讲的“iterative pressure”部分。旧版本地 benchmark 图和脚本已经
清理；当前 k-means 证据来自 MacBook Air `latest` 与 Linux server/A100
`long_safe_20260503_190133`。

## 算法定义

Lloyd k-means 重复两个步骤：

1. Assignment：每个样本分配给最近 centroid。
2. Update：每个 cluster 的 centroid 更新为该 cluster 的均值。

为了比较实现而不是比较初始化，本仓库的实现共享相同数据、相同初始
centroids、相同 `max_iter` 与 tolerance。优化路径必须和 reference 保持
inertia/labels 一致，才能进入 timing 解释。

## 三种 CPU 写法

### Reference

`experiments/kmeans/kmeans_reference.py` 是统计定义的锚点。它不是性能目标，
而是用来检查优化实现是否仍然在算同一个东西。

### NumPy broadcast

朴素广播会构造 `(N, K, d)` 临时张量：

```python
diff = X[:, None, :] - centroids[None, :, :]
dists_sq = np.einsum("nkd,nkd->nk", diff, diff)
```

这很好教，但内存随 `N*K*d` 增长。当前结果中，较大的 broadcast/reference
场景会被标记为 `skipped_memory_risk`，不再强行运行。

### NumPy matmul

矩阵恒等式

\[
\|x-c\|^2 = \|x\|^2 + \|c\|^2 - 2x^\top c
\]

把距离计算转成 `X @ C.T`。它通常减少临时数组；但当 `K` 与 `d` 很小，BLAS
启动和内存访问不一定占优。所以 slides 里要讲“shape changes the winner”。

### Numba

`experiments/kmeans/kmeans_numba.py` 把 assignment/update 循环编译为
machine code。它的价值在于 fused loop 与更低内存压力，而不是改统计方法。

## 当前证据

- MacBook Air `kmeans_correctness.csv`：1,440 scenarios；优化实现 pass rows
  与 reference 对齐，unsafe broadcast/reference rows 被显式 skip。
- MacBook Air `kmeans_shape_stress.csv`：额外 `K=20/50`, `d=100`,
  `N=50,000/100,000` 形状压力测试。
- Server CPU `kmeans_cpu_scaling.csv`：540 pass rows，用来讲 N/d/K 的
  scaling。
- Server CPU `kmeans_numba_thread_sweep.csv`：说明线程数要调，不是越大越好。
- Server A100 `kmeans_jax_gpu.csv`：135 pass rows，用来讲 GPU break-even。

## 演讲要点

1. 先展示 reference equivalence：优化不应该改变 inertia。
2. 再展示 geometry/failure modes：大数据前先证明统计行为稳定。
3. 最后展示 scaling：Numba、NumPy matmul、JAX/A100 的赢家随 shape 改变。

图入口：

- `experiments/results/macbook_air_long/latest/figures/kmeans_recovery_scenario_facets.png`
- `experiments/results/macbook_air_long/latest/figures/kmeans_shape_stress_runtime.png`
- `experiments/results/presentation_figures/server_kmeans_cpu_a100_summary.png`
