# k-means（Lloyd 算法）实现与内存深度剖析

本文档为演讲 **「循环密集型迭代模式」** 提供算法与实现对照，并深度剖析不同实现方式在内存与运行时间上的差异。**所有数字都来自 [`experiments/results/v2/kmeans_sweep.csv`](../experiments/results/v2/kmeans_sweep.csv)**。

## 1. Lloyd 算法（标准 k-means）

给定数据点矩阵 \(X \in \mathbb{R}^{N \times d}\)，簇数 \(K\)：

1. **初始化**：选取 \(K\) 个质心（例如从 \(X\) 中随机取 \(K\) 行）。
2. **重复直到收敛或达到最大迭代次数**：
   - **分配（E-step）**：每个点指派到最近质心（欧氏距离）。
   - **更新（M-step）**：每个簇的质心取该簇点的 **均值**。

为了让演讲里的对比公平，本仓库所有实现都接收同一份 `init_centroids`（由 `np.random.default_rng(seed).choice` 挑选），因此它们在同一个初始位置下收敛，**inertia 可以直接横向比较**。NumPy / Numba / JAX 在 N=10k, N=100k, N=1M 上都报出了同一个 inertia（三组数字逐小数位对齐）。

## 2. NumPy 两种写法：教科书 vs 实战

### 2.1 朴素广播（`numpy_naive`）

```python
diff = X[:, None, :] - centroids[None, :, :]  # (N, K, d) 中间张量
dists_sq = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
```

**【内存黑洞】**：每次迭代分配一个形如 `(N, K, d)` 的临时张量，其体积为 \(N \cdot K \cdot d \cdot 8\) 字节。在 N=1M、K=5、d=10 时，单步 400 MB；30 次迭代反复分配/释放，**tracemalloc 峰值 ~810 MB**（[`kmeans_memory.png`](../experiments/results/v2/kmeans_memory.png)）。

### 2.2 矩阵乘法式（`numpy_smart`）

利用恒等式 \(\|x - c\|^2 = \|x\|^2 + \|c\|^2 - 2\, x^\top c\)，把距离矩阵转写成 `X @ C.T`：

```python
X_sq = np.einsum("ij,ij->i", X, X)[:, None]   # (N, 1)
C_sq = np.einsum("ij,ij->i", C, C)[None, :]   # (1, K)
dists_sq = X_sq + C_sq - 2.0 * (X @ C.T)       # (N, K)
```

**内存**：只需 (N, K) 矩阵，N=1M 时约 40 MB，**比朴素广播小约 10×**。

**【但是：实战性能取决于 K×d】**
- 在 K=5、d=10 这种「极瘦」配置下，矩阵乘法的收益会被两次 norm 计算吃掉，在我们的 M1 上甚至**比朴素版慢 ~50%**。
- 如果改成 K=50、d=100（更接近真实生物统计模型的 EM 步骤），`X @ C.T` 会显著快于广播。演讲中可以把这一条作为 **「加速窍门不是万能药」** 的例证：你得知道自己的 (N, K, d) 是否撑得起一次 BLAS 调用。

## 3. 纯 Python 双层循环版本（用于 JIT 演示）

```python
for i in range(n):
    for j in range(k):
        s = 0.0
        for t in range(d):
            u = X[i, t] - centroids[j, t]
            s += u * u
```

**【专家深度点评：JIT 的用武之地】**
这种循环在传统 CPython 中极慢，因为涉及数以亿计的字节码分发（Dispatch）和对象装箱/拆箱（Boxing/Unboxing）。在 N=2000, d=10, K=5 上一次完整运行需 **0.84 s**（[`kmeans_sweep.csv`](../experiments/results/v2/kmeans_sweep.csv)），而 Numba 在 N=100k、更大 50× 的数据上只要 **0.010 s**——两者相差 4 个数量级。

Python 3.14 实验性 JIT（Copy-and-patch）理论上能把这 0.84 s 砍 2–3×，但**永远达不到 Numba 的水准**。这是演讲的重要边界：JIT 是在既有解释器开销上打折，不是换算法或换运行时。

## 4. Numba 版本：最纯粹的高效

- 将 **内层循环**、距离与部分归约用 `@njit` 编译。
- **内存优势**：由于 Numba 能够在每次外层循环处理一个点时就立即计算最近距离并归约，它**完全不需要生成 `(N, K, d)` 的中间张量**！它的空间复杂度从 NumPy 朴素的 \(O(NKd)\) 断崖式下降到 \(O(N + Kd)\)（只需 labels 与 centroids）。
- **实测**：N=1M 峰值分配 **22 MB**，对比 NumPy 朴素的 **810 MB** — 38× 的差距。
- **实测运行时**：N=1M 稳态 **1.21 s**，对比 NumPy 朴素的 12.6 s — 10× 的速度差。

## 5. JAX 版本：XLA 的威力

- 整体 `jax.jit`；使用 `jax.lax.scan` 固定 `max_iter`；质心作为 `carry`。
- **内存视角**：利用 `tracemalloc` 测试 JAX 的内存时，Python 解释器层面几乎没有任何峰值分配（N=1M 只有 2.2 MB）。这是因为 `jax.jit` 将计算完全下放到 XLA 运行时中，**从 Python 层看不到**。但 XLA 的设备内存仍然被占用——如果要报 GPU VRAM，需要用 JAX 自己的 profiler。
- **实测运行时**：N=1M 稳态 **1.35 s**，与 Numba 差距仅约 12%。JAX 追上 Numba 的转折点恰好在 N=1M 附近——小 N 时 Numba 领先，大 N 时 BLAS-backed XLA 追上来。
- **实测冷启动**：N=10k 的 `cold_s = 0.68 s`, warm 中位数 0.015 s。**首次调用 45× 慢于稳态**。对「一次性脚本」这是实打实的代价。

## 6. 正确性检查

- 固定随机种子与 `max_iter`；
- **比较 inertia（簇内平方和）**：本仓库所有实现在同一 `init_centroids` 下得到相同 inertia，差异 < 1 ULP。这是回答「换框架会不会改变统计结果？」时最硬的证据。
- 与 `sklearn.cluster.KMeans` 在相同初始化下的结果（允许浮点微小差异）也能对上，但 sklearn 带 k-means++ 默认初始化，直接对比需要关闭。

## 7. 演讲要点（用这份实测讲故事）

1. **先诚实描述「教科书 NumPy」的代价**：400 MB / 30 iter 的临时张量 → 大 N 下 OOM 风险。
2. **再展示 Numba 把空间复杂度从 \(O(NKd)\) 砍到 \(O(N)\) 的代码片段**——这是统计学家能看得懂、也能实施的最佳改造。
3. **JAX 的亮点不是在 CPU 上追上 Numba**，而是「同一份代码，加速器上就能再快一个量级」——留给会后讨论。

## 8. 延伸阅读

- 向量化与 profiling：[Speed Up K-Means with NumPy](https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/)（示例思路，非必须复现其倍数）
