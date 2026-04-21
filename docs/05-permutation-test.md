# 置换检验（Permutation Test）实现要点与并行策略实测

本文档支撑演讲 **「大规模数据并行模式」**：同一统计量在 **多次随机置换** 下重复计算，天然 **embarrassingly parallel**。所有数字来自 [`perm_sweep.csv`](../experiments/results/v2/perm_sweep.csv)，运行在 Apple Silicon 8 核心、Python 3.11.6。

## 1. 统计思想（极简）

- 在 **零假设** 下，观测标签或符号与数据「可交换」；通过 **重排** 观测或残差生成 **零分布**。
- **p 值**：观测统计量在零分布中的分位（或双侧/单侧规则依检验而定）。

演讲重点在 **计算与内存**，非证明理论。

## 2. 本仓库实验采用的简化设定

为便于跨实现对比：

- 合并两组样本为长向量 `x = np.concatenate([a, b])`，长度 `n = n1 + n2`。
- 每次置换：`perm = rng.permutation(n)`，`x_perm = x[perm]`。
- **统计量（示例）**：两样本均值差  
  `stat = x_perm[:n1].mean() - x_perm[n1:].mean()`。
- 重复 `R` 次，得到 `R` 个统计量，再与观测值比较（或仅测 **总耗时**）。

## 3. 算法层面的小聪明（演讲高光时刻）

注意一个恒等式：由于 `sum(x)` 是常量，

\[
\bar{x}_A - \bar{x}_B = \frac{S_1}{n_1} - \frac{S - S_1}{n_2}
\]

其中 \(S = \sum_i x_i\), \(S_1 = \sum_{i \in A} x_i\)。**整个统计量只依赖子集和 \(S_1\)。** 因此我们不需要完整的 `perm`（长度 n），只需要 *n1 个不重复的索引*（`np.random.choice(n, n1, replace=False)`）。

| 实现 | warm 时间（R=10k, n=10k） | 加速比 vs `numpy_naive` |
|------|--------------------------:|--------------------------:|
| `numpy_naive`（完整 shuffle） | 2.04 s | 1.0× |
| `numpy_trick`（子集和） | 2.13 s | 0.96× |
| `numpy_batched`（`argsort((R,n))`） | 12.13 s | 0.17× |
| `numpy_trick_batched`（`argpartition`） | 4.67 s | 0.44× |

**关键发现**：在纯 NumPy 层面，`np.random.choice(..., replace=False)` 内部也是 `permutation` 实现，算法技巧本身在单循环里 **没有速度收益**。但——

- **批处理版 `numpy_trick_batched` 比 `numpy_batched` 快 2.6×，内存 1907 MB vs 2289 MB**：一旦你下沉到 `argpartition`（O(n) 而非 O(n log n)），算法技巧立刻体现。
- 更重要的是，**算法技巧改变了心智模型**：它让我们从「我要置换一整个向量」变成「我要采样一批索引」，从而把问题映射到 `vmap` 或 GPU 上时心智负担更低。

**演讲用一句话**：*「每次改进都从重新审视统计量本身开始——再快的并行库也换不回一个恰当的恒等式。」*

## 4. 并行与内存模型（演讲核心对比）

| 方案 | 本仓库实测 warm (R=10k) | 数据拷贝 | 说明 |
|------|-----------------------:|---------|------|
| 串行 NumPy | 2.04 s | 单份 `x` | 基线 |
| `ProcessPoolExecutor` | **3.34 s** | **每进程独立数据 + pickle** → 实测子进程 RSS 总和 **757 MB** | 小 R 下启动成本主导 |
| `ThreadPoolExecutor`（GIL build） | **1.46 s** | **共享** 进程内数组 | 避免整表复制；**1.4×**，GIL 上的天花板 |
| Numba `prange` | **0.158 s** | 共享只读数组，一个进程 | **13×，全场最快** |
| JAX `vmap` + `jit`（CPU） | **75.5 s** | 设备内存 | CPU 上是反例，见 [`03-jax-guide.md`](03-jax-guide.md) |

图例：

- [`perm_scaling.png`](../experiments/results/v2/perm_scaling.png)：runtime vs R，所有实现在同一张对数图上对比。`multiprocessing` 的 ~3s 启动阶梯、Numba 的平坦、JAX 的陡峭上升一目了然。
- [`perm_memory.png`](../experiments/results/v2/perm_memory.png)：tracemalloc 峰值 + 子进程 RSS 加总。Multiprocessing 的 **757 MB 子进程 RSS** 与其他实现的 ~1 MB 形成鲜明对比。
- [`perm_speedup.png`](../experiments/results/v2/perm_speedup.png)：Numba 13× 的柱子，和 JAX 0.03× 的「负加速」放在同一坐标下。

## 5. 随机性与可复现

- **多进程**：需固定每 worker 种子或显式传递子种子序列。本仓库 [`permtest_multiprocessing.py`](../experiments/permutation_test/permtest_multiprocessing.py) 用 `seed + 100003 * j` 派生。
- **多线程**：竞态若写共享状态会破坏可复现；本仓库设计为 **每次置换独立写临时缓冲** 或 **只读 `x` + 写独立输出槽**。
- **JAX**：必须用 `jax.random` 与 **split key**，避免隐式全局 RNG。
- **Numba**：手写 RNG 时要确保独立高质量种子；本仓库用 **SplitMix64 一次混合，xorshift64 作为步进器**。我们曾经因为朴素的 `(seed ^ i*const)` 种子让置换分布均值漂移了 ~0.05σ——KS 检验抓住了这个问题。教训：**并行 RNG 的质量保证不是事后诸葛亮，必须与算法一起设计**。

## 6. SciPy 对照

[`scipy.stats.permutation_test`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)：通用接口，支持 `vectorized`、多种 `permutation_type`。本仓库基准脚本以 **手写循环 + 多种并行后端** 为主，便于控制 **内存与进程数**。

## 7. 演讲要点速查卡

1. **开场定义**：把置换检验画成「同一个统计量做 R 次」的图示；统计学背景不要超过 20 秒。
2. **算法层**：讲恒等式 → 把 R 份完整 shuffle 变成 R 份 subset-sum（不是永远值钱，但心智模型值钱）。
3. **工程层**：[`perm_scaling.png`](../experiments/results/v2/perm_scaling.png) 里挑两条做对比——Numba 和 multiprocessing——讲一次性启动成本 vs 并行收益的拐点。
4. **内存层**：[`perm_memory.png`](../experiments/results/v2/perm_memory.png) 讲 multiprocessing 的 757 MB 是「看不见的内存」，而 Numba / threads / free-threaded 都是「一份数据」。
5. **诚实的 JAX**：CPU 上的 `vmap` 是反例；GPU 上是王道。演讲如果有条件放一个 GPU 数字更好。
