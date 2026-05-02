# Numba 数值计算速查（含实测数据）

本文档面向演讲中的 **k-means（循环密集）** 与 **置换检验（embarrassingly parallel）** 两类 workload，总结 Numba 常用模式、实测数字与常见陷阱。

## 1. 核心概念

- **Numba**：通过 LLVM 将装饰的函数编译为机器码，适合 **NumPy 数组 + 数值循环**。
- **`@njit`**（即 `@jit(nopython=True)`）：**nopython 模式**，避免回退到对象模式，性能最可预期。
- **冷启动**：首次编译某函数通常有 **数百毫秒到数秒** 级延迟。例如本仓库 k-means Numba 核（d=10, K=5）在当前本机 `py312` 下首次调用约 0.36 s（[`kmeans_sweep.csv`](../experiments/results/v2/kmeans_sweep.csv) 中 `cold_s` 列）。必须靠 `warmup` 或 `cache=True` 抵消。

## 2. `@njit`：单线程加速

典型用途：

- 手写 `for` 循环遍历样本/特征；
- 内层距离计算、聚类分配、小数组上的归约。

注意：

- nopython 模式只支持 **Python 与 NumPy 的子集**（无任意 Python 对象、字符串、字典等）。
- 部分 NumPy API 在 Numba 中 **未实现或行为不同**，需查 [Supported NumPy features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)。
- `np.random.default_rng` 在 Numba 里不支持；我们在置换检验里改用手写 **SplitMix64 + xorshift64**（[`permtest_numba.py`](../experiments/permutation_test/permtest_numba.py)），并且记录了一次 RNG 陷阱：直接用 `(base_seed ^ i*const)` 作为种子会产生高度相关的流，导致 null 分布均值偏移 ~0.05σ。演讲时可以用这个例子提醒观众「并行 RNG 要从高质量种子产出独立流」。

## 3. `@njit(parallel=True)` + `prange`

用于 **独立迭代** 可并行的外层循环（例如多次置换、多 bootstrap 次）：

```python
from numba import njit, prange

@njit(parallel=True)
def parallel_kernel(...):
    for i in prange(n_iter):
        ...
```

要点：

- **`prange`**：类似 OpenMP 的并行 for；Numba 负责线程划分与部分归约。
- **归约变量**（如 `s += x[i]`）由 Numba 处理为安全的并行归约（需符合其规则）。
- **只读大数组**：通常所有线程共享只读视图，避免在并行区写同一元素造成数据竞争。
- Numba 有自己的线程池，即便 CPython 是 **GIL 构建**，它也能让外层 `prange` 并行执行——这是置换检验里它全场最快的原因。

## 4. 本仓库实测（Apple Silicon 8 核, Python 3.12.2）

| 实验 | Numba 版 warm 中位数 | 对照项 | 加速比 |
|------|---------------------|--------|--------|
| k-means N=1M, max_iter=30 | **0.482 s** | NumPy 朴素广播 5.22 s | **10.8×** |
| Permutation R=10000, n=10k | **0.064 s** | NumPy 朴素循环 0.856 s | **13.4×** |
| Permutation R=10000, n=10k | **0.064 s** | `multiprocessing` (8 workers) 1.71 s | **26.8×** |

[图表链接](../experiments/results/v2/perm_speedup.png) · [数据链接](../experiments/results/v2/perm_sweep.csv)

**演讲可引用的一句话**：*「改写一个 `@njit(parallel=True)` 的核心，往往比起 `multiprocessing` 省下 8 个进程的启动成本、几百 MB 的数据重复与一整套 pickle 噪声，还能多出一个数量级的速度」。*

## 5. 与演讲实验的对应关系

| 实验 | Numba 角色 |
|------|------------|
| k-means | `@njit` 编译 Lloyd 内层循环/距离计算，对比「纯 NumPy 向量化」 |
| 置换检验 | `@njit(parallel=True)` + `prange` 并行多次置换，对比 **multiprocessing 拷贝** 与 **JAX vmap** |

## 6. 调试与开发体验

- 编译失败时，错误信息指向 **不支持的 Python/NumPy 构造**；常需改写为显式循环或预分配缓冲区。新手的痛点 #1 是错误信息里掺杂着 Numba 的中间表示（IR），看不懂就容易放弃。**建议：先只用 `@njit` 单线程跑通，再加 `parallel=True`。**
- 可在 nopython 外使用 **print**（有限支持）或暂时关闭 `parallel` 缩小问题范围。
- 性能调优：注意 **内存布局**（`float64` 连续数组）、避免在热循环中分配新数组。`fastmath=True` 可开启重结合/矢量化（对 1 ULP 要求严格的统计代码需谨慎）。
- **Numba 的 `cache=True`**：把 AOT 结果写入 `__pycache__/*.nbi/*.nbc`，下次启动复用。我们在 benchmark 中会主动清除此缓存，才能得到诚实的「冷启动」时间。

## 7. 延伸阅读

- [Numba User Guide](https://numba.readthedocs.io/en/stable/user/index.html)
- [Parallel range (`prange`)](https://numba.readthedocs.io/en/stable/user/parallel.html)
- 社区博文示例：[Parallel Bootstrap Sampling with Numba](https://medium.com/@jose.a.poblete/parallel-bootstrap-sampling-in-python-using-numba-b0fe55928a58)
