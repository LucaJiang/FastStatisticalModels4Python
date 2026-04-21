# JAX 统计与高性能数值速查（含 CPU 实测）

本文档面向演讲中的 **JAX 版 k-means** 与 **JAX 版置换检验**，强调 API 模式、实测数字与常见陷阱。**我们在 CPU 上实测出的一些反直觉结论，值得在演讲中诚实说清楚**。

## 1. 设计哲学

- **函数式**：JIT 追踪的函数应尽量 **纯**（无副作用、不依赖全局可变状态）。
- **显式随机数**：使用 `jax.random` 与 **PRNG key**（`key, subkey = jax.random.split(key)`），避免隐式全局 RNG。
- **XLA**：`jax.jit` 将 Python 可追踪运算 lowering 到 XLA，在 CPU/GPU/TPU 上执行。

## 2. 核心 API

### `jax.jit`

- 将 Python 可追踪函数编译为 XLA 程序。
- 首次调用会 **追踪 + 编译**，耗时明显；基准需 **warmup** 并多次运行取中位数。
- 本仓库 k-means JAX 核 (N=10k, d=10, K=5) 冷启动 ≈ 0.68 s，稳态 0.015 s（45× 差距）。在 N=1M 稳态下，编译成本相对于一次运行已微不足道。

### `jax.lax.scan`

- 形式类似 **带状态的 fold/scan**：`carry` 在步间传递，适合 **固定步数** 或结构化的迭代（例如 k-means 外层迭代若固定 `max_iter`）。
- 与「Python 层 for + 每次 `jit` 子块」相比，常能减少重复编译开销；但 **单步依赖下一步** 的算法在 **GPU** 上可能不占优（串行链）。

### `jax.lax.while_loop`

- 适合 **由数据决定的停止条件**（迭代次数运行时才知道）。
- 注意：在 `while_loop` 内部不能调用 Python 副作用函数，调试远比普通 `for` 困难。

### `jax.vmap`

- 将「单次样本上的函数」批处理化为「沿 batch 维向量化」，常用于：
  - 多次置换 / 多随机种子并行；
  - 与 `jit` 组合：`jax.jit(jax.vmap(...))`。

## 3. 与两种演讲 workload 的映射

| Workload | JAX 模式 |
|----------|----------|
| k-means | `scan` 或 `while_loop` 表达 Lloyd 迭代；`jit` 包一层；质心作为 **carry** |
| 置换检验 | `vmap` 对「单次置换统计量」并行；或批量生成置换索引后向量化 |

## 4. **CPU 上的 JAX 陷阱：本仓库关键实测**

`jax.random.permutation(n)` 在 CPU-XLA 上被编译成一个无法很好向量化的顺序 Fisher–Yates shuffle。在 `jax.vmap` 下展开 R 份后，每份 shuffle 自己跑自己的串行 log-n 次 shuffle，总时间随 R 线性上升而 **每次置换都远慢于 Numba 的 Fisher–Yates**：

| R | JAX vmap (perm) | Numba prange | NumPy naive | JAX 对 NumPy 倍率 |
|---|-----------------|--------------|-------------|-------------------|
| 500 | 4.0 s | 0.018 s | 0.091 s | 44× **slower** |
| 2 000 | 16.4 s | 0.054 s | 0.42 s | 39× slower |
| 10 000 | 75.5 s | 0.158 s | 2.04 s | 37× slower |

（来源：[`perm_sweep.csv`](../experiments/results/v2/perm_sweep.csv)）

我们还测试了「算法小聪明」版本——用 `jax.random.choice(replace=False)` 代替 `permutation`，期望少做一半的 shuffle 工作，结果 **几乎没差别**。因为 `choice(replace=False)` 在 JAX 内部也是基于 permutation 实现的。

**演讲要点**：*「如果你的代码只跑在 CPU 上，JAX 不是默认选项；`jax.vmap` 是为加速器批处理写的，不是为 CPU 外层并行写的。」*

反过来，k-means 这种 **BLAS-heavy 内核**（矩阵乘法在 XLA 上能良好下降到 BLAS），CPU JAX 已经可以在 N=1M 上跑到 1.35 s，与 Numba 只相差约 10%。

## 5. GPU 注意事项

- **强串行** 的 `scan`（每步依赖上一步）在 GPU 上可能 **慢于 CPU**（同步与并行度不足），演讲中可以诚实展示「JAX + GPU 并非对所有统计迭代都更快」。
- **embarrassingly parallel** 的置换（`vmap` 批处理）更适合 GPU，这也是 JAX 的真正卖点。
- 如果演讲现场能准备一个 GPU 盒子跑同一份代码，把 CPU 75s 砍到 GPU 5s 就是极具冲击力的对照；否则，老老实实说「GPU 结果本次未包含」更加诚实。

## 6. 调试陷阱

- **Tracing 错误**：当你在 JIT 区间使用 Python 条件判断或打印，通常会得到「ConcretizationTypeError」。对统计学家来说这是新概念；教学中可以把它类比为：JAX 在追踪时看到的是「符号变量」而不是具体数值。
- **Shape 异常**：`vmap` 对哪一维做向量化需要用 `in_axes` 明说；漏写的默认值是 0，有时会悄悄改变结果。
- **`block_until_ready` 必不可少**：`jax.jit` 是异步 dispatch，如果你在计时时忘记阻塞，报上来的可能只是 **dispatch 时间**（毫秒级），而不是真正的计算时间。本仓库的 [`bench_kmeans.py`](../experiments/kmeans/bench_kmeans.py) / [`bench_permtest.py`](../experiments/permutation_test/bench_permtest.py) 都按此处理。

## 7. 延伸阅读

- [JAX documentation](https://jax.readthedocs.io/)
- [jax.lax.scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)
- [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)
- QuantEcon：[NumPy vs Numba vs JAX](https://python-programming.quantecon.org/numpy_vs_numba_vs_jax.html)
