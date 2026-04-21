# 基准测试与可复现性方法论（本仓库真实做法）

本文档说明 `experiments/` 中基准的 **通用原则** 与 **我们实际采用的技巧**；具体参数见各子目录 `README.md`。

## 1. 目标

- **Runtime**：跨实现、跨数据规模的 **公平对比**。
- **Memory**：突出 **多进程拷贝** vs **单进程共享**（置换检验部分尤其重要）。
- **可复现**：固定种子、记录 **Python/库版本与硬件**（本仓库每份 CSV 旁都有一份 env JSON）。

## 2. 计时工具

### 手写 `time.perf_counter` + median

本仓库 [`bench_kmeans.py`](../experiments/kmeans/bench_kmeans.py) / [`bench_permtest.py`](../experiments/permutation_test/bench_permtest.py) 采用如下结构：

```python
cold = 第一次调用的完整 wall time
for _ in range(warmup - 1): fn()    # 预热丢弃
for _ in range(repeat): 记录 perf_counter 差值
返回 cold, median(times), std(times)
```

**为什么自己写而不用 `timeit`/`pyperf`**：我们需要把 `cold` 和 `warm` 分开报，而两个标准工具默认都丢弃冷启动。

### JAX 异步执行的陷阱

`jax.jit` 等调用可能 **异步返回**；计时时应对返回值或 pytree 调用 **`jax.block_until_ready(...)`**，否则测得时间可能严重偏短。本仓库已按此处理。

### `pyperf`（推荐用于正式数据）

- 多进程 worker、预热、统计摘要；适合写入 JSON 供后续绘图。
- 安装：`pip install pyperf`（已列入 `experiments/setup/requirements-base.txt`）。

## 3. 内存测量：**三种都用**，互补

| 工具 | 捕获对象 | 适用 | 本仓库作用 |
|------|----------|------|------------|
| `tracemalloc` | 当前进程 Python-level 分配 | 看 broadcast / batched 版的 "峰值快照" | 主要数字，写入 `tracemalloc_peak_mb` |
| `resource.getrusage(RUSAGE_SELF).ru_maxrss` | 当前进程累计峰值 RSS | 进程级基线 | 仅作为 sanity，**累积性导致同进程多实现时不准确** |
| `psutil.Process(pid).children().memory_info().rss` | 子进程 RSS 实时采样 | 多进程专属：我们对 `multiprocessing` 专门开了这个探针 | 让 `children_rss_peak_mb` 列非零 |

**【关键教训：原 benchmark 的隐藏问题】**
我们最初把所有 impl 跑在同一个进程里；`numpy_batched` 一次分配 2.3 GB 后，`resource.ru_maxrss` 就永久停在 2.3 GB，让后续所有 impl 的 "RSS" 全都看上去相同。解决方案是 **把每个 impl 放到独立子进程**（[`sweep_kmeans.py`](../experiments/kmeans/sweep_kmeans.py) / [`sweep_permtest.py`](../experiments/permutation_test/sweep_permtest.py) 都通过 `subprocess.run` 一次只跑一个实现）。这个变化让 threads / numba 等轻量实现的时间数据**移动了 5×**，是演讲叙事可信度的关键。

## 4. 实验设计清单

1. **固定**：`n_samples`、`n_features`、`K`（k-means）、`n_permutations`（置换）、`max_iter`、`random_seed`。
2. **Warmup**：Numba/JAX/CPython JIT 至少 1–3 次全量调用；若要报"冷启动"成本，**单独测 `cold_s`** 而不是用第一次 warm 来算。
3. **重复**：同配置运行多次，报告 **median** 或 mean ± std。
4. **环境**：记录 `python -VV`、`numpy.__version__`、`numba.__version__`、`jax.__version__`、CPU 型号、核心数、是否 GPU。本仓库 `bench_*.py` 会把这些写到 `--output-json` 的 `env` 字段里。
5. **Apple Silicon 注意**：长时间高负载（>1 分钟）会导致 P-core 降频。演讲里最好承认这一点，并把「稳态」的误差棒放大一些，或在 Linux 机器上复现一遍做对照。

## 5. 数据规模梯度（本仓库实际使用）

| 档位 | 规模 | 用途 |
|------|------|------|
| Toy (k-means) | \(N = 10^4\) | 笔记本实时 demo；JAX/Numba 的「冷启动 vs 稳态」差距最夸张 |
| Medium (k-means) | \(N = 10^5\text{-}5 \times 10^5\) | 笔记本常规实验 |
| Large (k-means) | \(N = 10^6\) | 展示 NumPy 朴素版的内存墙（~810 MB tracemalloc） |
| Permutation | \(n = 10^4\), \(R \in \{500, 2000, 10^4\}\) | 展示 `multiprocessing` 的启动阶梯 (~3 s 固定) 和 Numba 的几乎水平线 |

## 6. 输出与可视化

- 原始结果建议保存为 **CSV/JSON**（`experiments/visualization/plot_*.py` 直接读取）。
- 图表输出：
  - `*_scaling.png`：runtime vs 规模，log–log。
  - `*_speedup.png`：相对于朴素 NumPy 的加速比柱状图。
  - `*_memory.png`：tracemalloc 峰值（+ 子进程 RSS 如果适用）。
  - `*_cold_vs_warm.png`：冷启动 vs 稳态对比，用最小 N / 最小 R 让差距最醒目。
  - `tradeoff_radar.png`：runtime / memory / debuggability / dev effort 四轴雷达，其中 runtime、memory 由实测对数打分，另外两轴来自演讲者主观判断。

## 7. 延伸阅读

- [pyperf 文档](https://pyperf.readthedocs.io/)
- [timeit 文档](https://docs.python.org/3/library/timeit.html)
- [tracemalloc 文档](https://docs.python.org/3/library/tracemalloc.html)
- [psutil 文档](https://psutil.readthedocs.io/)
