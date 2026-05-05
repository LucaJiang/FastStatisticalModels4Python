# Python 3.14: Free-threaded 与实验性 JIT

本文档保留 Python 3.14 在演讲中的定位，但不再引用已经清理掉的
历史本地 benchmark 结果。当前仓库的实验证据以 MacBook Air `latest` 结果与
Linux server/A100 `long_safe_20260503_190133` 结果为准。

## Free-threaded 的讲法

Free-threaded Python 让同一进程内的 Python 线程可以真正并行执行
Python bytecode。它对统计计算最有价值的场景不是“自动让所有 NumPy
代码变快”，而是：

- 多个线程共享同一份大型只读数组，避免 `multiprocessing` 的 pickle
  与进程内存复制。
- 每个 worker 写独立输出槽，最后合并结果，避免共享 list/dict 的锁争用。
- Python 层调度成本足够高，而且 C/NumPy 内核已经不是唯一瓶颈。

当前主线中，置换检验仍然使用 `experiments/permutation/` 的
reference/NumPy/JAX matrix 实现作为可验证统计任务。server CPU 结果里
`permutation_worker_sweep.csv` 说明 workers 不能盲目拉满：固定
`n=5,000, p=10,000, R=10,000` 时，本次运行的 8 workers 最快，更多
workers 增加开销但没有线性收益。

## 实验性 JIT 的讲法

Python 3.14 的实验性 JIT 更适合作为“纯 Python 循环为什么会慢”的背景：
JIT 能降低解释器 dispatch/boxing 开销，但不会替代算法重写、NumPy 的 C
内核、Numba 的 LLVM 编译，或 JAX/XLA 的 whole-function lowering。

在当前仓库里，k-means 的可讲证据来自：

- `experiments/kmeans/kmeans_reference.py`：接近统计定义的 reference。
- `experiments/kmeans/kmeans_numpy_broadcast.py`：展示临时数组风险。
- `experiments/kmeans/kmeans_numpy_matmul.py`：展示矩阵恒等式何时有用。
- `experiments/kmeans/kmeans_numba.py`：展示 fused loop 与较低内存压力。

主结论应该是：Python JIT 是运行时进步，但科学计算代码仍然要先识别
计算图、内存形状和统计等价性。

## 当前可引用证据

- MacBook Air `latest`：3,840 个 k-means pass rows、480 个预期
  `skipped_memory_risk` rows、450 个 permutation equivalence pass rows。
- Server CPU：540 个 k-means CPU pass rows、105 个 permutation CPU pass
  rows、3 个最大 permutation 形状 timeout rows。
- Server A100：135 个 k-means A100 pass rows、15 个 permutation matrix
  A100 pass rows。

这些数字对应的 README：

- `experiments/results/macbook_air_long/latest/README.md`
- `experiments/results/linux_server_cpu/long_safe_20260503_190133/README.md`
- `experiments/results/linux_server_a100/long_safe_20260503_190133/README.md`

## 口头边界

不要把 Python 3.14 讲成“免费提速按钮”。更稳的表达是：

1. 先用 reference 与 simulation 证明统计量没有变。
2. 再用 profile/shape evidence 判断瓶颈是 Python loop、temporary array、
   process copy、还是 accelerator batching。
3. Python 3.14、Numba、JAX 都是工具箱的一部分；谁赢取决于 workload
   形状，而不是 logo。
