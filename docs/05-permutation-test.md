# 置换检验实现与并行策略

本文档对应演讲的“resampling pressure”部分。旧版本地 benchmark 脚本与图片
已清理；当前实现位于 `experiments/permutation/`，当前结果以 MacBook Air
`latest` 与 server `long_safe_20260503_190133` 为准。

## 统计任务

置换检验在零假设下重排标签或符号，重复计算同一个统计量，形成零分布。
本仓库的简化任务关注计算结构：

- 固定 seed 与数据生成方式。
- reference 实现定义统计量。
- NumPy/JAX matrix paths 必须和 reference 在 p-value、observed statistic、
  null calibration 上对齐。

演讲里不要把它讲成“只测 runtime 的玩具”。它的价值是：统计正确性、随机性
与并行/内存模型会同时出问题。

## 当前实现

- `experiments/permutation/permutation_reference.py`：reference 与等价性检查。
- `experiments/permutation/permutation_numpy.py`：CPU NumPy matrix/batched path。
- `experiments/permutation/permutation_jax_matrix.py`：JAX matrix reformulation。
- `experiments/permutation/run_mac_validation.py`：MacBook correctness/calibration。
- `experiments/server/long_safe_orchestrator.py`：server CPU/A100 long-safe runs。

## 当前证据

MacBook Air `latest`：

- `permutation_equivalence.csv`：450 pass rows，45 个预期
  `skipped_memory_risk` rows。
- `permutation_calibration_extended.csv`：100 个额外 null calibration pass rows。
- `permutation_power_extended.csv`：168 个 power rows，覆盖更密的 delta 与
  signal-fraction。
- `permutation_runtime_scaling_extended.csv`：108 pass rows，27 个显式
  memory-risk skips。

Server CPU：

- `permutation_cpu_scaling.csv`：105 pass rows，最大
  `n=50,000, p=50,000, R=100,000` 角落有 3 个 timeout rows。
- `permutation_worker_sweep.csv`：固定 shape 下 8 workers 最快，说明并行度需要
  调参。

Server A100：

- `permutation_matrix_gpu.csv`：15 pass rows。
- matched `n=5,000, p=50,000, batch_R=512` slice 中，当前 GPU matrix path
  不赢 CPU。这是需要保留的负结果。

## 演讲主线

1. 先用 reference 与 calibration 说明 p-value/null behavior 没坏。
2. 再讲 runtime scaling：`R`、`p` 与 batch size 决定工作量。
3. 最后讲并行：threads/workers/GPU 都不是越多越好，必须看内存形状与
   overhead。

图入口：

- `experiments/results/macbook_air_long/latest/figures/permutation_calibration_extended.png`
- `experiments/results/macbook_air_long/latest/figures/permutation_power_extended.png`
- `experiments/results/macbook_air_long/latest/figures/permutation_runtime_scaling_extended.png`
- `experiments/results/presentation_figures/server_permutation_cpu_a100_summary.png`
- `experiments/results/presentation_figures/server_parallelism_tradeoff.png`

## SciPy 对照

[`scipy.stats.permutation_test`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
是通用接口，适合生产代码参考。本仓库保留手写 reference/NumPy/JAX paths，是
为了教学：只有自己控制数据布局、随机性和 batch shape，才能解释 runtime 与
memory 为什么变化。
