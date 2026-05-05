# JAX 统计与高性能数值速查

JAX 在本仓库里不是“默认更快”的答案，而是一个很好的教学对象：当统计量能
表达成 batched array/matrix computation，JAX 和 GPU 才有机会赢；当算法
包含很多串行随机置换或小 kernel 调度，CPU 可能更合适。

## API 口径

- `jax.jit`：把可追踪函数 lowering 到 XLA。计时必须等待结果完成，不能只
  测 dispatch。
- `jax.vmap`：把单个统计量批处理化，适合多 replicate、多 permutation、
  多 seed。
- `jax.lax.scan` / `while_loop`：表达固定步数或数据依赖的迭代，适合 k-means
  这类有 state 的算法。
- `jax.random`：显式传递 key，避免隐式全局 RNG。

## 当前仓库里的 JAX 角色

### k-means

Server A100 的 `kmeans_jax_gpu.csv` 有 135 个 pass rows。当前 talk-ready
图 `server_kmeans_cpu_a100_summary.png` 匹配 CPU 与 A100 的共同形状，并展示
`CPU warm time / A100 warm time`。结论是：

- A100 不是所有形状都赢。
- 它在足够大的 `N` 与较低/中等 `d` 上更有优势。
- 最大优势约出现在 `N=5,000,000, d=10, K=20`，但 `d=256` 时优势明显缩小。

### permutation

A100 的 `permutation_matrix_gpu.csv` 有 15 个 pass rows。当前实现把置换检验
改写成 batched contrast matrix multiplication，再流式累计 exceedance。
这是正确的 GPU 方向，但当前 matched slice 的结果仍然是负结论：

- 在 `n=5,000, p=50,000, batch_R=512` 的共同点上，CPU 快于 A100。
- 这个结果应当保留，因为它说明“把代码搬到 GPU”不是优化，算法形状才是优化。

## 演讲建议

讲 JAX 时不要把 CPU 负结果藏起来。更有说服力的主线是：

1. JAX 需要纯函数、静态 shape 与显式 RNG，这会改变开发体验。
2. GPU 要求 batched/matrix 形状；没有足够 work，启动和数据布局开销会吃掉收益。
3. k-means 给出正例，permutation matrix path 给出当前负例。二者一起构成
   “validate before accelerate”的证据。

## 当前可引用材料

- `experiments/results/linux_server_a100/long_safe_20260503_190133/README.md`
- `experiments/results/linux_server_a100/long_safe_20260503_190133/figures/kmeans_cpu_gpu_break_even.png`
- `experiments/results/linux_server_a100/long_safe_20260503_190133/figures/permutation_cpu_gpu_break_even.png`
- `experiments/results/presentation_figures/server_kmeans_cpu_a100_summary.png`
- `experiments/results/presentation_figures/server_permutation_cpu_a100_summary.png`
