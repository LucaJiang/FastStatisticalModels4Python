# 置换检验实验（embarrassingly parallel）

## 文件

| 文件 | 说明 |
|------|------|
| `data_gen.py` | 合并两组 1D 样本为 `x`，返回 `(x, n1)` |
| `permtest_numpy.py` | 串行 NumPy |
| `permtest_multiprocessing.py` | `ProcessPoolExecutor` + worker initializer（每进程持有一份数据） |
| `permtest_freethreaded.py` | `ThreadPoolExecutor`（共享内存；在 **free-threaded** Python 上测并行） |
| `permtest_numba.py` | `@njit(parallel=True)` + `prange` |
| `permtest_jax.py` | `jax.jit` + `jax.vmap` |
| `bench_permtest.py` | 统一基准 |

## 运行示例

```bash
conda run -n py312 python experiments/permutation_test/bench_permtest.py \
  --impl numpy_trick multiprocessing numba --n1 5000 --n2 5000 --r 2000
```

线程版本建议在 `py314t` 跑：

```bash
conda run -n py314t python experiments/permutation_test/bench_permtest.py \
  --impl numpy_trick threads --r 4000 --max-workers 8
```

JAX：

```bash
conda run -n py312 python experiments/permutation_test/bench_permtest.py \
  --impl jax_perm jax_trick --r 512 --n1 2000 --n2 2000
```

可用实现名：

- `numpy_naive`
- `numpy_batched`
- `numpy_trick`
- `numpy_trick_batched`
- `multiprocessing`
- `threads`
- `numba`
- `jax_perm`
- `jax_trick`

## 内存说明

- `bench_permtest.py` 可选打印 **RSS**（需 `psutil`）。多进程场景下应用系统工具观察 **总物理内存** 与 **进程数 × 数据拷贝** 的关系。

## 可复现性

- 各实现使用 `numpy` / `RandomState` / `jax.random` / Numba 内置 LCG 的不同 RNG 路径，**数值 checksum 不一定一致**；以 **耗时与内存行为** 为主。
- `permtest_numba.py` 使用 **确定性 LCG + Fisher–Yates** 生成置换，与 `numpy.random.Generator` **统计量分布相近但逐次不同**。
- `bench_permtest.py` 已改为**按需导入** `numba` / `jax`，所以 `py314t` 不安装 `numba` / `jax` 也可以只跑 `--impl threads`。
