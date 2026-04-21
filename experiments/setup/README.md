# 实验环境搭建

本仓库现在以 **conda 多环境** 为主，而不是本地 `venv`。原因很直接：演讲里的 3 类运行时本来就不该混在一个解释器里。

> **注意**：Python 3.14 的 **experimental JIT** 与 **free-threaded** 不是同一个构建目标。请始终把它们当成两个独立环境处理，并用 `python -VV` / `sys._is_gil_enabled()` / `sys._jit.is_available()` 做实测确认。

## 1. 推荐环境划分

| conda env | Python | 用途 |
|-----------|--------|------|
| `py312` | 3.12.x，标准 GIL | NumPy baseline、Numba、JAX、画图、主开发环境 |
| `py314` | 3.14.x，标准 GIL | `kmeans_loops`、纯 Python 3.14 对照 |
| `py314t` | 3.14.x，free-threaded | `ThreadPoolExecutor` + 共享大数组的置换检验 |

## 2. 本机实际状态

已确认：

- `py312` 已存在，并可导入 `numpy` / `numba` / `jax` / `scikit-learn` / `pyperf` / `psutil`
- `py314` 已创建成功，版本为 `Python 3.14.4`
- `py314t` 已存在，版本为 `Python 3.14.0 free-threading build`
- `py314t` 中 `sys._is_gil_enabled()` 返回 `False`
- `py314` 中 `sys._jit` 模块存在，但 `sys._jit.is_available()` 返回 `False`

最后一点很重要：**当前这台机器上的 `conda-forge` 标准 `py314` 还不能直接当成“JIT 已启用”环境**。它仍然适合作为 3.14 GIL build 对照环境，但如果你要做“CPython experimental JIT on/off”演示，仍需另外准备一个确认可用的 JIT 构建。

## 3. 创建 / 更新命令

下面是当前仓库已验证可用的命令。

### `py312`

如果 env 已存在，只补项目需要的包即可：

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda run -n py312 python -m pip install pyperf
```

如果你要从头新建，可参考：

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda create -n py312 -c conda-forge -y python=3.12 numpy scipy numba matplotlib scikit-learn psutil
conda run -n py312 python -m pip install jax jaxlib pyperf
```

### `py314`

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda create -n py314 -c conda-forge -y python=3.14 numpy scikit-learn matplotlib pyperf psutil
```

### `py314t`

若 env 已存在但缺少 benchmark 依赖：

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda clean -p -t -y
conda install -n py314t -c conda-forge -y numpy psutil pyperf
```

## 4. 验证命令

### 解释器特性

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"

conda run -n py314 python -c "import sys; from sys import _jit; print(sys.version); print('gil', sys._is_gil_enabled()); print('jit_available', _jit.is_available())"

conda run -n py314t python -c "import sys; print(sys.version); print('gil', sys._is_gil_enabled())"
```

预期：

- `py314` 的 `gil` 为 `True`
- `py314` 的 `jit_available` 当前为 `False`
- `py314t` 的 `gil` 为 `False`

### smoke test

这些命令已在本机跑通：

```bash
conda run -n py312 python experiments/kmeans/bench_kmeans.py \
  --impl numpy_smart numba jax --n-samples 200 --n-features 4 --k 3 --centers 3 \
  --max-iter 5 --warmup 1 --repeat 1

conda run -n py314 python experiments/kmeans/bench_kmeans.py \
  --impl loops --n-samples 200 --n-features 4 --k 3 --centers 3 \
  --max-iter 5 --loops-max-n 200 --warmup 1 --repeat 1

conda run -n py314t python experiments/permutation_test/bench_permtest.py \
  --impl numpy_trick threads --n1 40 --n2 40 --r 64 --warmup 1 --repeat 1 --max-workers 2
```

## 5. 关于 benchmark driver 的依赖策略

为了支持这种拆分环境，`bench_kmeans.py` 和 `bench_permtest.py` 现在已经改成：

- 只有在你选择 `numba` 实现时才导入 `numba`
- 只有在你选择 `jax` 实现时才导入 `jax`

因此：

- `py314` 不需要安装 `numba`，也可以跑 `--impl loops`
- `py314t` 不需要安装 `numba` / `jax`，也可以跑 `--impl threads`

## 6. 旧的 `venv` 脚本

`setup_envs.sh` 仍然保留，适合作为最简单的本地 `venv` 参考。但它**不是**当前这份仓库文档的主路径，也不能替代 `py314` / `py314t` 这类解释器级别差异。

## 7. 可复现性

每次正式 benchmark 至少记录：

- `python -VV`
- `pip freeze` 或 `conda list`
- CPU 型号、核心数、操作系统
- 是否为 `py312` / `py314` / `py314t`
- `sys._is_gil_enabled()` 与 `sys._jit.is_available()` 的检查结果
