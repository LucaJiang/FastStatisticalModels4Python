# k-means 实验（迭代 / 循环密集）

## 文件

| 文件 | 说明 |
|------|------|
| `data_gen.py` | `sklearn.datasets.make_blobs` 生成数据 |
| `kmeans_numpy.py` | NumPy 向量化 Lloyd baseline |
| `kmeans_loops.py` | 显式 Python 循环（用于 CPython JIT 对照；**大 N 极慢**） |
| `kmeans_numba.py` | `@njit` 编译内层循环 |
| `kmeans_jax.py` | `jax.lax.scan` + `jax.jit` |
| `bench_kmeans.py` | 统一基准脚本 |
| `sweep_kmeans.py` | 按 N 与实现批量跑 benchmark，每个实现进独立子进程 |
| `check_sklearn_equivalence.py` | 与 `sklearn.cluster.KMeans` 在固定初始质心下做 inertia sanity check |

## 运行示例

```bash
conda run -n py312 python experiments/kmeans/bench_kmeans.py \
  --impl numpy_smart numba --n-samples 100000 --k 5 --max-iter 50
```

纯 Python `loops` 对照建议放在 `py314`，并使用**小 N**：

```bash
conda run -n py314 python experiments/kmeans/bench_kmeans.py \
  --impl loops --loops-max-n 5000 --n-samples 100000
```

JAX 版本建议在 `py312` 跑：

```bash
conda run -n py312 python experiments/kmeans/bench_kmeans.py \
  --impl jax --n-samples 50000
```

可用实现名：

- `numpy_naive`
- `numpy_smart`
- `loops`
- `numba`
- `jax`

批量 sweep 支持只跑某些实现：

```bash
conda run -n py312 python experiments/kmeans/sweep_kmeans.py \
  --impls numpy_smart numba jax --skip-loops \
  --output-csv experiments/results/v2/kmeans_subset.csv
```

高维 shape stress 可跳过大 N 下的 naive broadcasting：

```bash
conda run -n py312 python experiments/kmeans/sweep_kmeans.py \
  --n-list 5000 20000 50000 --n-features 100 --k 50 \
  --max-iter 10 --skip-loops --max-numpy-naive-n 5000 \
  --output-csv experiments/results/v2/kmeans_shape_k50_d100.csv
```

## 输出

- 默认打印 JSON 行；可用 `--output-csv results.csv` 供 `experiments/visualization/plot_runtime.py` 使用。

## 正确性

- 各实现返回 **inertia**（簇内平方和）；同一 `seed` 下可横向比较数量级（浮点差异允许）。
- `bench_kmeans.py` 已改为**按需导入** `numba` / `jax`，所以 `py314` 不安装 `numba` 也可以只跑 `--impl loops`。
- `check_sklearn_equivalence.py` 用相同 `init_centroids` 调 sklearn，当前本机 sanity check 的 inertia 相对差约 `1.5e-16`。
