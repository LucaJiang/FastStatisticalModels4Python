# 结果可视化

读取 `bench_*.py` 导出的 **CSV**（`--output-csv`）并绘图。

## 依赖

与 `experiments/setup/requirements-base.txt` 相同（`matplotlib`）。

## 用法

```bash
# 先生成 CSV
python experiments/kmeans/bench_kmeans.py --impl numpy numba --output-csv experiments/visualization/kmeans.csv

python experiments/visualization/plot_runtime.py --input experiments/visualization/kmeans.csv --output experiments/visualization/kmeans_runtime.png
```

内存对比（置换检验）：

```bash
python experiments/permutation_test/bench_permtest.py --impl numpy multiprocessing numba --output-csv experiments/visualization/perm.csv
python experiments/visualization/plot_memory.py --input experiments/visualization/perm.csv --output experiments/visualization/perm_memory.png
```
