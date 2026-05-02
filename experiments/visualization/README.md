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

当前演讲用的 v2 图表主要由这些脚本生成：

```bash
python experiments/visualization/plot_kmeans.py --input experiments/results/v2/kmeans_sweep.csv --output-dir experiments/results/v2/
python experiments/visualization/plot_kmeans_shape.py --input experiments/results/v2/kmeans_shape_k50_d100.csv --output experiments/results/v2/kmeans_shape_k50_d100.png
python experiments/visualization/plot_permtest.py --input experiments/results/v2/perm_sweep.csv --output-dir experiments/results/v2/
python experiments/visualization/plot_thread_workers.py --input experiments/results/v2/perm_threads_workers.csv --output experiments/results/v2/perm_threads_py312_py314t.png
python experiments/visualization/plot_tradeoff.py --kmeans experiments/results/v2/kmeans_sweep.csv --permtest experiments/results/v2/perm_sweep.csv --output experiments/results/v2/tradeoff_radar.png
```
