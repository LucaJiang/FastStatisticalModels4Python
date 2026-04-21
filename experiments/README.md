# Experiments

| 目录 | 内容 |
|------|------|
| [`setup/`](setup/) | 依赖与虚拟环境说明 |
| [`kmeans/`](kmeans/) | k-means：NumPy / loops / Numba / JAX + `bench_kmeans.py` |
| [`permutation_test/`](permutation_test/) | 置换检验：串行 / 多进程 / 线程 / Numba / JAX + `bench_permtest.py` |
| [`devex/`](devex/) | 粗略 LOC 统计 |
| [`visualization/`](visualization/) | 从 CSV 绘制 runtime / memory 等 |
| [`results/`](results/) | 本机跑出的 CSV/JSON/图与 [`README.md`](results/README.md) |

环境安装：

```bash
bash experiments/setup/setup_envs.sh
# 可选：INSTALL_JAX=1 bash experiments/setup/setup_envs.sh
```
