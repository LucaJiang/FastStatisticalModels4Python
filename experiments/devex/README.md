# 开发者体验（Devex）度量

`code_metrics.py` 对 `experiments/kmeans/` 与 `experiments/permutation_test/` 下的实现文件做 **行数统计**，用于演讲「与 NumPy baseline 相比多写了多少行 / 结构复杂度」的**粗略**量化。

> 说明：LOC 不是质量指标；仅作辅助叙事。

## 运行

```bash
python experiments/devex/code_metrics.py
```

可选输出 CSV：

```bash
python experiments/devex/code_metrics.py --output experiments/devex/metrics.csv
```
