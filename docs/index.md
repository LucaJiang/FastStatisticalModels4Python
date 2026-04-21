# 文档目录

这里按主题整理了 `docs/` 下的所有文档，方便快速跳转。

## 主题目录

### 运行时与语言特性

- [Python 3.14：Free-threaded 与实验性 JIT](01-python314-features.md)

### 数值库速查

- [Numba 数值计算速查（含实测数据）](02-numba-guide.md)
- [JAX 统计与高性能数值速查（含 CPU 实测）](03-jax-guide.md)

### 核心算法与实现

- [k-means（Lloyd 算法）实现与内存深度剖析](04-kmeans-algorithms.md)
- [置换检验（Permutation Test）实现要点与并行策略实测](05-permutation-test.md)

### 方法论与参考

- [基准测试与可复现性方法论（本仓库真实做法）](06-benchmarking-methodology.md)
- [相关演讲与参考文献](07-related-talks-and-references.md)

## 建议阅读顺序

1. 先看 [06 基准测试与可复现性](06-benchmarking-methodology.md)，了解实验口径。
2. 再看 [02 Numba](02-numba-guide.md) 和 [03 JAX](03-jax-guide.md)，建立工具层认知。
3. 然后看 [04 k-means](04-kmeans-algorithms.md) 与 [05 置换检验](05-permutation-test.md)，对应两个主实验。
4. 最后补 [01 Python 3.14 特性](01-python314-features.md) 与 [07 参考资料](07-related-talks-and-references.md)。
