# 相关演讲与参考文献

本文档收集与 **Python 数值性能、Numba、JAX、并行** 相关的公开资料，供演讲对齐叙事与延伸阅读（**非**完整学术引用列表）。

## 1. PyCon 与社区演讲

| 活动 | 题目 | 链接 |
|------|------|------|
| PyCon US 2026 | Breaking the Speed Limit: Fast Statistical Models with Python 3.14, Numba, and JAX（本演讲） | [日程条目](https://us.pycon.org/2026/schedule/presentation/128/) |
| PyCon US 2025 Tutorial | Speed Up Your Code by 50x: A Guide to Moving from NumPy to JAX | [存档](https://pycon-archive.python.org/2025/schedule/presentation/54/) |
| PyCon UK 2025 | JIT compilers for scientific computing: Numba vs JAX | [pretalx](https://pretalx.com/pyconuk-2025/talk/CFFEDZ/) · [幻灯片笔记](https://ickc.github.io/RSE/PyAutoLens/2025-09-21-autojax.html) |
| PyCon DE & PyData 2026 | Demystifying Parallel Programming in Python | [pretalx](https://pretalx.com/pyconde-pydata-2026/talk/HBFL78/) |

## 2. 讲者往期资料（PyCon HK 2025）

- [Functional Programming in Python](https://lucajiang.github.io/functional_python/)
- [Shall We Upgrade? Navigating Python's Rapid Evolution](https://lucajiang.github.io/new_in_python/)

## 3. PEP 与官方文档

- [PEP 703 – Optional GIL](https://peps.python.org/pep-0703/)
- [PEP 744 – JIT Compilation](https://peps.python.org/pep-0744/)
- [PEP 774 – JIT stencils / LLVM](https://peps.python.org/pep-0774/)
- [Python 3.14 – Free threading howto](https://docs.python.org/3.14/howto/free-threading-python.html)
- [Python 3.14 – Configure](https://docs.python.org/3.14/using/configure.html)

## 4. 工具文档

- [Numba](https://numba.readthedocs.io/)
- [JAX](https://jax.readthedocs.io/)
- [NumPy](https://numpy.org/doc/stable/)
- [SciPy `permutation_test`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)

## 5. 博客与讨论（观点性，引用时需自行核实）

- [Free-Threading Python vs Multiprocessing](https://baarse.substack.com/p/free-threading-python-vs-multiprocessing) — 内存与开销讨论
- [The Optimization Ladder](https://cemrehancavdar.com/2026/03/10/optimization-ladder/) — 优化层次与决策

## 6. 使用建议

- 幻灯片中引用 **PEP 编号 + 版本** 比引用第三方博客倍数更稳妥。
- 所有 **性能数字** 优先来自 **本仓库 `experiments/` 实测**，并注明环境。
