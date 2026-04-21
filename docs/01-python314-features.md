# Python 3.14：Free-threaded 与实验性 JIT

本文档汇总与演讲「Breaking the Speed Limit」相关的 **Python 3.14** 运行时特性。不仅提供结论，更深入探讨其底层设计机制，为现场回答高级 Python 工程师的技术提问提供充足弹药。

## 1. Free-threaded 构建（PEP 703 / PEP 779）

### 1.1 核心机制：移除 GIL 带来的底层重构

- **内存分配器（mimalloc）**：为了在多线程环境下保证无锁的内存分配效率，Free-threaded 构建默认集成了 `mimalloc`。由于去除了 GIL，多个线程会并发分配/释放内存，如果继续使用旧版的 `pymalloc` 将会导致严重的锁争用。
- **偏置引用计数（Biased Reference Counting, BRC）与延迟引用计数（Deferred Reference Counting）**：
  - 传统 CPython 每个对象的赋值/销毁都会原子的增减 refcount，这在无 GIL 时会造成可怕的 CPU 缓存行伪共享（False Sharing）。
  - BRC 允许将引用计数偏向"创建对象的那个线程"，通过本地线程进行操作，只有跨线程访问时才使用原子的原子操作（atomic instructions）。
  - 对于常驻对象（如单例、内置类型或顶层函数），采用延迟引用计数或不追踪（Immortalization, PEP 683），进一步降低锁争用。

### 1.2 行为与性能代价

- 在 free-threaded 模式下，**单线程**代码相对传统 GIL 构建通常有 **约 5–10%** 的性能损失。原因在于即便有 BRC，依然会有不可避免的轻量级锁检查和内存屏障（Memory Barriers）开销。
- 本地自适应解释器（PEP 659 的 Specializing Adaptive Interpreter）在无 GIL 环境下面临并发覆写字节码的问题。在 Python 3.13/3.14 中，这需要精细的锁或 RCU (Read-Copy-Update) 机制来保证类型专化的线程安全。

### 1.3 实验中的表现（在 GIL 构建上的天花板）

本仓库的置换检验实验在标准 GIL 构建（Python 3.11.6, macOS 15.1, Apple Silicon 8 核心）上运行。即便在 GIL 下，`ThreadPoolExecutor` 仍然取得了 **~1.4× 加速**（n=10k, R=10000，详见 [`perm_scaling.png`](../experiments/results/v2/perm_scaling.png)）。原因：NumPy 的 `.sum()`、`.permutation()` 在 C 层释放 GIL，使线程可以重叠。**这就是 GIL 构建上线程加速的上限。** 在 free-threaded 构建上，同一段 Python 代码应该继续向 8 核心靠拢——这是演讲最具说服力的「升级即收益」论点。

---

## 2. 实验性 JIT（PEP 744 / PEP 774）

### 2.1 Copy-and-Patch 技术解析

Python 3.14 搭载的实验性 JIT 并非传统的追踪式 JIT（如 PyPy 的 Tracing JIT）或基于完整 LLVM 编译的 JIT，而是基于 **Copy-and-Patch** 技术的模板 JIT（Template JIT）。

- **Stencil（模板生成）**：在 CPython 的构建阶段（而非运行阶段），使用 LLVM 对 C 语言写好的解释器指令（Opcode）进行编译，提取出机器码模板（Stencils）。这正是 PEP 774 的核心——不再要求用户的运行环境有 LLVM，而是直接在 CPython 源码树中预编译机器码片段。
- **运行时 Patch**：当一段 Python 代码变"热"（Hot）时，JIT 引擎只需把预设的机器码模板像"拼图"一样拷贝到内存可执行页中，把函数指针和偏移量作为参数（Patch）填入即可。
- **优势**：编译开销极低（因为只是内存拷贝和重定位），预热极快，完全不需要运行时进行复杂的寄存器分配和指令选择。
- **劣势**：因为 stencil 在构建时生成，**无法进行跨 opcode 的优化（例如 loop hoisting、公共子表达式消除）**，因此对于数值代码，与 Numba / JAX 的完整 LLVM 编译差距依然显著。

### 2.2 演讲实验的启示：什么能被加速？

- JIT 对于 **纯 Python 密集循环**、条件分支（控制流）有着显著加速。
- **NumPy/C 扩展盲区**：如果在 CPython JIT 下运行高度 NumPy 向量化的代码，JIT **毫无作用**。因为运行时大部分时间在 `numpy` 的 C 库中。
- 在我们的 k-means 实验中，`kmeans_loops.py`（N=2000, d=10, k=5）在标准 CPython 3.11 下单次完整跑需 **0.84 s**；对照 Numba 版本的 **0.010 s**（N=100k，80 倍大）。这一巨大差距正是 Python 循环解释开销的量化，也是 CPython JIT 值得去优化的地方——但即便 3.14 JIT 把纯循环加速 3×，距 Numba 也还有一个数量级。**这是向观众传达的真实界限**：JIT 加速的是 Python 解释器层面的开销，而不是改变算法的渐进复杂度或 C 核心的执行速度。

---

## 3. 陷阱：JIT 与 Free-threaded 的正交与互斥

在 **Python 3.14** 的实验性阶段，你需要向观众诚实指出：

- **现阶段，JIT 与 Free-threading 是互相独立的，甚至部分版本构建存在排斥。**
- 并发修改字节码（JIT 编译时替换指令）在 Free-threaded 下需要非常复杂的同步原语，因此很多时候我们是在 **单线程 JIT** 和 **多线程无 JIT** 之间做选择。
- 演讲建议：将两者解耦讲解。K-means 用于讲 JIT 带来的"原生 Python"收益；Permutation Test 用于讲 Free-threaded 解决的多进程共享内存痛点。
- 验证方式：`python -VV`、`sys._is_gil_enabled()`、`sys.flags.jit`。3.15+ 可能会放宽此约束。

---

## 4. Free-threaded 与科学计算栈的真实交互

在 Permutation Test 的实验中，我们会发现 Free-threaded 版本的线程池性能非常优异，但有两个容易踩坑的点：

1. **Thread-Safety 并不免费**：即便释放了 GIL，如果在 Python 循环里依然对全局状态（如往 `list.append` 结果）频繁访问，会导致 CPython 内部的微观锁争用，从而拖慢速度。正确的做法是：**为每个线程分配独立的写入 Buffer，最后统一 merge**（本仓库 [`permtest_freethreaded.py`](../experiments/permutation_test/permtest_freethreaded.py) 即按此设计）。
2. **NumPy 的内部锁**：NumPy 在执行向量化运算时，内部的内存分配也可能遭遇 C 层面的锁。好在现代 NumPy 和底层 BLAS（如 OpenBLAS / MKL）对多线程有着不同程度的支持和隔离。

---

## 5. 专家视角结论

在准备这篇演讲时，应该让观众明确：Python 3.14 的到来并不意味着我们可以盲目写纯 Python 循环。它降低了"偶尔写出慢代码"的惩罚，并为构建高性能的多线程 Python 库（如在单进程内共享巨大矩阵的统计工具）提供了基础设施。

**实测数据要点（Apple Silicon 8 核, Python 3.11, NumPy 1.24, Numba 0.57, JAX 0.4）**：

- 即使在 GIL 上，`ThreadPoolExecutor` 在 NumPy-heavy 置换检验里仍能取得 1.4× 加速（见 [`perm_scaling.png`](../experiments/results/v2/perm_scaling.png)）。无 GIL 构建应进一步扩大此差距。
- `multiprocessing` 在 R=10000 时的子进程 RSS 加总 **757 MB**（8 worker × 10k-length float64 array + 解释器基础内存），对照同规模线程池的 **~1.4 MB**。[`perm_memory.png`](../experiments/results/v2/perm_memory.png) 把这一差距做成了一张直观的对数刻度条形图。
- 在 CPU 上运行基于 `jax.vmap` 的置换检验（R=10000）耗时 **~75 s**——比纯 NumPy 慢 37×。JAX 在加速器上才能发挥，它在 CPU 上并非默认选项。
