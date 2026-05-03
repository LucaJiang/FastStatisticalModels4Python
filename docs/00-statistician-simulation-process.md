# Simulation-Driven Statistical Computing：面向 PyCon US 演讲与 Codex Skill 的设计参考

> 版本定位：这份文档**不是直接给参会者看的 README**，而是给演讲者本人使用的详细参考资料。它有两个目的：
>
> 1. 帮助构建 PyCon US 演讲开场：向程序员解释统计学家、生物统计学家和数据科学家如何通过 simulation 设计、验证和迭代代码。
> 2. 作为未来制作 Codex / ChatGPT Skill 的基础材料：让 AI 编码工具按照统计 simulation 的思维方式来辅助写代码，而不是只做表面上的性能优化。

---

## 0. 一句话核心

统计学家的代码开发流程通常不是“先写最快的代码”，而是：

```text
1. 定义问题与目标；
2. 建立数学模型；
3. 生成数据；
4. 验证模型；
   4.1 在小数据上验证模型；
   4.2 在大数据上验证模型；
   4.3 在不同场景下验证模型；
5. 优化模型；
   5.1 在不同场景下优化模型；
   5.2 针对性优化失败案例
6. 验证优化后的模型；
7. 重复上述过程，直到模型收敛或达到预设的停止条件。
```

因此，对统计计算来说，理想的 Python 工具链不只是追求 benchmark 上的最低 runtime，而是要在以下目标之间取得平衡：

- 统计含义清楚；
- 结果可信、可复现；
- 代码容易修改；
- 失败模式容易定位；
- 在真实数据规模上足够快；
- 优化版本能被 reference implementation 验证；
- 统计学家不需要变成 C++ / Rust / compiler engineer，也能获得合理性能。

---

## 1. 这段开场要解决什么问题

你的演讲摘要已经说明了主题：Python 3.14 的 free-threaded build、experimental JIT，以及 Numba、JAX 等工具，给了数据科学家新的方式去挑战 “Python is slow” 的刻板印象。你会用两个 workload：

1. **k-means**：代表 complex iterative loops；
2. **permutation test**：代表 massive data parallelism。

但是在进入 Python 性能工具之前，最好先帮助程序员理解：统计学家的“性能问题”从哪里来，为什么 simulation workflow 会天然要求代码既要容易修改，又要在大规模重复计算中跑得足够快。

很多程序员熟悉的开发叙事是：

```text
specification -> implementation -> unit tests -> optimization -> deployment
```

而统计学家实际更常经历的是：

```text
scientific question -> mathematical model -> simulation design -> reference implementation -> empirical behavior check -> failure discovery -> algorithm revision -> larger simulation -> performance bottleneck -> optimized implementation -> validation against reference
```

这里的关键差别是：统计问题中的“specification”往往不是一开始就完全固定的。我们经常需要通过 simulation 才知道：

- 这个统计量在有限样本下是否稳定；
- 这个算法是否对初始化敏感；
- 这个方法是否在某些数据分布下失效；
- 这个理论上漂亮的 procedure 是否在真实数据规模中遇到数值、内存或 runtime 问题；
- 我们是否需要修改模型、停止条件、随机化策略、近似方法或实现方式。

所以，统计学家的代码不是单纯的 engineering artifact，它也是 scientific reasoning 的一部分。

---

## 2. 面向程序员的开场草稿：中文思路版

下面是一段可以在演讲开头使用的中文思路草稿。真正上台时可以压缩成 2–4 分钟，并翻译成英文。

### 2.1 开场版本 A：偏叙事

在讨论 Python 性能之前，我想先从统计学家的工作方式讲起。

很多时候，我们写统计代码时，第一件事不是问：“怎么让它最快？”而是问：“我能不能相信它？”

对统计学家来说，simulation 不只是生成一些玩具数据，也不是论文最后附上的一个实验。它是我们设计、测试和修改算法的核心方式。我们通常会先构造一个非常小、非常可控的数据集，比如两个明显分开的 cluster，或者两个理论上没有任何差异的 group。然后我们写一个最清楚、最接近数学定义的版本，即使它很慢。这个版本的目标不是 performance，而是 trust。

接着，我们会一点点把问题变难：增加噪声、增加维度、让 cluster overlap、制造 imbalance、加入 outliers，或者把 permutation 的次数从几千增加到几十万、几百万。很多算法在理论上看起来很好，但在这些 simulation 场景里会暴露非常实际的问题：不收敛、对随机种子敏感、内存爆掉、数值不稳定、边界情况处理错误，或者只是慢到无法使用。

这就意味着，我们需要的代码工具有两个看起来矛盾的要求：一方面，代码必须容易修改，因为模型和算法会反复变化；另一方面，代码也必须足够快，因为 simulation、resampling 和 iterative algorithms 很容易产生大量重复计算。

所以，对我们来说，问题不是简单地问 “Python 是快还是慢”。更实际的问题是：在这个 simulation cycle 里，哪一部分需要灵活性，哪一部分需要加速？我们能不能先用清楚的 Python 写出可信的 reference implementation，然后只把真正 expensive 的 computational pattern 交给更高性能的工具？

这就是今天这场 talk 的出发点。

### 2.2 开场版本 B：偏程序员语言

Before we talk about Python 3.14, Numba, or JAX, I want to describe a workflow that is very common in statistics and biostatistics.

When we write statistical code, we often do not begin with performance. We begin with controlled simulation.

We create data where we know what should happen. We write the clearest implementation first. We test easy cases. Then we deliberately search for bad cases. We make the data noisier, larger, more imbalanced, higher-dimensional, or more computationally repetitive. Only after the behavior makes sense do we scale up and optimize.

This means statistical code has to support a very specific cycle:

```text
simulate -> implement clearly -> test behavior -> find failure modes -> revise -> scale -> optimize -> validate again
```

That cycle is why pure speed is not enough. If a tool gives me fast code that I cannot inspect, modify, or validate, it is not actually useful for scientific work. But if the code is clear and too slow to run the simulation we need, it is also not enough.

So the real goal is not “make Python magically fast everywhere.” The goal is to keep Python’s flexibility where we need it, and accelerate the computational patterns that simulation shows are truly expensive.

---

## 3. 统计学家如何通过 simulation 设计和测试代码

### 3.1 Simulation 的角色

在统计计算里，simulation 至少有五个角色。

#### 角色 1：构造已知答案的测试环境

真实数据通常没有 ground truth。我们不知道真实 cluster label，不知道真实 effect size，也不知道某个复杂过程的精确 sampling distribution。因此，第一步往往是构造 synthetic data：

- 我们知道数据来自什么分布；
- 我们知道真实参数是多少；
- 我们知道 null hypothesis 是否成立；
- 我们知道 cluster 是否真的存在；
- 我们知道某个 algorithm 大概应该输出什么。

这和传统 unit test 中的 expected output 类似，但统计问题的 expected output 通常不是完全确定的单个值，而是一个行为区间、分布性质或概率性质。

例如：

- 在 permutation test 的 null setting 下，p-value 应该大致服从 Uniform(0, 1)，type I error 应该接近 nominal level；
- 在有 strong signal 的 k-means setting 下，算法应该能恢复明显分开的 clusters；
- 在没有 signal 的场景下，算法不应该“稳定地产生看起来很漂亮的假结构”；
- 在高噪声场景下，算法性能下降应该是可解释的，而不是由 bug 导致的。

#### 角色 2：把 statistical correctness 和 code correctness 分开

统计代码的错误有两类：

1. **代码没有实现我们想实现的数学定义**；
2. **数学方法本身在某些数据机制下表现不好**。

Simulation 可以帮助区分这两类错误。

例如，一个 permutation test 的 p-value 不对，可能是因为：

- shuffle 维度错了；
- 没有保留 paired structure；
- 随机种子使用不当；
- p-value 的 numerator/denominator convention 错了；
- statistic 的方向定义反了；
- Monte Carlo permutation 数量太少；
- exchangeability assumption 本身不成立。

其中有些是 implementation bug，有些是 method assumption 问题。Simulation 可以通过改变数据生成机制来定位问题。

#### 角色 3：暴露理论和实践之间的差距

很多算法理论上是合理的，但实际应用时会遇到有限样本、浮点数、内存和 runtime 的约束。

常见差距包括：

- asymptotic theory 很好，但样本量不够大时表现差；
- objective function 定义明确，但优化过程容易陷入局部最优；
- convergence criterion 写得太严格导致 runtime 爆炸；
- convergence criterion 写得太宽松导致结果不稳定；
- 理论上每次 permutation 独立，但实现时随机数或内存布局造成瓶颈；
- 理论上一个矩阵操作很简单，但实际数据太大时中间矩阵无法放进内存；
- GPU / accelerator 理论上快，但数据传输和编译开销抵消了收益。

Simulation 是发现这些差距的主要工具。

#### 角色 4：指导优化方向

优化之前要先知道瓶颈在哪里。统计 simulation 的好处是：它可以把 workload 分解成清楚的 computational pattern。

例如：

- k-means 的核心瓶颈通常是距离计算、assignment step、centroid update 和迭代次数；
- permutation test 的核心瓶颈通常是大量独立重复、statistic recomputation、random shuffle 和内存分配；
- bootstrap / resampling 的核心瓶颈通常是 repeated sampling + repeated model fitting；
- MCMC 的核心瓶颈通常是 transition kernel、likelihood evaluation 和 chain length；
- cross-validation 的核心瓶颈通常是 repeated split + repeated training/evaluation。

Simulation 可以帮助我们判断：

- 是 Python-level loop 慢？
- 是 NumPy 临时数组太多？
- 是内存带宽限制？
- 是随机数生成限制？
- 是编译开销大于计算收益？
- 是数据拷贝比计算更贵？
- 是算法复杂度本身需要改变？

#### 角色 5：验证优化版本没有改变统计含义

一旦我们把 reference implementation 改写成 Numba、JAX、multi-threaded Python 或其他优化版本，必须回头验证：

- 输出是否与 reference implementation 一致；
- 差异是否只来自允许的浮点误差或 Monte Carlo randomness；
- edge cases 是否仍然正确；
- 随机数行为是否可复现；
- p-value、cluster assignment、loss、iteration count 等关键结果是否在合理范围内；
- 优化没有偷偷改变统计定义。

这一步非常重要。AI coding tools 在优化时很容易“帮忙重构”代码，但统计学家必须确保它没有改变问题本身。

---

## 4. 一个完整的 simulation workflow

下面是可以作为 Codex Skill 的核心工作流。

### 4.1 总体流程

```text
1. Clarify statistical target
   明确要估计什么、测试什么、优化什么、比较什么。

2. Define data-generating mechanisms
   定义一个或多个 synthetic data generator。

3. Write a clear reference implementation
   写最接近数学定义的版本，不追求速度。

4. Run small deterministic checks
   在小数据、固定 seed、已知答案的场景下检查基本行为。

5. Run simulation scenarios
   系统改变 n、p、effect size、noise、imbalance、correlation、missingness 等因素。

6. Identify failure modes
   找出算法何时失败、为什么失败、是统计问题还是实现问题。

7. Profile performance
   在已经可信的版本上测 runtime、memory、scaling behavior。

8. Choose optimization strategy
   根据 computational pattern 决定 NumPy、Numba、JAX、Python 3.14 free-threading/JIT 等工具。

9. Implement optimized version
   只优化瓶颈 kernel，尽量保留 reference implementation。

10. Validate optimized version
    与 reference implementation 做 correctness comparison、regression test 和 statistical behavior check。

11. Report results
    输出 simulation summary、failure cases、performance table 和建议。
```

### 4.2 工作流表格

| 阶段              | 统计问题                        | 编程任务                       | 典型输出                 | 失败信号                      | AI / Codex 可以做什么                              |
| ----------------- | ------------------------------- | ------------------------------ | ------------------------ | ----------------------------- | -------------------------------------------------- |
| 1. 明确目标       | 我们要验证什么行为？            | 写下 method contract           | problem statement        | 目标模糊，metric 不清楚       | 把用户描述整理成统计目标和验收标准                 |
| 2. 构造数据生成器 | 数据机制是什么？                | 写 `simulate_data()`           | synthetic datasets       | generator 与目标不匹配        | 生成多种 scenario 的 data generator                |
| 3. 写参考实现     | 数学定义是什么？                | 写清楚、慢但可信的代码         | reference implementation | 代码过早复杂化                | 用直白 Python / NumPy 写 reference code            |
| 4. 小数据检查     | easy case 是否正确？            | fixed seed + small n tests     | sanity check results     | easy case 失败                | 写 deterministic smoke tests                       |
| 5. 扫描场景       | 哪些情况下表现差？              | loop over scenarios            | simulation grid          | 结果不稳定、异常值多          | 生成 scenario matrix 和批量运行脚本                |
| 6. 定位失败       | 是 bug 还是 method limitation？ | inspect intermediate values    | diagnostic plots/tables  | 错误原因不明                  | 添加 logging、assertions、intermediate checks      |
| 7. 性能剖析       | 慢在哪里？                      | profiling / benchmarking       | runtime table            | benchmark 不可复现            | 写 benchmark harness，固定 seed，记录环境          |
| 8. 选择优化       | 哪种工具匹配 pattern？          | 选择 NumPy/Numba/JAX/threading | optimization plan        | 盲目套工具                    | 根据 loop/data parallelism/memory pattern 建议工具 |
| 9. 实现优化       | 怎么加速而不改统计定义？        | 写 optimized kernel            | fast implementation      | 输出与 reference 不一致       | 生成 Numba/JAX/NumPy 改写版本                      |
| 10. 验证优化      | 统计含义是否保持？              | compare outputs                | equivalence report       | 误差超阈值、seed 不可控       | 写 comparison tests 和 tolerance policy            |
| 11. 汇报          | 如何解释结果？                  | summarize behavior             | markdown report          | 只报 runtime 不报 correctness | 生成 simulation report template                    |

---

## 5. 先小数据，再大数据：为什么这不是浪费时间

程序员可能会觉得：如果目标是 large-scale data，为什么不一开始就在 large-scale data 上测试？

统计学家的回答是：大数据会掩盖错误。

在大数据里：

- 一个错误的趋势可能看起来“很平滑”；
- 一个 bug 可能只影响少数 case，很难定位；
- runtime 太长导致无法快速迭代；
- memory pressure 可能让你以为是性能问题，但真正原因是算法定义不合理；
- 随机性让错误看起来像 Monte Carlo noise；
- 可视化和手工检查都变得困难。

小数据的价值是：

- 可以手算或半手算 expected behavior；
- 可以打印所有 intermediate values；
- 可以快速重复运行；
- 可以固定 seed 做 deterministic debugging；
- 可以用非常清楚的 reference implementation 对照；
- 可以把错误限制在一个很小的空间里。

一个实用原则是：

```text
如果算法在小而简单的数据上都不可信，那么它在大数据上跑得快没有意义。
```

但另一个原则同样重要：

```text
如果算法只在小数据上可信，却无法在真实规模上运行，那么它也无法服务实际科学问题。
```

所以 simulation workflow 必须包含 scale-up 阶段，而不是停留在 toy examples。

---

## 6. 典型 simulation scenario 设计

### 6.1 基本维度

在设计 simulation 时，通常会系统改变以下因素。

| 因素               | 含义                             | 为什么重要                        |
| ------------------ | -------------------------------- | --------------------------------- |
| `n`                | 样本量                           | 影响统计 power、runtime 和 memory |
| `p`                | 变量/特征维度                    | 高维会改变算法稳定性和计算复杂度  |
| signal strength    | effect size / cluster separation | 决定问题难度                      |
| noise level        | 误差方差、测量噪声               | 影响估计稳定性                    |
| imbalance          | group size / cluster size 不平衡 | 很多算法对 imbalance 敏感         |
| correlation        | 特征相关性                       | 独立假设常常不成立                |
| distribution shape | normal/heavy-tailed/skewed       | 检验 robustness                   |
| missingness        | 缺失机制                         | 生物医学数据常见问题              |
| outliers           | 极端值                           | 检验鲁棒性和数值稳定性            |
| random seed        | 随机性控制                       | 影响 reproducibility              |
| repetitions        | simulation 重复次数              | 决定 Monte Carlo error 和 runtime |

### 6.2 Scenario 分层

一个好的 simulation 设计通常不是随机堆很多参数，而是有层次地增加复杂度。

#### Level 0：toy deterministic case

目标：确保代码结构没错。

特点：

- 样本量很小；
- 数据可以直接看懂；
- expected output 近似明确；
- 固定 seed；
- 打印 intermediate values。

#### Level 1：easy statistical case

目标：确保方法在最友好的统计条件下表现正常。

特点：

- strong signal；
- low noise；
- balanced groups；
- no missing values；
- no extreme outliers。

#### Level 2：realistic case

目标：模拟真实应用中的常见困难。

特点：

- moderate signal；
- correlated features；
- mild imbalance；
- realistic sample size；
- moderate noise。

#### Level 3：stress case

目标：故意寻找失败模式。

特点：

- weak signal；
- high dimensionality；
- severe imbalance；
- heavy-tailed noise；
- outliers；
- large-scale repetitions。

#### Level 4：performance case

目标：测试 runtime、memory 和 scalability。

特点：

- 接近真实生产规模；
- 参数组合不一定很多，但规模大；
- benchmark 要可重复；
- 记录硬件、Python 版本、library 版本、threading 设置。

---

## 7. Reference implementation 的重要性

### 7.1 Reference implementation 是什么

Reference implementation 是一份尽量接近数学定义的实现。它可以慢，但必须清楚。

它的特点是：

- 变量名对应统计概念；
- 代码结构接近公式或算法描述；
- 少做技巧性优化；
- 中间结果容易检查；
- 可以作为 optimized implementation 的 oracle；
- 可以在小数据上运行；
- 可以被同事 review。

### 7.2 为什么不要一开始就写最高性能版本

过早优化会带来几个问题：

- 代码变得不透明，难以确认统计定义；
- 很难插入 intermediate diagnostics；
- 一旦方法需要修改，重构成本很高；
- AI coding tool 可能为了速度改变算法细节；
- benchmark 成功可能掩盖 correctness 问题。

### 7.3 Reference 和 optimized version 的关系

建议保留两套代码：

```text
reference implementation
    用于小数据、调试、验证、解释。

optimized implementation
    用于大数据、重复 simulation、benchmark、实际运行。
```

二者之间必须有 comparison tests：

- exact equality：适用于 deterministic integer / label / count 结果；
- approximate equality：适用于 floating point 结果；
- distributional equivalence：适用于随机算法或 Monte Carlo 结果；
- tolerance policy：明确允许误差范围；
- seed policy：明确随机数如何控制；
- failure logging：不一致时保存输入、seed、scenario 和中间结果。

---

## 8. 失败模式清单：simulation 要主动寻找什么

### 8.1 统计失败模式

- Type I error inflation：null 下假阳性过高；
- low power：有真实 effect 时检测不到；
- bias：估计量系统偏离真实值；
- high variance：结果对样本随机性极其敏感；
- poor calibration：p-value 或 confidence interval 不符合 nominal level；
- non-identifiability：不同参数产生类似数据；
- sensitivity to assumptions：轻微违反假设时结果严重变化；
- unstable clustering：不同 seed 产生完全不同结果；
- overfitting：simulation 中看似好，泛化场景差。

### 8.2 数值失败模式

- overflow / underflow；
- catastrophic cancellation；
- divide by zero；
- NaN / Inf 传播；
- ill-conditioned matrices；
- tolerance 设置不合理；
- float32 与 float64 结果差异过大；
- CPU/GPU 后端差异；
- parallel reduction 顺序不同导致结果不完全一致。

### 8.3 实现失败模式

- array shape 错误但被 broadcasting 掩盖；
- axis 参数写错；
- random seed 没有正确控制；
- train/test leakage；
- permutation 打乱了不该打乱的结构；
- paired data 被当成 independent data；
- group labels 和 observations 对不齐；
- missing values 被静默转成 0；
- sorting / indexing 改变了样本顺序；
- optimized version 改变了 reference semantics。

### 8.4 性能失败模式

- Python-level loop 成为瓶颈；
- NumPy 产生大量临时数组；
- memory allocation 比 computation 更贵；
- 数据拷贝频繁；
- cache locality 差；
- parallel overhead 大于收益；
- JIT compile time 淹没 runtime gain；
- GPU data transfer overhead 太高；
- batch size 不合理；
- benchmark 没有区分 warm-up 和 steady-state runtime。

---

## 9. 示例一：k-means 作为 iterative algorithm

### 9.1 为什么 k-means 是好例子

k-means 足够简单，听众容易理解；但它又包含很多真实的计算模式：

- repeated distance calculation；
- assignment step；
- centroid update；
- convergence check；
- data-dependent iteration count；
- sensitivity to initialization；
- possible empty clusters；
- memory vs speed trade-off。

它非常适合说明：一个算法的理论描述很短，但真正可用的实现需要处理大量细节。

### 9.2 k-means 的 simulation scenarios

#### Scenario A：明显分开的 clusters

目的：测试 easy case。

数据：

- 2 或 3 个 cluster；
- 每个 cluster 样本量相同；
- cluster means 距离很远；
- low variance；
- 维度低，例如 2D。

期望行为：

- algorithm 快速收敛；
- cluster assignment 与真实 label 高度一致；
- inertia 单调下降；
- 不同 initialization 结果差异小。

失败信号：

- easy case 都不能恢复 cluster；
- inertia 不下降；
- centroid update 错误；
- label permutation 导致评估指标写错。

#### Scenario B：overlapping clusters

目的：测试边界情况。

数据：

- cluster means 接近；
- variance 较大；
- clusters overlap。

期望行为：

- assignment accuracy 下降是合理的；
- inertia 仍应收敛；
- 不同 initialization 结果可能不同；
- simulation report 应说明统计难度，而不是简单判断 code 失败。

失败信号：

- 结果剧烈不稳定但没有 diagnostic；
- 误把 method limitation 当成 implementation bug；
- 只报告一个 seed 的结果。

#### Scenario C：imbalanced clusters

目的：测试 cluster size imbalance。

数据：

- 一个大 cluster，一个小 cluster；
- 或多个 cluster size 差异明显。

期望行为：

- k-means 可能倾向于切分大 cluster 或忽视小 cluster；
- 这是方法限制的一部分；
- simulation 应记录小 cluster recovery rate。

失败信号：

- empty cluster 未处理；
- centroid 出现 NaN；
- 小 cluster 被系统吞掉但报告没有体现。

#### Scenario D：high-dimensional data

目的：测试 curse of dimensionality 和性能。

数据：

- p 从 2 增加到 100、1000 或更高；
- n 也逐步放大。

期望行为：

- runtime 随 n、p、k 增长；
- distance computation 成为主要瓶颈；
- memory layout 和临时数组很重要。

失败信号：

- 构造完整 `(n, k, p)` 距离张量导致 memory 爆炸；
- vectorization 虽然减少 Python loop，却引入过大中间数组；
- optimized version 与 reference version 不一致。

### 9.3 k-means 验证指标

- final inertia；
- inertia 是否单调下降；
- number of iterations；
- convergence status；
- cluster size distribution；
- empty cluster count；
- adjusted rand index 或其他 label-invariant 指标；
- runtime；
- peak memory；
- results across random seeds；
- agreement between reference and optimized implementations。

### 9.4 k-means 的优化方向

可能比较的实现包括：

1. clear Python / NumPy reference；
2. vectorized NumPy；
3. Python 3.14 free-threaded build 下的并行策略；
4. Python 3.14 experimental JIT 的影响；
5. Numba-compiled loop；
6. JAX jit / vmap / array-based implementation。

讨论重点不是“哪个永远最快”，而是：

- 哪个版本最容易理解？
- 哪个版本最容易改 stopping rule？
- 哪个版本最容易插入 diagnostics？
- 哪个版本在小数据下 overhead 太大？
- 哪个版本在大数据下 scaling 最好？
- 哪个版本最容易保持和 reference semantics 一致？

---

## 10. 示例二：permutation test 作为 massive data parallelism

### 10.1 为什么 permutation test 是好例子

Permutation test 很适合说明统计计算中的 massive repetition。

它的核心思路通常很简单：

```text
observed statistic -> repeatedly shuffle labels -> recompute statistic -> compare observed statistic to null distribution
```

但是实际运行时会遇到：

- permutation 次数巨大；
- 每次 statistic 计算可能很简单，但重复次数多；
- permutation 可以独立并行；
- random number generation 很关键；
- p-value convention 必须明确；
- memory strategy 会决定是否可扩展。

### 10.2 permutation test 的 simulation scenarios

#### Scenario A：null hypothesis true

目的：检查 type I error 和 p-value calibration。

数据：

- 两组来自同一分布；
- effect size = 0；
- 多次 simulation repetition。

期望行为：

- p-value 近似 Uniform(0, 1)；
- 在 alpha = 0.05 时，reject rate 接近 5%；
- Monte Carlo error 随 repetition 数量下降。

失败信号：

- type I error 明显偏高；
- p-value 分布偏离 uniform；
- statistic 方向定义错；
- shuffle 破坏了数据结构。

#### Scenario B：alternative hypothesis true

目的：检查 power。

数据：

- 两组均值不同；
- effect size 从小到大变化；
- n 从小到大变化。

期望行为：

- power 随 effect size 增大而提高；
- power 随 n 增大而提高；
- 结果有 Monte Carlo uncertainty。

失败信号：

- effect size 增大但 power 不变；
- n 增大但 power 不变；
- p-value 方向反了。

#### Scenario C：imbalanced group sizes

目的：检查 group size imbalance。

数据：

- n1 和 n2 差异很大；
- 可以在 null 和 alternative 下都测试。

期望行为：

- test 仍应遵循正确 permutation scheme；
- power 可能下降；
- runtime 和 memory 可能受影响。

失败信号：

- permutation 后 group sizes 不保持；
- vectorized implementation 依赖 equal group size；
- optimized version silently assumes balance。

#### Scenario D：paired / blocked data

目的：检查 exchangeability structure。

数据：

- paired samples；
- repeated measures；
- block structure。

期望行为：

- permutation 必须限制在允许的 exchangeability group 内；
- 不能随意打乱所有 labels。

失败信号：

- AI 或优化代码把 blocked permutation 改成 global shuffle；
- null distribution 错误但 runtime 很快。

### 10.3 permutation test 验证指标

- observed statistic；
- permutation null distribution summary；
- p-value；
- p-value convention，例如 `(count_extreme + 1) / (B + 1)`；
- type I error under null；
- power under alternatives；
- Monte Carlo standard error；
- runtime per permutation；
- total runtime；
- memory usage；
- reproducibility across seeds；
- agreement between reference and optimized versions。

### 10.4 permutation test 的优化方向

可能比较的实现包括：

1. simple Python loop；
2. NumPy vectorized permutations；
3. batching strategy to control memory；
4. multi-threaded or multi-process execution；
5. Numba loop with random generation considerations；
6. JAX vmap / jit / random key splitting；
7. Python 3.14 free-threaded build 对 independent repetitions 的影响；
8. Python 3.14 JIT 对 tight loops 的影响。

这里的核心问题是：

- independent repetition 是否适合 parallelism？
- batch size 如何影响 memory？
- random number generation 是否成为瓶颈？
- JIT 编译开销是否值得？
- 结果是否与 reference implementation 保持一致？

---

## 11. 为什么统计学家需要“易修改”的高性能代码

很多统计项目中，算法不是一次写完的。

我们经常会在 simulation 中发现：

- 原来的 statistic 对 outliers 太敏感，需要换 robust version；
- 原来的 stopping rule 太慢，需要改 tolerance；
- 原来的 initialization 不稳定，需要多次 restart；
- 原来的 permutation scheme 不适合 paired data，需要改成 constrained permutation；
- 原来的 vectorization memory 太大，需要改成 batching；
- 原来的模型假设不适合真实数据，需要加入 covariate adjustment；
- 原来的 test 只适合 balanced groups，需要支持 imbalance。

如果高性能实现非常难改，那么它会拖慢统计迭代。更糟的是，研究者可能因为代码难改而不愿尝试更合理的方法。

所以我们想要的不是“一段神秘但很快的代码”，而是：

```text
清楚的统计接口 + 可验证的 reference implementation + 可替换的 optimized kernels + 自动化 simulation tests
```

这也是未来 Codex Skill 应该支持的核心设计。

---

## 12. Python 工具链应该如何放进这个叙事

### 12.1 不要把演讲讲成工具排名

这场 talk 最好不要被理解成：Numba vs JAX vs Python 3.14 谁赢。

更好的叙事是：

```text
不同工具适合 simulation workflow 的不同阶段。
```

| 工具                            | 最适合的位置                                    | 优点                           | 风险                                             |
| ------------------------------- | ----------------------------------------------- | ------------------------------ | ------------------------------------------------ |
| plain Python                    | reference implementation / teaching / debugging | 最清楚，最容易改               | Python-level loops 慢                            |
| NumPy                           | array operations / medium-scale baseline        | 生态成熟，代码短               | vectorization 可能制造巨大临时数组               |
| Python 3.14 free-threaded build | CPU-bound parallel workloads 的新可能           | 可能更好利用 threads           | 生态兼容性、debugging、library behavior 需要观察 |
| Python 3.14 experimental JIT    | runtime-level acceleration 的探索               | 可能减少解释器 overhead        | experimental，收益 workload-dependent            |
| Numba                           | numerical loops / CPU kernels                   | Python-like loop 可以编译加速  | 支持的 Python/NumPy subset 有限制                |
| JAX                             | array programming / jit / vmap / accelerator    | 很适合大规模 array computation | 需要 functional style，debugging 心智模型不同    |

### 12.2 面向听众的关键问题

每个工具都可以用以下问题评估：

1. 我能不能从清楚的数学代码开始？
2. 小数据下能不能 debug？
3. 能不能方便地插入 diagnostics？
4. 改 statistic、stopping rule、data generator 的成本高不高？
5. runtime 是否随 n、p、B 合理 scaling？
6. memory 是否可控？
7. 随机数和并行是否可复现？
8. optimized version 是否能被 reference version 验证？
9. 这个工具的学习成本是否适合 domain expert？
10. 它是否让统计学家必须变成 compiler engineer？

---

## 13. AI coding tools 在这个 workflow 中的角色

### 13.1 正确定位：copilot，不是统计判断替代品

AI coding tools 很适合做：

- 根据统计目标生成 simulation scaffold；
- 写 data generator；
- 写 reference implementation；
- 生成 scenario grid；
- 写 benchmark harness；
- 把清楚的 loop 改写成 Numba / JAX kernel；
- 添加 comparison tests；
- 生成 markdown report；
- 总结 runtime / memory / correctness trade-offs。

但 AI 不应该擅自决定：

- 统计假设是否成立；
- 哪个 p-value convention 应该使用；
- 是否可以忽略 paired / blocked structure；
- 是否可以把近似方法当成 exact method；
- 是否可以为了速度改变 estimator 或 test statistic；
- simulation scenario 是否足以支撑科学结论。

### 13.2 AI 工具最有价值的模式

一个好的 AI-assisted simulation workflow 是：

```text
User defines statistical intent
AI drafts clear implementation
User reviews statistical meaning
AI creates simulation scenarios
User inspects failure modes
AI profiles and proposes optimization
User approves statistical equivalence criteria
AI writes optimized kernel
AI compares optimized output with reference
User interprets results
```

### 13.3 对 Codex 的行为要求

Codex 在辅助 simulation 时应该遵循以下规则：

1. **先问统计目标，再写代码。** 需要明确 outcome、statistic、null/alternative、data-generating mechanism、sample size、repetition count。
2. **先写 reference implementation。** 除非用户明确要求，否则不要一开始就写最复杂的 JIT/GPU 版本。
3. **保留 reference implementation。** 不要用 optimized code 覆盖 reference code。
4. **所有优化都必须有 comparison test。** 输出不一致时要保存 seed、input scenario 和差异摘要。
5. **不要静默改变统计定义。** 任何算法近似、p-value convention、randomization scheme 的改变都必须显式说明。
6. **区分 correctness benchmark 和 performance benchmark。** 不能只报 runtime。
7. **记录 environment。** Python version、library version、hardware、threading settings、seed policy 都应该记录。
8. **报告 failure modes。** 不要只给成功案例。
9. **用 batch 控制 memory。** 尤其是 permutation / bootstrap / high-dimensional distance calculation。
10. **把可复现性当成一等目标。** seed 管理、random key splitting、parallel randomness 都要明确。

---

## 14. 未来 Skill 的设计草案

本节可以作为后续制作 Skill 的直接基础。

### 14.1 建议 Skill 名称

候选名称：

- `simulation-stat-copilot`
- `simulation-driven-statistics`
- `statistical-simulation-assistant`
- `simulation-performance-lab`

推荐：`simulation-stat-copilot`

原因：

- 足够短；
- 强调 simulation；
- 强调统计；
- 适合 Codex / ChatGPT 作为辅助工具，而不是替代统计判断。

### 14.2 Skill 的目标用户

- 生物统计学家；
- data scientist；
- computational statistician；
- 使用 Python 做 simulation、resampling、benchmark 或统计方法验证的人；
- 了解统计目标，但不一定熟悉 Numba/JAX/parallel runtime 的用户。

### 14.3 Skill 应该处理的输入

用户可能会提供：

- 一个统计方法描述；
- 一段 reference code；
- 一个 simulation idea；
- 一个性能瓶颈；
- 一个 Python script / notebook；
- 一个想比较的工具列表，例如 NumPy、Numba、JAX；
- 一个真实数据结构描述；
- 一个想测试的 failure mode；
- 一个 benchmark 结果。

### 14.4 Skill 应该输出什么

根据任务不同，Skill 可以输出：

- simulation plan；
- data-generating mechanism；
- reference implementation；
- scenario grid；
- sanity tests；
- failure-mode checklist；
- optimized implementation；
- comparison tests；
- benchmark harness；
- markdown report；
- performance/correctness trade-off summary；
- 给演讲或文档用的解释性文字。

### 14.5 Skill 文件结构建议

```text
simulation-stat-copilot/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── simulation-workflow.md
│   ├── failure-modes.md
│   ├── performance-tools.md
│   ├── validation-checklist.md
│   └── report-template.md
├── scripts/
│   ├── benchmark_harness.py
│   ├── compare_outputs.py
│   └── scenario_grid.py
└── assets/
    └── README-template.md
```

注意：Skill 的 `SKILL.md` 不应该塞入所有细节。主文件应该短而明确，把长清单放到 `references/` 中，需要时再加载。

---

## 15. `SKILL.md` 草案

下面是一个可作为后续 Skill entrypoint 的初始草案。真正打包前还需要根据实际使用方式修改。

```markdown
---
name: simulation-stat-copilot
description: assists with simulation-driven statistical computing workflows in python, especially for statisticians, biostatisticians, and data scientists who need to design simulations, write reference implementations, identify failure modes, optimize numerical kernels, compare numpy, numba, jax, and modern python runtime features, and validate optimized code against clear statistical behavior. use when the user asks for simulation design, statistical code testing, permutation/bootstrap/resampling workflows, iterative numerical algorithms, performance benchmarking, or ai-assisted conversion from clear mathematical code to faster python implementations.
---

# Simulation Stat Copilot

## Core principle

Always preserve the statistical meaning of the method before optimizing code. Start from a clear reference implementation, use simulation to validate behavior, then optimize only the computational pattern that has been shown to be expensive.

## Default workflow

1. Clarify the statistical target:
   - estimand, test statistic, loss function, clustering objective, or simulation question;
   - null and alternative settings when relevant;
   - data structure, sample size, dimensionality, and randomization constraints;
   - expected output and success criteria.

2. Propose a data-generating mechanism:
   - start with a small deterministic or easy scenario;
   - add realistic scenarios;
   - add stress scenarios that can reveal failure modes.

3. Write or preserve a reference implementation:
   - prioritize readability and statistical clarity;
   - use explicit variable names;
   - avoid premature optimization;
   - keep this version available for comparison.

4. Add validation checks:
   - deterministic small-data tests;
   - statistical behavior checks across repeated simulations;
   - edge-case tests;
   - reproducibility checks with explicit seed policy.

5. Profile performance only after correctness is plausible:
   - measure runtime and memory;
   - identify whether the bottleneck is Python loops, array allocation, random number generation, memory bandwidth, or algorithmic complexity.

6. Choose an optimization strategy based on computational pattern:
   - use NumPy for clear array operations;
   - use Numba for numerical loops and CPU kernels;
   - use JAX for jit/vmap/accelerator-friendly array programming;
   - consider modern Python runtime features when the workload involves interpreter overhead or thread-level parallelism.

7. Validate optimized code against the reference implementation:
   - compare exact outputs where possible;
   - use numerical tolerances for floating point results;
   - use distributional comparisons for stochastic algorithms;
   - report all deviations and possible causes.

8. Produce a concise report:
   - simulation scenarios;
   - correctness results;
   - failure modes;
   - benchmark results;
   - recommended implementation;
   - remaining risks.

## Rules

- Do not silently change the statistic, estimator, permutation scheme, convergence criterion, or p-value convention.
- Do not replace the reference implementation with optimized code.
- Do not report speedups without correctness checks.
- Do not assume global shuffling is valid for paired, blocked, clustered, or repeated-measures data.
- Do not hide random seed handling.
- Prefer batch-based designs when full vectorization would create large intermediate arrays.
- When performance results are requested, include enough environment information to make the benchmark interpretable.

## Common outputs

Depending on the user request, provide one or more of:

- simulation plan;
- reference implementation;
- optimized implementation;
- scenario grid;
- validation tests;
- benchmark harness;
- failure-mode analysis;
- markdown report;
- code review comments focused on statistical correctness and computational performance.
```

---

## 16. Skill references 内容草案

### 16.1 `references/simulation-workflow.md`

建议内容：

```markdown
# Simulation Workflow

Use this reference when the user asks to design or improve a simulation study.

## Minimum simulation plan

Require:

1. statistical target;
2. data-generating mechanism;
3. parameters to vary;
4. number of repetitions;
5. random seed policy;
6. metrics to record;
7. expected behavior under easy cases;
8. expected behavior under null/alternative or known truth;
9. failure modes to inspect;
10. output/report format.

## Scenario levels

- Level 0: deterministic toy case;
- Level 1: easy statistical case;
- Level 2: realistic case;
- Level 3: stress case;
- Level 4: performance case.

## Recommended outputs

- scenario table;
- simulation result table;
- diagnostic notes;
- plots if requested;
- benchmark summary;
- recommended next modifications.
```

### 16.2 `references/failure-modes.md`

建议内容：

```markdown
# Failure Modes

Use this reference when reviewing simulation results or designing stress tests.

## Statistical failures

- inflated type I error;
- low power;
- bias;
- poor calibration;
- unstable estimates;
- sensitivity to assumptions;
- overfitting to simulation design.

## Numerical failures

- NaN or Inf;
- overflow or underflow;
- cancellation;
- tolerance problems;
- dtype mismatch;
- CPU/GPU numerical differences.

## Implementation failures

- wrong axis;
- unintended broadcasting;
- broken seed policy;
- invalid permutation structure;
- leakage;
- incorrect indexing;
- optimized code changes semantics.

## Performance failures

- Python loop bottleneck;
- excessive temporary arrays;
- memory bandwidth bottleneck;
- random generation bottleneck;
- JIT warm-up hiding true cost;
- parallel overhead larger than speedup.
```

### 16.3 `references/performance-tools.md`

建议内容：

```markdown
# Performance Tool Selection

Use this reference when deciding how to optimize statistical simulation code.

## Tool choice by pattern

- Python / NumPy reference: use for clarity and validation.
- Vectorized NumPy: use for moderate array operations when intermediate memory is acceptable.
- Numba: use for numerical loops, custom kernels, iterative algorithms, and code that is close to scalar math.
- JAX: use for array programming, jit, vmap, accelerator-friendly workloads, and large batched computation.
- Threading / free-threaded Python: consider for independent tasks where Python-level threading overhead or GIL behavior matters.
- Experimental JIT: benchmark carefully; separate warm-up from steady-state runtime.

## Benchmark rules

- benchmark only after correctness checks;
- include warm-up runs;
- separate compile time from execution time;
- record Python and library versions;
- record hardware and thread settings;
- report memory when relevant;
- compare against reference outputs.
```

---

## 17. Codex prompt 模板

这些模板可以直接用于和 Codex 交互，也可以放进 Skill 的 examples。

### 17.1 设计 simulation plan

```text
I want to test a statistical method using simulation.

Statistical target:
[describe estimator/test/algorithm]

Data structure:
[describe observations, groups, features, paired/block structure if any]

Known truth:
[parameters/effect size/null setting]

Please create:
1. a small deterministic sanity-check scenario;
2. an easy statistical scenario;
3. a realistic scenario;
4. stress scenarios that may reveal failure modes;
5. metrics to record;
6. a clear Python simulation scaffold.

Do not optimize yet. Prioritize clarity and correctness.
```

### 17.2 写 reference implementation

```text
Write a clear reference implementation for the following statistical algorithm.

Algorithm:
[description or pseudocode]

Requirements:
- prioritize readability over speed;
- use explicit variable names;
- include comments that map code to the statistical steps;
- support fixed random seed;
- include small sanity tests;
- do not use Numba, JAX, multiprocessing, or advanced optimization yet.
```

### 17.3 寻找 failure modes

```text
Review this simulation code and propose failure modes to test.

Focus on:
- statistical assumptions;
- numerical stability;
- edge cases;
- random seed and reproducibility;
- data shape and indexing errors;
- performance bottlenecks that may appear at scale.

For each failure mode, propose a concrete simulation scenario that can reveal it.
```

### 17.4 优化代码但保持统计语义

```text
Optimize this reference implementation, but do not change the statistical definition.

Please:
1. identify the computational bottleneck;
2. propose whether NumPy, Numba, JAX, or batching is most appropriate;
3. keep the reference implementation unchanged;
4. create an optimized version with the same function contract;
5. write tests comparing reference and optimized outputs;
6. explain any numerical tolerance or stochastic differences.
```

### 17.5 生成 benchmark report

```text
Create a benchmark harness for these implementations.

Implementations:
- reference Python/NumPy
- optimized version 1
- optimized version 2

Benchmark dimensions:
- n: [...]
- p: [...]
- repetitions/permutations: [...]
- random seeds: [...]

Please report:
- runtime;
- memory if feasible;
- correctness agreement with reference;
- warm-up vs steady-state time when JIT is used;
- a markdown summary of trade-offs.
```

---

## 18. 适合放进演讲的主线

可以把整个 talk 的主线设计成：

```text
1. Statistics is simulation-driven.
2. Simulation creates both trust and computational pressure.
3. Trust requires clear, modifiable code.
4. Computational pressure requires acceleration.
5. Modern Python offers multiple acceleration paths.
6. AI tools can help translate clear statistical code into faster kernels.
7. But every optimized version must be validated against statistical behavior, not just benchmarked.
```

这条主线可以让听众理解：你不是单纯在比较工具，而是在展示一种 realistic workflow。

---

## 19. 可以在演讲中强调的句子

### 19.1 中文思路

- 对统计学家来说，simulation 不是附加实验，而是开发过程本身。
- 我们不是先问代码够不够快，而是先问代码能不能被信任。
- 大数据会放大计算成本，但小数据更容易暴露逻辑错误。
- 理论上漂亮的算法，在有限样本、随机初始化、浮点数和内存限制下，可能会表现得很不漂亮。
- 最快的代码如果无法修改、无法调试、无法验证，就不适合科学计算。
- 统计计算需要的是 clear enough to trust, fast enough to scale。
- AI coding tools 最适合做翻译器：把清楚的数学代码翻译成高性能 kernel，但不能替代统计判断。
- 不要只 benchmark runtime，也要 benchmark trust。

### 19.2 英文句子候选

- Simulation is not just how we evaluate statistical code; it is how we design it.
- The first question is not “How fast is this?” The first question is “Can I trust this?”
- In statistical computing, the reference implementation is part of the scientific argument.
- A fast implementation that changes the statistic is not an optimization. It is a different method.
- The goal is not to make every line of Python fast. The goal is to keep flexibility where we need it and accelerate the patterns that simulation reveals as expensive.
- For statisticians, the best tool is rarely the one with the lowest runtime alone. It is the one that balances correctness, debuggability, reproducibility, memory, speed, and cost of change.
- AI coding tools are useful when they translate clear mathematical intent into faster kernels, but every translation needs a statistical back-translation check.

---

## 20. README / Skill 后续迭代建议

如果要把这份文档真正做成一个可上传的 Skill，建议下一步补充三个具体信息。

### 20.1 典型输入

需要明确用户最常见的输入形式，例如：

- “我有一个 permutation test，帮我写 simulation”；
- “我有一段 NumPy code，帮我改成 Numba 并验证”；
- “我想比较 Python 3.14、Numba、JAX 的 performance”；
- “我想为 k-means 设计 simulation scenarios”；
- “我有 benchmark 结果，帮我解释 trade-offs”。

### 20.2 典型输出

需要明确 Skill 默认输出：

- 只给 plan？
- 直接写代码？
- 输出完整 repo skeleton？
- 输出 benchmark report？
- 输出 optimized code + tests？
- 是否生成 plots？

### 20.3 默认代码风格

建议定义：

- Python 版本；
- 是否默认使用 `numpy.random.Generator`；
- 是否默认使用 `pytest`；
- benchmark 是否默认使用 `time.perf_counter()`；
- 是否使用 `memory_profiler` 或 `tracemalloc`；
- Numba/JAX 是否作为 optional；
- 文件结构，例如 `src/`, `tests/`, `benchmarks/`, `reports/`。

---

## 21. 一个推荐的 simulation repo skeleton

如果让 Codex 生成项目结构，可以使用：

```text
simulation_project/
├── README.md
├── pyproject.toml
├── src/
│   ├── data_generators.py
│   ├── reference.py
│   ├── optimized_numpy.py
│   ├── optimized_numba.py
│   ├── optimized_jax.py
│   └── metrics.py
├── tests/
│   ├── test_sanity.py
│   ├── test_reference_vs_optimized.py
│   └── test_edge_cases.py
├── benchmarks/
│   ├── benchmark_kmeans.py
│   ├── benchmark_permutation.py
│   └── benchmark_utils.py
├── simulations/
│   ├── run_kmeans_scenarios.py
│   └── run_permutation_scenarios.py
├── reports/
│   └── simulation_report.md
└── results/
    └── .gitkeep
```

这个结构体现了前面说的原则：reference、optimized、tests、benchmarks、reports 分开，避免把探索性代码和最终性能代码混在一起。

---

## 22. 对这场 talk 的最终定位

这场 talk 可以被包装成：

> A workflow talk disguised as a performance talk.

表面上是在讲 Python 3.14、Numba、JAX 和 AI coding tools；更深层是在讲统计学家如何在真实科研代码中平衡：

- mathematical clarity；
- simulation-based trust；
- developer experience；
- performance engineering；
- reproducibility；
- AI-assisted optimization。

这会让你的身份优势非常突出：你不是站在系统工程师角度说“Python 怎样变快”，而是站在生物统计学家角度说“我们为什么需要一种既能快速迭代又能逐步加速的 Python workflow”。

---

## 23. 后续可扩展内容

后续可以继续补充：

1. k-means 的完整 reference / NumPy / Numba / JAX 代码；
2. permutation test 的完整 benchmark harness；
3. Python 3.14 free-threaded build 的实验记录模板；
4. Python 3.14 JIT 的 warm-up / steady-state benchmark 模板；
5. 一份真正可打包上传的 `simulation-stat-copilot` Skill；
6. 演讲版英文 opening script；
7. slide outline；
8. demo notebook 结构。

---

## 24. 最简版 takeaway

如果只能保留一句话：

> 统计学家需要的不是“最快的 Python”，而是一个 simulation-driven workflow：先用清楚代码建立信任，再用现代 Python 工具加速 simulation 证明真正昂贵的部分。

