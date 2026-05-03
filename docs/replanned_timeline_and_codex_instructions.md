# PyCon US Talk Replan: Simulation-driven Statistical Computing + GPU Experiments

> 目标：把这场 talk 从“比较 Python 3.14 / Numba / JAX 哪个快”，重新组织成“统计学家如何用 simulation 设计可信代码，再选择合适的现代 Python 性能工具”。
>
> 当前 slides 已经有这个方向，但 introduction 仍然偏短，实验也偏像两个单点 benchmark。下一版应让 simulation 的核心叙事成为主线，并让 GPU 环境服务于“实验设计更丰富、统计验证更完整、性能 trade-off 更真实”。

---

## 1. 当前版本的问题诊断

### 1.1 叙事方向是对的，但时间权重不够

当前 deck 已经把 talk 定义为：

- 不是 tool ranking；
- 而是 simulation workflow；
- 目标是 “Clear enough to trust. Fast enough to scale.”

这是很好的核心句。但 timeline 里 Introduction 只有 3 分钟，然后很快进入 k-means 12 分钟和 permutation test 10 分钟。这样听众会更容易把它理解成“工具 benchmark talk”，而不是“统计计算 workflow talk”。

建议把 introduction 扩展到 7–8 分钟，明确告诉程序员：统计 simulation 不是 demo data；它是一种 specification discovery、correctness testing、failure hunting 和 performance scaling 的方法。

### 1.2 实验设置目前偏简单

当前实验主要是：

- k-means: 比较 N 扩大时 NumPy / Numba / JAX 的 runtime；
- permutation test: 比较 R 扩大时 NumPy loop / threads / multiprocessing / Numba / JAX CPU。

这些结果有用，但还不够体现“simulation-driven statistical computing”。下一版应该加入：

1. **statistical behavior**：算法有没有恢复真实结构？p-value 在 null 下是否 calibration 正确？power 是否随 effect size 增加？
2. **scenario grid**：不只改变 N 或 R，还改变 signal、noise、dimension、imbalance、feature count、memory pressure。
3. **failure mode**：展示一个理论上合理、代码也能跑，但 simulation 发现它在某类场景下有问题的例子。
4. **iteration story**：从 reference implementation 到 bad cases，再到优化版本，最后回到验证。
5. **GPU 的合理定位**：GPU 不只是让 JAX 看起来快，而是展示“当统计问题可以改写成 batched array program / matrix program 时，JAX + GPU 才真正有意义”。

### 1.3 JAX/GPU 要避免错误叙事

当前 slides 里 JAX permutation on CPU 是反例，说明 `vmap` over random permutations 不一定好。GPU 环境来了以后，不应该简单地把同样的 `jax.random.permutation` vmap 搬到 GPU。更好的故事是：

> 先用统计代数重写问题，再让 GPU 做适合它的工作。

例如多特征 permutation test 可以从 “for each permutation, shuffle labels, compute many test statistics” 改写成：

```text
X: n_samples x p_features
W: B_permutations x n_samples   # each row is one permuted contrast/weight vector
T_null = W @ X                  # B x p null statistics
```

这让 permutation test 变成大矩阵乘法 / batch matrix computation，才是 GPU/JAX 的主场。

---

## 2. 新的 30 分钟 timeline

### 总体结构

| Time | Section | Core message | Slides |
|---:|---|---|---:|
| 0:00–0:45 | Title + thesis | This is not a speed contest. It is a workflow for trustworthy fast statistical code. | 1 |
| 0:45–7:30 | Introduction: simulation-driven statistical computing | Simulation is how statisticians turn vague scientific questions into testable software specifications. | 5–6 |
| 7:30–9:00 | Benchmark contract + experiment ladder | We measure correctness, statistical behavior, runtime, memory, and developer effort under one reproducible contract. | 1–2 |
| 9:00–16:00 | Workload 1: k-means as iterative simulation pressure | Iterative algorithms expose convergence, initialization, memory temporaries, and CPU/GPU break-even behavior. | 4–5 |
| 16:00–25:00 | Workload 2: high-dimensional permutation test as resampling pressure | Resampling becomes GPU-friendly only after we rewrite the statistic into batched linear algebra. | 5–6 |
| 25:00–28:30 | Developer experience + AI/Codex workflow | AI should help generate variants and experiments, but the human owns the statistical contract. | 2 |
| 28:30–30:00 | Decision guide + conclusion | Choose the smallest tool that preserves the statistic and removes the proven bottleneck. | 1–2 |

### Detailed timeline

#### 0:00–0:45 — Title and promise

**Slide:** Breaking the Speed Limit

Speaker message:

- “I am a biostatistician. I use Python, I understand my statistical model, but I do not want to become a full-time C++/Rust performance engineer.”
- “This talk is about a workflow: make the code clear enough to trust, then fast enough to scale.”

Keep the hero metrics, but consider replacing current metrics with neutral placeholders until GPU results are regenerated:

- “10x–100x speedups are possible, but only after we preserve the statistic.”
- “GPU wins only when the workload shape fits the accelerator.”

#### 0:45–2:30 — What simulation means to a statistician

**Slide:** Simulation is not fake data

Content:

- Synthetic data is not a toy; it is controlled reality.
- In real biomedical data, ground truth is often unknown.
- Simulation lets us know:
  - true clusters;
  - true null distribution;
  - true effect size;
  - expected failure modes;
  - whether an optimization changed the statistic.

Key sentence:

> For statisticians, simulation is how we write tests when real data does not come with an answer key.

#### 2:30–4:30 — Simulation workflow as software testing

**Slide:** Simulation-driven statistical computing loop

Use a stronger 6-step version:

1. **Define the estimand/statistic** — What quantity are we trying to compute?
2. **Build a reference implementation** — Slow, readable, close to math.
3. **Generate controlled scenarios** — Null, alternative, easy case, hard case, pathological case.
4. **Validate behavior** — Equivalence, calibration, power, convergence, invariants.
5. **Scale the workload** — Increase N, p, d, K, R, seeds, memory pressure.
6. **Optimize the proven bottleneck** — Numba, threads/no-GIL, JAX/GPU, or algorithmic rewrite.

Speaker analogy for programmers:

- unit tests = small exact cases;
- property tests = invariants under transformations;
- load tests = scale N/R/p;
- fuzz tests = bad seeds, outliers, imbalance;
- golden tests = reference implementation.

#### 4:30–6:00 — Why “fast but changed” is not optimization

**Slide:** Three kinds of failure

| Failure type | Example | Why simulation catches it |
|---|---|---|
| Implementation failure | Wrong labels after optimizing k-means distance computation | Reference equivalence test fails |
| Statistical failure | Permutation p-values are not uniform under null | Null calibration plot fails |
| Systems failure | Multiprocessing copies 8 large matrices | Memory scaling plot exposes it |

Key sentence:

> A faster implementation that changes the statistic is not an optimization. It is a different method.

#### 6:00–7:30 — Tool choice follows compute pattern

**Slide:** From statistical pattern to Python tool

| Pattern | Statistical example | Good first tool | When to move on |
|---|---|---|---|
| Small exact reference | Tiny k-means, exact permutation | Plain Python / NumPy | Never delete it; use it for tests |
| Scalar numerical loop | Assignment/update loop | Numba | CPU loop hotspot confirmed |
| Repeated independent work | Permutations, bootstrap | threads/no-GIL or Numba prange | shared data matters |
| Batched array program | many seeds/scenarios/features | JAX | GPU or accelerator likely helps |
| Large matrix algebra | multi-feature permutation | JAX/NumPy BLAS/GPU | rewrite statistic as matmul |

#### 7:30–9:00 — Benchmark contract

**Slide:** Measurement contract

Required details:

- environment JSON: CPU, GPU, RAM, Python, NumPy, Numba, JAX, CUDA, driver;
- correctness before performance;
- cold vs warm timings;
- `block_until_ready()` for JAX;
- device transfer included/excluded as separate rows;
- memory measured separately for host and GPU;
- each implementation in a fresh subprocess when feasible;
- no reported speedup without matching correctness check.

---

## 3. Revised experiment plan

## 3.1 Workload 1: k-means as iterative simulation pressure

### Purpose

Use k-means to demonstrate that iterative statistical algorithms are not just about runtime. We also care about convergence, initialization sensitivity, cluster recovery, memory behavior, and whether optimizations preserve the algorithm.

### Simulation design

Generate Gaussian mixture data with known labels.

Parameters:

| Parameter | Values | Why it matters |
|---|---|---|
| N | 10k, 100k, 1M, maybe 5M if GPU memory allows | scale |
| d | 10, 64, 256 | dimension pressure |
| K | 5, 20, 50 | centroid count |
| separation | 0.5, 1.0, 2.0, 4.0 | easy vs hard clustering |
| imbalance | balanced, 90/10, long-tail | empty/small clusters |
| covariance | spherical, anisotropic, correlated | distance geometry |
| outliers | 0%, 1%, 5% | robustness/failure mode |
| seeds | 5–20 | initialization sensitivity |

### Implementations to compare

1. `kmeans_reference.py`
   - small data only;
   - clear Python/NumPy implementation;
   - used for correctness tests.

2. `kmeans_numpy_broadcast.py`
   - readable vectorized version;
   - intentionally shows `O(N*K*d)` temporary.

3. `kmeans_numpy_matmul.py`
   - uses distance identity:
     `||x-c||^2 = ||x||^2 + ||c||^2 - 2 x @ c.T`;
   - better for large K/d.

4. `kmeans_numba.py`
   - scalar loops compiled with `@njit`;
   - possibly `parallel=True` for assignment step;
   - CPU-oriented.

5. `kmeans_jax_cpu_gpu.py`
   - `jax.jit` + `jax.lax.scan` for fixed max iterations;
   - run on CPU and GPU;
   - optional `vmap` over seeds or scenarios;
   - use float32 by default, optionally compare float64.

### Correctness/statistical validation

For each implementation:

- same generated data;
- same initial centroids;
- compare final inertia;
- compare iteration count if convergence rule is identical;
- compare cluster recovery using ARI against true labels;
- detect empty clusters;
- track run-to-run variability across seeds.

### New figures for slides

1. **kmeans_simulation_grid.png**
   - heatmap: separation x dimension or separation x imbalance;
   - color = ARI or failure rate;
   - purpose: show simulation reveals hard cases before optimization.

2. **kmeans_runtime_cpu_gpu.png**
   - runtime vs N/d/K;
   - implementations: NumPy broadcast, NumPy matmul, Numba CPU, JAX CPU, JAX GPU;
   - separate cold and warm JAX if possible.

3. **kmeans_memory_scaling.png**
   - memory vs N*K*d;
   - show broadcast temporary explosion.

4. **kmeans_gpu_break_even.png**
   - x-axis: problem size;
   - y-axis: runtime;
   - highlight where GPU overhead is worth it.

### Slide narrative

Do not present k-means as “Numba wins” or “JAX wins”. Present it as:

1. reference implementation defines meaning;
2. simulation grid finds hard cases;
3. small shape favors low-overhead CPU tools;
4. large batched shape can favor JAX/GPU;
5. vectorization can shift bottleneck from Python to memory.

---

## 3.2 Workload 2: high-dimensional permutation test as resampling pressure

### Purpose

Upgrade permutation test from a single statistic to a more realistic biostatistics workload: many features/genes, many permutations, null calibration, power, and GPU-friendly batching.

### Statistical setup

Generate a gene-expression-like matrix:

```text
X: n_samples x p_features
labels: two groups, n1 and n2
B: number of permutations
```

Scenarios:

| Parameter | Values | Why it matters |
|---|---|---|
| n | 100, 1k, 10k | sample scale |
| p | 100, 1k, 10k, 20k | feature/gene scale |
| B | 1k, 10k, 100k | permutation scale |
| effect size | 0, 0.2, 0.5, 1.0 | null vs power |
| affected features | 0%, 1%, 5% | sparse signal |
| group balance | 50/50, 80/20 | imbalance |
| correlation | independent, block-correlated | gene-expression realism |

### Core algebraic rewrite

For a two-sample mean difference, encode each permutation as a contrast vector:

```text
w_b[i] = +1/n1 if sample i is assigned to group A
w_b[i] = -1/n2 if sample i is assigned to group B
```

Then for all features:

```text
T_null = W @ X
```

where:

```text
W: B x n
X: n x p
T_null: B x p
```

This is the key GPU story: not “JAX makes Python loops fast”, but “simulation plus statistical algebra transforms resampling into batched linear algebra”.

### Implementations to compare

1. `perm_reference_exact.py`
   - exact enumeration for tiny n, or simple loop for small n;
   - validates p-values and statistic formula.

2. `perm_numpy_loop.py`
   - straightforward loop over permutations;
   - small p or small B only.

3. `perm_numpy_batched.py`
   - construct W in chunks and use NumPy matmul;
   - CPU BLAS baseline.

4. `perm_numba_prange.py`
   - CPU compiled loop;
   - useful for single/few-feature test or when W materialization is expensive.

5. `perm_threads_free_threaded.py`
   - optional if Python 3.14t is available;
   - compare shared input array vs multiprocessing copies.

6. `perm_jax_gpu.py`
   - JAX `jit` batched/chunked matrix multiplication;
   - run on GPU;
   - do not use naive `vmap(jax.random.permutation)` as the main GPU path;
   - may include it as an anti-pattern only if time permits.

### Correctness/statistical validation

1. **Exact small-n validation**
   - for n <= 12, enumerate all label assignments;
   - compare Monte Carlo estimates to exact p-values.

2. **Null calibration**
   - under effect size = 0, p-values should be approximately uniform;
   - produce QQ plot or histogram;
   - run KS statistic as a diagnostic.

3. **Power curve**
   - under increasing effect size, power should increase;
   - show power vs effect size.

4. **Multiple-feature behavior**
   - show null p-values for unaffected features;
   - show detection rate for affected features;
   - optionally add Benjamini-Hochberg FDR as a small applied hook, but do not let it dominate the talk.

5. **RNG/reproducibility**
   - CPU: use `numpy.random.SeedSequence` per worker;
   - JAX: use explicit PRNG keys and key splitting;
   - record seeds in result metadata.

### New figures for slides

1. **perm_null_calibration.png**
   - histogram or QQ plot of p-values under null;
   - purpose: correctness before speed.

2. **perm_power_curve.png**
   - power vs effect size for selected n/p/B;
   - purpose: simulation shows statistical behavior.

3. **perm_runtime_cpu_gpu.png**
   - runtime vs B*p or matrix size;
   - implementations: NumPy loop, NumPy batched, Numba, JAX CPU, JAX GPU.

4. **perm_memory_chunking.png**
   - show W materialization memory;
   - compare full W vs chunked W;
   - purpose: GPU speed requires memory planning.

5. **perm_gpu_break_even.png**
   - small problems: CPU simpler/faster;
   - large B x p problems: GPU wins.

### Slide narrative

Permutation section should be told as an iteration:

1. naive loop is clear but slow;
2. reference version defines p-value behavior;
3. simulation checks null calibration;
4. algebra rewrites the statistic into `W @ X`;
5. chunking controls memory;
6. JAX/GPU becomes useful because the workload shape changed;
7. result: speedup is credible because statistical behavior was checked first.

---

## 4. What to ask Codex to do

Below is the copy-paste prompt for Codex.

```text
You are working on the repository for my PyCon US 2026 talk: “Breaking the Speed Limit: Fast statistical models with Python 3.14, Numba, and JAX.”

I want to rework the talk around the narrative “simulation-driven statistical computing”, not a simple tool ranking. The current deck already has a simulation-first direction, but the introduction is too short and the experiments are too simple. I will run this in a GPU environment, so please redesign the experiment suite and update the slides accordingly.

Primary goals:

1. Give more weight to the opening narrative: statisticians use simulation to design, validate, stress-test, and optimize code.
2. Make the experiments richer statistically, not just faster computationally.
3. Use GPU/JAX only where the workload shape justifies it.
4. Keep all claims honest: no speedup without a correctness/statistical validation check; no JIT/GPU claims unless actually measured in the environment.
5. Preserve the final message: “Clear enough to trust. Fast enough to scale.”

Please implement a v3 revision with the following structure.

A. Documentation

Create or update these docs:

- docs/simulation_driven_statistical_computing.md
- docs/benchmark_contract_v3.md
- docs/experiment_plan_v3.md
- docs/gpu_notes_v3.md

The docs should explain:

- Simulation is how statisticians write tests when real data has no answer key.
- The workflow is: define statistic/estimand -> reference implementation -> controlled scenarios -> validation -> scale-up -> optimization.
- Correctness includes implementation equivalence, statistical calibration, convergence behavior, memory behavior, reproducibility, and developer effort.
- Tool choice should follow computational pattern, not personal preference.

B. Environment and reproducibility

Add a script:

- experiments/common/capture_environment.py

It should record:

- Python version and executable
- OS and CPU info
- RAM if easily available
- NumPy version
- Numba version
- JAX and jaxlib versions
- JAX default backend
- list of `jax.devices()`
- CUDA/cuDNN info if available
- `nvidia-smi` output if available
- whether Python appears to be free-threaded: use `sys._is_gil_enabled()` if present and `sysconfig.get_config_var("Py_GIL_DISABLED")`
- whether CPython JIT appears available/enabled: use `sys._jit.is_available()` and `sys._jit.is_enabled()` if present

Write the output as JSON under:

- experiments/results/v3/environment.json

C. Benchmark contract

For every benchmark:

- Run correctness/statistical checks before timing.
- Separate cold first-call time from warm execution time.
- For JAX, always call `.block_until_ready()` before stopping the timer.
- Report whether device transfer time is included.
- Record host memory and GPU memory where feasible.
- Store raw results as CSV/Parquet plus metadata JSON.
- Do not overwrite previous results; write under experiments/results/v3/.
- Each result row should include: workload, implementation, backend, device, n, p, d, k, B/R, seed, cold_time_s, warm_median_s, warm_iqr_s, host_peak_mem_mb, gpu_peak_mem_mb if available, validation_status, notes.

D. Workload 1: k-means simulation grid

Create or update:

- experiments/kmeans_v3/data_generation.py
- experiments/kmeans_v3/kmeans_reference.py
- experiments/kmeans_v3/kmeans_numpy_broadcast.py
- experiments/kmeans_v3/kmeans_numpy_matmul.py
- experiments/kmeans_v3/kmeans_numba.py
- experiments/kmeans_v3/kmeans_jax.py
- experiments/kmeans_v3/run_kmeans_v3.py
- experiments/kmeans_v3/validate_kmeans_v3.py

Simulation design:

Generate Gaussian mixture data with known labels. Include parameters:

- N: 10_000, 100_000, 1_000_000; optionally 5_000_000 if GPU memory allows
- d: 10, 64, 256
- K: 5, 20, 50
- separation: 0.5, 1.0, 2.0, 4.0
- imbalance: balanced, 90/10, long-tail
- covariance: spherical, anisotropic, correlated
- outlier fraction: 0, 0.01, 0.05
- seeds: at least 5 for statistical behavior plots; more if runtime allows

Validation:

- same generated data and same initial centroids across implementations
- compare final inertia
- compare cluster recovery using adjusted Rand index if sklearn is available; otherwise implement or skip with a clear note
- detect empty clusters
- record convergence iterations
- compare implementations on small and medium cases before large-scale timing

Implementations:

1. Reference implementation: readable, small data only.
2. NumPy broadcast: clear but memory-heavy; show O(N*K*d) temporary.
3. NumPy matmul identity: `||x-c||^2 = ||x||^2 + ||c||^2 - 2*x@c.T`.
4. Numba CPU implementation with `@njit`, optionally `parallel=True` for assignment.
5. JAX implementation using `jax.jit` and `jax.lax.scan`; run on CPU and GPU if available. Optional: use `vmap` over seeds or scenarios only if it makes sense and does not hide memory problems.

Figures to generate:

- experiments/results/v3/kmeans_simulation_grid.png
  Heatmap of statistical behavior, e.g. ARI or failure rate across separation and dimension/imbalance.

- experiments/results/v3/kmeans_runtime_cpu_gpu.png
  Runtime vs problem size for NumPy broadcast, NumPy matmul, Numba CPU, JAX CPU, JAX GPU.

- experiments/results/v3/kmeans_memory_scaling.png
  Peak memory vs N*K*d, highlighting broadcast temporaries.

- experiments/results/v3/kmeans_gpu_break_even.png
  Cold and warm runtime showing where JAX/GPU becomes worth the compilation and transfer overhead.

E. Workload 2: high-dimensional permutation test

Create a new richer workload:

- experiments/permutation_v3/data_generation.py
- experiments/permutation_v3/perm_reference_exact.py
- experiments/permutation_v3/perm_numpy_loop.py
- experiments/permutation_v3/perm_numpy_batched.py
- experiments/permutation_v3/perm_numba_prange.py
- experiments/permutation_v3/perm_threads_free_threaded.py
- experiments/permutation_v3/perm_jax_gpu.py
- experiments/permutation_v3/run_permutation_v3.py
- experiments/permutation_v3/validate_permutation_v3.py

Statistical setup:

Generate a gene-expression-like matrix:

- X: n_samples x p_features
- labels: two groups
- B: number of permutations
- effect_size: 0, 0.2, 0.5, 1.0
- affected feature fraction: 0, 0.01, 0.05
- optional block correlation among features

Parameter grid:

- n: 100, 1_000, 10_000
- p: 100, 1_000, 10_000, 20_000 if memory allows
- B: 1_000, 10_000, 100_000 if runtime/memory allows
- group balance: 50/50, 80/20
- seeds: at least 5 for statistical plots

Core algebraic rewrite:

Represent each permutation as a contrast vector w:

- +1/n1 for samples assigned to group A
- -1/n2 for samples assigned to group B

Then all null statistics can be computed by:

- T_null = W @ X

where W is B x n and X is n x p. This is the main GPU-friendly formulation. Generate W in chunks to avoid materializing an impossible B x n matrix when B and n are large.

Important: Do not make naive `vmap(jax.random.permutation)` the main JAX/GPU implementation. It can be included as an anti-pattern, but the main GPU implementation should use the contrast-matrix / chunked matmul approach.

Validation:

1. Exact small-n validation:
   - for tiny n, enumerate all label assignments where possible;
   - compare reference exact p-values to Monte Carlo p-values.

2. Null calibration:
   - under effect_size = 0, p-values should be approximately uniform;
   - produce histogram/QQ plot and a simple KS diagnostic.

3. Power curve:
   - under increasing effect sizes, detection power should increase.

4. Multi-feature behavior:
   - unaffected features should have calibrated null p-values;
   - affected features should show increased detection.

5. RNG/reproducibility:
   - NumPy/CPU workers should use `numpy.random.SeedSequence` or deterministic independent streams;
   - JAX should use explicit PRNG keys and key splitting;
   - record all base seeds.

Implementations:

1. Reference exact / reference loop for small cases.
2. NumPy loop baseline.
3. NumPy batched matrix multiplication with chunked W.
4. Numba prange CPU implementation for loop-style permutation.
5. Optional free-threaded Python ThreadPoolExecutor implementation if Python 3.14t/no-GIL is available.
6. JAX GPU chunked matrix multiplication implementation.

Figures to generate:

- experiments/results/v3/perm_null_calibration.png
  Histogram or QQ plot of null p-values.

- experiments/results/v3/perm_power_curve.png
  Power vs effect size.

- experiments/results/v3/perm_runtime_cpu_gpu.png
  Runtime vs B*p or problem scale across NumPy loop, NumPy batched, Numba, JAX CPU, JAX GPU.

- experiments/results/v3/perm_memory_chunking.png
  Full W vs chunked W memory model.

- experiments/results/v3/perm_gpu_break_even.png
  CPU vs GPU cold/warm break-even.

F. Slides update

Update index.html or create index_v3.html.

New 30-minute flow:

1. Title and thesis — 0:45
2. Simulation is not fake data — 1:45
3. Simulation-driven statistical computing loop — 2:00
4. Three kinds of failure: implementation/statistical/systems — 1:30
5. Tool choice follows compute pattern — 1:30
6. Benchmark contract — 1:30
7. k-means setup: controlled mixture simulation — 1:30
8. k-means statistical behavior / failure heatmap — 1:30
9. k-means implementation trade-offs — 1:30
10. k-means CPU/GPU runtime and memory — 2:30
11. k-means takeaway — 0:30
12. permutation setup: high-dimensional biostatistics workload — 1:30
13. correctness first: null calibration and exact check — 1:30
14. algebraic rewrite: permutations as W @ X — 2:00
15. permutation CPU/GPU runtime and memory — 2:30
16. permutation power curve / statistical behavior — 1:00
17. developer experience: AI as copilot, not driver — 2:00
18. decision guide — 1:30
19. conclusion — 1:00

Use speaker notes heavily. The slides should not contain too much text, but the notes should explain the statistical reasoning and why each benchmark exists.

G. Required cautionary language

Use language like:

- “This is not a universal benchmark. It is a reproducible local experiment under a stated contract.”
- “A fast implementation that changes the statistic is not an optimization.”
- “JAX/GPU helps when the workload is expressed as a large batched array program, not when we simply wrap a Python loop in vmap.”
- “Free-threaded Python removes one ceiling, but package support and shared-state design still matter.”
- “The CPython JIT should be reported only if `sys._jit.is_available()` and `sys._jit.is_enabled()` confirm it in this environment.”

H. Deliverables

At the end, provide:

1. Updated docs.
2. New experiment scripts.
3. Raw result CSV/Parquet files under experiments/results/v3/.
4. Environment JSON.
5. Figures under experiments/results/v3/.
6. Updated reveal.js deck: index_v3.html.
7. A short summary file: experiments/results/v3/summary.md, containing:
   - what was measured;
   - what passed validation;
   - what failed or was skipped;
   - which speedups are actually measured;
   - which claims should be avoided.

I. Priorities if time is limited

Priority 1:

- Environment capture
- k-means small/medium correctness checks
- permutation exact small-n check
- permutation null calibration
- JAX timing with block_until_ready
- updated introduction slides

Priority 2:

- GPU k-means runtime
- GPU permutation matrix formulation
- runtime/memory figures

Priority 3:

- full scenario grid
- power curves
- ARI phase diagram
- free-threaded Python comparison
- CPython JIT comparison

Do not fabricate missing results. If a tool or backend is unavailable, record it as unavailable and keep the claim conceptual.
```

---

## 5. Updated slide outline with suggested slide titles

### Slide 1 — Breaking the Speed Limit

Subtitle:

> Fast statistical models with Python 3.14, Numba, and JAX — told as a simulation workflow, not a tool ranking.

### Slide 2 — Simulation is how statisticians write tests

Main point:

> Real data rarely comes with an answer key. Simulation gives us a controlled world where the answer is known.

### Slide 3 — Simulation-driven statistical computing loop

Six stages:

1. Define statistic
2. Reference implementation
3. Controlled scenarios
4. Validate behavior
5. Scale workload
6. Optimize bottleneck

### Slide 4 — Failure is not always a crash

Three panels:

- implementation failure;
- statistical failure;
- systems failure.

### Slide 5 — Tool choice follows compute pattern

Matrix: pattern → example → tool → risk.

### Slide 6 — Benchmark contract

Correctness, calibration, warm/cold timing, memory, GPU sync, reproducibility.

### Slide 7 — Workload 1: k-means as iterative simulation pressure

Introduce mixture simulation and known labels.

### Slide 8 — First ask: does the algorithm behave?

Show ARI/failure heatmap across separation/imbalance/dimension.

### Slide 9 — Then ask: where is the bottleneck?

Show implementations and memory shape: broadcast vs matmul vs compiled loop vs JAX scan.

### Slide 10 — k-means CPU/GPU results

Runtime and memory plots. Include cold/warm distinction.

### Slide 11 — k-means takeaway

> Iterative algorithms reward the smallest optimization that preserves the convergence story.

### Slide 12 — Workload 2: permutation tests as resampling pressure

Move from single statistic to many-feature biomedical setting.

### Slide 13 — Correctness first: null calibration

Show p-values under null are uniform; exact small-n validation.

### Slide 14 — Statistical algebra before GPU

Show `T_null = W @ X`.

### Slide 15 — GPU-friendly resampling

Show chunked W, X on device, block_until_ready, memory planning.

### Slide 16 — Runtime, memory, and power

Combine runtime scaling and power curve.

### Slide 17 — AI/Codex as simulation copilot

Codex helps generate variants, scenario grids, plotting scripts, and refactors. Human owns the statistical contract.

### Slide 18 — Decision guide

Start with reference; then select NumPy/Numba/threads/JAX based on proven bottleneck.

### Slide 19 — Conclusion

> Clear enough to trust. Fast enough to scale.

---

## 6. What should be removed or de-emphasized

1. Do not spend 12 minutes only on k-means runtime tables.
2. Do not let Python 3.14 JIT dominate unless it is actually available and measured.
3. Do not make JAX CPU permutation anti-pattern the main JAX story after GPU becomes available.
4. Do not show too many raw timing cells; prefer pattern plots and one key table.
5. Do not claim GPU speedups without:
   - validation;
   - cold/warm distinction;
   - `block_until_ready()`;
   - memory/chunking explanation.

---

## 7. Short version for yourself

The talk should now feel like this:

1. **I am a statistician; I start with simulation because I need trustworthy behavior before speed.**
2. **Simulation gives me a reference, hard cases, validation criteria, and bottleneck evidence.**
3. **k-means shows iterative pressure: convergence, memory temporaries, CPU compiled loops, and GPU break-even.**
4. **Permutation tests show resampling pressure: null calibration, power, RNG, memory copies, and GPU-friendly algebra.**
5. **Python 3.14, Numba, and JAX are not competitors in a race; they are tools for different parts of the simulation loop.**
6. **Codex/AI helps generate variants, but the statistical contract remains human-owned.**
