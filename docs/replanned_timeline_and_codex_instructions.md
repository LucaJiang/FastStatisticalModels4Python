# Replanned Timeline and Codex Instructions

**Talk:** Breaking the Speed Limit: Fast statistical models with Python 3.14, Numba, and JAX  
**Audience:** PyCon US programmers, data scientists, scientific Python users  
**Speaker perspective:** biostatistician/statistician who writes Python, wants trustworthy code first and scalable code second  
**Working mode:** run local MacBook Air experiments and remote Linux server/A100 experiments in parallel  
**Document purpose:** give Codex concrete instructions for restructuring the repository, rebuilding experiments, regenerating figures, and revising the 30-minute talk around simulation-driven statistical computing.

---

## 0. One-paragraph direction for Codex

Refactor the talk and experiments around the following thesis:

> Statistical simulation is not fake data or a toy demo. It is how statisticians write tests when real data does not come with an answer key. We begin on a laptop with small, controlled, inspectable simulations; use those simulations to define the statistical contract; then scale the same contract on a server CPU and, only when the computation has the right shape, on an A100 GPU.

Do **not** turn the talk into a hardware leaderboard. The MacBook Air, server CPU, and A100 GPU should play different narrative roles:

- **MacBook Air:** correctness, debugging, reference implementation, smoke tests, developer experience.
- **Linux server CPU:** scaling, parallelism, memory behavior, process/thread/Numba trade-offs.
- **Linux server A100:** accelerator story, but only after the statistic is rewritten into a GPU-friendly batched array/matrix program.

The final talk should say:

> Small controlled simulation on laptop → validated statistical contract → scale on server CPU → rewrite the expensive pattern for A100 GPU.

---

## 1. Diagnosis of the current deck and timeline

The current deck already contains a strong seed of the right story:

- The title slide says the talk is “told as a simulation workflow rather than a tool ranking.”
- The current opening says statistical code starts with trust, not speed.
- The current deck already uses two workloads: k-means for iterative pressure and permutation tests for resampling pressure.
- The current deck already warns that “a fast implementation that changes the statistic is not an optimization.”

However, the current timeline underweights the core simulation story. It allocates only **3 minutes** to introduction, then moves quickly into **12 minutes of k-means** and **10 minutes of permutation tests**. That makes the talk still feel like a benchmark comparison. The revised version should spend more time upfront explaining how statisticians use simulation to design, test, break, and repair code.

The current experiments also need richer statistical behavior. They are good first benchmarks, but the next version should add:

1. **Scenario grids** instead of single scaling axes.
2. **Statistical validation metrics** instead of only runtime.
3. **Failure cases** that demonstrate why simulation matters.
4. **Environment-tiered results** instead of one local CPU result table.
5. **GPU-friendly reformulation** instead of simply moving the same CPU-style code to JAX/GPU.

---

## 2. Core narrative: simulation-driven statistical computing

### 2.1 Main message

For statisticians, simulation is a way to make software testable.

In many statistical problems, real data does not provide a known answer. A patient dataset does not tell us the true clustering. A p-value does not come with a runtime exception when the null distribution was generated incorrectly. A model can converge numerically while failing statistically.

Simulation solves this by letting us create controlled worlds where we know something important:

- the true clusters;
- the true null hypothesis;
- the true effect size;
- the intended invariants;
- the hard cases;
- the expected behavior as sample size, dimension, noise, or repetitions increase.

### 2.2 Programmer analogy

Explain simulation to programmers using software-testing language:

| Programming concept | Statistical simulation equivalent                                      |
| ------------------- | ---------------------------------------------------------------------- |
| Unit test           | Tiny dataset with exact known answer                                   |
| Golden test         | Slow reference implementation used as oracle                           |
| Property test       | Invariants under permutation, translation, scaling, or relabeling      |
| Fuzz test           | Bad seeds, outliers, imbalance, high dimension, edge cases             |
| Load test           | Increase N, d, p, K, R, seeds, workers, memory pressure                |
| Regression test     | Store scenario results and ensure optimized versions preserve behavior |

### 2.3 Six-step simulation loop

Use this as the central mental model throughout the talk:

1. **Define the statistical target**
   - What statistic, estimator, null distribution, or algorithmic behavior are we computing?
   - What does it mean for the result to be correct?

2. **Write the reference implementation**
   - Slow is acceptable.
   - Readable is mandatory.
   - Keep it close to the math.
   - This code becomes the oracle for optimized variants.

3. **Generate controlled scenarios**
   - Easy cases where the answer should be obvious.
   - Null cases where calibration should hold.
   - Alternative cases where power/recovery should improve.
   - Pathological cases where the method may break.

4. **Validate statistical behavior**
   - Equivalence to reference implementation.
   - Calibration under the null.
   - Power under alternatives.
   - Recovery of known clusters.
   - Numerical stability.
   - Invariants.

5. **Scale the workload**
   - Increase sample size, dimension, number of clusters, number of features, number of permutations, and number of seeds.
   - Measure runtime, memory, cold-start cost, compilation cost, and developer friction.

6. **Optimize the proven bottleneck**
   - Use the smallest tool that removes the bottleneck without changing the statistic.
   - Candidate tools: better NumPy formulation, Numba, threads/free-threaded Python, multiprocessing, JAX CPU, JAX GPU.

### 2.4 Key sentence for slides

Use this line more than once:

> A faster implementation that changes the statistic is not an optimization. It is a different method.

---

## 3. Revised 30-minute timeline

### 3.1 Summary table

|        Time | Section                                               | Core purpose                                                                       | Target slide count |
| ----------: | ----------------------------------------------------- | ---------------------------------------------------------------------------------- | -----------------: |
|   0:00–0:45 | Title + thesis                                        | Establish that this is a workflow talk, not a speed contest.                       |                  1 |
|   0:45–7:30 | Introduction: simulation-driven statistical computing | Give programmers the statistical testing mindset.                                  |                5–6 |
|   7:30–9:00 | Benchmark contract + environment ladder               | Explain correctness-first measurement and environment tiers.                       |                1–2 |
|  9:00–16:00 | Workload 1: k-means                                   | Iterative simulation pressure: convergence, bad cases, memory, CPU/GPU break-even. |                4–5 |
| 16:00–25:00 | Workload 2: high-dimensional permutation test         | Resampling pressure: calibration, parallelism, algebraic rewrite, A100 GPU.        |                5–6 |
| 25:00–28:30 | Developer experience + AI/Codex                       | Show how AI helps generate variants, not define truth.                             |                  2 |
| 28:30–30:00 | Decision guide + close                                | Choose the smallest tool that preserves the statistic and removes the bottleneck.  |                1–2 |

### 3.2 Detailed timeline

#### 0:00–0:45 — Title and promise

**Slide:** Breaking the Speed Limit  
**Goal:** Set expectations.

Speaker notes:

- “I am a biostatistician. I use Python, I understand the statistical model, but I do not want to become a full-time C++ or Rust performance engineer.”
- “This talk is about making statistical code clear enough to trust and fast enough to scale.”
- “The real story is not which tool wins. The real story is how we decide what is safe to optimize.”

Avoid putting final speedup metrics on the title slide until the new experiments are done. Use placeholders such as:

- “Laptop validation → server scale → A100 acceleration.”
- “Speedups only count after the statistic is preserved.”

#### 0:45–2:15 — Simulation is not fake data

**Slide:** Simulation is how statisticians write tests

Content:

- Real biomedical data rarely comes with known truth.
- Simulation creates a controlled world.
- Controlled data lets us ask: did the code compute the intended statistic?
- The point is not to pretend synthetic data is real; the point is to know what should happen.

Key line:

> For statisticians, simulation is how we write tests when real data does not come with an answer key.

#### 2:15–4:00 — The simulation loop

**Slide:** Simulation-driven statistical computing loop

Show six steps:

1. define target;
2. write reference;
3. generate scenarios;
4. validate behavior;
5. scale workload;
6. optimize bottleneck.

Speaker analogy:

- reference implementation = golden oracle;
- scenario grid = property tests + fuzz tests;
- large-scale sweeps = load tests;
- optimized kernels = production candidates.

#### 4:00–5:30 — Why programmers should care

**Slide:** Three kinds of failure

| Failure type           | Example                                                             | Simulation signal           |
| ---------------------- | ------------------------------------------------------------------- | --------------------------- |
| Implementation failure | Optimized k-means gives different inertia under same initialization | reference equivalence fails |
| Statistical failure    | Permutation p-values are not uniform under null                     | null calibration plot fails |
| Systems failure        | multiprocessing copies huge arrays into workers                     | memory scaling plot fails   |

Speaker note:

- A performance benchmark without a statistical contract can be misleading.
- A code path can be fast, deterministic, and wrong.

#### 5:30–7:30 — Environment ladder

**Slide:** Why laptop and server both matter

Use a ladder diagram:

1. **MacBook Air: small controlled simulation**
   - inspectable;
   - fast iteration;
   - reference implementation;
   - developer experience.

2. **Linux server CPU: scale and parallelism**
   - 512 cores;
   - process/thread/Numba scaling;
   - memory accounting;
   - worker sweeps.

3. **Linux server A100: accelerator only after reformulation**
   - GPU does not rescue every loop;
   - GPU helps when computation becomes large batched linear algebra;
   - use JAX GPU only for GPU-shaped workloads.

Key line:

> The laptop tells us what to trust. The server tells us what scales. The GPU tells us whether we found the right computational shape.

#### 7:30–9:00 — Benchmark contract

**Slide:** Measurement contract

Required measurement rules:

- Always record environment metadata.
- Always run correctness/statistical checks before reporting speed.
- Separate cold first call from warm median.
- For JAX, call `block_until_ready()` before stopping the timer.
- For GPU results, label host-to-device transfer separately from device compute when possible.
- For multiprocessing, measure parent and child RSS if possible.
- Do not mix MacBook, server CPU, and A100 results in one leaderboard unless the plot explicitly facets by environment.
- If Python 3.14 JIT, free-threaded build, CUDA, or A100 is unavailable, mark the row as `unavailable`, not as zero or missing.

#### 9:00–16:00 — Workload 1: k-means as iterative simulation pressure

**Narrative:** k-means is simple enough to explain, but rich enough to show convergence, initialization, memory temporaries, and CPU/GPU break-even.

Slide sequence:

1. **k-means as an iterative algorithm**
   - assign → update → repeat;
   - each iteration depends on previous centroids.

2. **Simulation grid reveals hard cases**
   - vary separation, dimension, cluster imbalance, covariance, outliers, seeds;
   - show ARI/failure heatmap, not just runtime.

3. **Reference/optimized split**
   - same data;
   - same initialization;
   - compare inertia, assignments, ARI, iterations, empty clusters.

4. **Runtime and memory scaling**
   - NumPy broadcast vs NumPy matmul vs Numba CPU vs JAX CPU/GPU;
   - show memory explosion from `O(N*K*d)` broadcast temporary.

5. **GPU break-even**
   - A100 does not automatically win small problems;
   - GPU becomes useful when N/K/d are large enough and the computation is regular.

#### 16:00–25:00 — Workload 2: high-dimensional permutation test

**Narrative:** permutation testing is a better biostatistics example when we make it high-dimensional, e.g. gene-expression-like `samples × features` matrix. The statistical rewrite is the star.

Slide sequence:

1. **From scalar permutation to high-dimensional permutation**
   - Real problem: many features/genes, same label permutation.
   - Need null distribution across features.

2. **Statistical algebra before runtime**
   - Naive: shuffle labels R times, recompute statistic feature by feature.
   - Better: encode permutations as contrast rows `W`, compute `T_null = W @ X`.
   - This turns resampling into batched matrix multiplication.

3. **Calibration first**
   - Under null, p-values should be approximately uniform.
   - Under alternatives, power should rise with effect size.
   - Compare optimized implementations to reference.

4. **CPU parallelism**
   - NumPy loop;
   - multiprocessing;
   - threads/free-threaded Python if available;
   - Numba `prange`;
   - memory and worker scaling.

5. **A100 GPU result**
   - JAX GPU matrix formulation;
   - batch size sweep;
   - GPU memory limits;
   - CPU-vs-GPU break-even.

6. **Anti-pattern warning**
   - Do not claim “JAX = fast” if using `vmap(random.permutation)` poorly.
   - The correct message is: JAX/GPU wins when simulation is reformulated into the right array program.

#### 25:00–28:30 — Developer experience + AI/Codex workflow

**Slide:** AI as copilot, not driver

Codex should help with:

- generating implementation variants;
- writing scenario-grid runners;
- creating benchmark harnesses;
- producing plots;
- checking environment metadata;
- adding reproducibility scripts;
- producing documentation.

The human statistician owns:

- definition of the statistic;
- validity of the null simulation;
- calibration criteria;
- interpretation of failure cases;
- decision about whether a speedup preserves the method.

#### 28:30–30:00 — Decision guide and close

**Final slide:** Clear enough to trust. Fast enough to scale.

Decision guide:

| If the bottleneck is...          | Prefer...                      | Why                                                  |
| -------------------------------- | ------------------------------ | ---------------------------------------------------- |
| understanding/correctness        | Python/NumPy reference         | debuggable and inspectable                           |
| scalar CPU loops                 | Numba                          | high ROI, low memory, familiar structure             |
| repeated work over shared arrays | threads/free-threaded or Numba | avoid process copies                                 |
| process-isolated workloads       | multiprocessing                | robust but memory-expensive                          |
| large batched array programs     | JAX/GPU                        | accelerator-friendly                                 |
| giant temporaries                | algebraic rewrite              | faster and more memory-stable than library switching |

Closing line:

> Choose the smallest tool that preserves the statistic and removes the proven bottleneck.

---

## 4. Environment split: what to run where

### 4.1 Environment roles

| Environment                 | Role                                         | What to optimize for                                               | What not to do                                                         |
| --------------------------- | -------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| MacBook Air                 | reference, smoke tests, developer experience | correctness, quick iteration, readable code                        | do not use it as final large-scale speed evidence                      |
| Linux server CPU, 512 cores | CPU scaling and parallelism                  | worker sweeps, memory accounting, Numba/process/thread comparisons | do not run unbounded 512-worker jobs without oversubscription controls |
| Linux server A100           | accelerator experiments                      | JAX/CUDA, batch size, device memory, matrix formulations           | do not simply move CPU-style random loops to GPU and call it success   |

### 4.2 Result labels

Every CSV row and every figure must include:

```text
environment_tier: macbook_air_validation | linux_server_cpu | linux_server_a100
machine_name: free text
python_executable: path
python_version: string
os: string
cpu_model: string
cpu_count_logical: int
ram_gb: float
gpu_name: string or none
gpu_memory_gb: float or none
jax_backend: cpu | gpu | unavailable
implementation: string
workload: kmeans | permutation_scalar | permutation_matrix
scenario_id: string
seed: int
correctness_status: pass | fail | unavailable
```

Do not create plots where results from different hardware appear as if they are directly comparable without a facet or explicit label.

---

## 5. Repository structure to create or update

Codex should reorganize experiments around environment tiers and reproducible outputs.

Recommended structure:

```text
experiments/
  common/
    env_report.py
    timing.py
    memory.py
    plotting_style.py
    scenario_schema.py
    correctness.py

  kmeans/
    README.md
    data_generation.py
    kmeans_reference.py
    kmeans_numpy_broadcast.py
    kmeans_numpy_matmul.py
    kmeans_numba.py
    kmeans_jax.py
    validate_kmeans.py
    run_mac_validation.py
    run_server_cpu.py
    run_server_a100.py
    plot_kmeans.py

  permutation/
    README.md
    data_generation.py
    permutation_reference.py
    permutation_numpy.py
    permutation_threads.py
    permutation_processes.py
    permutation_numba.py
    permutation_jax_scalar.py
    permutation_jax_matrix.py
    validate_permutation.py
    run_mac_validation.py
    run_server_cpu.py
    run_server_a100.py
    plot_permutation.py

  results/
    macbook_air_validation/
      env.json
      kmeans_correctness.csv
      kmeans_smoke_runtime.csv
      permutation_calibration.csv
      permutation_smoke_runtime.csv
      developer_experience_notes.md
      figures/

    linux_server_cpu/
      env.json
      kmeans_cpu_scaling.csv
      permutation_cpu_scaling.csv
      worker_sweep.csv
      memory_scaling.csv
      figures/

    linux_server_a100/
      env.json
      kmeans_jax_gpu.csv
      permutation_matrix_gpu.csv
      gpu_batch_sweep.csv
      cpu_gpu_break_even.csv
      figures/

    merged/
      all_results.csv
      figure_manifest.csv
      figures/

slides/
  index_v3.html
  speaker_notes_v3.md
  timeline_v3.md

docs/
  simulation_driven_statistical_computing.md
  benchmark_contract.md
  environment_ladder.md
  gpu_reformulation_notes.md
  codex_workflow.md
```

---

## 6. Measurement and correctness rules

### 6.1 Timing rules

For every implementation:

1. Run at least one cold call separately.
2. Run multiple warm calls and report median, min, max, and optionally IQR.
3. For JIT/compiled systems, record compile time separately from warm execution.
4. Use fresh subprocesses for full benchmark runs when feasible.
5. For JAX, call `.block_until_ready()` or `jax.block_until_ready()` before stopping the timer.
6. For GPU, distinguish:
   - host data generation time;
   - host-to-device transfer time;
   - compile time;
   - device compute time;
   - device-to-host transfer time if results are materialized.

### 6.2 Memory rules

Record what is feasible:

- Python-visible peak via `tracemalloc` for Python allocations.
- Process RSS via `psutil` or `/proc` on Linux.
- Child-process RSS for multiprocessing.
- GPU memory from NVIDIA tools or JAX memory info if available.
- For large matrix formulations, report theoretical matrix sizes and chosen batch sizes.

### 6.3 Correctness rules

Never report speedup without one of these correctness checks:

#### k-means

- same input data;
- same initial centroids;
- same max iteration and convergence tolerance where possible;
- final inertia close to reference;
- assignments close or equivalent up to label permutation;
- ARI against known labels;
- empty cluster count;
- number of iterations;
- failure flag if NaN/inf or non-convergence occurs.

#### permutation test

- same simulated null/alternative data;
- p-value calibration under null;
- power under alternatives;
- optimized statistic agrees with reference within tolerance for small scenarios;
- random stream design documented;
- if using matrix contrast formulation, validate against direct reference on small R/p.

### 6.4 Environment-availability rules

If a feature is missing, write an explicit `unavailable` row.

Examples:

```text
python314_jit_status = unavailable: sys._jit.is_available() returned false
free_threaded_status = unavailable: sysconfig.get_config_var('Py_GIL_DISABLED') != 1
a100_status = unavailable: jax.devices() contains no gpu device
gpu_memory_status = unavailable: nvidia-smi command failed
```

Never infer or invent results.

---

## 7. Workload 1: k-means experiment design

### 7.1 Purpose

Use k-means to show iterative simulation pressure:

- correctness depends on initialization and convergence;
- hard scenarios can be constructed by simulation;
- vectorization can introduce large temporaries;
- Numba helps scalar CPU loops;
- JAX/GPU requires enough regular work to amortize compile and transfer overhead.

### 7.2 Data generator

Implement Gaussian mixture simulation with known labels.

Parameters:

| Parameter          |                    Suggested values | Purpose                |
| ------------------ | ----------------------------------: | ---------------------- |
| `N`                |  1k, 10k, 100k, 1M, 5M+ server only | sample-size scaling    |
| `d`                |                      2, 10, 64, 256 | dimensional pressure   |
| `K`                |                        3, 5, 20, 50 | cluster-count pressure |
| `separation`       |                  0.5, 1.0, 2.0, 4.0 | easy vs hard recovery  |
| `imbalance`        |          balanced, 90/10, long-tail | empty-cluster risk     |
| `covariance`       |  spherical, anisotropic, correlated | geometry               |
| `outlier_fraction` |                       0, 0.01, 0.05 | robustness             |
| `seed`             | 5–20 seeds depending on environment | stability              |

### 7.3 Implementations

Codex should implement or update:

1. `kmeans_reference.py`
   - Simple, readable, used only for small data.
   - Should prioritize clarity over speed.

2. `kmeans_numpy_broadcast.py`
   - Uses broadcasted distance tensor.
   - Important for teaching memory blow-up.

3. `kmeans_numpy_matmul.py`
   - Uses identity:
     ```text
     ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x @ c.T
     ```
   - Important for high K/d.

4. `kmeans_numba.py`
   - `@njit`, optionally `parallel=True` for assignment step.
   - Avoid unsupported Python objects.

5. `kmeans_jax.py`
   - `jax.jit`.
   - `jax.lax.scan` or `jax.lax.while_loop` for iteration.
   - CPU and GPU backend detection.
   - Default `float32` for GPU; optional `float64` comparison if enabled.

### 7.4 MacBook Air k-means tasks

Run only small-to-medium validation:

```text
N: 1_000, 10_000, 50_000
d: 2, 10, 50
K: 3, 5
separation: 0.5, 1.0, 2.0, 4.0
imbalance: balanced, 90/10
outlier_fraction: 0, 0.01
seeds: 0, 1, 2, 3, 4
```

Required outputs:

```text
experiments/results/macbook_air_validation/kmeans_correctness.csv
experiments/results/macbook_air_validation/kmeans_smoke_runtime.csv
experiments/results/macbook_air_validation/figures/kmeans_ari_heatmap.png
experiments/results/macbook_air_validation/figures/kmeans_reference_equivalence.png
```

Focus:

- Does the reference implementation behave sensibly?
- Which scenarios are statistically hard?
- Which optimized implementations preserve the reference on small cases?
- What code was easiest to debug?

### 7.5 Server CPU k-means tasks

Run large CPU scaling:

```text
N: 100_000, 1_000_000, 5_000_000, optionally 10_000_000 if memory allows
d: 10, 64, 256
K: 5, 20, 50
separation: 1.0, 2.0
imbalance: balanced, 90/10
seeds: 0, 1, 2
implementations: numpy_matmul, numba, jax_cpu if installed
numba_threads: 1, 2, 4, 8, 16, 32, 64, 128
```

Required outputs:

```text
experiments/results/linux_server_cpu/kmeans_cpu_scaling.csv
experiments/results/linux_server_cpu/kmeans_numba_thread_sweep.csv
experiments/results/linux_server_cpu/kmeans_memory_scaling.csv
experiments/results/linux_server_cpu/figures/kmeans_cpu_runtime.png
experiments/results/linux_server_cpu/figures/kmeans_numba_threads.png
experiments/results/linux_server_cpu/figures/kmeans_memory_scaling.png
```

Guardrails:

- Do not run all 512 cores by default.
- Start with 1, 2, 4, 8, 16, 32.
- Expand to 64/128 only if scaling is still meaningful and memory bandwidth is not saturated.
- Record environment variables controlling thread pools: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS`, `XLA_FLAGS` if relevant.

### 7.6 Server A100 k-means tasks

Run only if JAX sees A100 GPU.

Scenarios:

```text
N: 100_000, 1_000_000, 5_000_000, optionally larger if memory allows
d: 10, 64, 256
K: 5, 20, 50
max_iter: 20 or fixed scan length for fair JAX comparison
seeds: 0, 1, 2
dtype: float32; optional float64 sensitivity check
```

Required outputs:

```text
experiments/results/linux_server_a100/kmeans_jax_gpu.csv
experiments/results/linux_server_a100/kmeans_gpu_batch_or_problem_sweep.csv
experiments/results/linux_server_a100/figures/kmeans_cpu_gpu_break_even.png
experiments/results/linux_server_a100/figures/kmeans_jax_cold_vs_warm.png
```

Interpretation goal:

- Show break-even, not just peak speed.
- Small problems should likely remain laptop/CPU territory.
- GPU is justified when problem size and computation regularity overcome compile/transfer overhead.

---

## 8. Workload 2: high-dimensional permutation test design

### 8.1 Purpose

Upgrade the permutation test from a scalar benchmark to a biostatistics-like high-dimensional simulation.

Original scalar story:

- one statistic;
- repeat R permutations;
- exposes repeated random work and parallel overhead.

New high-dimensional story:

- many features/genes;
- same label permutation applied to all features;
- exposes memory, matrix algebra, batching, and GPU suitability.

### 8.2 Statistical setup

Simulate matrix:

```text
X: n_samples x p_features
labels: n_samples binary group labels
```

Under the null:

```text
X_i,j ~ noise distribution independent of label
```

Under alternatives:

```text
a subset of features has mean shift delta in group 1
```

Statistics:

- difference in means;
- optionally standardized difference if variance handling is stable;
- empirical p-value from permutation null.

### 8.3 Direct reference formulation

For small cases:

```text
for r in range(R):
    permuted_labels = permute(labels)
    compute difference in means for each feature
```

Use this only as the correctness oracle for small `n`, `p`, and `R`.

### 8.4 Matrix contrast formulation

For high-dimensional GPU-friendly runs:

Let each permutation produce a contrast vector `w_r` of length `n_samples`, with positive weights for group A and negative weights for group B.

Stack contrasts:

```text
W: R x n_samples
X: n_samples x p_features
T_null = W @ X
```

For memory control, use batches:

```text
for batch in permutation_batches:
    W_batch: B x n_samples
    T_batch = W_batch @ X
    update exceedance counts or store summary only
```

Important: do not store the full `R x p` matrix if it is too large. Prefer streaming exceedance counts:

```text
counts += abs(T_batch) >= abs(T_observed)
p_values = (counts + 1) / (R + 1)
```

### 8.5 Correctness/statistical checks

Required:

1. Small-case equivalence:
   - direct reference vs matrix formulation.
   - compare statistics and p-values within tolerance.

2. Null calibration:
   - simulate many null datasets.
   - p-values should be approximately uniform.
   - report calibration plot or histogram.

3. Power curve:
   - vary effect size.
   - report detection power or rank of true signal features.

4. Randomness check:
   - no reused identical permutation batches unless intended.
   - seed handling documented.

### 8.6 MacBook Air permutation tasks

Run small validation:

```text
n: 100, 500, 1_000
p: 10, 100, 1_000
R: 100, 1_000, 2_000
null_replicates: 20 or 50 if fast
alternative_delta: 0.2, 0.5, 1.0
signal_fraction: 0.01 or 0.05
seeds: 0, 1, 2, 3, 4
```

Required outputs:

```text
experiments/results/macbook_air_validation/permutation_equivalence.csv
experiments/results/macbook_air_validation/permutation_calibration.csv
experiments/results/macbook_air_validation/permutation_smoke_runtime.csv
experiments/results/macbook_air_validation/figures/permutation_null_calibration.png
experiments/results/macbook_air_validation/figures/permutation_power_smoke.png
```

Purpose:

- validate statistical behavior;
- confirm matrix formulation agrees with direct reference;
- document code clarity and debugging.

### 8.7 Server CPU permutation tasks

Run CPU scaling and parallelism:

```text
n: 1_000, 5_000, 10_000, 50_000
p: 1_000, 10_000, 50_000
R: 1_000, 10_000, 100_000
workers: 1, 2, 4, 8, 16, 32, 64, 128
implementations:
  numpy_direct_loop
  numpy_matrix_batched
  multiprocessing
  threadpool_standard_python
  threadpool_free_threaded_if_available
  numba_prange
  jax_cpu_matrix_if_available
```

Required outputs:

```text
experiments/results/linux_server_cpu/permutation_cpu_scaling.csv
experiments/results/linux_server_cpu/permutation_worker_sweep.csv
experiments/results/linux_server_cpu/permutation_memory_scaling.csv
experiments/results/linux_server_cpu/permutation_calibration_server_subset.csv
experiments/results/linux_server_cpu/figures/permutation_cpu_runtime.png
experiments/results/linux_server_cpu/figures/permutation_worker_sweep.png
experiments/results/linux_server_cpu/figures/process_vs_thread_memory.png
```

Purpose:

- show process copies vs shared data;
- show when threads or free-threaded Python help;
- show Numba `prange` strengths and limitations;
- show CPU matrix formulation baseline for comparison to A100.

### 8.8 Server A100 permutation tasks

Run GPU matrix formulation as the flagship accelerator experiment.

Scenarios:

```text
n: 1_000, 5_000, 10_000, maybe 20_000 if memory allows
p: 10_000, 50_000, 100_000, maybe 500_000 if memory allows
R: 1_000, 5_000, 10_000, 50_000, maybe 100_000 with batching
batch_R: sweep values, e.g. 128, 256, 512, 1024, 2048, auto
dtype: float32; optional float64 comparison
```

Required outputs:

```text
experiments/results/linux_server_a100/permutation_matrix_gpu.csv
experiments/results/linux_server_a100/permutation_gpu_batch_sweep.csv
experiments/results/linux_server_a100/permutation_cpu_gpu_break_even.csv
experiments/results/linux_server_a100/permutation_gpu_memory.csv
experiments/results/linux_server_a100/figures/permutation_gpu_runtime.png
experiments/results/linux_server_a100/figures/permutation_gpu_batch_sweep.png
experiments/results/linux_server_a100/figures/permutation_cpu_gpu_break_even.png
experiments/results/linux_server_a100/figures/permutation_matrix_reformulation.png
```

Critical guardrails:

- Verify GPU availability before running:
  ```python
  import jax
  print(jax.devices())
  ```
- Confirm the selected device is A100 or a CUDA GPU.
- Ensure all timed JAX functions call `block_until_ready()`.
- Record compile time separately from warm execution.
- Use batched computation; do not materialize huge `R x p` outputs unless explicitly needed and memory permits.
- Report GPU memory usage and batch size.
- Do not compare GPU time to MacBook time as if hardware were identical.

---

## 9. Parallel Codex work plan

The goal is to let one Codex instance work locally and another Codex instance work on the remote server without blocking each other.

### 9.1 Branches or worktrees

Use separate branches or worktrees:

```text
branch: exp/mac-validation
branch: exp/server-cpu-a100
branch: exp/merge-slides-v3
```

If branches are inconvenient, use output directories and avoid editing the same files concurrently:

- Local Codex edits:
  - `experiments/common/`
  - `experiments/kmeans/*mac*`
  - `experiments/permutation/*mac*`
  - `docs/simulation_driven_statistical_computing.md`
  - `experiments/results/macbook_air_validation/`

- Server Codex edits:
  - `experiments/kmeans/*server*`
  - `experiments/permutation/*server*`
  - `docs/gpu_reformulation_notes.md`
  - `experiments/results/linux_server_cpu/`
  - `experiments/results/linux_server_a100/`

- Merge Codex edits after both finish:
  - `experiments/results/merged/`
  - `slides/index_v3.html`
  - `slides/speaker_notes_v3.md`
  - `slides/timeline_v3.md`

### 9.2 Shared contract file

Create one shared config/schema file before running either side:

```text
experiments/common/result_schema.md
experiments/common/scenario_schema.py
```

This prevents local and server outputs from becoming incompatible.

Minimum CSV columns:

```text
run_id, timestamp, environment_tier, machine_name, workload, implementation,
scenario_id, seed, n, p, d, k, r, batch_r, workers, dtype,
cold_time_s, warm_median_s, warm_min_s, warm_max_s,
compile_time_s, transfer_h2d_s, transfer_d2h_s,
peak_python_mb, peak_rss_mb, peak_child_rss_mb, peak_gpu_mb,
correctness_status, statistical_metric_name, statistical_metric_value,
notes
```

Use `NA` where a column does not apply.

---

## 10. Prompt for Local Codex: MacBook Air validation

Copy this section into Codex running on the MacBook Air.

```text
You are working on the local MacBook Air validation tier for the PyCon US talk repository.

Goal:
Create small, fast, inspectable simulation experiments that define the statistical correctness contract before optimization. Do not try to produce final large-scale speedup numbers on this machine. The MacBook role is reference implementation, smoke tests, correctness checks, and developer-experience notes.

High-level narrative to support:
Laptop simulation defines what we trust. Server experiments later test what scales.

Tasks:

1. Inspect the repository structure.
   - Do not delete existing experiments.
   - Add new files where needed under experiments/common, experiments/kmeans, experiments/permutation, docs, and experiments/results/macbook_air_validation.

2. Create or update a common environment reporter:
   - experiments/common/env_report.py
   - It should write experiments/results/macbook_air_validation/env.json.
   - Include OS, Python version, executable, package versions for numpy, scipy if installed, sklearn if installed, numba if installed, jax if installed, CPU info if available, RAM if available.
   - If Python 3.14 free-threaded support is available, detect it using sysconfig.get_config_var('Py_GIL_DISABLED') and sys._is_gil_enabled() when available.
   - If sys._jit exists, record sys._jit.is_available() and sys._jit.is_enabled() when available.

3. Create or update k-means validation scripts:
   - experiments/kmeans/data_generation.py
   - experiments/kmeans/kmeans_reference.py
   - experiments/kmeans/kmeans_numpy_broadcast.py
   - experiments/kmeans/kmeans_numpy_matmul.py
   - experiments/kmeans/validate_kmeans.py
   - experiments/kmeans/run_mac_validation.py

4. For k-means, run small scenario grid:
   N = [1000, 10000, 50000]
   d = [2, 10, 50]
   K = [3, 5]
   separation = [0.5, 1.0, 2.0, 4.0]
   imbalance = ['balanced', '90_10']
   outlier_fraction = [0.0, 0.01]
   seeds = [0, 1, 2, 3, 4]

   If the full grid is too slow, start with a representative subset and write the skipped scenarios into a TODO section.

5. K-means correctness outputs:
   - Compare all optimized variants to reference on small scenarios.
   - Record final inertia, ARI against true labels, iteration count, empty cluster count, numerical failure flags.
   - Write:
     experiments/results/macbook_air_validation/kmeans_correctness.csv
     experiments/results/macbook_air_validation/kmeans_smoke_runtime.csv
     experiments/results/macbook_air_validation/figures/kmeans_ari_heatmap.png
     experiments/results/macbook_air_validation/figures/kmeans_reference_equivalence.png

6. Create or update permutation validation scripts:
   - experiments/permutation/data_generation.py
   - experiments/permutation/permutation_reference.py
   - experiments/permutation/permutation_numpy.py
   - experiments/permutation/permutation_jax_matrix.py if JAX is installed, otherwise create stub with clear unavailable status
   - experiments/permutation/validate_permutation.py
   - experiments/permutation/run_mac_validation.py

7. For permutation, run small validation grid:
   n = [100, 500, 1000]
   p = [10, 100, 1000]
   R = [100, 1000, 2000]
   null_replicates = 20 if feasible
   alternative_delta = [0.2, 0.5, 1.0]
   signal_fraction = 0.01 or 0.05
   seeds = [0, 1, 2, 3, 4]

8. Permutation correctness outputs:
   - Validate direct reference vs matrix contrast formulation on small cases.
   - Under null, produce p-value calibration summary and histogram.
   - Under alternatives, produce a small power curve.
   - Write:
     experiments/results/macbook_air_validation/permutation_equivalence.csv
     experiments/results/macbook_air_validation/permutation_calibration.csv
     experiments/results/macbook_air_validation/permutation_smoke_runtime.csv
     experiments/results/macbook_air_validation/figures/permutation_null_calibration.png
     experiments/results/macbook_air_validation/figures/permutation_power_smoke.png

9. Developer-experience notes:
   - Create experiments/results/macbook_air_validation/developer_experience_notes.md.
   - For each implementation, write short notes on readability, debuggability, failure modes, and how far it moves away from plain NumPy.
   - Include examples of bugs or traps discovered during validation.

10. Docs:
   - Create or update docs/simulation_driven_statistical_computing.md.
   - Explain the six-step simulation loop.
   - Explain the laptop/server/GPU environment ladder.
   - Keep this document understandable to programmers without statistics background.

11. Do not fabricate results.
   - If Numba, JAX, Python 3.14 JIT, or free-threaded Python is unavailable locally, write explicit unavailable rows.
   - All figures should be generated only from actual CSV results.

12. Final local summary:
   - Write experiments/results/macbook_air_validation/LOCAL_SUMMARY.md.
   - Include what passed, what failed, what was skipped, and what should be run on the server.
```

---

## 11. Prompt for Server Codex: Linux CPU + A100

Copy this section into Codex running on the Linux server.

```text
You are working on the Linux server CPU + A100 tier for the PyCon US talk repository.

Goal:
Produce large-scale CPU and GPU experiments that reuse the same statistical correctness contract from the MacBook validation tier. The server is responsible for scale, parallelism, memory behavior, and A100 accelerator results. Do not turn the results into a single leaderboard across hardware.

High-level narrative to support:
The laptop defines what we trust. The server tests what scales. The A100 only wins when the statistical computation is reformulated into a GPU-friendly batched array/matrix program.

Tasks:

1. Inspect the repository structure.
   - Do not delete existing experiments.
   - Add server-specific runners under experiments/kmeans and experiments/permutation.
   - Write outputs only under experiments/results/linux_server_cpu and experiments/results/linux_server_a100.

2. Environment report:
   - Create or update experiments/common/env_report.py if needed.
   - Write:
     experiments/results/linux_server_cpu/env.json
     experiments/results/linux_server_a100/env.json
   - Include OS, CPU model, logical and physical core count if available, RAM, Python executable/version, package versions, BLAS/threading info if available, CUDA driver/runtime, nvidia-smi output, GPU name and memory.
   - For JAX, record jax.devices(), jax.default_backend(), jaxlib version, and whether a CUDA GPU is visible.
   - If A100 is not visible to JAX, mark A100 experiments unavailable and do not invent GPU numbers.

3. Threading controls:
   - Add a utility to run benchmarks with explicit thread settings:
     OMP_NUM_THREADS
     MKL_NUM_THREADS
     OPENBLAS_NUM_THREADS
     NUMBA_NUM_THREADS
   - Avoid oversubscription.
   - Do not start with 512 workers. Use worker/thread sweeps: [1, 2, 4, 8, 16, 32, 64, 128]. Expand only if safe.

4. Server CPU k-means:
   - Implement or update experiments/kmeans/run_server_cpu.py.
   - Compare numpy_matmul, numba, and jax_cpu if available.
   - Scenario grid:
     N = [100000, 1000000, 5000000]
     optional N = [10000000] only if memory allows
     d = [10, 64, 256]
     K = [5, 20, 50]
     separation = [1.0, 2.0]
     imbalance = ['balanced', '90_10']
     seeds = [0, 1, 2]
   - For Numba, run thread sweeps.
   - Validate final inertia and ARI against a trusted smaller or same-scenario reference where feasible.
   - Write:
     experiments/results/linux_server_cpu/kmeans_cpu_scaling.csv
     experiments/results/linux_server_cpu/kmeans_numba_thread_sweep.csv
     experiments/results/linux_server_cpu/kmeans_memory_scaling.csv
     experiments/results/linux_server_cpu/figures/kmeans_cpu_runtime.png
     experiments/results/linux_server_cpu/figures/kmeans_numba_threads.png
     experiments/results/linux_server_cpu/figures/kmeans_memory_scaling.png

5. Server A100 k-means:
   - Implement or update experiments/kmeans/run_server_a100.py.
   - Run only if JAX sees CUDA GPU/A100.
   - Use jax.jit and a fixed iteration structure if necessary.
   - Scenario grid:
     N = [100000, 1000000, 5000000]
     d = [10, 64, 256]
     K = [5, 20, 50]
     max_iter = 20
     seeds = [0, 1, 2]
     dtype = float32 by default
   - Record cold compile time, warm execution time, and memory.
   - Always call block_until_ready() before stopping timers.
   - Write:
     experiments/results/linux_server_a100/kmeans_jax_gpu.csv
     experiments/results/linux_server_a100/kmeans_gpu_problem_sweep.csv
     experiments/results/linux_server_a100/figures/kmeans_cpu_gpu_break_even.png
     experiments/results/linux_server_a100/figures/kmeans_jax_cold_vs_warm.png

6. Server CPU high-dimensional permutation:
   - Implement or update experiments/permutation/run_server_cpu.py.
   - Include direct loop only for small reference cases.
   - Include batched matrix contrast formulation for larger cases.
   - Include multiprocessing, threadpool, numba prange, and jax_cpu if available.
   - Scenario grid:
     n = [1000, 5000, 10000, 50000]
     p = [1000, 10000, 50000]
     R = [1000, 10000, 100000]
     workers = [1, 2, 4, 8, 16, 32, 64, 128]
   - Skip combinations that exceed memory budget; record skipped rows with reason.
   - Write:
     experiments/results/linux_server_cpu/permutation_cpu_scaling.csv
     experiments/results/linux_server_cpu/permutation_worker_sweep.csv
     experiments/results/linux_server_cpu/permutation_memory_scaling.csv
     experiments/results/linux_server_cpu/permutation_calibration_server_subset.csv
     experiments/results/linux_server_cpu/figures/permutation_cpu_runtime.png
     experiments/results/linux_server_cpu/figures/permutation_worker_sweep.png
     experiments/results/linux_server_cpu/figures/process_vs_thread_memory.png

7. Server A100 high-dimensional permutation:
   - This is the flagship GPU experiment.
   - Implement or update experiments/permutation/run_server_a100.py.
   - Use the matrix contrast formulation:
     W_batch: batch_R x n_samples
     X: n_samples x p_features
     T_batch = W_batch @ X
   - Do not materialize full R x p matrix if too large. Stream exceedance counts for p-values.
   - Scenario grid:
     n = [1000, 5000, 10000]
     optional n = [20000] only if memory allows
     p = [10000, 50000, 100000]
     optional p = [500000] only if memory allows
     R = [1000, 5000, 10000, 50000]
     batch_R = [128, 256, 512, 1024, 2048] or auto-tuned subset
     dtype = float32 by default
   - Validate small cases against direct reference before running large cases.
   - Always call block_until_ready() before stopping timers.
   - Record compile time separately from warm time.
   - Record GPU memory and batch size.
   - Write:
     experiments/results/linux_server_a100/permutation_matrix_gpu.csv
     experiments/results/linux_server_a100/permutation_gpu_batch_sweep.csv
     experiments/results/linux_server_a100/permutation_cpu_gpu_break_even.csv
     experiments/results/linux_server_a100/permutation_gpu_memory.csv
     experiments/results/linux_server_a100/figures/permutation_gpu_runtime.png
     experiments/results/linux_server_a100/figures/permutation_gpu_batch_sweep.png
     experiments/results/linux_server_a100/figures/permutation_cpu_gpu_break_even.png
     experiments/results/linux_server_a100/figures/permutation_matrix_reformulation.png

8. GPU anti-pattern check:
   - If there is an old JAX vmap(random.permutation) implementation, keep it as an anti-pattern only if it is useful and clearly labeled.
   - Do not present it as the recommended GPU approach.
   - The recommended GPU approach is the matrix contrast formulation.

9. Server summary:
   - Write experiments/results/linux_server_cpu/SERVER_CPU_SUMMARY.md.
   - Write experiments/results/linux_server_a100/A100_SUMMARY.md.
   - Include what ran, what failed, what was skipped due to memory/time, and what figures are ready for slides.

10. Do not fabricate results.
    - If JAX GPU is unavailable, write unavailable rows.
    - If a scenario is too large, skip with explicit reason.
    - If a result fails correctness, keep it and mark fail. Do not hide it.
```

---

## 12. Prompt for Merge Codex: integrate local + server results and rebuild slides

Run this after the Mac and server work are done.

```text
You are the merge/rebuild Codex instance for the PyCon US talk repository.

Goal:
Merge MacBook validation results, Linux server CPU results, and A100 GPU results into a coherent simulation-driven talk. Rebuild the timeline, figures, speaker notes, and reveal.js deck.

Inputs:
- experiments/results/macbook_air_validation/
- experiments/results/linux_server_cpu/
- experiments/results/linux_server_a100/
- docs/simulation_driven_statistical_computing.md
- docs/gpu_reformulation_notes.md
- existing index.html deck
- current timeline markdown

Tasks:

1. Validate result schemas.
   - Load all CSVs.
   - Confirm required columns exist.
   - Confirm environment_tier labels are present.
   - Create experiments/results/merged/all_results.csv.
   - Create experiments/results/merged/figure_manifest.csv.

2. Do not create a single cross-hardware leaderboard.
   - Use separate panels or facets:
     MacBook validation
     Linux server CPU
     Linux server A100
   - If a plot compares CPU and GPU, clearly label hardware and explain the comparison as a scale/break-even analysis, not a universal ranking.

3. Rebuild figures for the talk:
   Required figures:
   - simulation_loop.png
   - environment_ladder.png
   - kmeans_ari_heatmap.png
   - kmeans_memory_scaling.png
   - kmeans_cpu_runtime.png
   - kmeans_cpu_gpu_break_even.png
   - permutation_matrix_reformulation.png
   - permutation_null_calibration.png
   - permutation_worker_sweep.png
   - permutation_cpu_gpu_break_even.png
   - tradeoff_decision_matrix.png

4. Rebuild timeline:
   - Write slides/timeline_v3.md using the 30-minute structure in replanned_timeline_and_codex_instructions.md.
   - Introduction must receive about 6.5–7 minutes.
   - k-means should receive about 7 minutes.
   - permutation should receive about 9 minutes.
   - developer experience and AI/Codex should receive about 3.5 minutes.

5. Rebuild reveal.js deck:
   - Create slides/index_v3.html.
   - Preserve the visual style of the existing deck unless there is a strong reason to change it.
   - Replace old local-only result numbers with new environment-tiered figures.
   - Use explicit labels: MacBook Air validation, Linux server CPU, Linux server A100.
   - Update speaker notes so the story is simulation-first.

6. Speaker notes:
   - Create slides/speaker_notes_v3.md.
   - Notes should be detailed enough for rehearsal.
   - Avoid sounding like a benchmark report.
   - Emphasize: reference first, bad cases second, scale third, optimization last.

7. Final repository summary:
   - Write REPLAN_RESULTS_SUMMARY.md.
   - Include final timeline, figure list, open questions, and experiments that need rerun.

8. Integrity rules:
   - Every number in slides/index_v3.html must be traceable to a CSV.
   - Every figure must be generated from committed result files.
   - If a result is preliminary, label it preliminary.
   - Do not invent missing GPU/JIT/free-threaded results.
```

---

## 13. Slide-by-slide target outline for index_v3.html

Use approximately 20–22 slides.

### Slide 1 — Title

**Title:** Breaking the Speed Limit  
**Subtitle:** Fast statistical models with Python 3.14, Numba, and JAX, told as a simulation workflow.

Remove or de-emphasize old hero metrics until regenerated.

### Slide 2 — Simulation is how statisticians write tests

Core message:

> Real data often has no answer key. Simulation creates controlled worlds where code can be tested.

### Slide 3 — The simulation loop

Show six-step loop:

1. define statistic;
2. reference implementation;
3. controlled scenarios;
4. validation;
5. scale;
6. optimize.

### Slide 4 — Three failure modes

Implementation failure, statistical failure, systems failure.

### Slide 5 — Environment ladder

MacBook Air → Linux server CPU → Linux server A100.

### Slide 6 — Benchmark contract

Correctness first, then cold/warm time, memory, JAX synchronization, environment labels.

### Slide 7 — Two workloads

k-means and high-dimensional permutation.

### Slide 8 — k-means as iterative simulation pressure

Brief algorithm and why it is useful.

### Slide 9 — k-means simulation grid

ARI/failure heatmap from Mac validation.

### Slide 10 — k-means reference vs optimized

Show equivalence checks and memory question.

### Slide 11 — k-means CPU scaling

Server CPU figure.

### Slide 12 — k-means GPU break-even

A100 figure, if available. If unavailable, replace with planned experiment slide.

### Slide 13 — Permutation tests as resampling pressure

Move from scalar to high-dimensional biostatistics setting.

### Slide 14 — Statistical algebra before runtime

Show direct formulation and matrix contrast formulation `T_null = W @ X`.

### Slide 15 — Calibration first

Null p-value calibration and small-case equivalence.

### Slide 16 — CPU parallelism: process/thread/Numba

Worker and memory sweep.

### Slide 17 — A100 matrix formulation

GPU batch and break-even result.

### Slide 18 — Developer traps

JAX async timing, random streams, process memory, oversubscription.

### Slide 19 — AI/Codex workflow

AI generates variants and experiments; statistician owns the contract.

### Slide 20 — Decision guide

Choose smallest tool preserving the statistic.

### Slide 21 — Thank you

Clear enough to trust. Fast enough to scale.

---

## 14. Figures and what they should prove

| Figure                                 | Environment           | Claim                                                         |
| -------------------------------------- | --------------------- | ------------------------------------------------------------- |
| `simulation_loop.png`                  | conceptual            | Simulation is a workflow, not a benchmark trick.              |
| `environment_ladder.png`               | conceptual            | Laptop/server/GPU have different roles.                       |
| `kmeans_ari_heatmap.png`               | MacBook               | Small simulation reveals hard cases.                          |
| `kmeans_reference_equivalence.png`     | MacBook               | Optimized versions preserve behavior on small cases.          |
| `kmeans_memory_scaling.png`            | Server CPU            | Broadcast vectorization can create huge temporaries.          |
| `kmeans_cpu_runtime.png`               | Server CPU            | Numba/NumPy/JAX CPU trade-offs depend on shape.               |
| `kmeans_cpu_gpu_break_even.png`        | Server A100           | GPU helps only past a problem-size threshold.                 |
| `permutation_matrix_reformulation.png` | conceptual            | Statistical rewrite turns resampling into matrix computation. |
| `permutation_null_calibration.png`     | MacBook/server subset | Correctness means calibration, not just same runtime.         |
| `permutation_worker_sweep.png`         | Server CPU            | Worker count has diminishing returns and memory costs.        |
| `process_vs_thread_memory.png`         | Server CPU            | Processes can hide copies in child RSS.                       |
| `permutation_cpu_gpu_break_even.png`   | Server A100           | A100 payoff appears for high-dimensional batched workloads.   |
| `tradeoff_decision_matrix.png`         | merged                | No single winner; choose by computational pattern.            |

---

## 15. Speaker framing for each environment

### 15.1 MacBook Air framing

Say:

> This is where the statistical contract is born. I want a version that is small enough to inspect, quick enough to rerun, and clear enough to debug.

Do not say:

> This is the performance benchmark for modern Python.

### 15.2 Linux server CPU framing

Say:

> Once the contract is stable, the server shows how the same code behaves under scale: more rows, more features, more repetitions, more workers, and more memory pressure.

Do not say:

> 512 cores means everything should be 512 times faster.

### 15.3 Linux server A100 framing

Say:

> The GPU becomes useful only after the statistical computation is expressed as large batched array work. The speedup is as much a statistical algebra story as a hardware story.

Do not say:

> JAX makes every permutation test fast.

---

## 16. Important implementation notes for Codex

### 16.1 JAX timing

JAX dispatch is asynchronous. Timed JAX operations must block before recording elapsed time.

Use patterns like:

```python
start = time.perf_counter()
y = compiled_fn(*args)
y.block_until_ready()
elapsed = time.perf_counter() - start
```

or:

```python
import jax
start = time.perf_counter()
y = compiled_fn(*args)
jax.block_until_ready(y)
elapsed = time.perf_counter() - start
```

### 16.2 JAX GPU detection

Use:

```python
import jax
print(jax.devices())
print(jax.default_backend())
```

If no GPU is visible, write an unavailable result row.

### 16.3 Python 3.14 free-threaded detection

Use guarded detection:

```python
import sys
import sysconfig

supports_free_threading = sysconfig.get_config_var("Py_GIL_DISABLED") == 1
is_gil_enabled = None
if hasattr(sys, "_is_gil_enabled"):
    is_gil_enabled = sys._is_gil_enabled()
```

If unavailable, record `unavailable`. Do not infer.

### 16.4 Python 3.14 JIT detection

Use guarded detection:

```python
import sys

jit_available = None
jit_enabled = None
if hasattr(sys, "_jit"):
    if hasattr(sys._jit, "is_available"):
        jit_available = sys._jit.is_available()
    if hasattr(sys._jit, "is_enabled"):
        jit_enabled = sys._jit.is_enabled()
```

Do not claim CPython JIT speedup unless it is actually available and measured.

### 16.5 Randomness and reproducibility

Rules:

- Use explicit seeds.
- Store seed in every result row.
- For parallel RNG, avoid correlated streams.
- For JAX, use explicit PRNG keys and split/fold-in by batch/seed.
- For Numba, document random stream design.
- Check null calibration, not only equality of one random output.

### 16.6 Avoid memory explosions

Before allocating large matrices, estimate size:

```text
bytes = rows * cols * dtype_size
GB = bytes / 1024**3
```

For `W @ X`:

```text
W_batch: batch_R x n
X: n x p
T_batch: batch_R x p
```

Memory risk comes from all three arrays plus temporary buffers. Use batch size sweeps and fail gracefully.

---

## 17. Acceptance criteria

A successful rebuild should produce:

### Local MacBook outputs

- `env.json`
- k-means correctness CSV
- permutation equivalence CSV
- null calibration figure
- k-means ARI/failure heatmap
- developer-experience notes

### Server CPU outputs

- CPU scaling CSVs
- worker sweep CSVs
- memory scaling CSVs
- process/thread/Numba comparison figures

### Server A100 outputs

- JAX GPU environment confirmation
- k-means GPU break-even figure if feasible
- high-dimensional permutation matrix GPU results
- batch size sweep
- GPU memory summary

### Merged slide outputs

- `slides/timeline_v3.md`
- `slides/index_v3.html`
- `slides/speaker_notes_v3.md`
- `REPLAN_RESULTS_SUMMARY.md`

### Narrative acceptance test

A listener should be able to answer:

1. Why did we start on a laptop?
2. What statistical behavior did the simulation test?
3. Which failure modes did simulation reveal?
4. Which bottleneck did each optimization remove?
5. Why did the A100 help only after reformulation?
6. Why is there no single universal winner?

If the slides mostly answer “which library is fastest,” the rebuild failed. If the slides answer “how do we make statistical Python trustworthy and fast,” the rebuild succeeded.

---

## 18. Short commands/checklists to add to README files

### 18.1 MacBook README checklist

```text
# MacBook validation tier
python -m experiments.common.env_report --out experiments/results/macbook_air_validation/env.json
python -m experiments.kmeans.run_mac_validation
python -m experiments.permutation.run_mac_validation
python -m experiments.kmeans.plot_kmeans --tier macbook_air_validation
python -m experiments.permutation.plot_permutation --tier macbook_air_validation
```

### 18.2 Server CPU README checklist

```text
# Linux server CPU tier
python -m experiments.common.env_report --out experiments/results/linux_server_cpu/env.json
python -m experiments.kmeans.run_server_cpu
python -m experiments.permutation.run_server_cpu
python -m experiments.kmeans.plot_kmeans --tier linux_server_cpu
python -m experiments.permutation.plot_permutation --tier linux_server_cpu
```

### 18.3 Server A100 README checklist

```text
# Linux server A100 tier
nvidia-smi
python - <<'PY'
import jax
print(jax.devices())
print(jax.default_backend())
PY
python -m experiments.common.env_report --out experiments/results/linux_server_a100/env.json
python -m experiments.kmeans.run_server_a100
python -m experiments.permutation.run_server_a100
python -m experiments.kmeans.plot_kmeans --tier linux_server_a100
python -m experiments.permutation.plot_permutation --tier linux_server_a100
```

---

## 19. Final Codex instruction: protect the story

When changing code, figures, docs, or slides, keep the story in this order:

1. **Trust first:** reference implementation and statistical contract.
2. **Bad cases second:** simulation grid reveals where things fail.
3. **Scale third:** increase N, d, p, K, R, workers, and memory pressure.
4. **Optimize last:** choose NumPy rewrite, Numba, threads, multiprocessing, JAX CPU, or JAX GPU based on the actual bottleneck.

Do not optimize the talk into a tool ranking. The result should feel like a statistician showing programmers how simulation turns performance engineering into a disciplined, testable workflow.
