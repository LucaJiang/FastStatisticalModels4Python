# Simulation-driven statistical computing

The v3 talk frames performance work as a statistical workflow, not a tool ranking. The central claim is:

> Simulation is how statisticians write tests when real data has no answer key.

In biomedical and scientific data analysis, the real dataset often does not tell us the true clusters, true null distribution, true effect size, or exact failure modes. A simulation gives us a controlled world where those quantities are known.

## Workflow

1. Define the estimand or statistic.
2. Build a slow reference implementation close to the math.
3. Generate controlled scenarios: null, alternative, easy, hard, and pathological.
4. Validate behavior: equivalence, calibration, power, convergence, memory, and reproducibility.
5. Scale the workload along realistic axes.
6. Optimize the proven bottleneck with the smallest tool that preserves the statistic.

## What correctness means

Correctness is broader than matching one floating point number.

- Implementation equivalence: optimized code agrees with the reference on small cases.
- Statistical calibration: null p-values behave like null p-values.
- Power: signal becomes easier to detect as effect size increases.
- Convergence behavior: iterative algorithms stop for the same reason.
- Memory behavior: vectorization does not hide impossible temporaries.
- Reproducibility: seeds, backends, and environment are recorded.
- Developer effort: simpler code remains valuable when it is fast enough.

## Why this matters

A faster implementation that changes the statistic is not an optimization. It is a different method. The v3 experiments therefore put validation before timing, then use runtime and memory plots to decide whether NumPy, Numba, threads, or JAX/GPU is the right next tool.

## Local MacBook validation from 2026-05-03

The local tier was run in `py312` on a 16 GB MacBook Air with Python 3.12.2,
NumPy 1.26.4, Numba 0.59.1, and JAX 0.4.25 on CPU. Its role is correctness,
debugging, reference checks, and lightweight evidence, not final large-scale
speed claims.

k-means validation used simulated Gaussian mixtures with separation,
imbalance, outliers, dimensions, cluster counts, and seeds. The full MacBook
grid covered all 1,440 required scenarios. The reference implementation,
matrix NumPy implementation, and Numba implementation produced 3,840 passing
rows; 480 reference-broadcast rows were marked `skipped_memory_risk` because
they would allocate the intentionally unsafe `O(N*K*d)` distance tensor.

Permutation validation used a two-group feature-wise test. The loop reference,
NumPy matrix formulation, and JAX CPU matrix formulation all produced identical
p-values on the full equivalence grid after enabling JAX x64 for this
correctness tier. Null calibration stayed close to uniform: the extended
MacBook calibration pass has mean `p <= 0.05` rate 0.051.

The curated local outputs, targeted extra evidence, and 16:9 deck figures live
in `experiments/results/macbook_air_long/latest/`.

## Server and A100 scale evidence from 2026-05-03

The Linux server CPU and A100 tiers use the same validation-first contract at
larger shapes. The curated result directories are:

- `experiments/results/linux_server_cpu/long_safe_20260503_190133/`
- `experiments/results/linux_server_a100/long_safe_20260503_190133/`

The CPU server run on `BI103202` produced 540 passing k-means CPU rows, 105
passing CPU permutation rows, and 3 explicit permutation timeouts at the largest
`n=50,000`, `p=50,000`, `R=100,000` corner. Numba k-means improved up to about
32 to 64 threads, while 128 requested threads was slower, so maximum thread
count is not automatically optimal. The permutation worker sweep also showed a
non-monotone memory/runtime trade-off, with 8 workers fastest in the measured
shape.

The A100 run produced 135 passing JAX k-means rows and 15 passing matrix
permutation rows. A100 k-means is compelling only after enough work exists to
amortize startup and transfer overhead, reaching about 5.8x over the best CPU
baseline at `N=5,000,000`, `d=10`, `K=20`. The permutation A100 result is an
important negative result: the validated matrix reformulation runs on GPU, but
the matched `n=5,000`, `p=50,000` slice is still faster on CPU for `R=1,000`
and `R=10,000` in this implementation.
