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
