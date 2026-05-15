# Timeline - 30 minute talk

This timeline matches `slides/index.html`: 36 total slides, 29 main-path slides, and 7 backup slides. Video slides: slides 9 and 17 are short method-transition slides.

The narrative is an evidence ladder, not a hardware leaderboard: laptop simulation defines trust, server CPU tests scale and parallelism, and A100 is useful only when the validated computation has an accelerator-shaped pipeline.

## 0:00-1:00 - Title and thesis

Slide 1: Breaking the Speed Limit

Purpose: establish the thesis. Simulation is how statisticians write tests; speed is meaningful only when optimized code answers the same statistical question.

## 1:00-8:00 - Simulation-driven statistical computing

Slide 2: For statisticians, speed comes after validation
Slide 3: Simulation creates an answer key
Slide 4: Statistical CI
Slide 5: Python stays the scientific interface
Slide 6: Two workloads, two statistical computing shapes
Slide 7: Evidence ladder

Purpose: build the PyCon/programmer bridge, explain why Python remains the interface, and define validation before speed.

## 8:00-16:00 - Workload 1: k-means

Slide 8: Workload 1 - k-means as iterative model-fitting pressure
Slide 9: How k-means moves
Slide 10: Why k-means works here
Slide 11: First ask: did k-means recover the simulated clusters?
Slide 12: Before timing: does optimized k-means reproduce the reference?
Slide 13: How the same k-means step looks in Python
Slide 14: Server k-means: acceleration is shape-dependent
Slide 15: k-means takeaway

Purpose: show iterative model-fitting pressure: assignment/update dependence, recovery before runtime, equivalence before optimization, implementation shape, and conditional CPU/GPU scale behavior.

## 16:00-24:00 - Workload 2: permutation tests

Slide 16: Workload 2 - permutation tests as resampling inference pressure
Slide 17: How a permutation test scales
Slide 18: High-dimensional testing: one label shuffle, many features
Slide 19: Statistical algebra changes the kernel, not the statistic
Slide 20: Equivalence check: loop reference vs matrix formulation
Slide 21: Null calibration gate: type-I error stays near alpha
Slide 22: Local validation scale: methods are close enough
Slide 23: When does GPU help for permutation tests?
Slide 24: CPU parallelism is also a tuning parameter

Purpose: move from loop reference to matrix formulation, then through correctness and calibration before runtime. The MacBook validation-scale slide explains why readability dominates until the bottleneck is real; the A100 decision map answers where the streamed GPU path helps, and the CPU parallelism slide keeps the main path on practical scaling choices.

## 24:00-27:30 - Tool roles and AI/Codex

Slide 25: What each tool did in this talk
Slide 26: Codex can generate variants; the statistician owns the claims

Purpose: explain that NumPy/BLAS, Numba, threads/Python 3.14, and JAX (A100) solve different bottlenecks. AI tooling helps with experiment hygiene; the statistician owns the scientific claims.

## 27:30-30:00 - Decision guide and close

Slide 27: From simulation evidence to the next tool
Slide 28: Make the statistic testable. Then make the bottleneck fast.
Slide 29: Thank you

Purpose: leave the audience with a simulation-result-driven tool-selection rule and a clear close.

## Backup

Slide 30: Backup: what batch_R tuning showed
Use if someone asks about batch_R. Explain the trend: small batches were inefficient, larger batches amortized overhead, returns flattened, and 8192 was the fastest accepted value in the measured 128-8192 grid--not a global optimum.
Slide 31: Backup: A100 full-stage decomposition
Use if someone asks where full A100 scenario time goes; scope it as two committed representative rows, not a full shape grid.
Slide 32: Evidence map: each environment answers a different question
Slide 33: Backup: local permutation validation inventory
Slide 34: Backup: Numba wins when k-means stays loop-shaped
Use if someone asks when Numba beats the NumPy matrix path; scope it as local MacBook loop-shaped evidence, not a universal ranking.
Slide 35: Power increases with effect size
Slide 36: What the coding agent automated
