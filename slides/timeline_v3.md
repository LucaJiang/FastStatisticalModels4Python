# Timeline v3 - 30 minute talk

This timeline matches `slides/index.html`: 32 total slides, 27 main-path slides, and 5 backup slides. The main path intentionally keeps 27 slides because the k-means and permutation video slides are short method-transition slides.

The narrative is an evidence ladder, not a hardware leaderboard: laptop simulation defines trust, server CPU tests scale, and A100 is useful only when the validated statistical computation has an accelerator-shaped pipeline.

## 0:00-1:00 - Title and thesis

Slide 1: Breaking the Speed Limit

Purpose: establish the thesis. Simulation is how statisticians write tests; a faster implementation that changes the statistic is not an optimization.

## 1:00-8:00 - Simulation-driven statistical computing

Slide 2: For statisticians, speed is useful only after trust  
Slide 3: Simulation creates an answer key  
Slide 4: Statistical CI  
Slide 5: Python stays the scientific interface  
Slide 6: Two workloads, two statistical computing shapes  
Slide 7: Evidence ladder

Purpose: build the PyCon/programmer bridge, explain why Python remains the interface, and define the trust-before-speed measurement contract.

## 8:00-16:00 - Workload 1: k-means

Slide 8: Workload 1 - k-means as iterative model-fitting pressure  
Slide 9: How k-means moves  
Slide 10: Why k-means works here  
Slide 11: Recovery comes before runtime  
Slide 12: First prove optimized paths preserve the reference  
Slide 13: Server k-means: acceleration is shape-dependent  
Slide 14: k-means takeaway

Purpose: show iterative model-fitting pressure: assignment/update dependence, recovery before runtime, reference equivalence before optimization, and conditional CPU/GPU scale behavior.

## 16:00-24:00 - Workload 2: permutation tests

Slide 15: Workload 2 - permutation tests as resampling inference pressure  
Slide 16: How a permutation test scales  
Slide 17: Why permutation tests work here  
Slide 18: Statistical algebra changes the kernel, not the statistic  
Slide 19: First, prove the matrix path matches the reference  
Slide 20: Null calibration gate: type-I error stays near alpha  
Slide 21: When does GPU help for permutation tests?  
Slide 22: Where does A100 time go?  
Slide 23: CPU parallelism is also a tuning problem

Purpose: move from loop reference to matrix formulation, then through two correctness gates before runtime. The A100 story is now explicitly reconciled: the old matched slice was negative, and the follow-up changed the pipeline with streamed reduction, larger `batch_R`, and a broader shape sweep while preserving the statistic.

## 24:00-27:30 - Tool roles and AI/Codex

Slide 24: Python 3.14, Numba, and JAX solve different parts of the workflow  
Slide 25: Codex can generate variants; the statistician owns the contract

Purpose: explain that Python 3.14, Numba, threads/processes, JAX, and A100 solve different bottlenecks. AI tooling helps with variants and experiment hygiene; the statistician owns the statistical contract.

## 27:30-30:00 - Decision guide and close

Slide 26: Choose the smallest tool that preserves the statistic  
Slide 27: Make the statistic testable. Then make the bottleneck fast.

Purpose: leave the audience with a practical tool-selection rule: preserve the statistic first, then accelerate the proven bottleneck.

## Backup

Slide 28: Evidence map: keep tiers separate  
Slide 29: Shape stress: K and d become the bottleneck  
Slide 30: Power increases with effect size  
Slide 31: MacBook-only permutation runtime follows computational shape  
Slide 32: What the coding agent automated
