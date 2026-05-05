# Timeline v3 - 30 minute talk

This timeline follows `docs/replanned_timeline_and_codex_instructions.md`: laptop simulation defines trust, server CPU tests scale, and A100 is an evidence ladder rung, not a leaderboard trophy.

The timed path is 25 non-blank main slides. Five lower-priority evidence slides remain after the close as backup.

## 0:00-1:00 - Title and thesis

Slide 1: Breaking the Speed Limit

Purpose: establish the thesis. Simulation is how statisticians write tests; a faster implementation that changes the statistic is not an optimization.

## 1:00-7:00 - Simulation-driven statistical computing

Slide 2: Simulation-driven statistical computing is statistical CI  
Slide 3: Simulation is how statisticians write tests  
Slide 4: Six steps before optimization is allowed  
Slide 5: Three ways statistical software fails  
Slide 6: The environment ladder is not a leaderboard  
Slide 7: Each tier answers a different question

Purpose: build the PyCon/programmer bridge and explain why MacBook, server CPU, and A100 evidence should not be collapsed into one ranking.

## 7:00-8:30 - Benchmark contract and workloads

Slide 8: Measurement contract  
Slide 9: Two workloads, two pressure points

Purpose: define the benchmark contract, metadata, and the two statistical workloads before showing results.

## 8:30-15:30 - Workload 1: k-means

Slide 10: k-means as iterative simulation pressure  
Slide 11: Simulation grid reveals hard cases  
Slide 12: Optimized paths must preserve the reference  
Slide 13: Runtime is useful only after recovery is visible  
Slide 14: Server k-means: A100 wins only after enough work exists  
Slide 15: The right k-means implementation depends on shape

Purpose: show that recovery and reference preservation come before runtime, then use server/A100 evidence as conditional scale evidence.

## 15:30-23:30 - Workload 2: permutation tests

Slide 16: Permutation tests as resampling pressure  
Slide 17: Same permutations, different formulation  
Slide 18: Permutation correctness: equivalence and null calibration  
Slide 19: Server permutation: scale evidence includes negative results  
Slide 20: Why the A100 permutation result stays in the deck  
Slide 21: Parallelism is a tuning parameter, not a moral victory

Purpose: move from loop reference to matrix formulation, correctness/calibration, server scale, and a validated negative A100 result.

## 23:30-26:30 - Python 3.14, parallelism, and AI/Codex

Slide 22: Where Python 3.14 fits  
Slide 23: AI/Codex helps generate variants, not truth

Purpose: explain runtime ceilings, GIL/JIT availability logging, parallelism as a tuning parameter, and the boundary between Codex-generated variants and human-owned statistical contracts.

## 26:30-30:00 - Decision guide and close

Slide 24: Decision guide  
Slide 25: Choose the smallest tool...

Purpose: leave the audience with a practical tool-selection rule: choose the smallest tool that preserves the statistic and removes the proven bottleneck.

## Backup

Backup slide 1: Backup evidence divider  
Backup slide 2: Shape stress: K and d become the bottleneck  
Backup slide 3: Power rises with effect size, not wishful thinking  
Backup slide 4: Runtime scaling follows the computational shape  
Backup slide 5: What the agent actually changed
