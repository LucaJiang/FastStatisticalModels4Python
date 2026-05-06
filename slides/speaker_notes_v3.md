# Speaker Notes - Revised Main Deck

Target: 30 minutes. Main path is slides 1-29. Slides 30-33 are backup. Slides 9 and 16 are short method-transition video slides.

## Timing overview

- 0:00-1:00 - Title and thesis
- 1:00-8:00 - Simulation-driven statistical computing
- 8:00-16:00 - k-means: iterative model-fitting pressure
- 16:00-24:00 - permutation testing: resampling inference pressure, local validation scale, GPU map/decomposition, and parallelism
- 24:00-27:30 - tool roles and AI/Codex
- 27:30-30:00 - decision guide, close, and questions

---

## Slide 1 - Breaking the Speed Limit

Open as a statistician's workflow, not a library benchmark. Establish: simulation first, validation before speed, Python as the scientific interface, and different tools for different bottlenecks.

## Slide 2 - For statisticians, speed is useful only after trust

Use the rule on the slide exactly: a speedup only counts if the optimized code answers the same statistical question. Add verbally that same data, same statistic, same stopping rule, and same result within tolerance are what make a benchmark claim meaningful.

## Slide 3 - Simulation creates an answer key

Explain that real biomedical ground truth is often unavailable. The target behavior is what should be recovered, estimated, or calibrated; the scenario grid then turns that into validation checks and bottleneck discovery.

## Slide 4 - Statistical CI

Do not read every row. Use two examples: fixed-seed equivalence behaves like a unit test, while calibration or recovery behaves like a property test with statistical acceptance criteria.

## Slide 5 - Why Python

Python remains the control plane. NumPy, Numba, Python 3.14 threads, and JAX/A100 are ways to move a measured hotspot while preserving the validation target.

## Slide 6 - Why these two examples

k-means represents iterative model fitting: assign, update centroids, repeat. Permutation tests represent resampling inference: shuffle labels, compute a statistic, repeat across many features.

## Slide 7 - Evidence ladder

MacBook answers "can I trust it?", server CPU answers "what scales?", and A100 answers "did we find an accelerator-shaped pipeline?" Mention synchronized GPU timing, warm medians, memory, and explicit unavailable rows.

## Slide 8 - k-means section title

Transition to the first concrete workload. k-means is simple enough that both statistical recovery and performance pressure are visible.

## Slide 9 - How k-means moves

We use petal length and petal width from Iris only as a visual method example. The species labels help the audience understand that there are three biological groups, but k-means only sees the two numerical measurements. Setosa separates easily; Versicolor and Virginica overlap, which makes the iteration more interesting than three clean synthetic blobs.

Use the animation to emphasize the dependency: assign each flower to the nearest centroid, update centroids from assigned flowers, then repeat until assignments stabilize. Iteration t+1 depends on centroids from iteration t. Keep the tool explanation verbal here: that dependency is why explicit loops, temporary arrays, and distance computation connect naturally to Numba, NumPy matmul, and JAX/A100 later in the talk.

## Slide 10 - Why k-means is a useful test case

k-means is not the most advanced model, but it stands in for iterative algorithms such as EM, coordinate descent, and simulation-based estimators. The point is to separate statistical behavior from implementation performance: hold the data, initial centroids, stopping rule, and final inertia comparison steady before asking which Python implementation is faster.

## Slide 11 - Recovery surfaces

Define ARI: 1 is perfect recovery; near 0 is random-like. The takeaway is not that k-means is bad, but that speed is not meaningful where recovery is poor.

## Slide 12 - Before timing: compare against the reference

Explain why same initialization matters: k-means can legitimately differ when initialization changes. The validation logic is same data, same initial centroids, same stopping rule, then compare final inertia against the readable reference before timing optimized versions. If needed, mention the bookkeeping verbally: 3,840 checked rows, 480 expected memory-risk skips, and 0 optimized failures. Memory-risk skips are expected skips from unsafe reference-broadcast rows, not hidden failures.

## Slide 13 - Server k-means

This is a representative-shape comparison, not a universal ranking. Numba, BLAS, and A100 each match different implementation shapes. A100 is not automatically faster, and high dimension alone does not make a GPU win.

Explain the tool relationship: Numba helps when explicit loops dominate. BLAS helps when the distance computation can be rewritten as dense matrix algebra. A100 helps only when the work is large, regular, and device-friendly. These are representative measured rows; high dimension can increase memory traffic and reduce apparent GPU advantage.

## Slide 14 - k-means takeaway

Use this as a decision guide for iterative algorithms: fix untrustworthy results, compile scalar loops with Numba, rewrite dense distance algebra for BLAS, and use A100 only after large regular work exists.

## Slide 15 - Permutation section title

Transition from iterative dependence to repeated simulation.

## Slide 16 - How a permutation test scales

Walk through the animation in order: observed X matrix, group labels, one shuffled label vector, feature-wise group differences, repeated null-statistic vectors, then the later implementation view W @ X. The ordinary resampling loop comes first. Threads/processes help distribute independent repetitions, Numba can compile CPU kernels, and JAX/A100 matters only when the repeated work becomes large, batched, and device-friendly.

## Slide 17 - High-dimensional testing

Use this slide to make the computational size explicit: the input is n samples by p features, then R shuffled label vectors create R by p potential null statistics. In practice the optimized path can stream this down to p exceedance counts, but the work still scales with the repetition. Examples: samples are patients, features are genes, biomarkers, or voxels, and R is the number of null draws.

## Slide 18 - Same permutation stream

The optimized path does not change the statistical question. It avoids materializing the full R × p null matrix by streaming exceedance counts.

## Slide 19 - Equivalence check

Same simulated data, same permutation stream, same p-value definition. The recorded max p-value difference is 0.0, max statistic difference is 9.4e-16, with 45 expected memory-risk skips and 0 failed rows.

## Slide 20 - Null calibration

Equivalence checks implementation; calibration checks statistical behavior. Under the null, type-I error should stay near alpha = 0.05. The observed estimate is 0.051.

## Slide 21 - Local validation scale

This is the moved MacBook validation-scale result. NumPy matrix, batched NumPy matrix, and JAX CPU are close enough that readability and correctness should dominate until a real bottleneck appears.

## Slide 22 - GPU decision map

Explain that A100 helps only when the permutation pipeline becomes large, batched, and device-resident enough. Compile is excluded, transfer included, and kernel-only timing is not used for the decision map.

Two high-R cells at p=500,000 were rerun on May 6, 2026 with the CPU timeout raised to 14,400 seconds per cell. The CPU baselines completed, but the canonical A100 streamed end-to-end run at batch_R=8,192 still failed during JAX autotune/OOM, even with `TF_GPU_ALLOCATOR=cuda_malloc_async`. Those cells are labeled A100 OOM/unavailable in the evidence CSVs and should not be interpreted as CPU wins or hidden speedup estimates.

## Slide 23 - A100 pipeline decomposition

Show where full-scenario A100 time goes. A fast kernel is not enough; the full pipeline decides. The four rows are representative shapes from the canonical break-even grid: CPU-faster, near break-even, A100-faster, and largest/highest-speedup measured.

Say the timing semantics precisely: this is full scenario timing, compile excluded, transfer included, streamed reduction, and no kernel-only comparison. The named stages are reconciled to the recorded total with an explicit other-overhead segment, and each row reports total time plus the W @ X share.

## Slide 24 - CPU parallelism

Main message: CPU parallelism must be measured under resource constraints.

This is Linux server CPU evidence, not MacBook evidence. The expanded sweep shows 1, 4, 16, 64, and 128 workers or threads.

For k-means, Numba kept improving through 128 in this shared-server run. For permutation, runtime improved through 16 workers, then got worse at 64 and 128 while memory climbed to about 58 GiB at 128 workers.

The 128-worker point is only cleanly interpretable if the process had access to 128 CPUs and the server was not heavily loaded. Here affinity exposed 512 CPUs, but there was no exclusive scheduler allocation, so the high-count rows are marked shared-server evidence. They may reflect shared-server contention, NUMA placement, memory bandwidth, scheduler effects, or nested thread conflicts.

## Slide 25 - Tool roles

Use concrete examples: Numba for k-means assignment/update loops, NumPy/BLAS for distance algebra, threads/Python 3.14 for repeated work over shared arrays, and JAX/A100 for streamed W @ X after validation.

## Slide 26 - AI/Codex

Codex can automate implementation variants, scenario-grid runners, environment metadata, plot regeneration, and result manifests. The statistician owns the statistical target, null model, validation criteria, interpretation, and scientific claims.

## Slide 27 - Decision guide

Tie each branch back to simulation evidence: untrustworthy result, hot scalar loop, dense matrix identity, repeated shared-array work, large device-resident batch, or giant temporary.

## Slide 28 - Close

Close with: make the statistic testable, then make the bottleneck fast. Repeat: clear enough to trust, fast enough to scale.

## Slide 29 - Thank you

Leave the repo link visible and invite questions.

---

# Backup Slides

## Slide 30 - Evidence map

Use if someone asks how the validation, scale, and accelerator tiers differ.

## Slide 31 - Shape stress

Use if someone asks for definitions of N, d, or K, or for backup evidence about k-means shape pressure.

## Slide 32 - Power curve

Use if someone wants statistical validation beyond null calibration.

## Slide 33 - What the coding agent automated

Use if someone asks about the repository or Codex workflow.
