# Speaker Notes - Revised Main Deck

Target: 30 minutes. Current structure is 36 slides total: main path is slides 1-29, slides 30-36 are the 7 backup slides, and slides 9 and 17 are short method-transition video slides.

## Timing overview

- 0:00-1:00 - Title and thesis
- 1:00-8:00 - Simulation-driven statistical computing
- 8:00-16:00 - k-means: iterative assignment-update pressure
- 16:00-24:00 - permutation testing: resampling inference pressure, local validation scale, GPU map, and parallelism
- 24:00-27:30 - tool roles and AI/Codex
- 27:30-30:00 - decision guide, close, and questions

---

## Slide 1 - Breaking the Speed Limit

Open as a statistician's workflow, not a library benchmark. Establish the thesis: make the statistic testable, then speed up the bottleneck. Simulation creates controlled behavior and answer keys; it does not define the scientific target by itself.

## Slide 2 - For statisticians, speed comes after validation

Use the rule on the slide exactly: speed only counts when the statistic is preserved. Add verbally that same data-generating setup, same statistic, same stopping rule when applicable, and same result within tolerance are what make a benchmark claim meaningful. Reference code makes the target executable; it does not define the scientific target.

## Slide 3 - Simulation creates an answer key

Explain that real biomedical ground truth is often unavailable. The statistical target comes first: what should be estimated, recovered, preserved, or calibrated. Simulation scenarios create controlled behavior and answer keys; the reference implementation makes the target executable; validation checks decide whether optimized versions preserve it.

## Slide 4 - Simulation turns statistical behavior into tests

Do not read every row. Use two examples: fixed-seed equivalence behaves like a unit test, while calibration or recovery behaves like a property test with statistical acceptance criteria.

## Slide 5 - Why Python

Python remains the control plane. NumPy, Numba, threads/workers, and JAX / A100 are ways to move a measured hotspot while preserving the validation target.

## Slide 6 - Why these two examples

k-means represents iterative model fitting: assign, update centroids, repeat. Permutation tests represent resampling inference: shuffle labels, compute a statistic, repeat across many features.

## Slide 7 - Evidence ladder

MacBook answers "can I trust the statistic and implementation?", server CPU answers "what scales with larger shapes and parallelism?", and A100 answers "does the validated pipeline fit the accelerator?" Emphasize that timeouts, OOMs, and skipped rows stay explicit. This is not a hardware leaderboard.

## Slide 8 - k-means section title

Transition to the first concrete workload. k-means is simple enough that both statistical recovery and assignment-update pressure are visible. The contract is fixed while geometry and scale vary.

## Slide 9 - How k-means moves

We use Iris petal length and petal width as a visual method example. The image inset reminds the audience what petal and sepal refer to biologically, but k-means only sees petal length and petal width. Species labels help the audience understand the context, but they are not model inputs. The animation starts at iteration 0 with fixed pedagogical starting centroids that are not one-per-final-cluster, so the movement is visible. Lloyd iterations alternate assignment and update steps until assignments stop changing.

## Slide 10 - Why k-means is a useful test case

k-means is not the most advanced model, but it is a useful test case because the statistical and computational pressures are easy to see. The validation contract is fixed: same data, same initial centroids, same stopping rule, and comparable final inertia. The algorithm also becomes harder under overlap, imbalance, outliers, and higher dimension. Computationally, the repeated distance work scales with N times K times d every iteration. That lets us explain bottleneck shapes without turning Numba, NumPy, and JAX / A100 into a ranking.

## Slide 11 - Recovery surfaces

This slide is about statistical recovery, not runtime. Because the data are simulated, we know the true generating group for each point. ARI summarizes agreement between fitted clusters and simulated groups. High ARI means k-means recovered the simulated grouping; low ARI means the scenario is statistically hard for k-means. Low ARI is not an implementation bug by itself. Read this as a finite scenario difficulty summary, not a monotone law: the plotted clean high-d row at separation 4 and the outlier row values were verified from the source medians. Poor recovery means speed is not evidence of success.

## Slide 12 - Before timing: compare against the reference

Explain why same initialization matters: k-means can legitimately differ when initialization changes. The validation logic is same data, same initial centroids, same stopping rule, then compare final inertia against the readable reference as the executable oracle for the fixed-init comparison before timing optimized versions. If needed, mention the bookkeeping quietly: 3,840 checked rows, 480 expected memory-risk skips, and 0 optimized failures. Memory-risk skips are expected skips from unsafe reference-broadcast rows, not hidden failures.

## Slide 13 - The same distance step can be expressed three ways

This slide is not about teaching full k-means syntax. It is here to make the benchmark interpretable. Numba keeps the assignment step close to explicit loops and removes Python loop overhead. NumPy rewrites the distance computation as dense matrix algebra; the code is NumPy, while the heavy matrix multiply is BLAS-backed underneath. JAX / A100 uses nearly the same array expression on the accelerator, but it is only useful after validation and enough scale. The validation contract is unchanged: same data, same initial centroids, same stopping rule, and comparable final inertia.

## Slide 14 - Server k-means

This is a representative-shape comparison, not a universal ranking or hardware leaderboard. These are validated warm rows. The slide labels should describe what the statistician writes: Numba, NumPy, and JAX / A100. NumPy is the user-facing code path; for dense distance algebra, its matrix multiply is BLAS-backed. JAX / A100 means the JAX implementation running on A100 hardware. Numba can be slower than NumPy on dense algebra because explicit compiled loops do not automatically become a highly optimized GEMM. BLAS uses cache blocking, SIMD, threading, and mature matrix-multiply kernels. Numba is still useful when the bottleneck is loop-shaped or when avoiding giant temporary arrays matters. The point is shape matching: Numba for explicit loops, NumPy for dense algebra, and JAX / A100 for large regular device-friendly work.

## Slide 15 - k-means takeaway

Use this as a decision guide for iterative algorithms: fix poor recovery first, compile scalar loops with Numba, rewrite dense distance algebra for NumPy, and use JAX / A100 only after large regular array work exists. NumPy is the user-facing label; its dense matrix multiply is BLAS-backed underneath.

## Slide 16 - Permutation section title

Transition from iterative assignment-update dependence to repeated null resampling across many features.

## Slide 17 - How a permutation test scales

Keep this as the method slide. Walk through the ordinary resampling loop: shuffle labels, compute feature-wise differences, repeat to build the null. The main point is that independent repetitions still create real work. Save W @ X, streamed counts, and accelerator details for the following implementation and scale slides.

## Slide 18 - High-dimensional testing

Use this slide to make the computational size explicit: the input is n samples by p features, then R shuffled label vectors create R by p potential null statistics. The full matrix does not have to be materialized; the optimized path can stream results down to p counts. The work still scales like R times p. Examples: samples are patients, features are genes, biomarkers, or voxels, and R is the number of null draws.

## Slide 19 - Same permutation stream

The optimized path does not change the statistical question: same simulated data, same permutation stream, and same p-value definition. The shape chips are W_batch = batch_R by n, X = n by p, and T = batch_R by p. The streamed path avoids materializing the full R by p null matrix.

## Slide 20 - Equivalence check

Same simulated data, same permutation stream, same p-value definition. The recorded max p-value difference is 0.0 and max statistic difference is 9.4e-16 across 18 workloads times 5 seeds, with 45 expected memory-risk skips and 0 failed rows. This lets the optimized path enter runtime comparison.

## Slide 21 - Null calibration

This slide checks statistical behavior under the null. Equivalence is necessary but not enough; p-values must also be calibrated. The observed type-I error estimate is approximately 0.051 in this local null calibration check, near alpha = 0.05. This is still validation, not performance, and not a universal calibration guarantee. The visual band is feature-level binomial variation per null replicate, `0.05 +/- 1.96 * sqrt(0.05 * 0.95 / 1000)`, not a confidence interval for the mean across replicates. Equivalence checks code; calibration checks inference.

## Slide 22 - Local scale signal

Now that equivalence and calibration have passed locally, the question changes. This plot is MacBook Air local-tier evidence, not server or A100 evidence. The plotted implementation is the validated batched NumPy matrix path. n is fixed at 500; p and R vary, and warm median runtime is shown. The goal is not to compare tools here. The goal is to show that the local validated computation has a bottleneck shape. The next slides ask whether server CPU or A100 handles that shape better. Do not compare these MacBook runtimes directly with server or A100 speedups.

## Slide 23 - GPU decision map

This slide answers where A100 wins. Speedup means matched CPU matrix baseline divided by A100 streamed full end-to-end. Compile is excluded, transfer is included, and kernel-only timing is excluded. Break-even starts at the smallest-p A100 win shown here: p=10k, R=5k, batch_R=8192. Briefly mention that batch_R=8192 came from the measured batch-size sweep; the tuning detail is now in backup for Q&A. Largest slide-level measured speedup: 8.54x at n=5,000, p=500,000, R=5,000. The two high-R p=500k cells are A100 OOM/unavailable, not CPU wins. Some speedups are lightweight slide-level summaries; raw CPU/A100 timing pairs are not committed for every cell.

## Slide 24 - CPU parallelism

- This slide is about diminishing returns.
- Total time went down when we added parallelism, especially early.
- But each jump gave less benefit than the previous jump.
- For permutation workers, high counts also increased memory substantially.
- The practical rule is to use enough parallelism to capture most of the speedup.
- Keep the resource-isolation caveat in notes: these high-count rows were measured on a shared server, so do not treat exact 64/128 behavior as a universal scaling law.

## Slide 25 - Tool roles

Keep this as a tool-roles slide, not another decision guide. Numba solved explicit CPU loops in k-means assignment/update. NumPy / BLAS solved dense matrix algebra: the k-means distance identity and W @ X on CPU. Threads/workers handled repeated work over shared data in the permutation worker sweep. JAX / A100 solved large regular device-resident batches for streamed W @ X after validation. What not to infer: no tool is universally faster; each one matched a measured bottleneck shape.

## Slide 26 - AI/Codex

Codex is useful for multiplying workflow variants: implementation variants, scenario grids, metadata capture, plots, and manifests. But the scientific responsibility does not move to the model. The statistician still owns the statistical target, data-generating assumptions, validation criteria, hard-case interpretation, and claims.

## Slide 27 - Decision guide

This slide is the synthesis of the talk: simulate, validate the result, diagnose the bottleneck, then pick the smallest tool or strategy that addresses it. The audience does not need every system detail here; the slide should read in about ten seconds as a summary decision guide.

This slide is not saying every validation failure is a bug. A failed recovery or calibration check may mean the target, assumptions, or scenario need work before optimization. Once validation passes, the bottleneck pattern determines the next move: Numba for hot loops, BLAS for dense algebra, tuned workers for repeated shared-data work, JAX / A100 for large device-resident batches, and rewrite/streaming for memory blow-ups.

The longer rule is: validate the result; then move the bottleneck.

## Slide 28 - Close

Close with the two lines on the slide: make the statistic testable, then make the bottleneck fast. Clear enough to validate. Fast enough to scale.

## Slide 29 - Thank you

Leave the QR code and repo link visible. Label it as slides plus reproducible artifacts and invite questions.

---

# Backup Slides

## Slide 30 - Evidence map

Use if someone asks why the validation, scale, and accelerator tiers differ.

MacBook Air answers validation and local shape questions; server CPU answers scaling and parallelism questions; A100 answers whether a validated accelerator-shaped pipeline wins.

The A100 permutation phrase should point to streamed-reduction break-even evidence, not an earlier follow-up. Detailed row/status bookkeeping remains in result READMEs and speaker notes rather than the slide visual hierarchy.

## Slide 31 - Local permutation validation inventory

Use if someone asks what local permutation checks were run before scaling.

This combines the main-path equivalence, null calibration, and runtime-shape slides into one backup inventory.

Equivalence checks code; calibration checks inference; runtime shape motivates scale evidence.

## Slide 32 - Shape stress

Use if someone asks when Numba wins for k-means.

This backup slide is local MacBook shape-stress evidence. It does not contradict the server representative-shape slide. Here the comparison is Numba versus a NumPy matrix-style path on validation-scale shapes where explicit loops and temporary-array overhead can dominate, so Numba can be faster. The audit detail is 1,164 shape-stress rows, all passing. On larger server shapes, dense distance algebra can favor NumPy / BLAS, and sufficiently large regular workloads can favor JAX / A100. The conclusion is shape-dependent tool choice, not a universal ranking.

## Slide 33 - Power curve

Use if someone wants statistical validation beyond null calibration.

This is a local simulated power curve, not a speed result. Weak effects have low power; stronger simulated effects reach high power in this local curve.

## Slide 34 - Backup: why batch_R=8192?

Use this if someone asks why the A100 decision map used batch_R=8192.

This is a tuning detail for Q&A, not part of the main statistical narrative.

The committed sweep shows 8192 as the best measured batch_R for the decision-map setting; phrase it as a measured tuning choice, not a new statistical method.

The right cards explain that batching changes scheduling, not the statistic or p-value definition.

Full end-to-end timing remains the decision criterion: compile excluded, transfer included, kernel-only excluded.

## Slide 35 - A100 stage accounting

Use this only if someone asks how the full-stage accounting reconciles.

This is mechanism evidence only. It explains why full-scenario timing matters and why W @ X alone is not the whole claim.

The committed representative summary is limited, so do not use this slide to set the A100 decision boundary. The canonical decision boundary remains the break-even map based on matched CPU matrix baseline divided by A100 streamed full end-to-end timing.

## Slide 36 - What the coding agent automated

Use if someone asks about the repository or Codex workflow.

This is backup detail for the AI/Codex slide. Automation widened implementation variants, scenario grids, figures, manifests, and QA artifacts, while the statistician retained ownership of target, assumptions, validation criteria, interpretation, and claims.
