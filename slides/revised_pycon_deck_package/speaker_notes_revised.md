# Speaker Notes - Revised Main Deck

Target: 30 minutes. Main path is slides 1-25. Slides 26-30 are backup.

## Timing overview

- 0:00-1:00 - Title and thesis
- 1:00-8:00 - Why a statistician starts with simulation, and why Python fits this workflow
- 8:00-9:00 - Evidence ladder and measurement contract
- 9:00-16:00 - k-means: iterative simulation pressure
- 16:00-24:00 - permutation testing: resampling pressure and matrix reformulation
- 24:00-27:30 - parallelism, Python 3.14, Numba, JAX, and Codex
- 27:30-30:00 - decision guide and close

---

## Slide 1 - Breaking the Speed Limit

Open with the key framing: this is not a speed contest and not a tool ranking. The phrase to establish immediately is: clear enough to trust, fast enough to scale.

Say that you are speaking as a biostatistician who uses Python to move from mathematical/statistical ideas to executable experiments. You care about performance, but only after you know what computation you are trying to preserve.

Do not spend time on the names of every tool yet. Say that Python 3.14, Numba, and JAX will appear as answers to different bottlenecks, not as a universal ranking.

## Slide 2 - From a statistician's point of view, speed is only useful after trust

This slide directly answers why the talk starts from the statistician's point of view.

Emphasize that in statistical work, the method is often still being understood while the code is being written. Simulation helps reveal which assumptions matter, which scenarios fail, and which outputs should be stable.

Explain that readable Python or NumPy is not merely a slow first draft. It is the executable statistical definition. Optimized kernels are allowed into the benchmark only after they match that definition.

Say the rule clearly: a faster implementation that changes the statistic is a different method.

## Slide 3 - Simulation creates the answer key

Use this slide to explain why simulation is not fake data or toy data. It creates an answer key when real data cannot.

Examples:

- For k-means, the simulated answer key is the true cluster assignment.
- For permutation tests, the simulated answer key is the expected behavior under the null, especially p-value calibration.
- For systems behavior, the simulation grid tells us which rows are memory-risk skips rather than failures.

The four-step workflow is the short version of the whole talk: reference, scenarios, validate, optimize.

## Slide 4 - Statistical CI

Translate the workflow into language familiar to PyCon programmers.

Do not read every row. Pick two examples:

- Unit tests become fixed-seed reference equivalence.
- Load tests become increasing N, d, K, p, R, workers, and batch sizes.

Then say the important difference: the oracle is not always one exact scalar. Sometimes the oracle is calibration, recovery, invariance, or a tolerance bound.

## Slide 5 - Why Python

This is where the abstract and the deck connect.

The message is: Python remains the scientific interface. We keep code close enough to the math to inspect, then move hotspots into the appropriate engine.

Use the table to show the tool roles:

- Python/NumPy reference defines the statistic.
- NumPy/BLAS handles vectorized algebra when the arrays have the right shape.
- Numba handles explicit scalar CPU loops.
- Threads and free-threaded Python matter for repeated work over shared arrays.
- JAX/A100 matters after a statistic has become a large batched array program.

This slide should prevent the audience from thinking you are saying one library is always best.

## Slide 6 - Why these two examples

This slide is the main content fix.

Say: I chose k-means and permutation tests not because they are the most advanced statistical methods, but because they represent two common computational shapes.

k-means represents iterative model fitting. Each iteration depends on the previous centroids. That pattern appears in many algorithms: EM, coordinate descent, iterative optimization, simulation-based estimators.

Permutation testing represents resampling inference. The basic statistic is simple, but we repeat it many times, often across many features. That pattern appears in high-dimensional biostatistics, genomics, imaging, and biomarker analysis.

Then connect explicitly to tools:

- k-means gives us Numba loops, NumPy distance algebra, and JAX/A100 only after enough regular work exists.
- permutation tests give us threads/processes, Numba CPU kernels, and the JAX/A100 matrix formulation `W @ X`.

## Slide 7 - Evidence ladder

Explain that MacBook, server CPU, and A100 are not a hardware leaderboard. They answer different questions.

Laptop: what can we trust? Server CPU: what scales? A100: did we find an accelerator-shaped computation?

State the measurement contract briefly. The most important points are:

- correctness before runtime,
- warm median plus cold first call,
- memory including child RSS,
- unavailable rows are explicit,
- JAX/GPU timings must be synchronized.

## Slide 8 - k-means section title

Transition: now that the workflow is clear, use k-means as the first concrete case.

Say that k-means is deliberately simple: assignment, update, repeat. That simplicity makes it a good demonstration because failures and performance changes are easy to inspect.

## Slide 9 - Why k-means works here

Make the k-means-to-tools connection explicit.

Statistical target: recover known clusters and preserve fixed-init inertia.

Failure modes: imbalance, outliers, low separation, high dimension, initialization, empty clusters.

Python pressure: loops, temporaries, distance algebra, compiled kernels.

Do not read the whole table. Use one row as an example: the readable NumPy implementation defines the target, and Numba or matmul variants must match it before they can be compared on speed.

## Slide 10 - Recovery surfaces

Define ARI: adjusted Rand index; 1 means perfect recovery, around 0 means random-like recovery.

State the pattern: balanced clusters recover well when separation and dimension are favorable; imbalanced clusters are much harder.

The message is not that k-means is bad. The message is that a runtime chart alone would hide the statistical difficulty.

## Slide 11 - Reference gate

Explain why same initialization matters. k-means can legitimately produce different results if initialization changes, so fixed initialization is part of the contract.

Explain the memory-risk skips. The reference broadcast implementation is readable but can create an unsafe O(N*K*d) temporary. Skipping those rows is honest; forcing them to run would turn validation into a memory stress test.

## Slide 12 - Server k-means

This is the scale result.

The laptop established trust; the server asks what happens at larger N and different shapes.

State the observed pattern rather than reading the chart: Numba wins many CPU loop shapes; BLAS helps a different shape; A100 wins only when enough regular work exists.

Mention that GPU advantage is conditional. Do not imply high dimension automatically means GPU wins.

## Slide 13 - k-means takeaway

Use this as a short decision guide for iterative algorithms.

If the reference is unstable, stop and validate. If scalar loops dominate, try Numba. If the distance computation becomes dense algebra, try NumPy matmul. If the workload is huge and regular, try JAX/A100. If recovery is poor, the method or scenario needs attention before runtime.

Close the k-means section by saying it stands in for many iterative statistical algorithms.

## Slide 14 - Permutation section title

Transition from iterative dependence to repeated simulation.

Say: the second workload is different on purpose. The statistic is simple, but resampling makes the workload large.

## Slide 15 - Why permutation tests work here

This slide should make the biostatistics connection clear.

Use a concrete mental model: X is samples by features. Each feature could be a gene, biomarker, voxel, or measurement. We test group differences and use label permutations to approximate the null distribution.

The programming challenge is not just parallelism. It includes random index generation, shared arrays, process copies, calibration, batching, dtype, device transfer, and memory.

## Slide 16 - Same statistic, different formulation

Use this as the key code slide.

The reference loop is readable and close to the definition: make permutations, compute one statistic per permutation.

The matrix formulation uses the same permutations but encodes them as contrast rows in W, so the null statistics are computed as `W @ X`.

Say clearly: same permutations matter. Otherwise differences could be random-number differences rather than implementation differences.

## Slide 17 - Equivalence gate

Explain that the matrix path is not accepted because it looks clever. It is accepted only after it matches the reference.

State the result: 450 pass rows, 45 expected memory-risk skips, no failed rows. P-value differences are zero in this run; statistic differences are at floating-point scale.

## Slide 18 - Null calibration

Explain that equivalence against a reference is necessary but not enough. The reference itself must behave statistically.

Under the null, p-values should be calibrated. The mean replicate-level proportion with p <= 0.05 is 0.051, close to the nominal 0.05.

This is the moment where the server benchmark becomes meaningful: the statistic and null simulation have been checked.

## Slide 19 - Server permutation negative GPU result

State the result honestly. CPU cost grows quickly with R and p. The A100 matrix path is validated but slower than CPU on the matched slice.

Frame this as success of the workflow, not failure of the talk. The workflow prevented you from telling a false GPU story.

## Slide 20 - Negative result is evidence

Use this slide to explain what the negative result tells us.

The statistic passed. The speed claim failed. The next question is more precise: W construction, transfer, batch size, dtype, or matrix multiplication.

Be explicit that the timing-layer table is a next-bottleneck diagram unless measured decomposition exists. Do not imply measured decomposition if it has not been run.

## Slide 21 - Parallelism

Main message: more workers is not automatically better.

For k-means, Numba improves up to the middle of the thread sweep, but maximum thread count is not best. For permutation, workers add overhead and memory after speed saturates.

Connect to Python 3.14: free-threaded Python may help thread-friendly code, but the actual stack must be measured and logged.

## Slide 22 - Connecting the tools

This slide connects directly to the talk title and abstract.

Python 3.14, Numba, and JAX do not solve the same problem.

- Python 3.14 changes interpreter and threading ceilings.
- Numba compiles explicit numerical CPU loops.
- JAX/A100 is for batched array programs after reformulation.

Say: Python is not competing with compiled languages here. Python is the scientific interface; the hotspot moves to the appropriate engine.

## Slide 23 - AI/Codex

Keep this short and practical.

Codex can write variants, runners, metadata logging, plots, and READMEs. That is valuable because simulation work is repetitive and detail-heavy.

But the statistician owns the contract: statistic, null model, calibration, scenario grid, and interpretation of negative evidence.

## Slide 24 - Decision guide

Use the table as the practical takeaway. Do not read every row.

Pick three examples:

- Understanding/correctness -> reference implementation.
- Scalar CPU loops -> Numba.
- Large batched arrays -> JAX/GPU after reformulation.

Then add: if the problem is giant temporaries, algebraic rewrite may be better than switching libraries.

## Slide 25 - Close

Close with one sentence: make the statistic testable, then make the bottleneck fast.

Repeat: clear enough to trust, fast enough to scale.

End by inviting questions.

---

# Backup Slides

## Slide 26 - Evidence map

Use only if someone asks how much validation was run or how the tiers are separated.

## Slide 27 - Shape stress

Use only if someone asks more about K and d in k-means.

## Slide 28 - Power curve

Use only if someone wants more statistical validation beyond null calibration.

## Slide 29 - Local permutation runtime

Use only if someone asks about MacBook-only permutation performance.

## Slide 30 - What the agent changed

Use only if someone asks about the repository or Codex workflow.
