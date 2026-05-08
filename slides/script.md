# Verbatim draft — 30-minute talk

## Slide 1 — Breaking the Speed Limit

**Target: 0:00–1:00**

Hi everyone. I’m Luca, and today I want to talk about speeding up statistical models in Python.

But I want to start with a warning: this is not a hardware leaderboard.

This is not going to be, “Numba is always faster,” or “JAX wins because GPU,” or “just use more cores.”

The question I care about as a statistician is a little different:

**Can I still trust the statistic after I make the code faster?**

So the workflow for this talk is:

First, make the statistic testable.
Then validate that optimized code preserves it.
Then, and only then, speed up the bottleneck.

We’ll use two examples: k-means, which has iterative assignment-update pressure, and permutation tests, which have resampling pressure across many features.

The main idea is simple:

**Speed only counts when the statistic is preserved.**

---

## Slide 2 — For statisticians, speed comes after validation

**Target: 1:00–2:00**

Here is the rule for the whole talk:

**Speed only counts when the statistic is preserved.**

That sounds obvious, but it is very easy to forget when a chart says “10x faster.”

For a statistician, a benchmark is not just:

“Did this code run faster?”

It is:

“Did it answer the same statistical question?”

That means the same data-generating setup, the same statistic, the same stopping rule when the algorithm is iterative, and the same result within a tolerance that makes sense for the method.

The reference implementation is important, but not because it magically defines the scientific target.

The scientific target comes first.

The reference implementation makes that target executable. It gives us something readable, inspectable, and testable.

Only after that do I want to talk about acceleration.

So whenever you see a runtime chart today, please mentally ask:

**What validation gate did this result pass first?**

---

## Slide 3 — Simulation creates an answer key

**Target: 2:00–3:20**

Now, why simulation?

In many statistical problems, especially in biomedical data, real ground truth is not available.

If I cluster patients, I usually do not know the true patient subtypes.
If I test thousands of genes, I usually do not know which discoveries are truly null or truly non-null.

So simulation is how we create an answer key.

Not because simulation replaces real data, but because simulation gives us controlled behavior.

We can say:

Here is the target.
Here are the assumptions.
Here is the null case.
Here is the alternative case.
Here is an easy case.
Here is a hard case.

Then we write a readable reference implementation.

Then we run validation checks:

Does k-means recover the simulated groups?
Do optimized implementations match the reference?
Are p-values calibrated under the null?
Does power increase when signal gets stronger?

Only after those checks do we ask:

Where is the bottleneck?

Is it Python loops?
Dense algebra?
Repeated resampling?
Memory shape?
Transfer to a device?

That is the point where tools like NumPy, Numba, threads, and JAX become meaningful.

---

## Slide 4 — Simulation turns statistical behavior into tests

**Target: 3:20–4:30**

For the programmers in the room, I like to think of this as a kind of statistical CI.

Not CI as in confidence interval — although, yes, that joke is unavoidable.

I mean continuous integration, but for statistical behavior.

A fixed-seed equivalence check is like a unit test:

Same input, same seed, same expected result.

A null calibration check is more like a property test:

I do not expect one fixed scalar output, but I do expect p-values to behave like p-values under the null.

A recovery check is also a property test:

If I simulate well-separated clusters, the algorithm should recover them.
If I simulate overlapping or imbalanced clusters, recovery might get worse — and that is useful information, not necessarily a bug.

And stress tests are where things become interesting:

What happens when we increase samples, features, repetitions, workers, or batch size?

So simulation turns vague statistical expectations into concrete tests.

And that lets us make speed claims with a straight face.

---

## Slide 5 — Python stays the control plane

**Target: 4:30–5:40**

This is why Python is still the control plane.

The goal is not to abandon Python.

The goal is to keep the statistical workflow readable in Python, and move only the measured hotspot.

So the workflow looks like this:

Write a readable reference.
Validate the statistical behavior.
Measure the bottleneck shape.
Then choose the smallest engine that solves that shape.

If the bottleneck is dense matrix algebra, NumPy and BLAS are often the right layer.

If the bottleneck is explicit CPU loops, Numba can remove Python loop overhead.

If the bottleneck is repeated work over shared arrays, threads or worker pools become relevant.

If the bottleneck is large regular array work that fits the accelerator, JAX on an A100 can help.

But the key phrase is:

**after validation.**

No tool gets to skip the validation gate.

---

## Slide 6 — Two workloads, two statistical computing shapes

**Target: 5:40–6:30**

Today I’ll use two workloads.

The first is k-means.

K-means is simple, but it has a very useful shape: assign points to centroids, update centroids, repeat.

Each step depends on the previous step, so it exposes iterative assignment-update pressure.

The second is the permutation test.

A permutation test is also simple: shuffle labels, compute a statistic, repeat.

But if we do this across many features and many null draws, the workload becomes very large.

So these examples give us two different kinds of pressure:

k-means gives us iterative dependence.

Permutation tests give us repeated resampling across many features.

And together, they let us talk about several acceleration tools without pretending the tools are interchangeable.

---

## Slide 7 — Evidence ladder

**Target: 6:30–8:00**

One more framing slide before the examples.

Different environments answer different questions.

The MacBook results answer:

Can I trust the statistic and implementation?

This is where I want correctness checks, calibration checks, recovery surfaces, and small local runtime signals.

The server CPU results answer:

What scales when the shapes get bigger?
What happens with more threads or workers?
What happens to memory?

And the A100 results answer:

Does the validated computation actually fit an accelerator-shaped pipeline?

This is not a hardware leaderboard.

A laptop result, a server result, and a GPU result should not be mashed into one ranking.

They answer different questions.

Also, I keep timeouts, OOMs, and skipped rows explicit.

If a row is unavailable, it is not a hidden win.

That is the evidence ladder: validation, scale, acceleration.

Now let’s use it.

---

# Workload 1: k-means

## Slide 8 — k-means as iterative assignment-update pressure

**Target: 8:00–8:20**

First workload: k-means.

I’m using it because it is simple enough to explain visually, but rich enough to expose real computational choices.

Same validation contract.

Different geometry.

Different scale.

Different bottleneck shapes.

---

## Slide 9 — How k-means moves

**Target: 8:20–9:30**

Here is the method visually.

This is the Iris dataset. Each flower is a point, using petal length and petal width.

The image is just to remind us what petals and sepals are. K-means does not see the species label. It only sees the two numerical measurements.

We ask k-means for three clusters.

The algorithm starts with centroids. Then Lloyd’s algorithm alternates two steps:

Assign each point to the nearest centroid.

Then update each centroid to the mean of the assigned points.

Then repeat.

The important thing is that this is iterative.

The next assignment depends on the current centroids, and the next centroids depend on the current assignments.

That is why this example is useful for performance work.

It is not just one matrix multiply. It is an assignment-update loop.

---

## Slide 10 — Why k-means is a useful test case

**Target: 9:30–10:30**

Now, k-means is not the most advanced statistical model in the world.

That is exactly why it is useful here.

The validation contract is easy to state:

Same data.
Same initial centroids.
Same stopping rule.
Compare final inertia.

But the scenarios can still get harder:

Clusters can overlap.
Groups can be imbalanced.
There can be outliers.
Dimension can increase.

Computationally, the repeated distance work scales like:

N samples times K clusters times d dimensions, every iteration.

So k-means lets us separate several questions:

Did the method recover the simulated groups?
Did optimized code reproduce the reference?
Was the bottleneck loop-shaped, algebra-shaped, or accelerator-shaped?

That is the pattern we will reuse later.

---

## Slide 11 — Recovery surfaces

**Target: 10:30–11:50**

Before timing k-means, we ask a statistical question:

Did it recover the groups we simulated?

Because this is simulation, we know which generating group each point came from.

So we can compute ARI — adjusted Rand index — between fitted clusters and the simulated truth.

An ARI near 1 means strong recovery.
An ARI around 0 is random-like agreement.

This slide is not about runtime.

It is about whether speed would even mean anything.

If k-means does not recover the simulated structure in a scenario, then a faster implementation is not a success story for that scenario.

It might still be a correct implementation of k-means.
But the method did not recover the target.

Also, do not read this heatmap as a perfect monotone law.

These are finite scenario summaries over dimensions, imbalance, outliers, and seeds.

The pattern is the message:

Some scenarios are easy.
Some are hard.
Poor recovery comes before speed.

---

## Slide 12 — Before timing: compare against the reference

**Target: 11:50–13:00**

Now suppose the scenario is worth timing.

We still need an implementation gate.

For k-means, same initialization matters a lot.

If I change the starting centroids, I can get a legitimately different final solution.

So here the comparison is fixed:

Same data.
Same initial centroids.
Same stopping rule.

Then compare final inertia against the readable reference.

The reference is the executable oracle for this fixed-init comparison.

In these checks, the optimized paths reproduce the reference result within numerical tolerance.

The maximum relative inertia difference is tiny — on the order of roundoff.

The main point is not the exact number.

The point is the workflow:

Before I show a speedup, I show that the optimized version is still answering the same fixed-init k-means question.

---

## Slide 13 — The same distance step can be expressed three ways

**Target: 13:00–14:20**

Now we can talk about implementation shape.

This slide is not here to teach full k-means syntax.

It is here to show that the same validated distance step can be expressed in different computational languages.

On the left, Numba keeps the code close to explicit loops.

That is useful when the work is loop-shaped and we want to remove Python loop overhead.

In the middle, NumPy rewrites squared distances as dense matrix algebra:

x squared plus c squared minus two times X times C transpose.

The user-facing code is NumPy, but the heavy matrix multiply underneath is BLAS-backed.

On the right, JAX uses a very similar array expression, but now the question becomes:

Is there enough large regular array work to make the accelerator worthwhile?

The statistical contract has not changed.

Same data.
Same initial centroids.
Same stopping rule.

Only the expression of the bottleneck changed.

---

## Slide 14 — Server k-means

**Target: 14:20–15:30**

Here are representative server shapes.

This is not a universal ranking.

It is not “Numba always wins” or “GPU always wins.”

These are validated warm rows chosen to show different shapes.

For small or overhead-dominated cases, all paths can be close.

For dense distance algebra, NumPy can win because BLAS is very good at matrix multiply. It has cache blocking, SIMD, threading, and decades of optimization.

A compiled loop does not automatically become a highly optimized GEMM.

For large regular workloads, JAX on A100 can help because the array work is big enough and regular enough to fit the accelerator.

So the point is shape matching.

Numba for explicit loops.

NumPy and BLAS for dense algebra.

JAX and A100 for large regular device-friendly work.

Again: representative validated rows, not a hardware leaderboard.

---

## Slide 15 — k-means takeaway

**Target: 15:30–16:00**

So the k-means takeaway is this:

If recovery is poor, do not optimize yet. Understand the method and the scenario first.

If the code is a hot scalar loop, Numba is a good small move.

If the statistic becomes dense matrix algebra, NumPy and BLAS may be the right expression.

If there is enough large regular array work, JAX and A100 become relevant.

The same pressure appears in many iterative algorithms: EM, coordinate descent, optimization loops, and simulation-based estimators.

Now let’s switch to a different kind of pressure: resampling inference.

---

# Workload 2: permutation tests

## Slide 16 — Permutation tests as resampling inference pressure

**Target: 16:00–16:20**

Second workload: permutation tests.

Here the algorithm is simple, but the repetition is expensive.

A simple inference procedure becomes a computational problem when null resampling meets many features.

---

## Slide 17 — How a permutation test scales

**Target: 16:20–17:30**

Let’s start with the ordinary version.

We have samples by features.

Maybe samples are patients.
Maybe features are genes, biomarkers, or voxels.

We have labels, like treatment and control.

One permutation means: shuffle labels, compute feature-wise differences, and get one vector of statistics.

Then repeat.

Many shuffles build the null distribution.

This is conceptually simple.

But the repeated work is real.

And when we do it across many features, even a simple statistic can become expensive.

For now, do not worry about the matrix trick.

The statistical idea is just:

shuffle, compute, repeat.

---

## Slide 18 — Why this becomes expensive

**Target: 17:30–18:30**

This is where the shape appears.

The input matrix is n samples by p features.

If we run R shuffled label vectors, then the potential null statistics scale like R times p.

Now, we do not necessarily have to materialize a full R by p null matrix.

A good optimized path can stream results down to p exceedance counts.

But the work still scales like R times p.

So the problem is not that the statistic is complicated.

The statistic can be very simple.

The problem is that we repeat it many times across many features.

That is the resampling pressure.

---

## Slide 19 — Same permutations, different computation

**Target: 18:30–19:40**

Now we can change the computation without changing the statistical question.

The reference loop is straightforward:

For each permutation, shuffle labels, compute the statistic, update the p-value counts.

The optimized path uses the same simulated data, the same permutation stream, and the same p-value definition.

But it groups permutations into batches.

A batch of permutation weights is W_batch.

The data matrix is X.

Then T equals W_batch times X.

That gives a batch of feature-wise statistics.

Then we update exceedance counts and stream forward.

The important phrase is:

**same permutation stream, same p-value definition.**

The streamed path avoids materializing the full R by p null matrix.

So we changed the computation shape, not the statistic.

---

## Slide 20 — Equivalence check

**Target: 19:40–20:40**

Now we validate that statement.

Loop reference versus matrix formulation.

Same simulated data.

Same permutation stream.

Same p-value definition.

The maximum p-value difference recorded here is 0.

The maximum statistic difference is about 9.4e-16, which is roundoff scale.

That is why this path is allowed into runtime comparison.

This slide is the first correctness gate for permutation tests.

It answers:

Did the faster computation still answer the same permutation question?

Here, yes.

Same question.
Same result.
Then faster code.

---

## Slide 21 — Null calibration

**Target: 20:40–21:50**

Equivalence is necessary, but it is not enough.

A permutation implementation can match a reference and still be used in a bad statistical setup.

So next we check behavior under the null.

Under the null, if alpha is 0.05, the type-I error estimate should be near 0.05, up to simulation variation.

Here the observed estimate is about 0.051.

That is very close to nominal alpha.

This is still validation, not performance.

And it is not a universal guarantee that every future analysis is calibrated.

It says: in this local null calibration check, the p-values behave as expected.

So:

Equivalence checks code.

Calibration checks inference.

Now it makes sense to ask about scale.

---

## Slide 22 — Local scale signal

**Target: 21:50–22:45**

Now the question changes.

We are no longer asking:

Did the implementation preserve the statistic?

We already passed equivalence and calibration locally.

Now we ask:

Where does runtime start to bend?

This plot is MacBook Air local-tier evidence.

It is not server evidence.

It is not A100 evidence.

The plotted path is the validated batched NumPy matrix path.

n is fixed at 500.
p and R vary.
The y-axis is warm median runtime.

The point is not to compare tools here.

The point is to see the bottleneck shape.

As p and R grow, runtime grows.

That tells us why the next question belongs on server CPU and A100.

Local validation makes the statistic safe.

Local scaling shows why we need scale evidence.

---

## Slide 23 — GPU decision map

**Target: 22:45–24:00**

Now we ask:

When does the A100 help?

Each cell compares a matched CPU matrix baseline with the A100 streamed full end-to-end path.

The speedup is CPU time divided by A100 time.

Compile is excluded.

Transfer is included.

Kernel-only timing is excluded.

That last point is important.

We are not claiming victory because one matrix multiply is fast.

We are comparing the full statistical pipeline.

The break-even starts in the smaller p region at p equals 10k and R equals 5k, with batch_R 8192.

The largest slide-level measured speedup here is 8.54x at p equals 500k and R equals 5k.

The grey cells are A100 OOM or unavailable.

They are not CPU wins.

They are unavailable rows.

So the GPU story is not:

“GPU is faster.”

The GPU story is:

After validation, after batching, after streamed reduction, the A100 wins for the right R by p shape.

---

## Slide 24 — CPU parallelism

**Target: 24:00–24:50**

One more practical scaling lesson.

More parallelism helps — until returns flatten.

This slide is about diminishing returns.

When we add parallelism, total time can go down, especially early.

But each jump gives less benefit.

And for permutation workers, high counts can also increase memory substantially.

So the practical rule is not:

“Use the maximum number of CPUs you can see.”

The rule is:

Use enough parallelism to capture most of the speedup.

Then stop and measure.

The exact 64 and 128 behavior here is workload evidence from a shared server, not a universal law.

But the general lesson is very robust:

Parallelism is a tuning parameter, not a dial to max out.

---

# Closing section

## Slide 25 — Tool roles

**Target: 24:50–26:00**

Let’s step back.

What did each tool actually do in this talk?

Numba solved explicit CPU loops in k-means assignment and update.

NumPy and BLAS solved dense matrix algebra, like the k-means distance identity and W at X on CPU.

Threads and workers handled repeated work over shared data in the permutation worker sweep.

JAX and A100 solved large regular device-resident batches for streamed W at X after validation.

So this is not a ranking.

No tool wins by logo.

Each tool matched a measured bottleneck shape.

That is the way I want statisticians to think about performance tools.

Not:

“What is fastest?”

But:

“What shape is my validated computation?”

---

## Slide 26 — AI / Codex

**Target: 26:00–27:00**

Now, a quick note about AI coding tools.

Codex was useful in this project.

But not because I asked it:

“Please decide the scientific claim.”

That is not the job.

The useful part was multiplying the workflow.

Implementation variants.

Scenario-grid runners.

Metadata capture.

Plot regeneration.

Result manifests.

That is the left side.

But the right side stays human.

The statistician owns the statistical target.

The data-generating assumptions.

The validation criteria.

The interpretation of hard cases.

And the scientific claims.

So my view is:

AI can automate the workflow.

Statistical judgment decides what can be claimed.

That boundary matters.

---

## Slide 27 — Decision guide

**Target: 27:00–28:40**

Here is the whole talk as a decision guide.

Simulate.

Validate.

Find the bottleneck.

Pick the smallest tool.

If validation fails, do not optimize yet.

That does not always mean the code is wrong.

It may mean the statistic, assumptions, scenario, or reference needs work.

Debug before optimizing.

If the result matches the reference and a Python loop dominates, use Numba.

If the statistic becomes dense matrix operations, use NumPy and BLAS.

If you have many repeats over shared data, tune threads or workers.

If large batches fit the device and the work is regular enough, JAX and A100 can help.

If a temporary array becomes huge, rewrite or stream.

This is the rule:

Validate the result.

Then move the bottleneck.

The tool choice follows the evidence, not the logo.

---

## Slide 28 — Close

**Target: 28:40–29:20**

So if you remember one sentence from this talk, make it this one:

Make the statistic testable.

Then make the bottleneck fast.

That is the workflow I want to advocate for.

Clear enough to validate.

Fast enough to scale.

---

## Slide 29 — Thank you

**Target: 29:20–30:00**

Thank you.

The slides and reproducible artifacts are in the repository.

I’m happy to talk about the benchmark details, the A100 break-even map, the simulation setup, or the Codex workflow.

And if you remember one thing:

Do not ask only, “Which tool is fastest?”

Ask:

“What did the validated simulation make expensive?”

Thank you — questions?

---

# Backup transition lines, if asked

如果 Q&A 时需要切 backup，可以用这些短句，不要完整讲 backup：

## Backup 30 — Evidence map

“If the question is where each result came from, this backup slide separates the tiers: MacBook for validation, server CPU for scale and parallelism, A100 for accelerator-shaped pipelines.”

## Backup 31 — Local permutation validation inventory

“This is the compact inventory of local permutation checks: equivalence, calibration, and runtime shape.”

## Backup 32 — Shape stress

“This backup is useful if someone asks when Numba wins. It is local MacBook loop-shaped evidence, not a universal ranking.”

## Backup 33 — Power curve

“This is the alternative-side sanity check: stronger simulated effects become easier to detect.”

## Backup 34 — batch_R

“This answers why the decision map used batch_R 8192. It is a tuning choice, not a new statistical method.”

## Backup 35 — A100 stage accounting

“This is mechanism evidence for where A100 end-to-end time goes. It does not define the decision boundary.”

## Backup 36 — Coding agent

“This is what Codex helped automate: variants, grids, figures, manifests, and QA artifacts. The scientific claims stayed human-owned.”

---

# 练习时的节奏提醒

这版逐字稿按正常 PyCon 语速大约 **29–30 分钟**。真正上台时，如果你语速偏快，会在 27–28 分钟；如果你现场解释图多一点，会接近 30 分钟。

最容易超时的地方是：

* Slide 11 recovery heatmap；
* Slide 14 server k-means；
* Slide 23 A100 decision map；
* Slide 27 decision guide。

这四页每页只讲主结论，不要逐个读数字。
