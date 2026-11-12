# Verbatim script - PyCon HK 2026 (30 minutes)

## Slide 1 - Breaking the Speed Limit (0:00)

Hi everyone. I'm Wenxin, and this talk is joint work with my colleague Jian Yin.

It's about a problem every scientific Python user hits: our code is too slow, and when we make it faster, how do we know it still gives the same answer?

On the right are three numbers from this talk. A matrix rewrite that changes the p-values by exactly zero. Free-threaded Python 3.14 scaling threads by almost 5x. And a GPU that is 8.5 times faster, but only for the biggest problems. By the end you'll know where each number comes from and when it applies to you.

---

## Slide 2 - Wenxin Jiang (0:45)

A quick word about me. I'm a PhD student in biostatistics at City University of Hong Kong, so I'm local. I work on statistical genetics and genomics, where the data are big: hundreds of thousands of genetic variants, tens of thousands of genes.

I recently finished prusie, a Python package with a Rust core that reimplements SuSiE fine-mapping from the R package susieR. On 200 GWAS loci from FinnGen, the R package takes almost two hours; prusie takes under three minutes, 49 times faster. On 14,157 genes from GTEx, it goes from 21 hours to 48 minutes. It also uses about a fifth of the memory.

The line I care most about is the third one: the credible sets, which are the actual scientific output, are identical to susieR in every region.

That is what this talk is about: making statistical code fast while proving it still gives the same answer. prusie is on PyPI if you want to try it. Today I'll use two simpler examples, k-means and permutation tests, so we can walk through every step.

---

## Slide 3 - A readable permutation test is too slow at genomics scale (2:00)

Here's the kind of code we start with. A permutation test: shuffle the group labels, recompute the statistic for every feature, repeat R times, and count how often the shuffled statistic beats the observed one. That count is the p-value.

This code is good. It reads like the textbook definition, so we trust it. The problem is size. In genomics, p is hundreds of thousands of features and R is thousands of shuffles.

Even after we vectorized it with NumPy, one test with half a million features takes about 11 seconds on our server. That sounds fine until you run it thousands of times in a simulation study. And one larger configuration didn't finish within our 45-minute limit at all.

So we have two questions. How do we make this faster? And, as statisticians, the one we care about more: how do we prove the fast version answers the same question?

---

## Slide 4 - Validate against a reference before timing anything (3:30)

Our rule is simple: validate first, then time.

We define exactly what we're computing. We simulate data where we know the truth. We write a slow reference implementation that looks like the math. Every optimized version must match that reference before we look at its runtime.

Only then do we measure where the time goes, and pick the smallest tool that fixes that bottleneck. Not the most exciting tool. The smallest one.

The reason is in the orange box. If the fast version quietly changes the statistic, you haven't saved time. You've just produced a wrong scientific conclusion sooner.

---

## Slide 5 - Simulation is how statisticians write tests (4:30)

If you write software, you already know this idea under a different name.

Our outputs are random, so we can't assert one exact value. But we can still test. With a fixed seed, the fast and reference versions must agree: that's a unit test. Under the null hypothesis, a correct test rejects about 5 percent of the time at alpha 0.05: that's a property test. Outliers and imbalanced groups are stress tests. Growing the problem until it breaks is a load test. And every new engine has to pass all of this again: regression testing.

So simulation gives statistical code an answer key. That's what makes optimization safe.

---

## Slide 6 - Two workloads with different bottlenecks (5:30)

We use two workloads because they stress Python in different ways.

k-means is iterative. Each step depends on the previous one, so you can't just run iterations in parallel. The cost is distance computations: N points times K centres times d dimensions, every iteration.

A permutation test is the opposite. Every shuffle is independent, so it's embarrassingly parallel. The cost is R shuffles times the full data matrix, and with genomic p that gets very large.

Different shapes, so we'll see different tools win.

---

## Slide 7 - k-means: an iterative fit (6:15)

Let's start with k-means.

---

## Slide 8 - k-means alternates two steps until nothing changes (6:30)

Here's k-means on the iris flower data. Each dot is a flower, plotted by petal length and width. We want three clusters.

We start from fixed centres, so every implementation we compare starts from exactly the same place. Then two steps. Assign: each point goes to the nearest centre, and you can see colours change. Update: each centre moves to the mean of its points.

We repeat until assignments stop changing.

The assign step is where the time goes: every point against every centre in every dimension. And because each iteration needs the previous centres, we can parallelize inside an iteration but not across iterations.

---

## Slide 9 - Before speed: k-means fails on imbalanced clusters at any speed (7:45)

The first check isn't about code at all. Does the method recover the clusters we simulated?

We score recovery with the adjusted Rand index: 1 means perfect, 0 means no better than chance. Rows are how well separated the clusters are; columns are different stressors.

Clean, high-dimensional data: near-perfect, 0.9 to 1.0. But look at the two right columns. With a 90/10 imbalance in cluster sizes, recovery stays around 0.25 or lower, even when clusters are well separated.

That is a property of k-means, not of our code. No amount of Numba or GPU fixes it. If this were our real data, we'd fix the method before optimizing anything. For this talk we accept these limits and move on to checking the code.

---

## Slide 10 - Every optimized k-means matches the NumPy reference (9:00)

Now the code check. Every optimized version runs on the same data, starts from the same centres, and uses the same stopping rule as the reference.

We compare two things: the final value of the objective, called inertia, and the cluster labels.

The largest inertia difference we saw was 3 times 10 to the minus 14. That's floating-point rounding. Our tolerance was 10 to the minus 8. Labels agreed exactly.

One practical detail: for very large problems the naive reference would run out of memory. We record those rows as skipped instead of letting the run crash. That's 480 skipped rows, and they're labelled as skipped in the results, not hidden.

---

## Slide 11 - One assignment step, written three ways (10:00)

Here is the hot step written three ways.

NumPy: we expand the squared distance into x squared plus c squared minus 2 x dot c. The expensive part becomes one matrix multiply, which goes to an optimized BLAS library.

Numba: we keep the explicit loops, which is closest to the textbook, and Numba compiles them to machine code and runs the outer loop in parallel. No big temporary matrix.

JAX: almost the same code as NumPy, but jit-compiled to run on the GPU.

All three passed the equivalence check on the previous slide. The question now is which one is fastest, and the answer depends on the shape of the data.

---

## Slide 12 - The fastest engine changes with the problem shape (11:15)

So which engine wins? Here are three shapes on our server.

Small problem: everything finishes in under half a second. The differences don't matter. Don't spend a day optimizing this.

Dense algebra: a million points, 256 dimensions, 50 clusters. Now the distance matrix is a big matrix multiply, and NumPy with BLAS wins at 32 seconds. Numba's explicit loops take twice as long. And notice the GPU doesn't win here either.

Very large: five million points. Now there's enough regular work to keep an A100 busy, and it's 2.8 times faster than the best CPU.

So there's no single best tool. The shape of the problem picks the tool, and you only find out by measuring.

If someone asks where Numba wins: on loop-shaped problems on a laptop. That's in the backup slides.

---

## Slide 13 - Permutation tests: many independent repeats (12:30)

Now the permutation test, the example we started with.

---

## Slide 14 - Shuffle the labels many times to build the null distribution (12:45)

Here's the permutation test in pictures. Each row is a sample, each column a feature, and samples belong to group A or B.

We compute the observed difference between groups for every feature. Then we shuffle the labels, keeping the data fixed, and recompute every difference. That gives one draw from the null distribution: what differences look like when group membership doesn't matter.

Repeat R times. The p-value for each feature is the fraction of shuffles that beat the observed value.

The shuffles are independent, which is good for parallelism. But each one touches the whole matrix, so work grows with R times n times p.

---

## Slide 15 - Each shuffle is a weight vector, so a batch is one matrix multiply (14:00)

This is the key rewrite. A difference in group means is just a weighted sum of the rows: plus one over n A for samples in group A, minus one over n B for group B.

So each shuffle is a weight vector. Stack a batch of them into a matrix W, and W times X gives you the statistics for the whole batch in one matrix multiply. That's exactly the operation BLAS and GPUs are built for.

The second change is streaming. The reference stores the full R by p null matrix, which for half a million features doesn't fit in memory. The optimized version keeps only a running count per feature: how many shuffles beat the observed value so far.

The math is the same. The shape of the computation is completely different. That's why we have to prove it's still the same test.

---

## Slide 16 - The matrix rewrite gives exactly the same p-values (15:15)

First check: does the rewrite produce the same numbers?

Both versions get the same data and, importantly, the same sequence of shuffles. Otherwise we'd be comparing random noise. We ran 18 different problem shapes, each with 5 seeds: 450 comparisons passed, none failed, and 45 were skipped because the reference would have run out of memory.

The p-values matched exactly: maximum difference zero. The statistics differed by at most 10 to the minus 16, which is rounding in the last bit of a double. We set a tolerance of 10 to the minus 6 before running anything, and we're ten orders of magnitude inside it.

So the code is equivalent. But equivalent code can still implement a bad test, which brings us to the second check.

---

## Slide 17 - Under the null, the test rejects 5.1% of the time at α = 0.05 (16:15)

Second check: is it a valid test?

Equivalence tells us the fast code matches the slow code. It doesn't tell us the slow code was right. So we simulate data where there is no real difference between groups. A correct test at alpha 0.05 should flag about 5 percent of features, purely by chance.

We observed 0.051. The dashed line is 0.05, the dot is our estimate, and the green band is the range you'd expect for a single feature just from binomial noise.

So now we have both: the code is equivalent, and the inference is calibrated. Only now do we look at speed.

---

## Slide 18 - The A100 pays off once R × p is large: up to 8.5× faster (17:15)

Now speed. This map compares our server CPU against an A100 GPU, running the same validated, streamed permutation test. Columns are the number of features, rows the number of shuffles. Each cell is how many times faster the GPU is, end to end, including copying data to the GPU.

Bottom left, the smallest problem: the CPU wins. The GPU is 0.7 times as fast because transfer and launch overhead dominate.

As p and R grow, the GPU pulls ahead. The best cell, half a million features and 5,000 shuffles, goes from 10.7 seconds to 1.25 seconds: 8.5 times faster.

Top right, grey: the GPU ran out of memory. The CPU still finished. So even the GPU has a shape it doesn't fit.

The lesson: the GPU is worth it only when there's a lot of regular work. And it only helped because we had already rewritten the test as a matrix multiply.

---

## Slide 19 - More workers help until about 16, then stop paying (18:45)

What about plain CPU parallelism on the server?

Going from one to four workers is a big win: 58 percent faster for k-means, 46 percent for the permutation test. After that, each doubling buys less. For the permutation test, going past 16 workers actually made it slower.

And look at memory. The permutation version uses processes, and each process gets its own copy of the data. At 128 workers that's 58 gigabytes.

So: take the early speedup, then stop. And that memory cost is exactly what free-threaded Python addresses.

---

## Slide 20 - Free-threaded Python 3.14t lets threads scale: 4.77× on 16 threads (19:45)

This is the Python 3.14 part of the talk.

Here's a CPU-bound resampling workload written in plain Python, run with a ThreadPoolExecutor. On standard CPython 3.14, the grey line: 16 threads are no faster than one. That's the GIL: only one thread executes Python at a time.

Same code on the free-threaded build, 3.14t, the blue line: 16 threads are 4.77 times faster. Nothing changed in our code. Only the interpreter.

The usual workaround for the GIL is processes. But processes copy the data, which is the 58 gigabytes we saw on the last slide. Compared with a ProcessPool on standard Python, threads on 3.14t were 2.4 times faster and used a third less memory, because all threads share one copy.

A caveat: this helps code that spends its time in Python bytecode. If your hot loop is already inside NumPy or BLAS, those release the GIL anyway, and you'll see less difference. And as always, we re-ran the validation on 3.14t.

---

## Slide 21 - Pick the tool from the bottleneck you measured (21:15)

Here's everything in one slide.

If your profile shows tight numerical loops, try Numba. If the math can be written as matrix products, rewrite it and let NumPy and BLAS do the work. If you have many independent repeats over shared data, threads on free-threaded Python are now a real option. And if you have huge, regular batches, a GPU with JAX can pay off, but only once the work outweighs the cost of moving data.

I think of it like travel. You don't fly across Victoria Harbour, and you don't walk to Tokyo. You look at the trip first, then pick the vehicle. Here the trip is your bottleneck, and the only way to know it is to measure.

And every row assumes the validation already passed. Speed is the last step.

---

## Slide 22 - AI speeds up writing fast code. It doesn't skip the validation. (22:30)

We used AI coding agents a lot in this project, and they really did speed us up. Translating a NumPy reference into Numba or JAX used to take us a day; an agent gives a working first draft in minutes. That means we tried more variants than we ever would have written by hand. It also built the experiment runners and regenerated the plots.

But notice that nothing in our rule changed. To us, AI-written code is just one more candidate implementation. It goes through exactly the same steps as code we write ourselves: match the slow reference on the same seeds, within a tolerance we fixed before looking; check calibration under the null; and only then measure end-to-end time.

One small example. The agent's GPU timings for the kernel alone looked much better than our end-to-end numbers. They were real numbers, but they skip copying data to the GPU, so they answer a different question. We left them out.

So AI makes writing fast code cheaper. It doesn't make trusting that code any cheaper. That still takes a reference and a test, which is the whole point of this talk.

---

## Slide 23 - Make the statistic testable. Then make the bottleneck fast. (23:45)

To wrap up.

Simulate, so you have an answer key, and keep the slow reference around.

Check that the fast version is equivalent and that the test is still calibrated, before you look at a single timing.

Then measure where the time goes and pick the smallest tool that fixes it. Sometimes that's NumPy, sometimes Numba, sometimes free-threaded Python, and occasionally a GPU.

Make the statistic testable. Then make the bottleneck fast.

---

## Slide 24 - Questions? (24:30)

Thank you. Everything is on GitHub: the slides, the code, the raw CSVs, and the scripts that generate every figure. The QR code goes to the repository. Happy to take questions.

---

## Backup 1 - On a laptop, runtime grows with both R and p

If asked how runtime scales locally: on the MacBook, with the batched NumPy path at n equals 500, runtime grows roughly linearly in R, and higher p shifts the curve up. This is what told us to move the large runs to the server and GPU.

---

## Backup 2 - Evidence map. Different questions.

If you are wondering why we didn't just build one massive hardware leaderboard, the answer is scope. Each environment answers a different question. The MacBook establishes local validation. It gives us local shape signals. The server CPU answers scaling and parallelism questions. The GPU only enters the picture to ask: does this validated, streamed pipeline actually win? It is an evidence map, not a leaderboard.

---

## Backup 3 - Validation inventory.

If you want to see the full local permutation checks, here is the inventory. It combines main-path equivalence, null calibration, and the local runtime shapes. This is all local MacBook evidence. We run all of this before we ever touch a server or a GPU.

---

## Backup 4 - Numba wins loop shapes.

Someone usually asks why Numba looks so fast locally but loses on the server. Look at the shape. This is local MacBook shape-stress evidence. Over a thousand rows. All passing. Numba dominates when the shape is pure explicit loop work. But when you move to a server and the shape widens into dense distance algebra, BLAS takes over. And at massive device-friendly scale, the GPU wins. It never contradicts. It just changes shape.

---

## Backup 5 - Power vs. effect size.

If you want statistical validation beyond just the null calibration, here is the simulated power curve. This is not a speed result. It is pure statistics. The reading is very simple. Weak effects have low power. Stronger simulated signals are easier to detect. The inference behaves exactly as it should.

---

## Backup 6 - Backup: what batch_R tuning showed

This backup slide is about the trend, not a proof of optimality. batch_R is the number of permutation rows processed in one A100 batch. In this committed sweep, small batches were slower because the streamed path paid more scheduling and loop overhead. Increasing batch_R reduced that overhead, but the gains flattened after roughly 512-1024 and the curve is not perfectly monotone. 8192 was the fastest accepted value in the measured 128-8192 grid, so it was used as the chosen setting for the later A100 decision map. Larger batch_R values were not part of this committed sweep; they could be faster, slower, or fail because of memory or autotuning. batch_R changes scheduling, not the permutation statistic or p-value definition. Full end-to-end timing remains the criterion: compile excluded, transfer included, kernel-only excluded.

---

## Backup 7 - Scaling evidence.

The py313 control is not present in the local benchmark outputs, so this backup uses only the
available py314 and py314t data. The chart normalizes each interpreter to its own one-worker baseline. The standard
GIL build stays flat. The free-threaded build scales on this CPU-bound Python workload. The pool rows show why the
execution model matters: ProcessPoolExecutor can work around the GIL, but threads avoid the memory and serialization
costs for shared data.

---
