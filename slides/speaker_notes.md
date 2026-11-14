# Speaker notes - PyCon HK 2026

Generated from the `<aside class="notes">` blocks in `index.html`. Edit the deck, then regenerate this file.

Target: 30-minute slot. Notes run about 25 minutes at a relaxed pace, leaving about 5 minutes of buffer for transitions and timing variance. 24 main slides, 7 backup slides.

## Slide 1 - Breaking the Speed Limit (0:00)

Hi everyone. I'm Wenxin, and this talk is joint work with my colleague Jian Yin.

It's about a problem every scientific Python user hits: our code is too slow, we want to make it faster. And when we make it faster, how do we know the code is correct?

## Slide 2 - Wenxin Jiang (0:45)

A quick word about me. I'm a PhD student in biostatistics at City University of Hong Kong. I work on statistical genetics and genomics, where the data are really big like hundreds of thousands of genetic variants and genes.

I recently finished prusie, a Python package with a Rust core that reimplements a R package. On 200 GWAS loci, the R package takes almost two hours; prusie takes under three minutes. On 14k genes dataset, it reduce the runtime from 21 hours to 48 minutes. It also reduces peak memory use by about 80%.

It's faster and uses less memory. Most importantly, although there are small numerical differences from reference, the key statistical features remain the same.

That is what this talk is about: making statistical code fast while proving it still gives the same answer. Today I'll use two simpler examples, k-means and permutation tests, so we can walk through every step.

## Slide 3 - A readable permutation test is too slow at genomics scale (2:00)

Here's the kind of code we start with. A permutation test: we have a group of labels, shuffle the group labels with rng.permutation, compute the statistic we call null, repeat it R times, and count how often the shuffled statistic beats the given one. That count is the p-value.

This code is good. It follows the textbook definition, so we trust it. The problem is size. In genomics, p is hundreds of thousands of features and R is thousands of shuffles. Even after we vectorized it with NumPy, one test with half a million features takes about 11 seconds on our server. That sounds fine until you need to run it thousands of times. Also, as the size goes larger, we can find that it can not finish within 45-minute.

So we have two questions. How do we make this faster? And, as statisticians, we care about that: how do we prove the fast version answers the same question?

## Slide 4 - Validate against a reference before timing anything (3:30)

Our rule is simple: validate first, then measure performance.

First, we define what we want to compute and how to evaluate the results. We simulate data, so we know the ground truth. Then we write a simple, slow reference implementation that follows the math.

Before testing speed, every optimized version must agree with the reference, within a small numerical tolerance. We also check that the key statistical results stay the same.

Only after that do we measure where the time goes and choose the right tool to speed up the bottleneck.

## Slide 5 - Simulation is how statisticians write tests (4:30)

If you write software, you probably know these ideas by different names.

In statistics, we often work with random data, and optimized code may produce small numerical differences. So we cannot always expect an exact match. But we can still test the code.

With a fixed seed, we can reproduce the same data. We compare individual functions with a simple reference. That's unit testing.

We generate many datasets and check properties that should always hold. That's property-based testing.

We test unusual cases, like outliers and imbalanced groups. Those are edge-case tests.

We increase the data size to see how the program performs. That's load testing.

We push the program beyond its normal limits. That's stress testing.

And after every code change, we run these tests again. That's regression testing.

So simulation helps us check correctness before we focus on speed.

## Slide 6 - Two workloads with different bottlenecks (5:30)

We use two examples because they have different performance bottlenecks.

First, k-means. It runs in iterations, and each iteration depends on the previous one. So we cannot run all iterations in parallel. But within each iteration, we can calculate distances in parallel. For N data points, K cluster centers, and d dimensions, the cost is roughly N times K times d per iteration.

Second, a permutation test. Each permutation is independent, so we can run many of them in parallel.

We repeat the calculation R times, often across many variables. When R and p are large, the cost can be very high.

These two workloads have different bottlenecks, so different optimization tools may work best.

## Slide 7 - k-means: an iterative fit (6:15)

Let's start with k-means.

## Slide 8 - k-means alternates two steps until nothing changes (6:30)

Here's k-means on the Iris dataset. Each dot represents a flower, based on its petal length and width. The dataset has three known species, so we set the number of clusters to three.

We use the same starting centers for every implementation, to make the comparison fair.

Then we repeat two steps.

First, assign each point to its nearest center. You can see the colors change.

Second, update each center by taking the mean of its assigned points.

We repeat these steps until the assignments stop changing.

Most of the computation happens in the assignment step, because we need to calculate the distance from every point to every center.

And because each iteration depends on the previous one, we can run calculations in parallel within an iteration, but the iterations themselves must run in order.

## Slide 9 - Before speed: k-means fails on imbalanced clusters at any speed (7:45)

First, we check the statistical performance. Can k-means recover the clusters we simulated?

We measure this using the adjusted Rand index, or ARI. A score of 1 means perfect recovery, while 0 means about the same as random guessing.

The rows show different levels of cluster separation, and the columns show different data conditions.

On clean, high-dimensional data, the results are very good, with ARI between 0.9 and 1.0.

But look at the two columns on the right. When cluster sizes are imbalanced, with a 90-to-10 ratio, ARI drops to around 0.25 or lower, even when the clusters are well separated.

This shows a limitation of k-means itself. Making the code faster with Numba or GPU won't solve this problem.

In real research, we should address this statistical limitation first. For this talk, we'll keep it in mind and move on to checking our implementations.

## Slide 10 - Every optimized k-means matches the NumPy reference (9:00)

Now let's check the code.

Every optimized version uses the same data, the same starting centers, and the same stopping rule as the NumPy reference.

We compare two things: the final objective value and the cluster labels.

We set a numerical tolerance of 10 to the minus 8 before running the tests.

The largest difference in the final objective was only 3 times 10 to the minus 14, much smaller than our tolerance. And the cluster labels also matched.

One practical issue: for very large datasets, the NumPy reference needs too much memory. So we mark those cases as skipped instead of letting the program crash.

In total, 480 rows were skipped. For all the cases we could compare, the results matched.

## Slide 11 - One assignment step, written three ways (10:00)

Here is the same k-means assignment step, implemented in three different ways.

First, NumPy. We rewrite the squared distance as x squared plus c squared minus two times x dot c. This turns most of the work into matrix multiplication, which uses an optimized BLAS library.

Second, Numba. We keep the simple loops, just like the original algorithm. Numba compiles them into machine code and runs the loop over data points in parallel. This also avoids creating a large distance matrix.

Third, JAX. The code looks similar to NumPy, but we use JIT compilation to run it on the GPU.

All three implementations matched the reference in our tests.

Now the question is: which one is the fastest?

As we'll see, the answer depends on the size and shape of the data.

## Slide 12 - The fastest engine changes with the problem shape (11:15)

So, which one is the fastest? Let's look at three different problem sizes on our server.

First, a small problem. All three versions finish in less than half a second. The differences are very small, so there's usually no need to spend time optimizing further.

Second, a larger problem: one million data points, 256 dimensions, and 50 clusters.

Here, NumPy with BLAS is the fastest, finishing in 32 seconds. Numba takes about twice as long. And even the GPU is slower than NumPy in this case.

Finally, a very large problem with five million data points.

Now the GPU has enough work to make good use of the A100. It runs 2.8 times faster than the best CPU version.

So there is no single best tool. Performance depends on the size and shape of your data. We need to measure it to find the best choice.

And if you're wondering where Numba wins, it can perform very well on loop-heavy problems, even on a laptop.

## Slide 13 - Permutation tests: many independent repeats (12:30)

Now the permutation test, the example we started with.

## Slide 14 - Shuffle the labels many times to build the null distribution (12:45)

Here is how the permutation test works.

Each row is a sample, each column is a feature, and each sample belongs to group A or group B.

First, we calculate the difference between the two groups for each feature.

Then we randomly shuffle the group labels, keeping the data unchanged, and calculate the differences again.

We repeat this R times to build a null distribution. This shows what differences we might see if the group labels didn't matter.

For each feature, the p-value tells us how often the shuffled results are at least as extreme as the observed result.

Each shuffle is independent, so we can easily run them in parallel.

But if every shuffle processes the full data matrix, the total work grows with R times n times p.

## Slide 15 - Each shuffle is a weight vector, so a batch is one matrix multiply (14:00)

Here is the key optimization.

The difference between two group means can be written as a weighted sum.

Skipping the math, W times X calculates the test statistics for the whole batch using just one matrix multiplication.

This is exactly what BLAS and GPUs are good at.

The second optimization is memory usage.

The reference stores the results of all R permutations for all p features. When we have half a million features, this can use a lot of memory.

Instead, we process the permutations in batches and keep a running count for each feature. We only need to count how many shuffled results are at least as extreme as the observed result.

The statistical test stays the same, but the computation is very different.

So we need to check that the optimized results still match the reference.

## Slide 16 - The matrix rewrite gives exactly the same p-values (15:15)

Well, let's check it.

Both versions use the same data and the same sequence of shuffles. This allows us to compare their results directly.

We tested 18 different problem sizes and shapes, each with 5 random seeds.

In total, 450 comparisons passed.

The p-values matched exactly, with a maximum difference of zero.

The test statistics had a maximum difference of only 10 to the minus 16, due to floating-point rounding.

Our tolerance was 10 to the minus 6, so the difference is ten orders of magnitude smaller.

This shows that the optimized version matches the reference in our tests.

But we still need to check whether the statistical test itself works correctly. That's our next step.

## Slide 17 - Under the null, the test rejects 5.1% of the time at α = 0.05 (16:15)

Now the second check about the statistical test itself.

We simulate data where there is no real difference between the two groups.

With a significance level of 0.05, a well-calibrated test should reject the null hypothesis about 5 percent of the time.

In our simulations, the rejection rate was 5.1 percent, very close to the expected 5 percent. The difference is within the range we'd expect from random variation.

So we have two checks: the optimized code matches the reference, and the statistical test gives the expected false-positive rate under the null.

Now we can move on to performance.

## Slide 18 - The A100 pays off once R × p is large: up to 8.5× faster (17:15)

Now let's look at speed.

This heatmap compares our server CPU with an A100 GPU, using the same permutation test.

The columns show the number of features, and the rows show the number of shuffles.

Each cell shows how much faster the GPU is compared with the CPU. We measure the total runtime, including data transfer to the GPU.

For the smallest problem, the CPU is faster. The GPU is only 0.7 times as fast, because the overhead is larger than the time saved.

But as the number of features and shuffles increases, the GPU becomes faster.

In the best case, with half a million features and 5,000 shuffles, the runtime drops from 10.7 seconds to 1.25 seconds. That's about 8.5 times faster.

The grey cell in the top right means the GPU ran out of memory, while the CPU could still finish.

So GPUs work best when there is enough parallel work. And our matrix rewrite makes it possible to use the GPU efficiently.

## Slide 19 - More workers help until about 16, then stop paying (18:45)

Now let's look at CPU parallelism on our server.

Going from one to four workers makes a big difference. K-means runs 58 percent faster, and the permutation test runs 46 percent faster.

But adding more workers gives smaller improvements. For the permutation test, using more than 16 workers actually makes it slower.

Now look at memory usage.

Our permutation test uses multiple processes, and in our implementation, each process has its own copy of the data.

With 128 workers, memory usage reaches 58 gigabytes.

So more workers are not always better. We need to balance speed and memory usage.

This is where free-threaded Python becomes interesting. It allows multiple threads to run in parallel while sharing data in the same process.

Let's see how that works.

## Slide 20 - Free-threaded Python 3.14t lets threads scale: 4.77× on 16 threads (19:45)

Now let's look at Python 3.14.

Here's a CPU-bound resampling task written in plain Python.

On standard Python, 16 threads are no faster than one because of the GIL.

But with the same code on free-threaded Python 3.14t, we get a 4.8 times speedup.

Compared with a process pool on standard Python, it's 2.4 times faster and uses about one-third less memory, because threads can share data.

This mainly helps CPU-bound Python code. NumPy and BLAS often release the GIL already, so the benefit may be smaller.

And of course, we also validated the results on 3.14t.

## Slide 21 - Pick the tool from the bottleneck you measured (21:15)

Here's a simple guide to choosing the right tool.

For numerical loops, try Numba. For matrix operations, use NumPy and BLAS.

For independent tasks that share data, consider free-threaded Python.

For large, parallel workloads, JAX on GPU may be the best choice.

Think of it like travel. You wouldn't swim to Taiwan. You choose the right tool for the trip.

And remember, all these choices come after validation.

## Slide 22 - AI helps write the code. Validation decides what we keep. (22:30)

We also used AI coding agents to develop Numba and JAX implementations, explore different optimizations, and automate benchmarks.

AI helps us write code, but we still need to validate every implementation.

We check numerical results against the reference, test statistical properties, and measure runtime and memory usage.

We also measure end-to-end performance, because a faster function doesn't always mean a faster program.

AI can write code quickly, but it may be wrong. Validation tells us whether to believe it.

## Slide 23 - Make the statistic testable. Then make the bottleneck fast. (23:45)

To wrap up, three main points.

First, simulate data so we know the ground truth, and keep a simple reference implementation.

Second, validate both numerical results and statistical properties.

Third, measure the bottleneck and choose the right tool.

Our workflow is simple: validate first, measure second, and optimize last.

Make the statistic testable. Then make the bottleneck fast.

## Slide 24 - Thank you! (24:30)

Thank you. Please find me at my poster in room LW661 on the sixth floor. We can continue the conversation there. You can scan the QR code for the slides, code and other materials.

## Backup 1 - On a laptop, runtime grows with both R and p

If asked how runtime scales locally: on the MacBook, with the batched NumPy path at n equals 500, runtime grows roughly linearly in R, and higher p shifts the curve up. This is what told us to move the large runs to the server and GPU.

## Backup 2 - Evidence map. Different questions.

If you are wondering why we didn't just build one massive hardware leaderboard, the answer is scope. Each environment answers a different question. The MacBook establishes local validation. It gives us local shape signals. The server CPU answers scaling and parallelism questions. The GPU only enters the picture to ask: does this validated, streamed pipeline actually win? It is an evidence map, not a leaderboard.

## Backup 3 - Validation inventory.

If you want to see the full local permutation checks, here is the inventory. It combines main-path equivalence, null calibration, and the local runtime shapes. This is all local MacBook evidence. We run all of this before we ever touch a server or a GPU.

## Backup 4 - Numba wins loop shapes.

Someone usually asks why Numba looks so fast locally but loses on the server. Look at the shape. This is local MacBook shape-stress evidence. Over a thousand rows. All passing. Numba dominates when the shape is pure explicit loop work. But when you move to a server and the shape widens into dense distance algebra, BLAS takes over. And at massive device-friendly scale, the GPU wins. It never contradicts. It just changes shape.

## Backup 5 - Power vs. effect size.

If you want statistical validation beyond just the null calibration, here is the simulated power curve. This is not a speed result. It is pure statistics. The reading is very simple. Weak effects have low power. Stronger simulated signals are easier to detect. The inference behaves exactly as it should.

## Backup 6 - Backup: what batch_R tuning showed

This backup slide is about the trend, not a proof of optimality. batch_R is the number of permutation rows processed in one A100 batch. In this committed sweep, small batches were slower because the streamed path paid more scheduling and loop overhead. Increasing batch_R reduced that overhead, but the gains flattened after roughly 512-1024 and the curve is not perfectly monotone. 8192 was the fastest accepted value in the measured 128-8192 grid, so it was used as the chosen setting for the later A100 decision map. Larger batch_R values were not part of this committed sweep; they could be faster, slower, or fail because of memory or autotuning. batch_R changes scheduling, not the permutation statistic or p-value definition. Full end-to-end timing remains the criterion: compile excluded, transfer included, kernel-only excluded.

## Backup 7 - Scaling evidence.

The py313 control is not present in the local benchmark outputs, so this backup uses only the
available py314 and py314t data. The chart normalizes each interpreter to its own one-worker baseline. The standard
GIL build stays flat. The free-threaded build scales on this CPU-bound Python workload. The pool rows show why the
execution model matters: ProcessPoolExecutor can work around the GIL, but threads avoid the memory and serialization
costs for shared data.
