# Proposal

Breaking the Speed Limit: Fast Statistical Models with Python 3.14, Numba, and JAX

## Section Type

30 minute talk

## Abstract

Data scientists and domain experts often face a dilemma: we understand the models, and we use Python, but we aren't C++ or Rust engineers. We need code that is quick to write, easy to work with, and still fast enough to run on large, real‑world datasets. How do we choose the right tool without getting lost in low‑level details?

With the new free‑threaded build and experimental JIT in Python 3.14, combined with tools like Numba and JAX, we finally have a realistic way to push back against the "Python is slow" stereotype. In this talk, we'll use two concrete workloads to illustrate this modern stack: complex iterative loops (k‑means) and massive data parallelism (permutation test). The focus is on computational patterns rather than statistical theory.

We'll compare plain NumPy, Python 3.14 (with free-threaded or JIT configurations), Numba, and JAX across varying data scales, highlighting the trade-offs in runtime, memory, debuggability, and developer experience. Along the way, we'll also demonstrate how AI coding tools can serve as a copilot, helping to translate clear mathematical code into high-performance kernels without requiring deep compiler expertise.

-----


# Outline

## Introduction (3 minutes)

* **Background and motivation:** The tension between development speed, execution speed, and safety/debuggability: we want code that is fast to write, fast to run, and still easy to reason about.
* **Evaluation matrix:**

  * **Runtime:** Behavior across different data sizes.
  * **Memory:** Extra temporaries and **copies vs shared data** in parallel code.
  * **Debuggability:** How easy it is to troubleshoot or handle numerical issues.
  * **Developer effort:** How far you have to move away from "plain NumPy".
* **Two compute patterns** (no background in statistics required):

  * **k‑means for the iterative pattern:** A loop‑heavy, sequential algorithm where each step depends on the previous step. Good for testing how different runtimes handle *tight Python loops* over arrays.
  * **Permutation Test for the parallel pattern:** A "run the same function thousands of times on different resamples" scenario. Embarrassingly parallel computation with a lot of big array math.
* **Data scales:**

  * **Toy datasets:** tens of thousands of points (great for illustration and debugging, runs on a laptop).
  * **Large-scale datasets:** tens of millions of data points or gene expression matrices (typical research scale, usually needs a server).


## Loop‑heavy k‑means (12 minutes)

* **Baseline:** k‑means clustering implemented in plain NumPy, highlighting the iterative pattern: assign points to clusters, then update centroids.
* **Numba:** Add `@njit` to the assignment and centroid‑update functions, fix unsupported features, and show how compiling the inner loops changes performance and code structure.
* **Python 3.14 with JIT (GIL build):** Take a more loop‑heavy k‑means variant using explicit Python `for` loops and run it on a JIT‑enabled CPython 3.14 build. Demonstrate that the JIT accelerates native Python control flow but has minimal impact on the NumPy‑heavy baseline, where most work is already in C.
* **JAX:** Rewrite k‑means in a functional style using JAX arrays, `jax.lax.scan` for the refinement steps, and `jax.jit` for compilation.
* **Discussion of trade‑offs:**
  * Show where the CPython JIT helps and where it doesn’t.
  * Compare speedups and costs: type restrictions, debugging experience, and how much each approach diverges from the simple NumPy baseline.


## Parallel Permutation Test (10 minutes)

* **Baseline:** Implement a permutation test using `multiprocessing` (e.g. `ProcessPoolExecutor` or `multiprocessing.Pool`) on a standard GIL build. Highlight how large arrays are serialized and effectively copied into multiple processes.
* **Free‑threaded Python 3.14:** Switch to `concurrent.futures.ThreadPoolExecutor` on the free‑threaded (no‑GIL) CPython 3.14 build, where all threads share a single in‑process copy of the data while running permutations in parallel.
* **Numba:** Package the permutation test into a compiled kernel using `@njit(parallel=True)` and `prange` to parallelize across resamples. Discuss how Numba's own thread pool gives parallel speedups while keeping everything in one process and one shared array, independent of whether CPython is free‑threaded.
* **JAX:** Express the permutation test in JAX and use `jax.vmap` (with `jax.jit`) to vectorize over permutations and batch them into a single compiled computation, optionally on a GPU/TPU.
* **Discussion of trade‑offs:**

  * Show how **naive multiprocessing** tends to create multiple large copies of the data, while free‑threaded threads, Numba, and JAX all operate on **one shared dataset** (in RAM or on device).
  * Compare performance and memory behavior between these approaches, and discuss where each one wins (e.g., simplicity vs maximum speedup).
  * Briefly touch on randomization and reproducibility: how random initialization and parallelism can change the order of operations and random draws, and simple strategies to solve this.
  * Compare developer effort and style shifts.


## Developer Experience (2 minutes)

Practical tips for working with these tools on numerical code:

* **Start simple:** Build and test your model in plain NumPy (or NumPy plus a small Numba kernel) to validate correctness before chasing performance.
* **Profile first:** Use basic profiling to identify hotspots and verify that JIT/Numba/JAX changes actually improve the end‑to‑end runtime and memory behavior.
* **AI as copilot, not driver:** Use AI tools to help translate statistical formulas to NumPy code and further Numba or JAX variants, but keep human control over correctness, numerical stability, and performance checks.


## Conclusion (1 minute)

Wrap up with a compact "take‑home" decision guide:

* When a **small refactor + Numba** is the best trade‑off for loop‑heavy numerical work.
* When the **CPython 3.14 JIT** is "good enough" as a low‑effort speedup for Python‑heavy glue code, and when the **free‑threaded build** is worth using to replace `multiprocessing` with threads for parallel workloads.
* When it's worth adopting JAX's functional style for **maximum speed and scalability**, especially with accelerators.
* How AI tools can accelerate refactoring and exploration, but should not replace human judgment.

The goal is not to crown a single winner, but to give an honest picture of **what you gain for each extra unit of complexity**, so attendees can choose the right toolchain for their own numerical workloads.

-------

# Current local findings

Latest local refresh:

- `py312`: Python 3.12.2, NumPy 1.26.4, Numba 0.59.1, JAX 0.4.25 (CPU).
- `py314t`: Python 3.14.0 free-threaded build, confirmed `sys._is_gil_enabled() == False`.
- k-means at `N=1M`, `k=5`, `d=10`: Numba `0.482 s`, JAX `0.499 s`, NumPy naive `5.22 s`.
- permutation test at `n=10k`, `R=10k`: Numba `0.064 s`, ThreadPool on `py314t` `0.173 s`, NumPy loop `0.856 s`, JAX CPU `37.4 s`.
- See [`experiments/results/README.md`](experiments/results/README.md) for the refreshed figures and commands.

-------

# Notes

## What motivates us to submit this proposal

We are PhD candidates in Biostatistics working with high‑dimensional genomic data. This talk is based on our experience with statistical modeling in Python, particularly in the context of large‑scale genomic data analysis, where runtime and memory efficiency are crucial.

Unlike software engineers, statisticians prioritize **correctness, reproducibility, and ease of collaboration** over raw performance. However, as datasets grow larger, the need for efficient computation becomes unavoidable. Historically, the answer was "rewrite the hot path in C/C++". This is the current situation for most statisticians in our research area: they rewrite their R code using Rcpp, which not only requires a deep knowledge of C++ but also results in a fragile package dependency. Is there a better way?

With the advent of Python 3.14's free‑threaded build and experimental JIT, along with mature tools like Numba and JAX, we now have a more accessible path to high performance without leaving Python. This talk aims to share practical insights and benchmarks that help statisticians and data scientists make informed decisions about optimizing their Python code.

In our research group, we have successfully applied these techniques to accelerate various statistical models, including EM algorithms and moment-based estimators, demonstrating significant speedups while maintaining code clarity and reproducibility. We believe that our experiences can provide valuable guidance to others facing similar challenges in the Python ecosystem. Thus, we are motivated to share our findings and help the broader community navigate the evolving landscape of high-performance Python for statistical computing.

While the examples come from Biostatistics, the core ideas are general to anyone doing numerical Python. We focus on **runtime behavior, memory layout, and developer workflow**, not on the underlying statistical theory. Developers of numerical computing packages can also benefit from this talk by gaining insight into the user's perspective on performance trade-offs in Python. Therefore, we believe this talk will resonate with a wide audience in the Python community.

## Previous Experience

We have given talks at PyCon HK 2025, focusing on functional programming in Python and new features in recent Python releases (especially Python 3.14). Although the YouTube video is not available yet, here are our slides for reference:

1. [PyCon HK 2025: Functional Programming in Python](https://lucajiang.github.io/functional_python/)
2. [PyCon HK 2025: Shall We Upgrade? Navigating Python's Rapid Evolution](https://lucajiang.github.io/new_in_python/)

---

## Repository layout

- **`experiments/`** — Current experiment surface: shared k-means implementations (`kmeans/`), permutation validation (`permutation/`), server/A100 long-safe orchestration (`server/`), visualization, setup, and curated results. See each subdirectory `README.md`.
- **`slides/`** — Reveal.js HTML deck for the talk ([`slides/index.html`](slides/index.html)); see [`slides/README.md`](slides/README.md).
