# Timeline - PyCon HK 2026, 30-minute slot

The deck has 23 main slides and 7 backup slides. At a relaxed pace the speaker notes run about 24 minutes, which leaves about 5 minutes for questions.

Story: a slow but trusted permutation test → our validation rule → k-means (method check, code check, shape-dependent speed) → permutation tests (matrix rewrite, equivalence, calibration, GPU, parallelism, free-threaded 3.14) → decision guide → AI tools → close.

| Time | Slides | Section |
| --- | --- | --- |
| 0:00-2:45 | 1-2 | Title with three headline numbers; the slow reference permutation test |
| 2:45-5:30 | 3-5 | Validation rule; simulation as testing; the two workloads |
| 5:30-12:00 | 6-11 | k-means: animation, recovery heatmap, equivalence, three implementations, server shape results |
| 12:00-21:45 | 12-19 | Permutation tests: animation, `W @ X` rewrite, equivalence, calibration, A100 decision map, CPU parallelism, free-threaded 3.14t |
| 21:45-23:00 | 20 | Decision guide |
| 23:00-23:45 | 21 | AI coding tools, with two concrete examples |
| 23:45-24:30 | 22-23 | Takeaway and questions |

If running long, shorten slide 5 (two workloads) and slide 18 (CPU parallelism) first.

## Backup slides

1. On a laptop, runtime grows with both R and p: local MacBook scaling.
2. Evidence map: why three environments.
3. Validation inventory: equivalence, calibration, and runtime on one figure.
4. Numba wins loop shapes: the answer to "when does Numba beat NumPy?"
5. Power vs. effect size.
6. batch_R tuning: 8192 was fastest within the measured 128-8192 grid only.
7. Standard GIL vs free-threaded: the full timing and memory table.
