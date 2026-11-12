# Timeline - PyCon HK 2026, 30-minute slot

The deck has 24 main slides and 7 backup slides. At a relaxed pace the speaker notes run about 25 minutes, which leaves about 5 minutes for questions.

Story: about me and prusie → a slow but trusted permutation test → our validation rule → k-means (method check, code check, shape-dependent speed) → permutation tests (matrix rewrite, equivalence, calibration, GPU, parallelism, free-threaded 3.14) → decision guide → AI tools → close.

| Time | Slides | Section |
| --- | --- | --- |
| 0:00-0:45 | 1 | Title with three headline numbers |
| 0:45-2:00 | 2 | About me; prusie as a real-world example of "match the reference, then make it fast" |
| 2:00-3:30 | 3 | The slow reference permutation test |
| 3:30-6:15 | 4-6 | Validation rule; simulation as testing; the two workloads |
| 6:15-12:45 | 7-12 | k-means: animation, recovery heatmap, equivalence, three implementations, server shape results |
| 12:45-22:30 | 13-20 | Permutation tests: animation, `W @ X` rewrite, equivalence, calibration, A100 decision map, CPU parallelism, free-threaded 3.14t |
| 22:30-23:45 | 21 | Decision guide |
| 23:45-24:30 | 22 | AI coding tools: AI speeds up writing code, validation still comes first |
| 24:30-25:15 | 23-24 | Takeaway and questions |

If running long, shorten slide 6 (two workloads) and slide 19 (CPU parallelism) first.

## Backup slides

1. On a laptop, runtime grows with both R and p: local MacBook scaling.
2. Evidence map: why three environments.
3. Validation inventory: equivalence, calibration, and runtime on one figure.
4. Numba wins loop shapes: the answer to "when does Numba beat NumPy?"
5. Power vs. effect size.
6. batch_R tuning: 8192 was fastest within the measured 128-8192 grid only.
7. Standard GIL vs free-threaded: the full timing and memory table.
