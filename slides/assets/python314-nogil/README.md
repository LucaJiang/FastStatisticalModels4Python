# Python No-GIL Slide Figures

This directory contains PNG-only slide figures generated from local benchmark
outputs. The deck uses the generated PNG files and `derived_metrics.json`; it
does not perform live plotting.

## Data Provenance

Source summary CSV files:

- `experiments/results/python314_interpreter_effects/latest/summary_interpreter_effects_py314.csv`
- `experiments/results/python314_interpreter_effects/latest/summary_interpreter_effects_py314t.csv`

Source metadata JSON files:

- `experiments/results/python314_interpreter_effects/latest/metadata_py314.json`
- `experiments/results/python314_interpreter_effects/latest/metadata_py314t.json`

Column mapping:

- `env_label`: `py313` is standard CPython 3.13 with the GIL enabled when present; `py314` is standard CPython 3.14 with the GIL enabled; `py314t` is CPython 3.14 free-threaded / no-GIL with the GIL disabled.
- `experiment`: `thread_scaling` feeds the thread-scaling figures; `pool_memory_runtime` feeds the execution-model metric cards/table.
- `pool`: `process` identifies the ProcessPoolExecutor row; `thread` identifies the ThreadPoolExecutor row.
- `workers`: worker count for each measurement row.
- `median_wall_time_sec`: repeat-only median runtime used for plotted values and speedups.
- `iqr_wall_time_sec`: repeat-only interquartile range used for runtime uncertainty bands where shown.
- `max_peak_rss_gb`: peak RSS value used for memory comparison.
- `jit_claim_allowed`: verifies whether any Python 3.14 JIT acceleration claim is allowed.

Rows used:

- `env_label == "py313"` and `experiment == "thread_scaling"`: optional standard CPython 3.13 GIL thread-scaling control.
- `env_label == "py314"` and `experiment == "thread_scaling"`: standard CPython 3.14 GIL thread-scaling series.
- `env_label == "py314t"` and `experiment == "thread_scaling"`: free-threaded / no-GIL thread-scaling series.
- `env_label == "py314"`, `experiment == "pool_memory_runtime"`, `pool == "process"`: ProcessPoolExecutor comparison row.
- `env_label == "py314t"`, `experiment == "pool_memory_runtime"`, `pool == "thread"`: ThreadPoolExecutor comparison row.

## How To Regenerate

Run from the repository root:

```bash
conda run -n py312 python slides/assets/python314-nogil/make_figures.py
```

Required Python packages: `numpy`, `pandas`, and `matplotlib`.

## Generated Files

- `thread_scaling_runtime_hero.png`
- `thread_scaling_speedup_backup.png`
- `derived_metrics.json`
- `README.md`

## Missing py313 Control Data

py313 control data was not found. To answer the version-baseline question, rerun
the thread-scaling benchmark under standard CPython 3.13 and regenerate the
figures:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n py313 python -m experiments.interpreter_effects.run_suite \
  --env-label py313 \
  --output-dir experiments/results/python314_interpreter_effects/latest \
  --experiments thread \
  --repeats 5 \
  --warmups 1

conda run -n py312 python slides/assets/python314-nogil/make_figures.py
```

## Interpretation

- py313 vs py314 is a control for ordinary CPython version changes when py313 data exists locally.
- py314 vs py314t isolates the free-threaded/no-GIL execution-model change within the Python 3.14 generation.
- If py313 and py314 both stay flat across workers, the result supports the claim that standard GIL threads do not scale this CPU-bound Python workload shape.
- If py314t scales while py313/py314 do not, the result supports the execution-model claim: no-GIL makes ThreadPool viable for this workload shape.
- This does not claim Python 3.14 JIT acceleration.
- This does not claim no-GIL makes all Python code faster.

Slide callouts generated in `derived_metrics.json`:

- Standard GIL card: `~1.02×`; `py314 stays flat`.
- py314t no-GIL card: `4.77×`; `0.565s → 0.119s`.
- Execution-model payoff card: `2.43× faster`; `32% less peak RSS`.

Standard py314 changes from 0.584s
at 1 worker to 0.574s
at 16 workers, a 1.02×
speedup. In this local run, that is little/no thread scaling.

py314t changes from 0.565s
at 1 worker to 0.119s
at 16 workers, a 4.77×
speedup against its own 1-worker baseline.

For the 4-worker execution-model comparison, py314 ProcessPool
took 0.571s and py314t
ThreadPool took 0.235s, so the
thread-pool row is 2.43× faster.
Peak RSS changes from 1.017 GiB to
0.692 GiB, a
32.0% reduction.

## Limitations / Non-Claims

- This benchmark does not claim Python 3.14 JIT acceleration.
- This benchmark does not show that no-GIL makes all Python code faster.
- This benchmark is relevant when work can be split into reasonably independent thread tasks over shared data.
- Workloads dominated by NumPy, BLAS, GPU kernels, I/O, locks, synchronization, or shared mutable state may behave differently.
- A plateau at higher worker counts should be interpreted as overhead, memory bandwidth, scheduling, or task-granularity limits, not as a failure of free-threaded Python.
