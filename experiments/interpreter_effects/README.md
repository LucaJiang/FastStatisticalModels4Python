# CPython 3.14 Interpreter Effects

This suite compares standard CPython 3.14 (`py314`) with free-threaded
CPython 3.14 (`py314t`) on statistical-computing shaped workloads.

It is deliberately scoped to interpreter effects. Do not describe a result as a
JIT speedup unless the generated metadata reports both:

- `python.jit.available: true`
- `python.jit.enabled: true`

The runner also writes `jit_claim_allowed` into metadata and CSV rows.

## Outputs

Each interpreter run writes:

- `raw_interpreter_effects_<env>.csv`: warmup and repeated measurements.
- `summary_interpreter_effects_<env>.csv`: repeat-only median and IQR summaries.
- `metadata_<env>.json`: environment report from `experiments.common.env_report`.

The plotting command writes slide-ready PNG/SVG files under `figures/`.

## Exact Commands

Run from the repository root. The BLAS thread pins are repeated here so the
intent is visible before Python imports NumPy; the runner also forces these
values inside the process.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Standard CPython 3.14 GIL build:

```bash
conda run -n py314 python -m experiments.interpreter_effects.run_suite \
  --env-label py314 \
  --output-dir experiments/results/python314_interpreter_effects/latest \
  --experiments negative thread memory \
  --repeats 5 \
  --warmups 1
```

Free-threaded CPython 3.14:

```bash
conda run -n py314t python -m experiments.interpreter_effects.run_suite \
  --env-label py314t \
  --output-dir experiments/results/python314_interpreter_effects/latest \
  --experiments negative thread memory \
  --repeats 5 \
  --warmups 1
```

Optional contention backup:

```bash
conda run -n py314t python -m experiments.interpreter_effects.run_suite \
  --env-label py314t \
  --output-dir experiments/results/python314_interpreter_effects/latest \
  --experiments contention \
  --repeats 5 \
  --warmups 1
```

Plot with the repository default development environment:

```bash
conda run -n py312 python -m experiments.interpreter_effects.plot_interpreter_effects \
  --results-dir experiments/results/python314_interpreter_effects/latest
```

For smoke testing only:

```bash
conda run -n py312 python -m experiments.interpreter_effects.run_suite \
  --env-label py312_smoke \
  --output-dir /tmp/fsm4py_interpreter_effects_smoke \
  --experiments negative thread memory contention \
  --quick
```

## Experiments

1. Single-thread negative control at `workers=1`
   - Pure Python CPU loop.
   - NumPy/BLAS-heavy matrix path.
   - Small statistical loop.

2. Thread scaling
   - `ThreadPoolExecutor` workers `1, 2, 4, 8, 16`.
   - CPU-bound Python permutation/bootstrap-like statistic.
   - Independent per-worker accumulators.

3. ProcessPool vs ThreadPool memory/runtime
   - `py314` defaults to `ProcessPoolExecutor`.
   - `py314t` defaults to `ThreadPoolExecutor`.
   - A large simulated NumPy array is shared by threads and copied into spawned
     worker processes.
   - Measures `wall_time_sec` and parent-plus-child `peak_rss_gb`.

4. Optional backup
   - Thread-local accumulation compared with shared mutable counter/list/dict.
   - Intended to show that contention can erase no-GIL speedups.

## Notes

- The process-pool memory comparison uses the `spawn` start method so child
  workers receive their own copy of the simulated array. This makes the memory
  contrast explicit and portable across platforms.
- The generated figures are suitable for slides, but slide benchmark numbers
  should only be updated after inspecting the generated CSVs.
