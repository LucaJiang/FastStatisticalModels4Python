# Guideline of this project

Use `conda` to manage Python environments for this repository.

## Environments on this machine

```bash
conda activate py312   # baseline: NumPy / Numba / JAX / plotting
conda activate py314   # standard CPython 3.14 (GIL build)
conda activate py314t  # CPython 3.14 free-threaded build
```

## What each env is for

- `py312`: main development environment. Use this for NumPy, Numba, JAX, plotting, and most benchmark runs.
- `py314`: standard CPython 3.14 environment for interpreter-feature checks.
- `py314t`: free-threaded CPython 3.14 environment for optional thread-scaling checks.

## Important notes

- The current `conda-forge` `py314` build exposes `sys._jit`, but `sys._jit.is_available()` is still `False` on this machine. Treat it as a standard 3.14 environment, not a confirmed JIT-enabled build.
- `py314t` is confirmed free-threaded: `sys._is_gil_enabled()` returns `False`.
- Current curated experiments use `py312` locally. Historical benchmark drivers were removed after the MacBook correctness and server long-safe result tiers became the active evidence.

## Verified commands

```bash
conda run -n py312 python -m experiments.run_macbook_evidence_extra \
  --output-dir experiments/results/macbook_air_long/latest --checkpoint-every 20 --max-iter 15

conda run -n py312 python -m experiments.visualization.plot_macbook_air_evidence \
  --results-dir experiments/results/macbook_air_long/latest

conda run -n py312 python -m experiments.visualization.plot_server_talk_evidence
```
