# Benchmark contract v3

This is not a universal benchmark. It is a reproducible local experiment under a stated contract.

## Required environment record

Every formal run records:

- Python version and executable.
- OS, CPU, RAM, and filesystem availability.
- NumPy, SciPy, Numba, scikit-learn, matplotlib, pandas, pyarrow, JAX, and jaxlib versions.
- JAX default backend and `jax.devices()`.
- `nvidia-smi` output when available.
- Free-threaded status from `sys._is_gil_enabled()` and `sysconfig.get_config_var("Py_GIL_DISABLED")`.
- CPython JIT status from `sys._jit.is_available()` and `sys._jit.is_enabled()` when present.

Use `experiments/common/capture_environment.py` and write JSON under `experiments/results/v3/`.

## Timing rules

- Run validation before timing.
- Separate cold first-call time from warm execution time.
- Report median and IQR for warm runs.
- For JAX, always call `block_until_ready()` before stopping the timer.
- Record whether device transfer time is included.
- Use chunking for workloads where full materialization would dominate memory.
- Record unavailable tools and skipped cases instead of fabricating results.

## Result row fields

Rows should include:

`workload, implementation, backend, device, n, p, d, k, B, seed, cold_time_s, warm_median_s, warm_iqr_s, host_peak_mem_mb, gpu_peak_mem_mb, validation_status, notes`

Additional workload-specific fields may be added, but these fields are the shared reporting surface for v3 plots and summaries.

## Claim policy

No speedup should be reported without a matching validation status. GPU, free-threaded Python, and CPython JIT should only be described as measured when the environment capture and benchmark rows confirm them.
