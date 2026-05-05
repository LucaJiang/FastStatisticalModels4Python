# CPU vs A100 permutation break-even

Generated/updated: 2026-05-05T21:23:58+0800

## CPU baseline
- Matched CPU matrix baseline used here: `numpy_matrix_same_stream` batched matrix path.
- Scope: this is not an exhaustive best-of-all-CPU search; speedup means matched CPU matrix baseline divided by A100 streamed full end-to-end.
- CPU rows recorded: 27.

## Break-even
- A100 becomes faster at n=5000, p=10000, R=5000, batch_R=8192.
- Largest measured speedup: 8.54x at n=5000, p=500000, R=5000.

## Streamed reduction
- `a100_streamed_reduction` computes `T_null_batch = W_batch @ X_device`, accumulates exceedance counts on device, and collects final p-values/counts only.
- It preserves the same statistic and same host W permutation stream used for CPU checks.
- The break-even map uses this streamed full end-to-end path. A separate full-collection A100 break-even row was not used, so no speedup is claimed from streaming alone; the measured benefit is that the full `R x p` null matrix is not collected.

## Kernel-only vs end-to-end
- Kernel-only rows are labeled as not full permutation tests and are not used for CPU/A100 speedup decisions.

## Timing semantics
- CPU/A100 comparisons are full scenario end-to-end, warm timing, compile excluded, transfer included for A100.
- Representative decomposition rows report named stages plus residual Python/JAX loop overhead. Figures include this residual as `other overhead` so stacked bars reconcile to `total_end_to_end_time_s`.
- Kernel-only rows time only `W @ X` with device-resident inputs and are labeled as hypotheses, not full permutation tests.

## Batch_R
- Best safe batch_R from Stage 1: 8192.

## Correctness
- Correctness check rows: 97.
  - check: 97
- Historical `check` rows mean accepted bounded CPU/JAX comparisons under the then-current status vocabulary; they are not exact `pass` rows.
- New accepted GPU rows should be emitted as `pass_gpu_tolerance`.
- The result CSVs with `max_abs_p_diff` and `max_abs_stat_diff` are generated on the experiment server but are not committed in this repository snapshot.

## OOM / memory-risk / timeout
- `batch_R_sweep.csv`: 0 timeout/skipped/memory-risk/fail rows.
- `break_even_shape_sweep.csv`: 2 timeout/skipped/memory-risk/fail rows.
- `n_sensitivity_sweep.csv`: 0 timeout/skipped/memory-risk/fail rows.
- `cpu_matched_baselines.csv`: 0 timeout/skipped/memory-risk/fail rows.
