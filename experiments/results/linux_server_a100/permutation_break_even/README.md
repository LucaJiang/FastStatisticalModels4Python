# CPU vs A100 permutation break-even

Generated/updated: 2026-05-08T01:34:00+0800

## CPU baseline
- Matched CPU matrix baseline used here: `numpy_matrix_same_stream` batched matrix path.
- Scope: this is not an exhaustive best-of-all-CPU search; speedup means matched CPU matrix baseline divided by A100 streamed full end-to-end.
- CPU summary rows committed: 2.

## Break-even
- A100 becomes faster in the streamed-reduction grid at n=5,000, p=10,000, R=5,000, batch_R=8,192.
- Largest slide-level measured speedup: 8.54x at n=5,000, p=500,000, R=5,000.
- Speedups in lightweight summary CSVs are slide-level summaries where raw CPU/A100 timing pairs are not committed unless the CSV says otherwise in `timing_note`.

## Streamed reduction
- `a100_streamed_reduction` computes `T_null_batch = W_batch @ X_device`, accumulates exceedance counts on device, and collects final p-values/counts only.
- It preserves the same statistic and same host W permutation stream used for CPU checks.
- The break-even map uses this streamed full end-to-end path. A separate full-collection A100 break-even row was not used, so no speedup is claimed from streaming alone; the measured benefit is that the full `R x p` null matrix is not collected.

## Kernel-only vs end-to-end
- Kernel-only rows are labeled as not full permutation tests and are not used for CPU/A100 speedup decisions.

## Representative A100 decomposition
- Representative summary rows committed: 2.
- The committed lightweight summary covers one CPU-faster row and the largest/highest-speedup row. The raw 4-row representative decomposition CSV is not committed in this repository snapshot, so do not claim four representative categories from the committed evidence.
- `decomposition_representative_shapes_summary.csv` includes explicit other overhead and stage-sum reconciliation for the two committed summary rows.

## Timing semantics
- CPU/A100 comparisons are full scenario end-to-end, warm timing, compile excluded, transfer included for A100.
- Representative decomposition rows report named stages plus residual Python/JAX loop overhead. Figures include this residual as `other overhead` so stacked bars reconcile to `total_end_to_end_time_s`.
- Kernel-only rows time only `W @ X` with device-resident inputs and are labeled as hypotheses, not full permutation tests.

## Batch_R
- Best safe batch_R from Stage 1: 8192.
- The main deck uses batch_R=8192 in the A100 decision map and keeps the batch-size sweep in backup for Q&A.
- `batch_R_sweep_summary.csv` is the committed lightweight source for the backup batch-size tuning figure.
- batch_R is an A100 pipeline tuning choice, not a new statistical method; it changes scheduling while preserving the same permutation statistic and p-value definition.

## Targeted rerun policy
- 2026-05-06 targeted rerun covered only cells previously marked timeout/skipped/unavailable/memory-risk in the Stage 2 break-even grid.
- Targeted CPU timeout was raised to 14,400 seconds (4 hours) per cell; both targeted CPU baselines completed.
- Targeted A100 reruns kept the canonical definition: streamed full end-to-end path, compile excluded, transfer included, kernel-only excluded, batch_R=8,192.
- The targeted A100 rerun used `XLA_PYTHON_CLIENT_PREALLOCATE=false` and retried with `TF_GPU_ALLOCATOR=cuda_malloc_async`; the two p=500,000 high-R cells still failed during JAX autotune/OOM at canonical batch_R.
- `targeted_rerun_audit.csv` records the old memory-risk state and the new explicit CPU-completed/A100-OOM state.

## Correctness
- Correctness rows represented in the committed lightweight correctness summary: 97 accepted bounded/GPU-tolerance rows.
- Two high-R p=500,000 cells are A100 OOM/unavailable in the break-even shape summary after the targeted rerun; they are not CPU wins and not hidden speedups.
- No committed lightweight summary row records a correctness failure.
- Accepted rows use the explicit status vocabulary `pass_exact`, `pass_gpu_tolerance`, `manual_check`, and `fail`.
- Older generated raw rows may show `check`; treat that as a historical accepted bounded-check status, not an exact pass.
- Lightweight summary CSVs with slide-level rows are committed in this directory.

## OOM / memory-risk / timeout
- `break_even_shape_sweep_summary.csv`: 2 timeout/skipped/memory-risk/unavailable/OOM/fail rows.
- `cpu_matched_baselines_summary.csv`: 0 timeout/skipped/memory-risk/unavailable/OOM/fail rows.
