# Break-even Slide QA

Generated: 2026-05-05

- Figure type: full scenario timing for the CPU-vs-A100 decision map and representative A100 decomposition.
- Compile time: excluded from warm end-to-end comparisons; compile is recorded separately in the decomposition CSV.
- Transfer: included for A100 end-to-end and streamed-reduction rows.
- Stage sums: named decomposition stages do not exhaust `total_end_to_end_time_s`; the figure adds an explicit `other overhead` segment so stacked bars reconcile to the recorded total.
- CPU comparison: matched full end-to-end CPU matrix baseline, not per-batch, not kernel-only, and not an exhaustive best-of-all-CPU search.
- Kernel-only rows: labeled separately as `kernel-only hypothesis, not end-to-end permutation test` and excluded from speedup decisions.
- Correctness: accepted GPU-tolerance rows use `pass_gpu_tolerance`; older bounded-check rows may show `check`; no speedup plot uses failed rows.
- Targeted rerun: on 2026-05-06, the two previously memory-risk Stage 2 cells (`n=5000`, `p=500000`, `R=10000/50000`, `batch_R=8192`) were rerun with CPU timeout 14,400 seconds. CPU completed; canonical A100 streamed full end-to-end still failed during JAX autotune/OOM and is labeled `A100 OOM` on the decision map.
- Representative decomposition: updated to four rows from the canonical break-even grid: CPU-faster, near break-even, A100-faster, and largest/highest-speedup. The summary CSV records `other_overhead_s`, `stage_sum_s`, `stage_sum_delta_s`, and `wx_share`.
- Visual check: updated decision and decomposition slides were exported to PDF and screenshotted at 1280x720.
