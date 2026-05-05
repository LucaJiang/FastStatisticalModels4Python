# Break-even Slide QA

Generated: 2026-05-05

- Figure type: full scenario timing for the CPU-vs-A100 decision map and representative A100 decomposition.
- Compile time: excluded from warm end-to-end comparisons; compile is recorded separately in the decomposition CSV.
- Transfer: included for A100 end-to-end and streamed-reduction rows.
- Stage sums: named decomposition stages do not exhaust `total_end_to_end_time_s`; the figure adds an explicit `other overhead` segment so stacked bars reconcile to the recorded total.
- CPU comparison: matched full end-to-end CPU matrix baseline, not per-batch, not kernel-only, and not an exhaustive best-of-all-CPU search.
- Kernel-only rows: labeled separately as `kernel-only hypothesis, not end-to-end permutation test` and excluded from speedup decisions.
- Correctness: historical rows are marked `check`; future accepted GPU-tolerance rows should be marked `pass_gpu_tolerance`; no speedup plot uses failed rows.
- Visual check: updated decision and decomposition slides were exported to PDF and screenshotted at 1280x720.
