# A100 permutation follow-up

Generated/updated: 2026-05-05T20:24:42+0800

This suite preserves the feature-wise two-group mean-difference statistic and uses host-built W batches from one NumPy seed stream for the A100 path and trusted CPU checks.

This is an earlier follow-up suite. It is useful for pipeline decomposition and for documenting why streamed reduction changed the story, but it is not the canonical slide-level break-even boundary. The canonical slide boundary is in `experiments/results/linux_server_a100/permutation_break_even/README.md`.

## CSV status
- `a100_permutation_decomposition.csv`: 2 rows
  - check: 2
- `a100_permutation_batch_sweep.csv`: 6 rows
  - check: 6
- `a100_permutation_shape_sweep.csv`: 29 rows
  - check: 29
- `a100_permutation_kernel_only.csv`: 35 rows
  - check: 35
- `cpu_matched_permutation_baseline.csv`: 2 rows
  - check: 2

## Correctness
- Accepted small matched CPU/JAX subset rows: 74.
- Historical raw rows in this suite use `check`; these are accepted bounded checks under the older status vocabulary, not exact pass rows.
- New runs emit `pass_exact`, `pass_gpu_tolerance`, `manual_check`, or `fail`. Accepted A100 float32 rows should normally be `pass_gpu_tolerance`; tiny statistic differences can flip about one permutation count in p-values.
- `max_abs_p_diff` and `max_abs_stat_diff` are from a bounded small matched subset for each benchmark row; large rows are timed without changing the statistic or permutation stream.

## End-to-end vs kernel-only
- End-to-end rows include W construction, host-to-device transfer, `W @ X`, p-value reduction, and collection of reduced results.
- Kernel-only rows are labeled `kernel-only hypothesis, not end-to-end permutation test`; they time only device-resident `W_batch @ X_device`.

## Bottleneck summary
- Largest recorded named A100 end-to-end stage in this run: `permutation_generation_time_s`.
- Earlier follow-up boundary observed here: A100 becomes faster at n=5000, p=50000, R=10000, batch_R=4096.
- Final slide-level canonical boundary is superseded by the break-even suite: n=5000, p=10000, R=5000, batch_R=8192.

## Unavailable / OOM / timeout rows
- None recorded in this run; Stage 3 stress rows completed within the memory guard.

## Figures
- `figures/figure1_a100_end_to_end_decomposition.png`
- `figures/a100_permutation_decomposition_clean.png` and `experiments/results/presentation_figures/a100_permutation_decomposition_clean.{png,svg}`
- `figures/figure2_a100_batch_R_sweep.png`
- `figures/figure3_cpu_vs_a100_break_even_map.png`
- `figures/figure4_kernel_only_vs_end_to_end.png`
