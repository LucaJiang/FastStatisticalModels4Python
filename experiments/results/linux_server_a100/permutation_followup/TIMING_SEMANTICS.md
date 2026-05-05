# A100 permutation timing semantics audit

Generated/updated: 2026-05-05T20:24:42+0800

## Source

- Decomposition CSV: `a100_permutation_decomposition.csv`.
- Producing script: `experiments/server/a100_permutation_followup.py`, function `run_decomposition`.
- Figure script path: `make_clean_decomposition_figure` in the same file.

## Timing meanings

- `compile_time_s`: JAX compile and warm-up time for the staged functions; excluded from the plotted full-scenario bars.
- `warm_time_s`: one warm preflight call on the first batch shape; not a scenario runtime and not plotted as a total.
- `end_to_end_time_s` and `total_end_to_end_time_s`: warm full-scenario runtime for the follow-up A100 path, excluding compile and including transfer of X, observed statistics, and each host-built W batch.
- `permutation_generation_time_s`: host NumPy permutation index generation summed across all batches in the scenario.
- `W_build_host_time_s`: host construction/filling of W batches summed across the scenario.
- `host_to_device_transfer_time_s`: transfer of X, observed statistics, and W batches to A100 during the timed scenario.
- `device_compute_time_s`: device-resident `W @ X`/absolute-statistic work summed across the scenario.
- `device_to_host_collect_time_s`: collection of the reduced p-values/statistics needed by the benchmark, not the full `R x p` null matrix.
- `pvalue_reduction_time_s`: device-side exceedance count reduction summed across the scenario.
- Kernel-only rows live in `a100_permutation_kernel_only.csv` and are explicitly not used as end-to-end totals.

## Stage sum check

- R=1,000, batch_R=4,096, n_batches=1: stage sum 0.180335s vs recorded total 0.181576s; delta 0.001240s; match=True.
- R=10,000, batch_R=4,096, n_batches=3: stage sum 0.823712s vs recorded total 0.827491s; delta 0.003779s; match=True.

## Reconciliation with previous A100 matched-slice result

The previous long-safe A100 rows in `linux_server_a100/long_safe_20260503_190133/permutation_matrix_gpu.csv` measured the prior `perm_a100` JAX GPU permutation path. That path generated permutations/W on device with `jax.random.permutation`, used the old matched slide setting `batch_R=512`, and reported warm scenario runtimes in the tens of seconds for `n=5,000, p=50,000`.

This follow-up decomposition measures a different correctness-preserving implementation, `jax_host_w_same_stream`: W is built on the host from the same NumPy seed stream used by the trusted CPU matrix check, W batches are transferred to A100, JAX compile is excluded from the warm full-scenario total, and `batch_R=4,096` was selected from the batch-size sweep. The new decomposition should therefore not be used as a direct replacement for the old implementation's tens-of-seconds runtime without this label.

CPU comparison rows in `cpu_matched_permutation_baseline.csv` are full-scenario CPU end-to-end timings for the same host-W stream and `batch_R=4,096`; they are not compared against kernel-only A100 timings.
