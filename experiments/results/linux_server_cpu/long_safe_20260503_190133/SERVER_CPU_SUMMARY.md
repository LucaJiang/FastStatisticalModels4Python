# Linux server CPU long-safe summary

Generated/updated: 2026-05-04T14:48:02+0800

## CSV status

- `kmeans_cpu_scaling.csv`: 540 rows
  - pass: 540
  - duplicate scenario_id rows: 0
- `kmeans_numba_thread_sweep.csv`: 8 rows
  - pass: 8
  - duplicate scenario_id rows: 0
- `permutation_calibration_server_subset.csv`: 2 rows
  - pass: 2
  - duplicate scenario_id rows: 0
- `permutation_cpu_scaling.csv`: 108 rows
  - pass: 105
  - timeout: 3
  - duplicate scenario_id rows: 0
- `permutation_worker_sweep.csv`: 8 rows
  - pass: 8
  - duplicate scenario_id rows: 0

## Notes

- Quick-run outputs are separate; this directory is the long-safe run surface.
- Skipped rows are intentional memory/load guardrail outcomes.
