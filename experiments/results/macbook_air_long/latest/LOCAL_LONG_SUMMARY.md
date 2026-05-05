# MacBook Air evidence summary

Run directory: `experiments/results/macbook_air_long/latest`  
Base full run: `experiments/results/macbook_air_long/20260503_full_x64`  
Updated: 2026-05-03

## Environment

- Python: 3.12.2
- NumPy: 1.26.4
- Numba: 0.59.1
- JAX: 0.4.25 on CPU with x64 enabled for correctness checks
- CPU cores: 8
- RAM GB: 16.0

## k-means

Full validation grid:

- Required scenarios covered: 1,440 / 1,440
- Rows: 4,320
- Status counts: `{'pass': 3840, 'skipped_memory_risk': 480}`
- Mean ARI by separation: `{'0.5': 0.2662, '1.0': 0.4084, '2.0': 0.4869, '4.0': 0.5199}`
- The 480 skipped rows are the intentionally unsafe reference-broadcast path on large `N*K*d` cases.

Targeted shape-stress evidence:

- Rows: 1,164
- Status counts: `{'pass': 1164}`
- Scenarios add `K=20/50`, `d=100`, `N=50,000`, and selected `N=100,000` cases.
- Median warm runtime over shape-stress rows: NumPy matmul 0.2485 s, Numba 0.1020 s.

## permutation test

Full equivalence and validation grid:

- Equivalence rows: 495, status counts: `{'pass': 450, 'skipped_memory_risk': 45}`
- Calibration rows: 20, status counts: `{'pass': 20}`
- Power rows: 15, status counts: `{'pass': 15}`
- Mean null `p <= 0.05`: 0.048 in the full grid

Extended evidence:

- Calibration rows: 100, status counts: `{'pass': 100}`
- Extended calibration mean `p <= 0.05`: 0.051
- Power rows: 168, status counts: `{'pass': 168}`
- Runtime scaling rows: 135, status counts: `{'pass': 108, 'skipped_memory_risk': 27}`
- Extended power at delta 0.1 is still low, around 0.15-0.19 depending on signal fraction; by delta 0.5 it reaches 1.0 across the tested signal fractions.

## Primary figures

Use `figure_manifest.csv` as the canonical figure list.

- `figures/kmeans_recovery_scenario_facets.png`
- `figures/kmeans_shape_stress_runtime.png`
- `figures/kmeans_runtime_recovery_tradeoff.png`
- `figures/kmeans_reference_equivalence.png`
- `figures/permutation_calibration_extended.png`
- `figures/permutation_power_extended.png`
- `figures/permutation_runtime_scaling_extended.png`
- `figures/permutation_equivalence_detail.png`

Deprecated quick-validation outputs and pre-x64 failed intermediate runs were removed from `latest`.
