# A100 permutation decomposition QA note

Generated/updated: 2026-05-05T20:24:42+0800

- Figure scope: warm full scenario, not per batch and not kernel-only.
- Compile time: excluded from the plotted bars; `compile_time_s` remains in the CSV.
- Transfer: included. The transfer segment includes X/observed-stat transfer plus host-built W batch transfers during the timed scenario.
- CPU comparison: full CPU end-to-end rows from `cpu_matched_permutation_baseline.csv`; no kernel-only A100 time is compared to CPU end-to-end.

## Stage Sum Check
- R=1,000: stage sum 0.180335s, recorded total 0.181576s, delta 0.001240s.
- R=10,000: stage sum 0.823712s, recorded total 0.827491s, delta 0.003779s.

## CPU Rows
- R=1,000: CPU full end-to-end 0.436249s; timeout_status=completed.
- R=10,000: CPU full end-to-end 2.545338s; timeout_status=completed.
