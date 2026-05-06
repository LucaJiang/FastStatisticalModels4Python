# Linux server CPU expanded parallelism

Generated/updated: 2026-05-06T18:21:25+0800

## Provenance
- Run tier: Linux server CPU, not MacBook.
- Hostname: `BI103202`.
- Effective CPU count from affinity: 512.
- Scheduler: `none`; allocated CPUs: ``.
- Resource isolation: `shared_server`.

## Availability
- 64 available under affinity: `True`.
- 128 available under affinity: `True`.
- No exclusive scheduler allocation was detected, so 64/128 are marked as shared-server evidence.

## Workloads
- k-means: `n=1,000,000`, `d=64`, `K=20`, `max_iter=20`, fixed data/init/stopping, Numba parallel method.
- permutation: `n=5,000`, `p=10,000`, `R=1,000`, `batch_R=256`, deterministic same-stream process-pool method.

## Results After 16
- k-means 16/64/128 rows: `[{'thread_count': 16, 'median_warm_time_s': 6.42437343, 'row_status': 'completed'}, {'thread_count': 64, 'median_warm_time_s': 4.74813531, 'row_status': 'completed'}, {'thread_count': 128, 'median_warm_time_s': 4.42132746, 'row_status': 'completed'}]`.
- permutation 16/64/128 rows: `[{'worker_count': 16, 'median_warm_time_s': 0.902549223, 'total_peak_rss_mb': 7397.73047, 'row_status': 'completed'}, {'worker_count': 64, 'median_warm_time_s': 0.981275251, 'total_peak_rss_mb': 28120.5469, 'row_status': 'completed'}, {'worker_count': 128, 'median_warm_time_s': 1.12579955, 'total_peak_rss_mb': 57864.8203, 'row_status': 'completed'}]`.
- Permutation memory grows with worker count; see `total_peak_rss_mb`.

## Safest Slide Interpretation
- Because this was a shared server run without exclusive allocation, 128-worker/thread results are shared-server evidence, not clean evidence.
- High worker counts on a shared server are hard to interpret without CPU affinity and load checks.
- Do not claim that 128 is intrinsically bad for the algorithm; the point is to measure parallelism under explicit resource constraints.
