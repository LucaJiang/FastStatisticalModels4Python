# Linux server CPU targeted parallelism

Generated/updated: 2026-05-06T18:07:48+0800

## Provenance
- Run tier: Linux server CPU, not MacBook.
- Hostname: `BI103202`.
- Slide 23 main counts: 1, 4, and 16 workers/threads.

## Workloads
- k-means: `n=1,000,000`, `d=64`, `K=20`, `max_iter=20`, fixed seed/init, same stopping rule.
- permutation: `n=5,000`, `p=10,000`, `R=1,000`, `batch_R=256`, fixed deterministic per-permutation seed stream.

## Methods
- k-means: `numba_cpu_serial`, `numba_cpu_parallel`, and `numpy_blas_or_matmul`.
- permutation: `numpy_matrix_same_stream` baseline and `process_pool_same_stream` for 1/4/16 process workers.

## Thread isolation
- Numba rows set `NUMBA_NUM_THREADS` to the requested count and set BLAS/OpenMP thread env vars to 1.
- NumPy/BLAS rows set BLAS/OpenMP env vars to the requested count and `NUMBA_NUM_THREADS=1`.
- Process-worker permutation rows set BLAS/OpenMP/Numba env vars to 1 inside each worker row.

## Shared-server load
- Load status values observed: `ok`.
- Rows are not discarded for shared-server load; `load_status` records whether the row was run while the server appeared busy.

## Safest interpretation
- This is controlled Linux server CPU evidence for small realistic counts, not MacBook performance.
- More workers are not automatically better; runtime and memory should be measured under controlled threading.
- Very high counts such as 128 are intentionally absent from the main slide; high-count behavior on a shared many-core server can reflect memory bandwidth, NUMA placement, scheduler contention, nested threading, or other users.
