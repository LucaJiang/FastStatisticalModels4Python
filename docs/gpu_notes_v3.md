# GPU notes v3

GPU is useful when the statistical workload has the right shape. The v3 talk should avoid saying "JAX makes loops fast." The better story is:

> JAX/GPU helps when the workload is expressed as a large batched array program, not when we simply wrap a Python loop in `vmap`.

## K-means

K-means can benefit from JAX/GPU for large dense distance computations and fixed-iteration scans. Small problems may lose to CPU implementations because compilation and transfer overhead dominate. The v3 results therefore separate cold and warm timings and include CPU/GPU break-even plots.

## Permutation tests

The high-dimensional permutation test becomes accelerator-friendly after the statistic is rewritten as matrix multiplication:

```text
W: B x n
X: n x p
T_null = W @ X
```

`W` must be generated in chunks for large `B` and `n`. Full materialization can be a memory failure even when the arithmetic is GPU-friendly.

## Public-server hygiene

Before GPU runs:

- Check `nvidia-smi`.
- Avoid unnecessary preallocation with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Consider `XLA_PYTHON_CLIENT_MEM_FRACTION` for large sweeps.
- Record unavailable or occupied GPU states as skipped, not failed.

## Claim policy

Only report GPU speedups when:

- JAX reports a GPU backend.
- The benchmark calls `block_until_ready()`.
- Validation passed first.
- Cold and warm timings are separated.
- Memory/chunking behavior is documented.
