# k-means MacBook Evidence

This directory contains the shared k-means implementations used by the local
MacBook evidence runs and the Linux server/A100 orchestration.

## Active Files

| File | Role |
| --- | --- |
| `data_generation.py` | Gaussian mixture scenarios with known labels, including `KMeansScenario` for server runs. |
| `kmeans_reference.py` | Readable fixed-init reference implementation. |
| `kmeans_numpy.py` | Naive broadcast and matmul NumPy variants. |
| `kmeans_numpy_broadcast.py` | Explicit broadcast-path helper. |
| `kmeans_numpy_matmul.py` | Explicit matmul-distance helper. |
| `kmeans_numba.py` | Numba Lloyd implementation. |
| `kmeans_jax.py` | JAX implementation used by server/A100 runs. |
| `validate_kmeans.py` | ARI, inertia, and equivalence summaries. |
| `run_mac_validation.py` | Full MacBook validation grid and local diagnostic figures. |

## Current Run Commands

```bash
/Users/lucajiang/anaconda3/envs/py312/bin/python -m experiments.run_macbook_long \
  --mode full --output-dir experiments/results/macbook_air_long/latest

/Users/lucajiang/anaconda3/envs/py312/bin/python -m experiments.run_macbook_evidence_extra \
  --output-dir experiments/results/macbook_air_long/latest
```

Historical local benchmark drivers were removed. The commands above are the
current local evidence entry points.
