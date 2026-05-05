# Experiments

Current experiment surface is split by role:

| Directory | Role |
| --- | --- |
| [`kmeans/`](kmeans/) | MacBook correctness/evidence implementation for k-means. |
| [`permutation/`](permutation/) | MacBook permutation validation plus shared matrix methods used by server runs. |
| [`kmeans_v3/`](kmeans_v3/) | Server/A100 k-means kernels used by the long-safe orchestrator. |
| [`server/`](server/) | Linux CPU/A100 long-safe orchestration and plotting. |
| [`visualization/`](visualization/) | 16:9 figures for current slides/poster. |
| [`results/`](results/) | Curated MacBook, server, A100, and presentation figures. |
| [`setup/`](setup/) | Local environment notes. |

Historical local benchmark scripts and result images were removed. Use
`experiments/results/README.md` to decide which result tier belongs in slides or
poster material.
