# Experiments

Current experiment surface is split by role:

| Directory | Role |
| --- | --- |
| [`kmeans/`](kmeans/) | Shared k-means implementations for MacBook validation plus server/A100 runs. |
| [`permutation/`](permutation/) | MacBook permutation validation plus shared matrix methods used by server runs. |
| [`interpreter_effects/`](interpreter_effects/) | CPython 3.14 GIL vs free-threaded interpreter-effect experiments and plots. |
| [`server/`](server/) | Linux CPU/A100 long-safe orchestration and plotting. |
| [`visualization/`](visualization/) | 16:9 figures for current slides/poster. |
| [`results/`](results/) | Curated MacBook, server, A100, and presentation figures. |
| [`setup/`](setup/) | Local environment notes. |

Historical local benchmark scripts and result images were removed. Use
`experiments/results/README.md` to decide which result tier belongs in slides or
current talk material.
