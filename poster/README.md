# Poster

This directory builds the PyCon US 2026 hallway poster for the
simulation-driven statistical computing talk.

The current poster is a three-column, CityU-themed `beamerposter` summary:

- simulation contract and evidence ladder
- k-means as iterative model fitting
- permutation tests and the scoped CPU/A100 break-even result
- bottom talk spine and short evidence-pack paths

## Source

- `poster.tex`: canonical poster source.
- `poster.pdf`: rebuilt poster output.
- `assets/repo_qr.png`: QR code for the public repository.
- `cityu_logo.pdf`: CityU logo used in the header.
- `beamercolorthemeimsa.sty`, `beamerthemegemini.sty`: local theme files.

## Build

Use the repository default `py312` environment for Python-side figure work. Build
the poster from the repository root with:

```bash
make -C poster poster.pdf
```

The `Makefile` places TeX cache directories under `/private/tmp` so LuaLaTeX does
not write generated font caches into the repository.

## Figure Inputs

Poster-specific figures live in `experiments/results/presentation_figures/`:

- `poster_kmeans_recovery.*`: derived from `experiments/results/macbook_air_long/latest/kmeans_correctness.csv`.
- `poster_kmeans_a100_speedup.*`: derived from the Linux CPU k-means scale CSV and `experiments/results/linux_server_a100/long_safe_20260503_190133/kmeans_jax_gpu.csv`.
- `poster_permutation_gpu_breakeven.*`: derived from `experiments/results/linux_server_a100/permutation_break_even/break_even_shape_sweep_summary.csv`.

The QR target is:

```text
https://github.com/LucaJiang/FastStatisticalModels4Python
```

## A100 Wording Source

Use the root `README.md` and
`experiments/results/linux_server_a100/permutation_break_even/README.md` as the
canonical source for poster-level A100 wording.

The poster uses the scoped break-even definition:

```text
speedup = matched CPU matrix baseline / A100 streamed full end-to-end
compile excluded; transfer included; kernel-only excluded
```

The old negative matched slice is historical context, not the main poster
permutation result.

## QA

The final PDF was rendered at 200 dpi for visual inspection:

```bash
pdftoppm -png -r 200 poster/poster.pdf poster/qa/poster_200dpi
```

Expected QA artifact:

- `poster/qa/poster_200dpi-1.png`
