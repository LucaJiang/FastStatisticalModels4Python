# Fast Statistical Models for Python

This repository contains the PyCon US 2026 deck and reproducibility artifacts for **Breaking the Speed Limit: Python 3.14, Numba, and JAX in Statistical Computing**.

## Public pages

The GitHub Pages version is the easiest way to view the talk materials:

| Material | Open online | Source |
| --- | --- | --- |
| Slides | [View slides](https://lucajiang.github.io/FastStatisticalModels4Python/slides/) | [`slides/index.html`](slides/index.html) |
| Poster | [View poster](https://lucajiang.github.io/FastStatisticalModels4Python/poster/) | [`poster/index.html`](poster/index.html) |
| Slides print mode | [Open print/PDF view](https://lucajiang.github.io/FastStatisticalModels4Python/slides/index.html?print-pdf) | [`slides/README.md`](slides/README.md) |

Core thesis:

- simulation is how statisticians write tests;
- speed only counts after the statistic is preserved;
- Python remains the scientific interface;
- Numba, JAX, Python 3.14, threads, processes, and A100 solve different bottlenecks.

## Local preview

Use the project conda environment by default:

```bash
conda activate py312
python -m http.server 8000
```

Open the local pages:

- Slides: <http://localhost:8000/slides/>
- Poster: <http://localhost:8000/poster/>
- Slides print/PDF mode: <http://localhost:8000/slides/index.html?print-pdf>

The canonical deck is [`slides/index.html`](slides/index.html). Current structure is 36 slides total: 29 main-path slides and 7 backup slides. Slides 9 and 17 are video method-transition slides.

## Results layout

- MacBook/local trust tier: [`experiments/results/macbook_air_long/latest/`](experiments/results/macbook_air_long/latest/)
- Linux server CPU scale tier: [`experiments/results/linux_server_cpu/long_safe_20260503_190133/`](experiments/results/linux_server_cpu/long_safe_20260503_190133/)
- Historical A100 long-safe tier: [`experiments/results/linux_server_a100/long_safe_20260503_190133/`](experiments/results/linux_server_a100/long_safe_20260503_190133/)
- A100 permutation break-even notes: [`experiments/results/linux_server_a100/permutation_break_even/`](experiments/results/linux_server_a100/permutation_break_even/)
- Slide-ready figures: [`experiments/results/presentation_figures/`](experiments/results/presentation_figures/)

## Current result status

- MacBook k-means and permutation correctness/calibration results are the trust tier used in the main talk.
- Linux server CPU and k-means A100 results are scale evidence used in the main talk.
- The old A100 permutation matched slice is historical negative evidence: `n=5,000`, `p=50,000`, `batch_R=512`, before streamed reduction and the broader shape sweep.
- The current A100 permutation break-even narrative uses `a100_streamed_reduction`, larger `batch_R`, and a measured shape sweep. Speedup is scoped to the matched CPU matrix baseline divided by A100 streamed full end-to-end. Compile is excluded, transfer is included, and kernel-only rows are not used for CPU/A100 decisions.
- A100 permutation correctness wording is intentionally scoped: new accepted rows use `pass_exact` or `pass_gpu_tolerance`, while ambiguous rows use `manual_check`; historical rows marked `check` appear only in older follow-up notes.

## Regenerate figures

MacBook figures:

```bash
python -m experiments.visualization.plot_macbook_air_evidence \
  --results-dir experiments/results/macbook_air_long/latest
```

Server CPU and k-means/A100 presentation figures:

```bash
python -m experiments.visualization.plot_server_talk_evidence
```

A100 permutation break-even figures, when the experiment CSVs are available:

```bash
python -m experiments.server.a100_permutation_break_even plot \
  --out-dir experiments/results/linux_server_a100/permutation_break_even \
  --presentation-dir experiments/results/presentation_figures
```

## Deck QA

Review screenshots are local generated artifacts under `slides/review/screenshots/`. The QA pass checks browser mode, video poster fallbacks, broken media URLs, clipping, overlap, console errors, and A100 pending/experimental wording.
