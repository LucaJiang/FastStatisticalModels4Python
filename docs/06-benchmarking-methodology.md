# Benchmarking and Reproducibility Methodology

This repository now uses a validation-first benchmark contract. Historical
local benchmark drivers were removed; current results come from the MacBook
correctness/evidence tier and the Linux CPU/A100 long-safe tier.

## Principles

- Record the environment tier for every row: MacBook Air, Linux server CPU, or
  A100.
- Separate correctness status from runtime. A fast row is not useful if it
  changes the statistic.
- Keep cold first-call time separate from warm median time.
- Record memory-risk skips explicitly instead of treating skipped rows as zero
  timings.
- For JAX timings, block on device work before stopping the timer.
- For process or worker experiments, record memory in a way that makes hidden
  process cost visible.

## Current Producers

| Producer | Role |
| --- | --- |
| `experiments/run_macbook_long.py` | Full MacBook correctness validation. |
| `experiments/run_macbook_evidence_extra.py` | Targeted local evidence after visual review. |
| `experiments/server/long_safe_orchestrator.py` | Linux CPU/A100 long-safe orchestration. |
| `experiments/server/long_safe_plots.py` | Server raw diagnostic plots and summaries. |
| `experiments/visualization/plot_macbook_air_evidence.py` | 16:9 MacBook slide figures. |
| `experiments/visualization/plot_server_talk_evidence.py` | 16:9 server/A100 slide and poster figures. |

## Result Tiers

- `experiments/results/macbook_air_long/latest/`: local correctness,
  calibration, power, and lightweight runtime evidence.
- `experiments/results/linux_server_cpu/long_safe_20260503_190133/`: large CPU
  scaling, thread/worker sweeps, and memory behavior.
- `experiments/results/linux_server_a100/long_safe_20260503_190133/`: A100
  accelerator evidence after matrix or batched-array reformulation.
- `experiments/results/presentation_figures/`: normalized 16:9 summaries for
  slides and poster.

## Reproducibility Checklist

1. Use `py312` for local MacBook reruns unless a document explicitly says
   otherwise.
2. Preserve `env.json` or `extra_env.json` beside result CSVs.
3. Keep source CSVs and plotting code together in the result README.
4. Rebuild presentation figures from CSVs, not by manually editing PNGs.
5. Treat server/A100 speedups as measured only for matched shapes.
6. Keep negative results, such as the A100 permutation path losing to CPU, in
   the narrative when they clarify the computational shape.
