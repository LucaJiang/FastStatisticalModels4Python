# Experiment Environment Setup

Default local environment for this repository is conda env `py312`.

## Recommended Environments

| conda env | Use |
| --- | --- |
| `py312` | Main development, MacBook evidence, plotting, and local checks. |
| `py314` | Optional CPython 3.14 interpreter-feature checks. |
| `py314t` | Optional free-threaded CPython checks. |

Current curated experiments no longer depend on the removed historical
benchmark drivers. Use the current module entry points instead.

## Local Commands

```bash
conda run -n py312 python -m experiments.run_macbook_evidence_extra \
  --output-dir experiments/results/macbook_air_long/latest --checkpoint-every 20 --max-iter 15

conda run -n py312 python -m experiments.visualization.plot_macbook_air_evidence \
  --results-dir experiments/results/macbook_air_long/latest

conda run -n py312 python -m experiments.visualization.plot_server_talk_evidence
```

## Environment Capture

```bash
conda run -n py312 python -m experiments.common.env_report \
  --environment-tier macbook_air_validation \
  --machine-name macbook_air \
  --out experiments/results/macbook_air_long/latest/env.json
```

The Linux server/A100 long-safe orchestrator writes its own environment reports
into the timestamped result directories.
