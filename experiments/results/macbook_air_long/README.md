# MacBook Air results

This directory contains the local MacBook Air validation and evidence outputs for the simulation-driven statistical computing talk.

## What to use

Use `latest/` for current slides and follow-up work. It contains:

- the full MacBook correctness grid after enabling JAX x64 for CPU equivalence checks;
- targeted extra evidence for larger k-means shapes and denser permutation calibration/power/runtime sweeps;
- cleaned, deck-ready figures listed in `latest/figure_manifest.csv`;
- `latest/README.md` and `latest/LOCAL_LONG_SUMMARY.md` with run notes.

## Cleaned up

Older quick-validation outputs, the incomplete long run, and the pre-x64 full run with JAX CPU float32 boundary failures were removed from the curated result set. The current correctness tier enables JAX x64 so that p-value equivalence is checked against the float64 reference path.

## Reproduce current curated figures

```bash
python -m experiments.run_macbook_evidence_extra \
  --output-dir experiments/results/macbook_air_long/latest \
  --checkpoint-every 20 --max-iter 15

python -m experiments.visualization.plot_macbook_air_evidence \
  --results-dir experiments/results/macbook_air_long/latest
```

Both scripts are resumable by `run_id`; rerunning them should skip existing completed rows.
