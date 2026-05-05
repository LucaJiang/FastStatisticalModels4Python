# Tiered Experiment Result Schema

Minimum shared columns for MacBook validation, Linux CPU, and A100 rows:

```text
run_id, timestamp, environment_tier, machine_name, workload, implementation,
scenario_id, seed, n, p, d, k, r, batch_r, workers, dtype,
cold_time_s, warm_median_s, warm_min_s, warm_max_s,
compile_time_s, transfer_h2d_s, transfer_d2h_s,
peak_python_mb, peak_rss_mb, peak_child_rss_mb, peak_gpu_mb,
correctness_status, statistical_metric_name, statistical_metric_value,
notes
```

Use `NA` when a column does not apply. Do not compare hardware tiers in a
single leaderboard unless the figure is explicitly faceted or labeled.
