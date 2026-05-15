# Slide QA Notes

## Backup batch_R tuning

- Backup batch_R slide now interprets the trend instead of implying global optimality.
- It states that small batches were inefficient and larger batches amortized stream overhead.
- It states that gains flattened after the early batch_R increases.
- It states that 8192 was the fastest accepted value within the measured 128-8192 grid.
- It states that larger batch_R values were not tested in the committed sweep.
- It states that batch_R is pipeline tuning, not a statistical parameter.
- Full end-to-end timing remains the decision criterion: compile excluded, transfer included, kernel-only excluded.
