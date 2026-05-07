# A100 Break-Even Audit

Updated: 2026-05-06

Scope: committed lightweight evidence for the A100 permutation break-even slides. Raw full experiment CSVs for several slide-map cells are not committed in this repository snapshot, so summary-only speedups must not be treated as raw CPU/A100 timing rows.

| Slide-level claim | Committed source | Audit note |
|---|---|---|
| Speedup means matched CPU matrix baseline / A100 streamed full end-to-end. | `experiments/results/linux_server_a100/permutation_break_even/README.md`; `break_even_shape_sweep_summary.csv`; `cpu_matched_baselines_summary.csv`; `slides/index.html` | This is the only speedup scope used for the slide-level A100 decision map. |
| Compile excluded, transfer included, kernel-only excluded. | `experiments/results/linux_server_a100/permutation_break_even/README.md`; `break_even_shape_sweep_summary.csv` columns `compile_excluded`, `transfer_included`, `kernel_only_excluded`; `slides/index.html` | Kernel-only timing is not used for CPU/A100 speedup decisions. |
| A100 becomes faster in the streamed-reduction grid at n=5,000, p=10,000, R=5,000, batch_R=8,192. | `experiments/results/linux_server_a100/permutation_break_even/README.md`; `break_even_shape_sweep_summary.csv`; `cpu_matched_baselines_summary.csv` | The row has speedup 1.5 and winner `a100`; timing note says the raw CPU/A100 timing pair is not committed. |
| Largest slide-level measured speedup: 8.54x at n=5,000, p=500,000, R=5,000. | `experiments/results/linux_server_a100/permutation_break_even/README.md`; `break_even_shape_sweep_summary.csv`; `cpu_matched_baselines_summary.csv`; `decomposition_representative_shapes_summary.csv` | The committed summary records A100 total 1.25s and CPU time 10.675s, with timing note marking CPU time as derived from speedup times A100 total rather than a committed raw CPU timing row. |
| Two high-R p=500,000 cells are A100 OOM/unavailable, not CPU wins. | `break_even_shape_sweep_summary.csv` rows `(p=500000, R=10000)` and `(p=500000, R=50000)`; `experiments/results/linux_server_a100/permutation_break_even/README.md`; `slides/speaker_notes.md` | These rows have winner `timeout_skip` and timing notes saying A100 OOM/unavailable is not a CPU win or hidden speedup. |
| Representative A100 decomposition has 2 committed lightweight summary rows. | `decomposition_representative_shapes_summary.csv`; `experiments/results/linux_server_a100/permutation_break_even/README.md`; `experiments/results/linux_server_a100/permutation_break_even/QA_NOTE.md` | The raw four-row representative decomposition CSV is not committed, so slides and notes must not claim four representative categories from committed evidence. |
| Correctness rows represented in committed lightweight correctness summary: 97 accepted rows. | `correctness_checks_summary.csv`; `experiments/results/linux_server_a100/permutation_break_even/README.md` | The previous 101/99/2 wording was not supported by the committed lightweight correctness summary. The two OOM/unavailable slide-map cells are tracked in `break_even_shape_sweep_summary.csv`, not as accepted correctness rows. |

