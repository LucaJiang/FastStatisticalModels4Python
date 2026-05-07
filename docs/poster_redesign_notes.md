# Poster v5 redesign notes

## Design intent

`poster/poster_v5.tex` rebuilds the PyCon US 2026 poster as an academic conference poster about statistical simulation practice:

1. simulation gives statistical code an answer key;
2. statisticians must specify the target, ground truth, acceptance rule, failure meaning, and scale stress before timing;
3. k-means and permutation tests show two different workload shapes without teaching the algorithms;
4. committed evidence examples are scoped to the files used in the talk;
5. the closing blocks emphasize conservative tool choice and reporting discipline.

The visual language follows the older `beamerposter` academic style: full-width institutional header, three columns, titled blocks, tables, and small supporting figures. The slide colors remain, but the poster avoids an infographic-first layout.

## Quantitative claims used

- K-means equivalence: max relative inertia difference `3.1e-14`, below `1e-8` tolerance.
  Source: `experiments/results/macbook_air_long/latest/LOCAL_LONG_SUMMARY.md` and slide evidence summaries.
  Scope shown on poster: validation contract before speed claims.

- Permutation equivalence: max p-value difference `0.0`; max statistic difference `9.4e-16`.
  Source: `experiments/results/macbook_air_long/latest/LOCAL_LONG_SUMMARY.md` and slide evidence summaries.
  Scope shown on poster: same permutation stream and p-value definition.

- Permutation null calibration: mean `p <= 0.05` was `0.051`.
  Sources: `experiments/results/macbook_air_long/latest/LOCAL_LONG_SUMMARY.md` and `slides/speaker_notes.md`.
  Scope shown on poster: MacBook null calibration.

- Permutation A100 break-even: `n=5,000, p=10,000, R=5,000, batch_R=8,192`.
  Source: `experiments/results/linux_server_a100/permutation_break_even/README.md`.
  Scope shown on poster: A100 streamed reduction first win.

- Largest permutation speedup: `8.54x at n=5,000, p=500,000, R=5,000`.
  Source: `experiments/results/linux_server_a100/permutation_break_even/README.md` and committed summary CSVs in the same directory.
  Scope shown on poster: matched CPU matrix baseline divided by A100 streamed full end-to-end; compile excluded; transfer included; kernel-only excluded.

No main-poster text includes bookkeeping counts such as pass-row totals or memory-risk skip counts.

## Assets

- `poster/assets/poster_v4_kmeans_iris.png`
- `poster/assets/poster_v4_permutation_workflow.png`
- `poster/assets/repo_qr.png`
- `poster/cityu_logo.pdf`

The poster keeps the existing `beamerposter` custom size: `width=120,height=72`. No PyCon-specific poster size file was found in the repository.

## Build and QA

Build from the repository root:

```bash
make -C poster poster.pdf
```

Render preview PNG:

```bash
pdftoppm -png -singlefile -r 150 poster/poster.pdf poster/poster_v5_preview
```

QA checklist:

- compiled from `poster/poster_v5.tex`;
- rendered to `poster/poster.pdf`;
- preview rendered to `poster/poster_v5_preview.png`;
- inspect preview for clipped text, missing images, overfull boxes, tiny text, and QR readability.

Final QA on 2026-05-07:

- `make -C poster poster.pdf` completed successfully.
- `pdftoppm -png -r 160 poster/poster.pdf poster/qa/poster_v5_redesign_final` produced `poster/qa/poster_v5_redesign_final-1.png`.
- LaTeX log was checked for `Overfull`, `LaTeX Error`, and `Fatal error`; none remained after the column-grid and table-column adjustment.
- The remaining log messages are font substitutions and one underfull line in a small gate box.
- Rendered preview shows the three-column academic layout, both workload-shape diagrams, CityU logo, QR code, and source-scoped quantitative claims without memory-risk skip bookkeeping.
