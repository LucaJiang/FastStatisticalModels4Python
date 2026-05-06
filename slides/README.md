# HTML slides (Reveal.js, 16:9)

## Canonical files

- [`index.html`](index.html) is the canonical Reveal.js deck.
- [`speaker_notes.md`](speaker_notes.md) is the canonical speaker-notes file.
- [`speaker_notes_v3.md`](speaker_notes_v3.md) is retained for compatibility and should match `speaker_notes.md`.
- [`timeline_v3.md`](timeline_v3.md) is the 30-minute talk timeline.

Current structure: 33 slides total, with slides 1-29 on the main path and slides 30-33 as backup. Slides 9 and 16 are short method-transition video slides.

## View locally

Use the project conda environment by default:

```bash
conda activate py312
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/slides/index.html
```

Print/PDF mode:

```text
http://localhost:8000/slides/index.html?print-pdf
```

PDF export should show 33 non-blank pages. Video slides use MP4/WebM in browser mode and poster PNG fallbacks in print/PDF mode.

## Current figure sources

- MacBook figures: [`experiments/results/macbook_air_long/latest/figures/`](../experiments/results/macbook_air_long/latest/figures/)
- Server and A100 presentation figures: [`experiments/results/presentation_figures/`](../experiments/results/presentation_figures/)
- Video assets: [`assets/videos/`](assets/videos/)
- Video poster frames: [`assets/posters/`](assets/posters/)

## Regenerate figures

Regenerate MacBook evidence figures:

```bash
python -m experiments.visualization.plot_macbook_air_evidence \
  --results-dir experiments/results/macbook_air_long/latest
```

Regenerate server CPU and k-means/A100 presentation figures:

```bash
python -m experiments.visualization.plot_server_talk_evidence
```

This also creates the representative k-means shape comparison used on slide 13.

Regenerate the current A100 permutation break-even figures from existing break-even CSVs when those CSVs are present on the experiment server:

```bash
python -m experiments.server.a100_permutation_break_even plot \
  --out-dir experiments/results/linux_server_a100/permutation_break_even \
  --presentation-dir experiments/results/presentation_figures
```

The break-even map uses `a100_streamed_reduction`; speedup is scoped to the matched CPU matrix baseline divided by A100 streamed full end-to-end. Compile is excluded, transfer is included, and kernel-only rows are not used for CPU/A100 speedup decisions.

## Update checklist

- Keep `index.html`, `speaker_notes.md`, `speaker_notes_v3.md`, `timeline_v3.md`, and this README in sync.
- Do not change statistical claims or benchmark numbers unless regenerated CSVs support the change.
- Keep the old A100 matched-slice negative result labeled as pre-break-even evidence: `n=5,000`, `p=50,000`, `batch_R=512`, before streamed reduction and broader shape sweep.
- Keep A100 correctness wording precise: accepted rows use `pass_exact` or `pass_gpu_tolerance`; ambiguous rows use `manual_check`; historical `check` appears only as a legacy status in older result notes.
- Put detailed metadata in notes or READMEs, not dense slide captions.
