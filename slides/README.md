# HTML slides (Reveal.js, 16:9)

## Files

- [`index.html`](index.html) - 1280x720 Reveal deck with 25 main slides plus 5 clearly marked backup slides; uses Reveal.js from jsDelivr CDN (requires network for first load).
- [`timeline_v3.md`](timeline_v3.md) - 30-minute talk plan aligned to `docs/replanned_timeline_and_codex_instructions.md`.
- [`speaker_notes_v3.md`](speaker_notes_v3.md) - standalone speaker notes matching the embedded Reveal notes.

## View locally

**Option A — open the file**

```bash
open slides/index.html   # macOS
xdg-open slides/index.html
```

Use a local static server if browser security blocks relative images:

**Option B — HTTP server (recommended)**

```bash
cd /path/to/FastStatisticalModels4Python
python -m http.server 8000
# Visit http://localhost:8000/slides/
```

## Print / PDF export

Open the deck through the local server, then add `?print-pdf`:

```text
http://localhost:8000/slides/?print-pdf
```

Exported PDFs should have 30 non-blank pages: 25 main slides and 5 backup slides.

## Controls

| Key | Action |
|-----|--------|
| Arrow keys | Next / previous slide |
| `Space` | Next fragment / slide |
| `Esc` | Slide overview |
| `S` | Speaker view (notes + timer) |
| `F` | Fullscreen |

## Updating numbers and figures

1. Re-run benchmarks under `experiments/` and refresh CSV/PNG in `experiments/results/`.
2. For the current MacBook evidence figures, use [`experiments/results/macbook_air_long/latest/figure_manifest.csv`](../experiments/results/macbook_air_long/latest/figure_manifest.csv).
3. For the server/A100 slides, use the 16:9 summaries in [`experiments/results/presentation_figures/`](../experiments/results/presentation_figures/).
4. Regenerate 16:9 MacBook figures with:
   ```bash
   /Users/lucajiang/anaconda3/envs/py312/bin/python -m experiments.visualization.plot_macbook_air_evidence \
     --results-dir experiments/results/macbook_air_long/latest
   ```
5. Regenerate 16:9 server figures with:
   ```bash
   /Users/lucajiang/anaconda3/envs/py312/bin/python -m experiments.visualization.plot_server_talk_evidence
   ```
6. Edit `slides/index.html`, `timeline_v3.md`, and `speaker_notes_v3.md` together.
7. Keep a one-line footnote with Python / hardware version for PyCon slides.

## Offline / USB copy

To bundle without `../experiments/results/`, copy needed PNGs into `slides/assets/` and update `src` attributes in `index.html`.
