# HTML slides (Reveal.js)

## Files

- [`index.html`](index.html) — single-file deck; uses Reveal.js from jsDelivr CDN (requires network for first load).

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

## Controls

| Key | Action |
|-----|--------|
| Arrow keys | Next / previous slide |
| `Space` | Next fragment / slide |
| `Esc` | Slide overview |
| `S` | Speaker view (notes + timer) |
| `F` | Fullscreen |

## Updating numbers and figures

1. Re-run benchmarks under `experiments/` and refresh CSV/PNG in `experiments/results/` (see [`experiments/results/README.md`](../experiments/results/README.md)).
2. Edit the tables and optional figure paths in `slides/index.html` to match new outputs.
3. Keep a one-line footnote with Python / hardware version for PyCon slides.

## Offline / USB copy

To bundle without `../experiments/results/`, copy needed PNGs into `slides/assets/` and update `src` attributes in `index.html`.
