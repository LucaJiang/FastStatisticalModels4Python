# Poster

This directory builds the PyCon US 2026 hallway poster for the
simulation-driven statistical computing talk.

The current poster is `poster_v5.tex`, a clean 3-column academic
`beamerposter` that follows the talk narrative:

- full-width header with CityU logo and repository QR code
- simulation as the statistical answer key
- decisions statisticians make before timing
- k-means and permutation tests as two workload shapes, not algorithm tutorials
- scoped validation and A100 evidence examples
- conservative tool choice and reporting discipline

## Source

- `poster_v5.tex`: canonical editable poster source.
- `poster.tex`: earlier poster source kept for reference.
- `poster.pdf`: rebuilt poster output.
- `assets/repo_qr.png`: QR code for the public repository.
- `assets/poster_v4_kmeans_iris.png`: k-means method visual.
- `assets/poster_v4_permutation_workflow.png`: permutation workflow visual.
- `generate_poster_v4_assets.py`: regenerates the poster-specific method diagrams.
- `cityu_logo.pdf`: CityU logo used in the header.
- `beamercolorthemeimsa.sty`, `beamerthemegemini.sty`: local theme files.
- `../docs/poster_redesign_notes.md`: v5 evidence and QA notes.

## Build

Use the repository default `py312` environment for Python-side figure work. Build
the poster from the repository root with:

```bash
make -C poster poster.pdf
```

The `Makefile` compiles `poster_v5.tex`, copies the result to `poster.pdf`, and
places TeX cache directories under `/private/tmp` so LuaLaTeX does not write
generated font caches into the repository.

## Figure Inputs

The revised poster does not compress slide benchmark figures into the layout or
show bookkeeping row counts. It uses two small workload-shape diagrams and the
repository QR code.

Regenerate the method diagrams with:

```bash
/Users/lucajiang/anaconda3/bin/conda run -n py312 python poster/generate_poster_v4_assets.py
```

The QR target is:

```text
https://github.com/LucaJiang/FastStatisticalModels4Python
```

## QA

Render a preview PNG for visual inspection:

```bash
pdftoppm -png -singlefile -r 150 poster/poster.pdf poster/poster_v5_preview
```

Expected QA artifact:

- `poster/poster_v5_preview.png`
