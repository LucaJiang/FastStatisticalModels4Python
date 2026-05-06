# Poster

This directory builds the PyCon US 2026 hallway poster for the
simulation-driven statistical computing talk.

The current poster is a clean 3-column CityU-themed `beamerposter`:

- full-width header with CityU logo and repository QR code
- simulation workflow column with validation/evidence badges
- k-means method column with scatter, centroid movement, validation box, and tool mapping
- permutation method column with ordinary shuffle logic before the streamed `W @ X` implementation
- bottom practical rule strip: simulate, preserve, identify, choose
- large QR/repo link

## Source

- `poster.tex`: canonical poster source.
- `poster.pdf`: rebuilt poster output.
- `assets/repo_qr.png`: QR code for the public repository.
- `assets/poster_kmeans_diagram.png`: static k-means method diagram.
- `assets/poster_permutation_diagram.png`: static permutation workflow diagram.
- `generate_poster_assets.py`: regenerates the poster-specific method diagrams.
- `cityu_logo.pdf`: CityU logo used in the header.
- `beamercolorthemeimsa.sty`, `beamerthemegemini.sty`: local theme files.

## Build

Use the repository default `py312` environment for Python-side figure work. Build
the poster from the repository root with:

```bash
make -C poster poster.pdf
```

The `Makefile` places TeX cache directories under `/private/tmp` so LuaLaTeX does
not write generated font caches into the repository.

## Figure Inputs

The revised poster does not compress slide benchmark figures into the layout. It uses two central method diagrams and the repository QR code.

Regenerate the method diagrams with:

```bash
/Users/lucajiang/anaconda3/bin/conda run -n py312 python poster/generate_poster_assets.py
```

The QR target is:

```text
https://github.com/LucaJiang/FastStatisticalModels4Python
```

## QA

The final PDF was rendered at 200 dpi for visual inspection:

```bash
pdftoppm -png -r 200 poster/poster.pdf poster/qa/poster_200dpi
```

Expected QA artifact:

- `poster/qa/poster_200dpi-1.png`
