# Poster

This directory builds the PyCon poster for the simulation-driven statistical
computing story. The current version is a CityU-themed Gemini/beamerposter
layout with the CityU logo in the title band and a red/orange/maroon palette
derived from the logo.

## Source Files

- `poster.tex`: poster content and four-column layout.
- `beamercolorthemeimsa.sty`: CityU color theme.
- `beamerthemegemini.sty`: title-band layout and logo placement.
- `cityu_logo.pdf`: local PDF conversion of `pycon_template/CityU_Horizontal_Logo_CMYK.svg`.

## Evidence Inputs

- MacBook Air figures: `../experiments/results/macbook_air_long/latest/figures/`
- Server summary figures: `../experiments/results/presentation_figures/`
- Server raw results:
  - `../experiments/results/linux_server_cpu/long_safe_20260503_190133/`
  - `../experiments/results/linux_server_a100/long_safe_20260503_190133/`

## Rebuild

Use the project default `py312` environment for Python-side plot regeneration.
Build the poster from this directory:

```bash
make poster.pdf
```

`Makefile` sets TeX cache directories under `/private/tmp` so LuaLaTeX does not
write generated font caches into the repository.
