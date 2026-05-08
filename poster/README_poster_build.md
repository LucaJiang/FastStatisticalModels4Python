# PyCon US 2026 Poster Build

This directory builds the A0 landscape conference poster:

- `poster/pyconus2026_poster.pdf`
- `poster/pyconus2026_poster.png`
- `poster/poster.tex`
- `poster/poster_data_audit.json`

## Build Command

From the repository root:

```bash
conda run -n py312 python poster/scripts/prepare_poster_figures.py
conda run -n py312 python poster/scripts/build_poster_assets.py
make -C poster pyconus2026_poster.pdf pyconus2026_poster.png
```

The Makefile also runs the Python preparation steps when their outputs are stale.

## Geometry

`poster/poster.tex` is a single-page A0 landscape LaTeX poster:

```tex
\usepackage[paperwidth=118.9cm,paperheight=84.1cm,margin=0cm]{geometry}
```

The generated PDF was checked with `pdfinfo`; the page size is reported as A0 landscape.

## Assets

Branding and QR:

- `poster/cityu_logo.pdf`
- `poster/assets/repo_qr.png`

Vector icons:

- `poster/assets/icon_macros_tikz.tex`
- Proof sheet: `poster/generated/icon_sheet.tex`

The poster uses TikZ icon macros such as `\IconTarget`, `\IconCode`, `\IconGauge`, `\IconScale`, `\IconCheck`, `\IconGear`, `\IconLightning`, `\IconMatrix`, `\IconPeople`, `\IconChip`, `\IconRobot`, and `\IconBrainPerson`.

## Scientific Figures

Poster figures live in `poster/figures/`. They are prepared by:

```bash
conda run -n py312 python poster/scripts/prepare_poster_figures.py
```

Figure manifest:

- `poster/figures/figure_sources.json`

Primary poster figures:

| Poster figure | Source/provenance |
| --- | --- |
| `poster/figures/kmeans_task_schematic.png` | Existing slide/poster visual from `slides/assets/posters/kmeans_animation_poster.png` |
| `poster/figures/permutation_task_schematic.png` | Existing slide/poster visual from `slides/assets/posters/permutation_animation_poster.png`; equivalent slide screenshots are also available under `slides/review/screenshots/` |
| `poster/figures/equivalence_validation.png` | Cropped from existing local validation result figure |
| `poster/figures/null_calibration.png` | Cropped from existing null calibration result figure |
| `poster/figures/kmeans_runtime_evidence.png` | Regenerated from committed server CPU/GPU CSVs using public `JAX / GPU` labels |
| `poster/figures/gpu_permutation_decision_map.png` | Regenerated from committed permutation break-even summary; labels keep `A100` because the measured experiment was run on A100 |

Screenshots in `slides/review/screenshots/` may be used as visual references or replacements when a slide-rendered figure is preferred, but numeric claims should still be traced through `poster/poster_data_audit.json`.

## Numeric Data Sources

Every numeric claim used in live poster text or quantitative figures is audited in:

- `poster/poster_data_audit.json`

Primary source files:

- `experiments/results/macbook_air_long/latest/kmeans_correctness.csv`
- `experiments/results/macbook_air_long/latest/permutation_equivalence.csv`
- `experiments/results/macbook_air_long/latest/permutation_calibration_extended.csv`
- `experiments/results/linux_server_cpu/long_safe_20260503_190133/kmeans_cpu_scaling.csv`
- `experiments/results/linux_server_a100/long_safe_20260503_190133/kmeans_jax_gpu.csv`
- `experiments/results/linux_server_a100/long_safe_20260503_190133/env.json`
- `experiments/results/linux_server_a100/permutation_break_even/break_even_shape_sweep_summary.csv`
- `experiments/results/linux_server_a100/permutation_break_even/cpu_matched_baselines_summary.csv`
- `experiments/results/linux_server_a100/permutation_followup/TIMING_SEMANTICS.md`

Claim macros are generated at:

- `poster/generated/poster_claims.tex`

## Timing Semantics

The permutation GPU/A100 decision-map claims use the committed summary semantics:

- compile excluded
- transfer included
- kernel-only excluded

The poster describes the public tool class as `GPU` / `JAX / GPU`. The A100 label is retained in the permutation decision-map figure and small provenance note because those measurements were actually run on A100.

## Verification

Commands used for the final build:

```bash
make -C poster pyconus2026_poster.pdf pyconus2026_poster.png
pdfinfo poster/pyconus2026_poster.pdf
file poster/pyconus2026_poster.pdf poster/pyconus2026_poster.png
python3 - <<'PY'
import json
data = json.load(open("poster/poster_data_audit.json"))
print(len(data))
PY
```

Final checked outputs:

- PDF: one page, A0 landscape
- PNG preview: `7022 x 4967`
- Audit entries: 47
