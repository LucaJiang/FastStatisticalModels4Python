# Reveal.js Deck QA - Latest

Generated: 2026-05-05

## Summary

- Total slide count: 32.
- Main/backup split: 27 main-path slides, 5 backup slides.
- Web screenshots: 32 saved in `slides/qa/latest/web/`.
- Print/PDF screenshots: 32 saved in `slides/qa/latest/pdf/`.
- Exported PDF: `slides/qa/latest/deck-export.pdf`.
- PDF page count: 32.
- PDF page size: 1017.12 x 571.92 pts.

## Required Consistency Checks

- Root README title is aligned: Python 3.14, Numba, and JAX are the title technologies; Codex remains a workflow topic.
- `slides/speaker_notes.md`, `slides/speaker_notes_v3.md`, and `slides/timeline_v3.md` use the same timing overview.
- Canonical A100 slide-level break-even source: `experiments/results/linux_server_a100/permutation_break_even/README.md`.
- `permutation_followup/README.md` is explicitly labeled as an earlier/non-canonical follow-up boundary.
- Correctness statuses use the updated vocabulary: `pass_exact`, `pass_gpu_tolerance`, `manual_check`, and `fail`; legacy `check` is documented only for older rows.
- Lightweight break-even summary CSVs are committed: `break_even_shape_sweep_summary.csv`, `decomposition_representative_shapes_summary.csv`, `cpu_matched_baselines_summary.csv`, and `correctness_checks_summary.csv`.
- Decision-map colorbar wording is scoped correctly: `speedup = matched CPU matrix baseline / A100 streamed full end-to-end`.

## Broken URLs / Console

- Broken image/video/poster URLs: 0 of 14 checked assets.
- Browser console errors/warnings: 0.
- Failed network requests: 0.
- Print/PDF console errors/warnings: 0.
- Print/PDF failed network requests: 0.

## Video Slides

- Video 1: autoplay=True, muted=True, loop=True, playsinline=True, readyState=4, paused=False, size=851x437.
- Video 2: autoplay=True, muted=True, loop=True, playsinline=True, readyState=4, paused=False, size=851x437.

PDF poster fallback:

- Slide 9: video display `none`, poster display `block`, poster `assets/posters/kmeans_animation_poster.png`, box 851x437.
- Slide 16: video display `none`, poster display `block`, poster `assets/posters/permutation_animation_poster.png`, box 851x437.

Result: browser mode plays the videos, and print/PDF mode hides videos and shows poster PNGs.

## Layout / Readability

- Automated overflow scan: 0 slides with text overflow.
- Correctness Gate 1 and Correctness Gate 2 have no automated overflow hits after the font metric fixes.
- No clipped text, label overlap, broken poster frames, or blank PDF video boxes found in the final screenshots.
- Correctness Gate 1 does not plot zero p-values on a log axis; JAX correctness tier is visible as CPU/x64, not A100.
- Correctness Gate 2 uses the calibration card/ruler, not a sorted-replicate S-curve.
- Slide 23 now includes the transition sentence: GPU is not the only scaling knob; CPU parallelism also needs tuning.

## A100 Narrative Check

- Slide 21 reconciles the old negative matched slice with the new streamed-reduction shape sweep.
- Slide 21 scopes speedup as matched CPU matrix baseline divided by A100 streamed full end-to-end; compile excluded, transfer included, kernel-only excluded.
- Slide 22 labels full-scenario A100 decomposition semantics and includes residual `other overhead` in the figure.
- No slide presents the old matched A100 permutation result as the final conclusion.

## Remaining Notes

- The lightweight summary CSVs preserve slide-level evidence and avoid committing huge raw artifacts. Some raw CPU/A100 timing cells are marked unavailable because the raw experiment CSVs were not in this repository snapshot; where a time is derived from a committed figure and speedup, the CSV says so in `timing_note`.
