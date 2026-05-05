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

- Automated overflow scan: 6 slides with minor font-metric overflows: 1, 8, 15, 19, 20, 27.
- Visual review of the flagged slides found no clipped text after fixing Correctness Gate 1.
- No broken poster frames or blank PDF video boxes found.
- Correctness Gate 1 no longer plots zero p-values on a log axis; JAX correctness tier is visible as CPU/x64, not A100.
- Correctness Gate 2 uses the calibration card/ruler, not a sorted-replicate S-curve.
- The PDF export initially produced a blank extra page; print CSS now overrides Reveal's `.pdf-page` layout and exports 32 pages.

## A100 Narrative Check

- No slide is marked as pending.
- Slide 21 scopes speedup as matched CPU matrix baseline divided by A100 streamed full end-to-end; compile excluded, transfer included, kernel-only excluded.
- Slide 21 reconciles the old negative matched slice with the new streamed-reduction shape sweep.
- Slide 22 labels full-scenario A100 decomposition semantics and includes residual `other overhead` in the figure.
- Repository READMEs now label historical `check` rows as accepted bounded checks, not exact `pass` rows; future accepted GPU rows should use `pass_gpu_tolerance`.

## Remaining Notes

- The committed break-even result CSVs are not present in this repository snapshot; the scripts now add the requested CPU-baseline scope fields when rerun.
- The decision-map PNG still contains a small internal colorbar phrase from the original figure, but the slide caption and script source now use the scoped matched CPU matrix baseline wording.
