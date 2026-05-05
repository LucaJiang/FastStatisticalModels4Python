# Reveal Deck QA - latest

Generated from local server `http://localhost:8000/slides/index.html`.

## Summary

- Total slide count: 32
- Web screenshots: 32 files in `slides/qa/latest/web/`
- PDF screenshots: 32 files in `slides/qa/latest/pdf/`
- Exported PDF: `slides/pycon_deck.pdf`
- PDF blank-page check: none

## Broken Assets And Console

- Broken local image/video/poster URLs: none detected
- HTTP 4xx/5xx local resource responses: none
- Browser console errors/warnings: none
- Request failures: none

## Video Slide QA

- Slide 9: `kmeans_animation.webm` autoplay=True, muted=True, loop=True, playsinline=True, paused=False, size=800x411
- Slide 16: `permutation_animation.webm` autoplay=True, muted=True, loop=True, playsinline=True, paused=False, size=800x411

PDF/print fallback:

- Slide 9: poster visible=True, size=937x437, `kmeans_animation_poster.png`
- Slide 16: poster visible=True, size=937x437, `permutation_animation_poster.png`

Result: video assets autoplay in web mode and poster fallbacks are visible in print/PDF mode.

## Layout Findings

- Clipped text after fixes: none detected in manual review.
- Overlapping labels after fixes: none detected in manual review.
- Tiny unreadable text: none detected below the 13.5px QA threshold in deck text.
- Automated heuristic notes: Slide 6 flagged inline `W @ X` text inside its sentence; manual screenshot review found no visual collision.

## PDF Versus Web

- PDF export uses poster frames for the two method animation slides; no blank video boxes were found.
- PDF screenshot review of the revised correctness gates and A100/parallelism slides did not show bad divergence from web mode.

## Correctness Gate Checks

- Gate 1 is a validation dashboard and no longer plots zero p-value differences on a log axis.
- Gate 2 is a calibration dashboard/ruler and no longer plots sorted null replicate index as an S-curve.

## A100 Pending State

A100-related pending slides:

- Slide 21: `Server permutation: CPU scale is measured; A100 follow-up is pending`
- Slide 22: `Use the current slice to ask the next question, not to close the case`

Both slides mark the current matched A100 slice as non-final and state that decomposition / batch-size follow-up is pending.
