# Reveal.js Deck QA - Latest

Generated: 2026-05-07

## Summary

- Total slide count: 33.
- Main/backup split: 29 main-path slides, 4 backup slides.
- Video slides: slides 9 and 16.
- Web screenshots: 33 saved in `slides/qa/latest/web/`.
- Print/PDF screenshots: 33 saved in `slides/qa/latest/pdf/`.
- Exported PDF: `slides/qa/latest/deck-export.pdf`.
- PDF page count: 33.
- QA workflow: local static server at `http://localhost:8000/slides/index.html`, Playwright Chromium screenshots for web and `?print-pdf`, plus PDF export.

## Required Consistency Checks

- Root README title is aligned: Python 3.14, Numba, and JAX are the title technologies; Codex remains a workflow topic.
- `slides/speaker_notes.md`, `slides/speaker_notes_v3.md`, and `slides/timeline_v3.md` use the same timing overview.
- Canonical A100 slide-level break-even source: `experiments/results/linux_server_a100/permutation_break_even/README.md`.
- `permutation_followup/README.md` is explicitly labeled as an earlier/non-canonical follow-up boundary.
- Correctness statuses use the updated vocabulary: `pass_exact`, `pass_gpu_tolerance`, `manual_check`, and `fail`; legacy `check` is documented only for older rows.
- Lightweight break-even summary CSVs are committed: `break_even_shape_sweep_summary.csv`, `decomposition_representative_shapes_summary.csv`, `cpu_matched_baselines_summary.csv`, and `correctness_checks_summary.csv`.
- Decision-map colorbar wording is scoped correctly: `speedup = matched CPU matrix baseline / A100 streamed full end-to-end`.
- Current deck structure remains 33 slides: slides 1-29 main path, slides 30-33 backup.
- Slide 9 k-means video uses Iris petal length and petal width, K=3.
- Species labels are context only; k-means sees only the two measurements.
- Slide 13 is a representative-shape comparison, not a universal ranking.
- Slide 16 explains ordinary permutation logic before W @ X.
- Slide 22 scopes speedup as matched CPU matrix baseline / A100 streamed full end-to-end; compile excluded, transfer included, kernel-only excluded.
- Slide 23 committed lightweight decomposition summary has 2 representative rows.
- Slide 24 uses the expanded Linux server CPU sweep at 1/4/16/64/128 workers or threads.
- Slide 24 marks high-count rows as shared-server evidence because affinity exposed 512 CPUs but no exclusive scheduler allocation was detected.
- Slide 24 high-count CPU parallelism is shared-server evidence, not a claim that 128 workers are intrinsically bad.
- No canonical slide/notes/QA/poster source uses the banned contract phrasing.
- No canonical source says the Slide 9 video uses four synthetic groups.

## Broken URLs / Console

- Broken image/video/poster URLs: 0.
- Browser console errors/warnings: 0.
- Failed network requests: 0.
- Print/PDF console errors/warnings: 0.
- Print/PDF failed network requests: 0.

## Video Slides

- k-means video: autoplay=True, muted=True, loop=True, playsinline=True, readyState=4, encoded 1280x720, duration 10.0s.
- permutation video: autoplay=True, muted=True, loop=True, playsinline=True, readyState=4, encoded 1280x720.
- Headless Chromium advanced both video current times during QA; the `paused` flag is not used as a pass/fail signal because headless capture can pause media when the tab is inactive.

PDF poster fallback:

- Slide 9: video display `none`, poster display `block`, poster `assets/posters/kmeans_animation_poster.png`, poster loaded at 1280x720.
- Slide 16: video display `none`, poster display `block`, poster `assets/posters/permutation_animation_poster.png`, poster loaded at 1280x720.

Result: browser mode plays the videos, and print/PDF mode hides videos and shows poster PNGs.

## Presentation Polish Check

- Slide 1 now uses the talk framing `Trust -> speed -> scale`.
- Slides 2-7 reduce text density and use pipeline/tool-map/evidence-ladder layouts instead of dense explanatory cards.
- Slide 9 k-means animation uses Iris petal length and petal width with K=3; species labels are context only, and k-means sees only the two measurements.
- Slides 10, 14, 24, 25, and 26 replace dense tables or long prose with cards, rules, and a decision tree.
- Slides 19 and 20 remain validation dashboards with no log-zero p-value plot and no sorted-replicate S-curve.
- Slide 24 now shows the expanded Linux server CPU parallelism evidence.

## Layout / Readability

- Automated overflow scan: 0 slides with text overflow.
- Automated tiny-text scan: 0 slides flagged.
- Correctness Gate 1 and Correctness Gate 2 have no automated overflow hits after the font metric fixes.
- No clipped text, detected label overlap, broken poster frames, or blank PDF video boxes found in the final screenshots.
- Correctness Gate 1 does not plot zero p-values on a log axis; JAX correctness tier is visible as CPU/x64, not A100.
- Correctness Gate 2 uses the calibration card/ruler, not a sorted-replicate S-curve.
- No main-path slide uses a tiny dense table after this pass.

## A100 Narrative Check

- Slide 22 reconciles the old negative matched slice with the new streamed-reduction shape sweep.
- Slide 22 scopes speedup as matched CPU matrix baseline divided by A100 streamed full end-to-end; compile excluded, transfer included, kernel-only excluded.
- Slide 22 states: A100 becomes faster in the streamed-reduction grid at n=5,000, p=10,000, R=5,000, batch_R=8,192.
- Slide 22 states: largest slide-level measured speedup is 8.54x at n=5,000, p=500,000, R=5,000.
- Slide 22 states: two high-R p=500,000 cells are A100 OOM/unavailable, not CPU wins.
- Slide 23 labels full-scenario A100 decomposition semantics and includes residual `other overhead` in the figure; committed lightweight decomposition evidence has 2 representative summary rows, not four committed categories.
- Slide 24 is explicitly Linux server CPU evidence, not MacBook evidence, and does not claim that 128 workers are intrinsically bad.
- No slide presents the old matched A100 permutation result as the final conclusion.

## Remaining Notes

- The lightweight summary CSVs preserve slide-level evidence and avoid committing huge raw artifacts. Some raw CPU/A100 timing cells are marked unavailable because the raw experiment CSVs were not in this repository snapshot; summary-only speedups and derived times are marked in `timing_note`.
