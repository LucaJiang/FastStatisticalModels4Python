# Codex Next Steps After Revised Deck Prototype

Use `/mnt/data/index_revised.html`, `/mnt/data/pycon_deck_revised.pdf`, and `/mnt/data/speaker_notes_revised.md` as the new content and layout prototype.

## What changed in this prototype

1. The introduction now explains why the talk starts from a statistician's workflow.
2. The deck now explicitly explains why k-means and permutation tests were chosen:
   - k-means = iterative model-fitting pressure.
   - permutation test = resampling / high-dimensional inference pressure.
3. The connection to Python tools is clearer:
   - Python/NumPy reference = executable statistical definition.
   - NumPy/BLAS = algebraic rewrite and vectorized matrix work.
   - Numba = compiled CPU loops.
   - Python 3.14/free-threaded = runtime/thread ceiling for thread-friendly shared-array work.
   - JAX/A100 = large batched array programs after reformulation.
4. Permutation correctness has been split into two slides:
   - equivalence against reference;
   - null calibration.
5. The layout has larger text, fewer tiny footnotes, side-note cards, and larger figures.
6. The old print/PDF local file path has been removed.

## Important implementation note

The revised HTML is a static HTML/PDF prototype. It uses cropped PNGs in `revised_assets/` extracted from the current PDF only so that the revised PDF can be previewed immediately.

For the real repository version, replace these cropped PNGs with regenerated source figures from the CSVs. Do not keep PDF-cropped figures as the final source of truth.

## Required Codex tasks

### 1. Port the revised content back into the repository slide system

If the repo uses reveal.js, port the content and CSS style from `index_revised.html` into the repo's `slides/index.html` rather than treating the static HTML as final.

Preserve the main deck order:

1. Title
2. Statistician point of view: speed after trust
3. Simulation creates the answer key
4. Statistical CI
5. Why Python
6. Why these two examples
7. Evidence ladder and measurement contract
8. k-means section title
9. Why k-means works here
10. k-means recovery surface
11. k-means reference equivalence
12. server k-means CPU/A100
13. k-means takeaway
14. permutation section title
15. why permutation works here
16. same statistic, different formulation
17. permutation equivalence
18. null calibration
19. server permutation negative result
20. negative result decomposition hypothesis
21. parallelism tuning
22. Python 3.14 / Numba / JAX tool connection
23. AI/Codex
24. decision guide
25. close
26-30. backup slides

### 2. Regenerate figures from CSVs with presentation-first readability

Do not simply reuse the current figures at the same size. Regenerate them with:

- larger figure titles;
- larger axis labels and ticks;
- thicker lines and larger markers;
- fewer panels per slide where possible;
- no tiny legends;
- export at high DPI or as SVG if the slide system supports it.

Recommended minimums:

- title font size: 18-22 pt;
- axis label font size: 15-18 pt;
- tick label font size: 13-15 pt;
- legend font size: 13-15 pt;
- line width: >= 2.5;
- marker size: >= 6.

### 3. Replace combined plots with simpler slide-specific figures

For permutation correctness, keep equivalence and calibration as separate main slides. Do not recombine them unless labels remain readable from the back of the room.

For parallelism, consider replacing the 2x2 figure with either:

- one slide for runtime vs workers;
- one backup slide for memory/RSS;

or keep 2x2 only if labels are visibly readable in the exported PDF.

### 4. Keep the stronger narrative

Do not remove the following ideas:

- Simulation is how statisticians create an answer key.
- Readable reference code is part of the scientific specification.
- k-means and permutation tests represent two computational shapes, not just two demos.
- Numba, JAX, Python 3.14, threads, and NumPy are chosen by bottleneck shape.
- A validated negative GPU result is useful evidence.

### 5. Python 3.14 handling

If actual `py314` or `py314t` measurements exist, add them only if environment detection is logged.

Record:

- Python version;
- `sys._is_gil_enabled()` when available;
- `Py_GIL_DISABLED`;
- `sys._jit.is_available()` and `sys._jit.is_enabled()` when `sys._jit` exists;
- extension-module behavior if any package re-enables the GIL.

If results are not measured, mark rows as unavailable. Do not infer speedups from the existence of Python 3.14.

### 6. JAX/A100 handling

All JAX timing must synchronize device work before timing is recorded. Label:

- device;
- dtype;
- whether timing is end-to-end or kernel-only;
- batch size;
- whether W and X are device-resident.

For the A100 permutation result, keep the negative result unless a measured, statistically equivalent implementation beats CPU. Do not manufacture a GPU win by changing the statistic.

### 7. Export checks

After rebuilding:

1. Export the PDF.
2. Render every page to PNG.
3. Check manually:
   - no blank trailing slide;
   - no clipped titles;
   - no tiny labels;
   - no local `file:///...` print link;
   - all slide numbers correct;
   - backup starts after the close.
4. Rebuild speaker notes from `speaker_notes_revised.md`.

## Desired final style

Use the revised deck style as the visual target:

- bigger titles;
- fewer words per slide;
- no dense evidence-map tables in the main path;
- figures paired with explanatory side cards;
- only one central point per slide;
- backup slides for extra evidence.
