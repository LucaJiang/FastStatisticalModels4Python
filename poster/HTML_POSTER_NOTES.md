# HTML Poster Notes

Updated source: `poster/index.html`

Generated outputs:

- `poster/html_poster.png`
- `poster/html_poster.pdf`

## Final copy-edit and layout polish

1. Final wording changes
   - Main thesis: "Make the statistical task testable; then accelerate the measured bottleneck."
   - Thesis subline: "Define the target, check agreement, then accelerate only the measured computational hotspot."
   - Python control-plane line: "Python stays the control plane; only the validated computational hotspot is accelerated."
   - Simulation section: "Simulation Provides the Test Harness"; lead text now says simulation creates a controlled setting for checking recovery, calibration, and agreement.
   - Workload section: "Two Workloads, Two Bottleneck Patterns."
   - Evidence section: "After Validation, Performance Depends on Workload Structure."
   - Tool section: "Tool Choices in This Talk" with subtitle "Examples, not a ranking."
   - AI section: "AI Scales the Workflow; the Statistician Owns the Claim."
   - Reporting contract kept: "Report Every Speed Claim" and "Timing semantics: compilation excluded; data transfer included; kernel-only timings not reported."

2. Central thesis replacement
   - Replaced the previous callout, "Same statistical task. Same validated result. Different computational bottleneck. Different smallest sufficient tool."
   - New callout:
     - "Validate the analysis before optimizing the implementation."
     - "Speed only counts after agreement with the reference passes the validation criteria."
   - This avoids implying exact identity when the supported claim is agreement under the stated validation criteria.

3. PyCon logo decision
   - Retained the local PyCon US 2026 logo from `poster/assets/pyconus_logo.png`, originally copied from `pycon_template/pyconus_logo.png`.
   - The asset is high resolution enough for the poster source: 1338 x 1333 px.
   - Reduced its rendered width from 148% to 112% of its masthead column so it reads as conference branding rather than a decorative sticker.
   - Reduced the title-block typography slightly and increased the masthead row height so the affiliation line no longer collides with the masthead divider.
   - CityU logo, QR code, authors, affiliation, and footer text were left unchanged.

4. AI/statistician image placement
   - Re-cropped `slides/assets/AI_duty.png` to the central automation subject and saved it as `poster/assets/ai_duty_thumb.png` at 640 x 640 px.
   - Re-cropped `slides/assets/human_duty.png` to the central statistician subject and saved it as `poster/assets/human_duty_thumb.png` at 640 x 640 px.
   - Increased the poster thumbnail boxes from 2.25cqw to 2.75cqw so the subjects are recognizable while remaining subordinate to the section text.
   - Left image placement aligned with "What AI can automate" and "What the statistician decides."

5. Validation numbers retained and sources
   - k-means: max relative inertia difference `3.1e-14` and tolerance `1e-8`.
     Source: `experiments/results/macbook_air_long/latest/kmeans_correctness.csv`; tolerance source: `experiments/kmeans/validate_kmeans.py`; both audited in `poster/poster_data_audit.json`.
   - permutation p-value agreement: max `|p diff| = 0.0`.
     Source: `experiments/results/macbook_air_long/latest/permutation_equivalence.csv`, audited in `poster/poster_data_audit.json`.
   - permutation test statistic agreement: max `|stat diff| = 9.4e-16`.
     Source: `experiments/results/macbook_air_long/latest/permutation_equivalence.csv`, audited in `poster/poster_data_audit.json`.
   - null calibration: estimated type-I error `0.051` near nominal alpha `0.05`.
     Source: `experiments/results/macbook_air_long/latest/permutation_calibration_extended.csv`, audited in `poster/poster_data_audit.json`.
   - permutation A100 speedup: largest validated speedup in this grid `8.54x`.
     Source: `experiments/results/linux_server_a100/permutation_break_even/cpu_matched_baselines_summary.csv`, audited in `poster/poster_data_audit.json`.

6. Claims removed or softened
   - Removed "same validated result" from live poster text.
   - Removed "different smallest sufficient tool" from live poster text.
   - Removed "pressure shapes" from live poster text.
   - Removed "What Each Tool Solved" from live poster text.
   - Removed "hotspot moves" from live poster text.
   - Removed "equivalence" from the six-box validation mini-label where formal statistical equivalence was not the intended claim.
   - No numeric claims were removed because the requested retained values were verified against committed result files or the poster audit.
