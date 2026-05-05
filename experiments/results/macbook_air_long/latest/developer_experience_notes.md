# Developer experience notes

- Long runs append to CSV and can be resumed without rerunning completed scenario/implementation rows.
- Extra evidence runs also append by stable `run_id`; rerunning `experiments.run_macbook_evidence_extra` resumes instead of duplicating rows.
- Rows that would allocate unsafe broadcast or full null matrices are recorded as `skipped_memory_risk`.
- Per-scenario exceptions are recorded as `fail` rows with `notes`; they do not abort the whole run.
- JAX is labeled CPU-only in this MacBook tier and uses x64 for equivalence checks.
- Deprecated quick-validation outputs were removed from `latest` so downstream agents cite the curated figures and CSVs.
