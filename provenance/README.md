# Audited study provenance

These files summarize author-held source records audited on 16 October 2026. They contain no source filesystem paths or raw dermoscopy images.

- [archived_run.json](archived_run.json): exact dataset size, split counts, generation procedure, training settings, checkpoint identity, archived test results and source hashes.
- [lesion_split_manifest.csv](lesion_split_manifest.csv): reconstructed historical membership for all 78 modelled lesion IDs, with one paired map per ID. Source-cohort labels are strata; `group_id` is the lesion unit. This table is not a public-trainer input manifest.
- [supplementary_fig21.json](supplementary_fig21.json): numerical provenance for the two figure cases, including reconstruction and probability-map source hashes.
- [audit_results.json](audit_results.json): independent CPU recomputation checks and differences from saved outputs.

The archived run uses a 64 × 64 numerical reconstruction grid. Its 8 × 8 field displays are bicubic resampling products. The current public package uses an 8 × 8 input and a 32 × 32 output. These records preserve that distinction.

Original source arrays, checkpoint weights and artwork are not included. Hashes establish file identity for anyone with access to the original assets; they are not download links or a substitute for those assets. The historical code revision is recorded for traceability. The study records do not establish patient-level independence or a measured clinical modulus ground truth.
