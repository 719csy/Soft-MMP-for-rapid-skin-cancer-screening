# Reproduction workflows and archived evidence

The commands below exercise the public 8 × 8 → 32 × 32 implementation. The separate archive audit documents the experiment behind the Supplementary Fig. 21 numerical source maps. All commands run from the repository root.

## Public environment

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements.txt` pins the tested public-package versions. Run summaries capture installed versions and platform. This environment is not a record of the original training machine. CPU execution is recommended for the example; deterministic operations do not guarantee identical floating-point output across platforms.

## Regenerate the synthetic pair

```bash
python -m srcnn_8x8_to_32x32.generate_toy
```

This writes one synthetic 8 × 8 input, one 32 × 32 target, an 8 × 8 input heatmap, a one-row manifest and `toy_provenance.json` in `srcnn_8x8_to_32x32/examples/`. [DATASET_CARD.md](DATASET_CARD.md) gives the formula. No measured lesion or phantom is represented.

## Explicit random-weight smoke inference

```bash
python -m srcnn_8x8_to_32x32.infer --input srcnn_8x8_to_32x32/examples/toy_input_8x8.csv --out_dir smoke_test_artifacts/random_demo --smoke_test --device cpu
```

Outputs are `prediction_32x32.csv`, `prediction_32x32.png`, `prediction_32x32_input_8x8.png` and `inference_metadata.json`. The input heatmap displays 64 array cells without smoothing. The reconstructed image is labeled `DEMO ONLY`; no checkpoint was loaded and its output has no calibrated physical meaning. Omitting both a checkpoint and `--smoke_test` fails before output is written.

## One-epoch smoke training

```bash
python -m srcnn_8x8_to_32x32.train --manifest srcnn_8x8_to_32x32/examples/toy_manifest.csv --epochs 1 --out_dir smoke_test_artifacts/toy_training --smoke_test --device cpu
python -m srcnn_8x8_to_32x32.infer --input srcnn_8x8_to_32x32/examples/toy_input_8x8.csv --checkpoint smoke_test_artifacts/toy_training/last.pt --out_dir smoke_test_artifacts/trained_demo --device cpu
```

Use a new, empty training output directory. The run writes `last.pt`, `train_log.jsonl` and `training_summary.json`. There is no `best.pt` or test metric because this one-pair example has no validation/test splits. The resulting checkpoint is an illustration, not the archived manuscript checkpoint.

## New grouped training with supplied data

Provide a manifest meeting [DATASET_CARD.md](DATASET_CARD.md), with nonempty train, val and test splits. Related maps must remain in the same independent group. Set actual units and data descriptions in a copy of `srcnn_8x8_to_32x32/config.example.yaml`.

```bash
python -m srcnn_8x8_to_32x32.train --config path/to/study_config.yaml --manifest path/to/study_manifest.csv --out_dir checkpoints/study_run --device cpu
python -m srcnn_8x8_to_32x32.infer --config path/to/study_config.yaml --input path/to/input_8x8.csv --checkpoint checkpoints/study_run/best.pt --out_dir checkpoints/study_inference --device cpu
```

These paths denote user-supplied assets. The code computes training-only min/max, selects `best.pt` on validation group-mean PSNR and evaluates test maps after selection. Inference reuses checkpoint normalization without fitting to test inputs. Run and inference metadata preserve hashes and provenance. These are new public-code runs; `manuscript_figure_reproduction` remains false.

## What the archive audit checked

The audit read the author-held `checkpoint_srcnn_realbz_mean_kpa` archive and the Supplementary Fig. 21 source packages. It did not retrain the model or alter the source files.

1. Counted 81 candidate IDs, 78 paired lesions and 3 exclusions from `paired_samples.json`.
2. Reconstructed the historical source-stratified split from saved IDs and seed 42; obtained 55/8/15 and exactly matched all 15 saved test IDs.
3. Recomputed training-only extrema from the 55 training caches and matched `config.json` and `best.pt`.
4. Matched the checkpoint's epoch 196 and validation score to the maximum across all 200 logged epochs.
5. Ran the selected model on all 15 cached test cases using the archived 64 × 64 numerical grid; compared per-case PSNR, SSIM, MAE and RMSE with the saved test table.
6. Checked all 78 saved 8 × 8 field displays against bicubic resampling of their cached field samples.
7. Recomputed the two Supplementary Fig. 21 reconstruction arrays, compared them with saved figure CSVs, matched probability maps to their stage-1 sources, and matched source hashes to the artwork revision manifest.

The audit used CPU inference and a restricted `weights_only=True` checkpoint load with the serialized Windows path type explicitly allowlisted. [audit_results.json](provenance/audit_results.json) records the audit environment, comparison tolerances and actual numerical differences. These audit versions are separate from the original training environment.

## Public evidence files

| File | Contents |
|---|---|
| [archived_run.json](provenance/archived_run.json) | Cohort counts, exclusions, split procedure, sample generation, training settings, normalization, checkpoint hash and archived metrics |
| [lesion_split_manifest.csv](provenance/lesion_split_manifest.csv) | All 78 modelled lesion IDs, cohorts, lesion groups, split membership, cache hashes and 8 × 8 display hashes |
| [supplementary_fig21.json](provenance/supplementary_fig21.json) | Two figure cases, selected checkpoint, map/script hashes and figure recomputation checks |
| [audit_results.json](provenance/audit_results.json) | Independent audit environment, checks and numerical comparison errors |

Source hashes refer to the original byte contents before removal of private filesystem paths from this public metadata. The original files are retained in the author-held archive. The released lesion split manifest is a provenance table; it is not the input manifest for the new public trainer, because the archived numerical grid differs.

## Full-figure regeneration boundary

The public repository includes metadata, hashes and the synthetic example. It does not bundle original field arrays, modulus targets, dermoscopy probability maps, the epoch-196 weights or manuscript artwork. Regenerating the original numerical maps requires those assets and the archived processing settings. Reproducing a particular manuscript layout additionally requires that layout's source artwork. Contact the corresponding author for study-asset enquiries.

Do not substitute a newly trained 8 × 8 → 32 × 32 model for the archived checkpoint, or substitute resampled field displays for the archived model inputs when comparing the reported metrics. [CLAIM_MAP.md](CLAIM_MAP.md) states the figure-specific scope.
