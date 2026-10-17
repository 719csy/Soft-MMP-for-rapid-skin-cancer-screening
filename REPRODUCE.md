# Reproduce the public SRCNN workflows

These commands verify software execution with an 8 × 8 sensor input and a 32 × 32 reconstruction. They do not reproduce manuscript figures or performance. All commands run from the repository root. The default configuration is `srcnn_8x8_to_32x32/config.example.yaml`.

## Environment

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements.txt` pins the tested package versions. Run summaries also capture the installed versions and platform. CPU execution is recommended for the example; seeds and deterministic operations do not guarantee bit-identical results across different hardware or library versions.

## Regenerate the one-pair example

```bash
python -m srcnn_8x8_to_32x32.generate_toy
```

This writes one synthetic 8 × 8 input, one 32 × 32 target, an 8 × 8 input heatmap, a one-row manifest and `toy_provenance.json` in `srcnn_8x8_to_32x32/examples/`. There is one analytic pattern and no measured lesion or phantom. [DATASET_CARD.md](DATASET_CARD.md) specifies the exact formula and counts.

## Explicit random-weight smoke inference

```bash
python -m srcnn_8x8_to_32x32.infer --input srcnn_8x8_to_32x32/examples/toy_input_8x8.csv --out_dir smoke_test_artifacts/random_demo --smoke_test --device cpu
```

Outputs are `prediction_32x32.csv`, `prediction_32x32.png`, `prediction_32x32_input_8x8.png` and `inference_metadata.json`. The input heatmap displays the 64 sensor sites without smoothing. The reconstructed image is labeled `DEMO ONLY` and records that no checkpoint was loaded; its output has no calibrated physical meaning. Omitting both a checkpoint and `--smoke_test` fails before any prediction is written.

## One-epoch training smoke test

```bash
python -m srcnn_8x8_to_32x32.train --manifest srcnn_8x8_to_32x32/examples/toy_manifest.csv --epochs 1 --out_dir smoke_test_artifacts/toy_training --smoke_test --device cpu
python -m srcnn_8x8_to_32x32.infer --input srcnn_8x8_to_32x32/examples/toy_input_8x8.csv --checkpoint smoke_test_artifacts/toy_training/last.pt --out_dir smoke_test_artifacts/trained_demo --device cpu
```

Use a new, empty training output directory on reruns. The training command writes `last.pt`, `train_log.jsonl` and `training_summary.json`. It does not write `best.pt` or test metrics because the example contains one training pair and no validation/test pairs. The loaded-checkpoint demo remains a one-pair illustration, not an independently validated model or the original manuscript checkpoint.

## Grouped training with a supplied study dataset

Provide a manifest meeting [DATASET_CARD.md](DATASET_CARD.md), with nonempty train, val and test splits. Keep every map derived from the same independent specimen or simulation source in the same group and split. Copy the example configuration and set the actual input units, target units and data description. The code consumes prepared absolute field-difference inputs; calibration and source simulation are not supplied here.

```bash
python -m srcnn_8x8_to_32x32.train --config path/to/study_config.yaml --manifest path/to/study_manifest.csv --out_dir checkpoints/study_run --device cpu
python -m srcnn_8x8_to_32x32.infer --config path/to/study_config.yaml --input path/to/input_8x8.csv --checkpoint checkpoints/study_run/best.pt --out_dir checkpoints/study_inference --device cpu
```

These paths are user-supplied placeholders, not bundled manuscript assets. Normal training shuffles only the training loader, computes min/max only from training data, uses negative PSNR loss, selects `best.pt` on validation group-mean PSNR and evaluates test maps once after selection. Metrics include individual samples, groups and their means. The configured optimizer defaults must not be represented as verified historical training settings.

Inference requires checkpoint metadata and matching data/model configuration; it does not silently fit new normalization to test inputs. Both run and inference metadata retain hashes and provenance. Checkpoints generated here are marked as new public-code runs, with manuscript provenance unverified.

## Manuscript reproduction status

Neither these demonstrations nor newly trained user models reproduce Fig. 4 or Supplementary Fig. 21 without the original source assets. See [CLAIM_MAP.md](CLAIM_MAP.md) for panel-level requirements. Do not use synthetic sample counts, newly generated weights or example hyperparameters to fill missing manuscript Methods information.
