# Soft-MMP for Rapid Skin Cancer Screening

This repository provides a public SRCNN implementation for **8 × 8 Hall-sensor input arrays → 32 × 32 reconstructed maps**. All sensor-input heatmaps use the 8 × 8 array. A 32 × 32 reconstruction is an inferred output grid, not an additional sensor array or a directly measured spatial resolution.

**Release scope:** the code and synthetic interface example are available. The manuscript training dataset, original split manifest, trained checkpoint and figure-generation provenance are not included. This checkout cannot independently regenerate the manuscript figures or establish their reported performance.

## What is available

- A three-layer sigmoid SRCNN with bicubic upsampling, negative-PSNR training loss and strict input dimensions.
- Explicit sample and independent-group identifiers, checks for groups or identical source files crossing splits, training-only normalization and validation-based checkpoint selection.
- Run summaries containing split counts, file hashes, effective settings, software versions and checkpoint provenance.
- One reproducible analytic input/target pair for a smoke test: one synthetic pattern, zero biological specimens and zero physical phantoms. This is not the manuscript dataset.

The manuscript reports a 70%/10%/20% split, but the total number of maps, independent lesion/phantom/specimen counts, exact split membership and original checkpoint identifier are not established by the public files. See [DATASET_CARD.md](DATASET_CARD.md) for the evidence gaps and [CLAIM_MAP.md](CLAIM_MAP.md) for figure-specific limits, including Supplementary Fig. 21.

## Quick start

Use Python 3.11 or 3.12. The following commands run from the repository root; use a fresh output directory for each training run.

```bash
python -m pip install -r requirements.txt
python -m srcnn_8x8_to_32x32.infer --input srcnn_8x8_to_32x32/examples/toy_input_8x8.csv --out_dir smoke_test_artifacts/random_demo --smoke_test --device cpu
python -m srcnn_8x8_to_32x32.train --manifest srcnn_8x8_to_32x32/examples/toy_manifest.csv --epochs 1 --out_dir smoke_test_artifacts/toy_training --smoke_test --device cpu
python -m srcnn_8x8_to_32x32.infer --input srcnn_8x8_to_32x32/examples/toy_input_8x8.csv --checkpoint smoke_test_artifacts/toy_training/last.pt --out_dir smoke_test_artifacts/trained_demo --device cpu
```

The first command uses random weights only because `--smoke_test` is explicit. The one-pair training run has no validation or test split and saves `last.pt`; it cannot measure generalization. Every demo image is labeled accordingly. Normal inference requires a checkpoint with its training normalization, configuration and provenance.

See [REPRODUCE.md](REPRODUCE.md) for expected outputs, data formats, grouped training and verification commands. The default file is `config.example.yaml`: its optimizer settings are public example defaults, not a recovered configuration from the manuscript experiments.

## Code availability scope

Source code for the 8 × 8 to 32 × 32 SRCNN implementation, input/output formats and synthetic interface tests is available in this repository. The simulation training data, original sample and specimen split assignments, calibration inputs, manuscript-trained model weights and per-panel figure-generation records are not included in this release. The public example validates software execution only and does not reproduce the manuscript's numerical results, Fig. 4 or Supplementary Fig. 21. Requests for study data or manuscript-trained assets should be directed to the corresponding author, as stated in the manuscript; availability and access conditions must be confirmed by the authors.

## Repository contents

| File or directory | Purpose |
|---|---|
| `srcnn_8x8_to_32x32/` | Model, training, inference, metrics and provenance |
| `srcnn_8x8_to_32x32/examples/` | One synthetic pair, manifest and 8 × 8 input heatmap |
| `DATASET_CARD.md` | Public dataset counts and missing study information |
| `CLAIM_MAP.md` | Which manuscript outputs can currently be reproduced |
| `REPRODUCE.md` | Executable workflows and their limits |
| `tests/` | Shape, split leakage, checkpoint and end-to-end checks |

Hardware acquisition, calibration, finite-element model generation and downstream experimental analysis are not supplied here. No future release date or access condition is implied.
