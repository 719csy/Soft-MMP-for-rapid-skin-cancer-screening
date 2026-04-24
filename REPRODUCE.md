# Reproduce the Public SRCNN Smoke Tests

These commands validate the public package interface without using private data or manuscript checkpoints.

## 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 2. Toy Inference

This command reads a synthetic 8x8 Hall input CSV and writes a 32x32 prediction CSV plus heatmap. No checkpoint is required for the public smoke test; random initialization is allowed only for this interface check.

```bash
python -m srcnn_8x8_to_32x32.infer \
  --config srcnn_8x8_to_32x32/config.paper.yaml \
  --input srcnn_8x8_to_32x32/examples/toy_input_8x8.csv \
  --out_dir release/toy_outputs
```

Expected outputs:

- `release/toy_outputs/prediction_32x32.csv`
- `release/toy_outputs/prediction_32x32.png`
- `release/toy_outputs/inference_metadata.json`

## 3. Toy Training Smoke Test

```bash
python -m srcnn_8x8_to_32x32.train \
  --config srcnn_8x8_to_32x32/config.paper.yaml \
  --manifest srcnn_8x8_to_32x32/examples/toy_manifest.csv \
  --epochs 1 \
  --out_dir smoke_test_artifacts/srcnn_8x8_smoke
```

The training smoke test should create a small local checkpoint under `smoke_test_artifacts/`. That directory is intentionally ignored by Git.

## 4. Privacy Check Before Upload

```bash
rg "H[:][\\\\]|G[:][\\\\]|One[D]rive|M[y] Drive|Clincal_ontolog[y]|pati[e]nt|r[a]w" \
  README.md REPRODUCE.md srcnn_8x8_to_32x32
```

The command should return no private paths. Mentions of excluded data classes in docs are acceptable only when they describe what is not included.
