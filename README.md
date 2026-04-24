# Soft-MMP for Rapid Skin Cancer Screening

This repository is the manuscript-facing code release for the Soft-MMP rapid skin cancer screening workflow. The current public priority is the SRCNN reconstruction interface used to map sparse Hall-sensor magnetic measurements to a higher-resolution modulus map.

## Current Public Status

- `srcnn_8x8_to_32x32/` contains the manuscript-aligned SRCNN package interface: true 8x8 Hall input CSV to 32x32 output map.
- The 8x8 simulation dataset is still pending, so this release includes code, schemas, and synthetic toy examples only.
- `srcnn_modulus_baseline_5x5.py` is retained as a legacy baseline from the earlier public repository state. It is not the manuscript-matched SRCNN package.
- No restricted human-source data, identifying metadata, private caches, or trained manuscript weights are included.

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m srcnn_8x8_to_32x32.infer \
  --config srcnn_8x8_to_32x32/config.paper.yaml \
  --input srcnn_8x8_to_32x32/examples/toy_input_8x8.csv \
  --out_dir release/toy_outputs
```

Run a one-epoch smoke training pass on the synthetic toy manifest:

```bash
python -m srcnn_8x8_to_32x32.train \
  --config srcnn_8x8_to_32x32/config.paper.yaml \
  --manifest srcnn_8x8_to_32x32/examples/toy_manifest.csv \
  --epochs 1 \
  --out_dir smoke_test_artifacts/srcnn_8x8_smoke
```

See [REPRODUCE.md](REPRODUCE.md) for the complete smoke-test workflow.

## Repository Map

- `srcnn_8x8_to_32x32/`: SRCNN package, CLI entry points, config, and toy examples.
- `CLAIM_MAP.md`: public claim-to-code map for the manuscript-facing release.
- `REPRODUCE.md`: setup and smoke-test instructions.

Hardware acquisition, embedded firmware, and finite-element simulation code are expected to live in companion repositories or future releases once cleared.
