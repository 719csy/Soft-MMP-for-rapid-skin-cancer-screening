# Manuscript figure and code correspondence

This map describes the current public checkout. An implemented interface does not establish that it produced the manuscript results. All sensor-input heatmaps in this release use an 8 × 8 array; reconstructed outputs use a 32 × 32 grid.

| Manuscript output or claim | Public implementation | What can be reproduced now | What is missing |
|---|---|---|---|
| SRCNN architecture and 8 × 8 input interface | `model.py`, `infer.py`, `config.example.yaml` | Input/output dimensions and forward execution | Original trained weights and historical run settings |
| Training, normalization and held-out evaluation | `train.py`, `dataset.py`, `metrics.py` | One-pair smoke training; grouped workflow on user-supplied data | Actual study dataset, independent-unit identifiers, original splits and counts |
| Fig. 4g architecture | `model.py` | Three-layer sigmoid architecture and bicubic upsampling | Original schematic artwork; this is architectural correspondence only |
| Fig. 4h reconstruction comparison | `infer.py` provides the numerical interface | No manuscript panel or numerical result | Paired source data, calibration, checkpoint, panel mapping and plotting script |
| Fig. 4i–k phantom validation, including the 8 × 8 input heatmap | Shape-checked input and reconstructed-output export | No manuscript phantom result | Independent phantom records, reference/loaded maps, calibration, targets and original checkpoint |
| Supplementary Fig. 21 modelled lesion inputs and modulus reconstructions | Generic 8 × 8 to 32 × 32 interface only | No numbered panel; historical generating pipeline remains unverified | Original inputs, ground truth, group identities, checkpoint, generation script and display settings |
| Fig. 4a–f and other experimental results | Not implemented here | No corresponding manuscript result | Acquisition, calibration, specimen/phantom data and analysis scripts |
| Morphology, histology and PET comparisons | Not implemented here | No corresponding manuscript result | Registration, source data and analysis workflows |
| Hardware/firmware and finite-element simulations | Not implemented here | No corresponding acquisition or simulation result | Hardware, firmware, model files and export procedures |

## Interpretation of generated files

The synthetic example can reproduce only the repository's demonstration input and execution workflow. `--smoke_test` predictions and checkpoints are demonstrations, regardless of their visual appearance. They must not be relabeled as manuscript panels or used to validate the reported improvements.

No public command currently claims exact Fig. 4 or Supplementary Fig. 21 reproduction. Once source assets are available, each panel should be linked to its input/target identifiers, independent group, split, preprocessing configuration, checkpoint SHA-256, script revision, command and plotting limits. The original figure provenance must be verified from those records, not inferred from this updated implementation.
