# Dataset and checkpoint provenance

The public synthetic example and the manuscript study dataset are separate resources. Counts below must not be substituted for the manuscript's sample size.

## Public synthetic example

| Item | Exact public value |
|---|---|
| Input/target pairs | 1 |
| Independent synthetic patterns | 1 |
| Biological lesions/specimens | 0 |
| Physical phantoms | 0 |
| Training / validation / test pairs | 1 / 0 / 0 |
| Training / validation / test groups | 1 / 0 / 0 |
| Sensor-input grid | 8 × 8 |
| Reconstruction-target grid | 32 × 32 |
| Included trained checkpoints | 0 |

`generate_toy.py` samples an analytic Gaussian g(x,y) = exp(-((x - 0.15)^2 + (y + 0.1)^2) / 0.18), on regularly spaced coordinates from -1 to 1. The synthetic input is 0.2 + 0.8g on the input grid; the illustrative target is 80 + 120g on the target grid. The latter is labeled synthetic modulus in kPa for interface demonstration only. It is not a calibrated magnetic-field-to-modulus relation, COMSOL simulation, measurement or manuscript sample. There is no random augmentation, cropping, strain series or test-set performance claim in this example.

`toy_manifest.csv` assigns the pair to `synthetic_pattern_001` and the training split. Regeneration is deterministic. A local smoke-training checkpoint is newly trained from this one pair and is never a manuscript-trained checkpoint.

## Manuscript dataset information still required

The supplied revised manuscript describes deformation-induced inputs, the absolute difference between loaded and unloaded Bz maps, bicubic upsampling to 32 × 32, three sigmoid convolutional layers, a negative-PSNR loss, a 70%/10%/20% split, training-only normalization and selection on validation data. These statements do not establish the following missing information:

| Required study record | Current status |
|---|---|
| Total paired maps before and after exclusions/augmentation | Not provided in the public release; exact count unverified |
| Number of independent simulated lesions and physical phantoms/specimens | Not provided; these categories must be counted separately |
| Exact train/validation/test map and independent-unit counts | Not provided; percentages alone do not establish counts |
| Unit used to assign the original study split | Unverified; evaluating one lesion at a time does not prove a grouped split |
| Sample-generation procedure | Complete simulation parameters, sampling/channel order, units, load/reference pairing, calibration and augmentation records are absent |
| Repeated measurements and derived samples | Relationship between specimen, lesion, load/strain level, crop, augmentation and source simulation must be recorded |
| Original training run | Exact settings, software versions, seed, normalization values and saved split manifest are absent |
| Manuscript checkpoint | File, identifier/hash, validation-selection record and accessible source are absent |
| Fig. 4 and Supplementary Fig. 21 provenance | Per-panel input, target, checkpoint, processing script and plotting settings are absent |

The original checkpoint source and Supplementary Fig. 21 generation path therefore remain **unverified**. Shared dimensions or similar appearance do not establish that a current script generated a published panel. The manuscript describes a common 71–240 kPa display scale for Supplementary Fig. 21; that display range is not evidence of the training normalization range and is not imposed on the synthetic example.

## Manifest contract for a study release

Required CSV columns are `sample_id,group_id,split,input_8x8_csv,target_32x32_csv`. `sample_id` must be unique; `split` must explicitly be `train`, `val` or `test`. Paths are relative to the manifest. `group_id` must identify the independent experimental unit: for example, a specimen or independently generated simulated lesion. All related maps, repeated loads, crops and augmentations must stay with that unit. When lesions share a specimen or simulation source, use the shared parent as the grouping unit. The code cannot infer these relationships from images.

Training rejects missing split information, groups crossing splits and byte-identical input or target files crossing splits. Hash checks do not detect all semantic duplicates; authors must supply valid provenance and related-sample identifiers. No automatic random row split is performed. Report both map counts and independent-unit counts, with any rounding or unequal group sizes explaining deviations from nominal percentages. The present code enforces this contract for new runs; it does not establish how the historical study was split.

The loader accepts an explicitly axis-labeled matrix (`y_norm` header), a headerless numeric matrix, or a target long table with `x_norm,y_norm,mean_kpa`. Inputs must be finite 8 × 8 matrices and targets finite 32 × 32 matrices. Input CSVs must already contain the intended loaded-minus-unloaded absolute difference; the model does not calculate it from an unpaired map or infer channel ordering.

## Checkpoint and evaluation contract

A checkpoint stores `model_state`, `norm_stats`, `config`, `epoch` and `provenance`. The run summary records the manifest/config hashes, sample hashes and group membership, actual settings, seed, software environment and selected-checkpoint SHA-256. `best.pt` is selected by validation PSNR averaged first within each group and then across groups. The test split is evaluated only after training finishes and the selected validation checkpoint is loaded. The smoke test uses `last.pt` when validation data are absent and reports no test metrics.

Min/max statistics come only from training maps. They are reused unchanged for held-out data and inference. Values outside the training range are not clipped; the sigmoid output remains bounded by the training target range. Metrics are reported on this normalized scale, with PSNR data range 1; they are not physical-unit errors. Public SSIM uses a local uniform window of size 7 and constants 0.01² and 0.03². The exact historical preprocessing and metric implementation still require verification.

To establish manuscript reproducibility, release or provide a documented access route for the actual dataset, split manifest and original checkpoint, with hashes and panel-level mapping. If release is restricted, state the specific unavailable assets, reason and access conditions once confirmed. The current repository does not assert that these assets are unavailable in principle.
