# Dataset and checkpoint provenance

This record distinguishes the archived modelled-lesion experiment from the current public 8 × 8 → 32 × 32 implementation. Counts and metrics from one must not be attributed to the other. The source archive was audited on 16 October 2026; file hashes are in [archived_run.json](provenance/archived_run.json).

## Archived dataset size and unit of analysis

The pairing inventory contains 81 candidate lesion IDs. Three ISIC2018 cases were excluded because their Bz source records were absent. The retained dataset contains **78 different modelled lesion IDs and 78 input–target pairs**, one pair per ID.

| Source cohort | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| ISIC2018 | 27 | 4 | 8 | 39 |
| MIDAS | 19 | 3 | 5 | 27 |
| PH2 | 9 | 1 | 2 | 12 |
| **Total** | **55** | **8** | **15** | **78** |

The actual fractions are 70.51%, 10.26% and 19.23%; the stated 70%/10%/20% proportions are nominal. The same counts apply to pairs and lesion IDs.

The partition unit was the **modelled lesion ID**, stratified by source cohort. There are zero physical phantoms and zero measured-modulus specimens in this training cohort. These counts do not describe separate phantom experiments elsewhere in the manuscript. The archived partition uses lesion IDs, with no patient/specimen parent identifier; it supports lesion-level separation, not a claim of patient-level or specimen-level independence.

[lesion_split_manifest.csv](provenance/lesion_split_manifest.csv) records every retained ID, source cohort, split and source-file hash. Membership was reconstructed from the saved paired IDs and the historical split routine using seed 42. All 15 reconstructed test IDs exactly match the saved per-sample test table. This is a reconstructed provenance manifest, not a newly randomized split or a contemporaneously saved training manifest.

Within each sorted source cohort, the historical routine shuffled sorted IDs using one `random.Random(42)` generator. It assigned the first rounded 20% to test and rounded 12.5% of the remaining cases to validation; the rest went to training. Cohorts were processed in sorted order. All maps associated with one retained lesion were combined into its single paired sample before this split.

## Sample generation and array meaning

Each lesion's modelled modulus target was the `mean_kpa` field from `interp_mean_kpa.csv`. These dermoscopy-derived modulus priors were inputs to the forward model, not experimentally measured clinical modulus ground truth.

For each lesion, COMSOL Bz exports carrying load labels 0 and 0.08 supplied the reference and loaded fields. Their pointwise absolute difference formed the input. The archived configuration records a sensor plane at z = 0.5 mm and 25 field-sampling sites within the model's 0–10 mm field of view. The field differences were bicubically interpolated to the **64 × 64 reconstruction grid**, and paired with the lesion's modulus target on that grid. One full-map pair per lesion was used; the historical training routine applied no crop or augmentation expansion.

The later `hall8_conditioning_exports` process generated **8 × 8 field-display arrays** by bicubic resampling of the cached field differences, with `align_corners=False`. All 78 saved display arrays were checked against that operation. The export produced no new independent simulation, measurement or trained checkpoint. Thus an 8 × 8 displayed array must not be described as evidence of 64 independently acquired field samples in this archived experiment.

The public software accepts prepared **8 × 8 input matrices** and reconstructs on a **32 × 32 output grid**. That interface is separate from the archived experiment above.

## Archived training and selected checkpoint

| Setting | Recorded value |
|---|---|
| Seed | 42 |
| Epochs / batch size | 200 / 8 |
| Optimizer | Adam |
| Learning rate / weight decay | 0.0001 / 0 |
| Loss | Negative PSNR from normalized batch MSE, data range 1, epsilon 1e-8 |
| Network | Three convolution layers, channels 1 → 64 → 32 → 1; sigmoid after each |
| Kernel widths | 9, 5, 5 |
| Augmentation / dropout / early stopping / learning-rate schedule | None |
| Input normalization range | 8.737906455993652–32.03804397583008 mT |
| Target normalization range | 71.17301940917969–240.06137084960938 kPa |
| Normalization source | Training maps only |
| Checkpoint selection | Highest mean PSNR across the 8 validation lesions |
| Selected file / epoch | `best.pt` / 196 |
| Selected validation PSNR | 13.593811869621277 dB |

Checkpoint source: the author-held `checkpoint_srcnn_realbz_mean_kpa` run archive. The saved checkpoint is 695,297 bytes and has SHA-256:

```text
bb1b02629de706c05e22b7f3d44b6f7528d5fcbd985f9cec3b7ee05dc66212bd
```

Its stored epoch and validation score agree with the maximum in the complete 200-epoch log. Training-split extrema were recomputed from the 55 cached pairs and exactly equal the saved normalization values. The historical dataset loader applies training min/max without clipping. The comparison baseline clips normalized field inputs to [0, 1] before mapping them to the target range.

The archived checkpoint belongs to the 64 × 64 reconstruction experiment. Its convolutional weights must not be described as a checkpoint trained for the current 8 × 8 → 32 × 32 interface merely because the convolution architecture is similar.

## Archived held-out performance

The saved metrics are means over the 15 test lesions, evaluated after validation-based checkpoint selection.

| Metric | Selected SRCNN | Range-matched field baseline |
|---|---:|---:|
| PSNR (dB) | 14.184524 | 9.851483 |
| SSIM | 0.415746 | 0.209405 |
| MAE (kPa) | 26.922670 | 44.439748 |
| RMSE (kPa) | 33.344205 | 54.862346 |

PSNR and SSIM use training-normalized targets with data range 1. SSIM uses a uniform window of width 7, stride 1, zero padding of 3, and constants 0.01² and 0.03². MAE and RMSE use denormalized values in kPa. The figures above are archived experiment results, not results of the public synthetic example or the current interface. Independent CPU inference on all 15 test cases was compared with the saved per-case metrics; numerical differences are recorded in [audit_results.json](provenance/audit_results.json).

## Supplementary Fig. 21

The numerical source chain is the archived training cache and epoch-196 checkpoint, the `SI_Fig21_regenerated` output package, and the recorded October 5 presentation revision. The two modelled lesions are:

| Case | Modelled lesion category | Split | Saved test PSNR (dB) |
|---|---|---|---:|
| `midas_lesion_626` | Keratinocytic carcinoma in situ | Test | 13.274233 |
| `midas_lesion_218` | Melanoma in situ | Test | 11.938630 |

The 8 × 8 field-display exports, SRCNN numerical reconstructions, modelled targets and dermoscopy probability maps have separate source records. Reconstruction and target arrays use the archived 64 × 64 numerical grid. The shared modulus scale is 71.17301940917969–240.06137084960938 kPa.

The probability map is a **stage-1 dermoscopy segmentation probability**, averaged over the lesion's views and bilinearly resampled to the reconstruction grid. It is not SRCNN reconstruction confidence or uncertainty. Both figure reconstructions were independently recomputed from the identified checkpoint and compared with the saved figure CSVs. The figure script clips its normalized input to [0, 1]; neither of these two cases has values changed by that clipping. Source hashes also match the preserved-data entries in the presentation revision manifest.

[supplementary_fig21.json](provenance/supplementary_fig21.json) records the two cases, checkpoint identity, numerical-source hashes, original generation-script hash and comparison errors. It identifies the numerical source chain; it does not assert that the October 5 presentation layout is byte-identical to every later manuscript layout.

## Public synthetic example

| Item | Exact public value |
|---|---:|
| Input/target pairs / synthetic patterns | 1 / 1 |
| Biological lesions/specimens / physical phantoms | 0 / 0 |
| Training / validation / test pairs and groups | 1 / 0 / 0 |
| Included trained checkpoints | 0 |

The analytic pattern is g(x,y) = exp(-((x - 0.15)² + (y + 0.1)²) / 0.18), on regularly spaced coordinates from -1 to 1. The illustrative input is 0.2 + 0.8g on an 8 × 8 grid; the target is 80 + 120g on a 32 × 32 grid. This is an interface demonstration, not a calibrated field-to-modulus relation, COMSOL result or manuscript sample. The example is assigned to `synthetic_pattern_001` in the training split. A smoke-training checkpoint is newly fitted to this one pair.

## Contract for new public-interface runs

Required manifest columns are `sample_id,group_id,split,input_8x8_csv,target_32x32_csv`. IDs must be unique and splits explicit. Paths are relative to the manifest. The group must identify the independent experimental unit; related loads, maps, crops and augmentations remain in that group. Lesions sharing a specimen or simulation parent must use the shared parent ID. No automatic row split is performed.

The loader checks finite dimensions and rejects groups or byte-identical inputs/targets crossing splits. Hashes do not identify every semantic duplicate. Supported CSV layouts are axis-labeled matrices (`y_norm` header), headerless matrices, and target long tables with `x_norm,y_norm,mean_kpa`. Inputs must already contain the intended absolute field difference.

New checkpoints store `model_state`, `norm_stats`, `config`, `epoch` and `provenance`. The public trainer selects `best.pt` by validation PSNR averaged within groups and then across groups; test evaluation follows selection. A smoke run without validation saves `last.pt`. Public PSNR/SSIM are normalized image metrics. The public code and metadata describe new runs; the archived records above describe the historical experiment.
