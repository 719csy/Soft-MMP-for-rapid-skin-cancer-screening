# Manuscript figure and code correspondence

The public 8 × 8 → 32 × 32 package, the archived modelled-lesion run and the manuscript artwork are distinct resources. The archive audit establishes source provenance for Supplementary Fig. 21; publishing metadata does not distribute its numerical assets.

| Output or claim | Code or source record | Supported reproduction and release scope |
|---|---|---|
| Public 8 × 8 input interface | `model.py`, `infer.py`, `config.example.yaml` | Input/output checks and forward execution; one synthetic pair is bundled |
| Public training and grouped evaluation | `train.py`, `dataset.py`, `metrics.py` | Smoke training and new runs on supplied data; these are separate from the archived experiment |
| Archived modelled-lesion dataset and split | [Split manifest](provenance/lesion_split_manifest.csv), [run record](provenance/archived_run.json) | 78 paired lesion IDs; 55/8/15 split; counts, membership and hashes public; original arrays author-held |
| Fig. 4g architecture | Public three-layer sigmoid model | Architecture correspondence; original schematic artwork not bundled |
| Fig. 4h reconstruction comparison | Public inference interface | No exact panel-generation workflow distributed; the archived Supplementary Fig. 21 checkpoint is not assigned to this panel |
| Fig. 4i–k phantom validation | Public shape-checked input/output export | Phantom acquisition, calibration, target data and panel-specific checkpoint records are outside this release; the 78 modelled lesions are not phantom counts |
| Supplementary Fig. 21 field-input display | `hall8_conditioning_exports` source record | 8 × 8 bicubic field-display arrays traced to the archive; these are resampled displays, not evidence of native 8 × 8 acquisition |
| Supplementary Fig. 21 modulus reconstruction | `best.pt` epoch 196, cached model inputs, `make_si21.py` | Cases 626 and 218 numerically checked; archived reconstruction grid is 64 × 64; metadata public, weights and arrays author-held |
| Supplementary Fig. 21 probability map | Stage-1 dermoscopy segmentation outputs | Mean probability over lesion views, bilinearly resampled to the reconstruction grid; not reconstruction uncertainty |
| Supplementary Fig. 21 presentation | `SI_Fig21_regenerated` → `SI_Fig21_revised_20261005` | Source CSV hashes preserved in the recorded artwork revision; later layout identity is outside this numerical provenance record |
| Fig. 4a–f and other experimental results | Outside the public SRCNN package | Acquisition, calibration and experimental analysis not distributed here |
| Morphology, histology and PET comparisons | Outside the public SRCNN package | Registration, source data and comparison workflows not distributed here |
| Hardware, firmware and finite-element model generation | Outside the public SRCNN package | No corresponding acquisition or simulation workflow distributed here |

## Supplementary Fig. 21 source chain

The archived run `checkpoint_srcnn_realbz_mean_kpa` supplies the selected checkpoint and lesion caches. `SI_Fig21_regenerated/make_si21.py` generates numerical maps for `midas_lesion_626` and `midas_lesion_218`. Both IDs occur in the saved 15-case test table. The subsequent October 5 artwork revision records unchanged source CSV hashes. [The figure provenance record](provenance/supplementary_fig21.json) identifies the checkpoint, cases, generation-script hash, source-file hashes and independent recomputation results.

The two probability maps come from dermoscopy segmentation. The modelled modulus targets are forward-model priors. Neither should be relabeled as measured clinical modulus or reconstruction confidence.

## Interpretation of generated files

The synthetic example reproduces the repository's demonstration input and execution workflow only. `--smoke_test` predictions and checkpoints are demonstrations. Newly trained public-interface models retain their own provenance and do not inherit a manuscript-figure claim.

No bundled command regenerates the manuscript panels from the original study assets, because those arrays and weights are not distributed in this release. The archive audit recovers and checks their identities and numerical relationships; the public 8 × 8 → 32 × 32 interface does not replace the archived 64 × 64 numerical reconstruction used for the figure-source maps.
