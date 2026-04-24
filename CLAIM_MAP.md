# Public Claim Map

This public claim map tracks the code artifacts needed to support the manuscript-facing repository. It is intentionally conservative: experimental wet-lab protocols, raw human data, patient-linked metadata, and uncleared fabrication files are not expected in this public code release.

## Status Legend

- **GREEN**: public and manuscript-matched.
- **YELLOW**: public but partial, legacy, or waiting on final data.
- **RED**: not yet public in this repository.
- **N/A**: experimental or wet-lab method where public code is not required.
- **MOVE**: canonical public home should be a companion repository.

## Claims

| ID | Method / claim | Public artifact | Current status | Minimum public artifact |
|---|---|---|---|---|
| NS-01 | Hall-sensor signal to stiffness-map pipeline | Documentation and interface files | RED | Add measurement pipeline docs and schemas |
| NS-02 | Device optimization | Optional data/doc | N/A | Optional cleared figure data |
| NS-03 | Phantom calibration to Young's modulus | Calibration script | RED | Add toy phantom calibration example |
| NS-04 | Depth-resolved phantom evaluation | Analysis script | RED | Add depth-vs-contrast toy workflow |
| NS-05 | Wireless 64-channel acquisition stack | Companion hardware repo | MOVE | Publish protocol and channel map in hardware repo |
| NS-06 | AI-assisted SRCNN super-resolution | `srcnn_8x8_to_32x32/` | GREEN for code interface; YELLOW for final data | Publish true 8x8-to-32x32 code, config, toy data, and smoke tests |
| NS-07 | Bx/By/Bz field reconstruction and export handling | Field mapping utilities | RED | Add export parser and sample schema |
| NS-08 | In vivo/ex vivo stiffness analysis | De-identified analysis scripts | RED | Add toy tables and representative analysis script |
| NS-09 | Morphology, anisotropy, boundary extraction | Morphology analysis script | RED | Add contour and anisotropy example |
| NS-10 | Histology-linked comparison | Optional cross-link | YELLOW | Add note or alignment script after clearance |
| NS-11 | COMSOL finite-element simulation | Companion simulation repo | MOVE | Publish redacted exports and post-processing scripts |
| NS-12 | Animal/human validation protocols | Protocol note only | N/A | Optional non-sensitive methods note |

## SRCNN Release Scope

The current upload closes the main public-code gap for NS-06 by adding a manuscript-aligned package interface for 8x8 Hall input to 32x32 map output. The package does not claim final figure reproduction while the 8x8 simulation dataset remains pending.

