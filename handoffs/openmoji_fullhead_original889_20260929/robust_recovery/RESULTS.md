# OpenMoji original-full-head robustness continuation (2026-09-29)

All model variants use the original 381,976-parameter shared electronic head,
17 µm / 10 cm optical geometry, and the common bounded amplitude/BMP contract.
The G1/G2 source is exactly SHA-256
`579b19e78befe93cd258ce02308235f1a9176316f37bc42fc2e1b6ff06d38e9f`.
G1 ideal and G2 direct-deployment share the same PT; G2 is **not yet measured**
on the optical bench. G3, G4, G5 are three separate full-head continuations.
All checkpoint epochs were chosen on the 1000-item TRAIN-derived holdout;
the 1000-item TEST was not used for gradients or epoch selection. The common
pretrained source previously saw original TRAIN and historical TEST feedback,
so this is an engineering comparison, not a new blind benchmark.

| Weight / training change | Clean TEST accuracy | CCD-only pressure | CCD+DC30% pressure | CCD+DC30%+grid pressure |
| --- | ---: | ---: | ---: | ---: |
| G1/G2 original, no new continuation | 0.8880 | 0.8865 | 0.8875 | 0.8812 |
| Diagnostic: same 4×40 steps, no added measure | 0.9435 | 0.9127 | 0.9012 | 0.9000 |
| G3: CCD noise plus paired clean/noisy consistency | 0.9455 | 0.9163 | 0.9027 | 0.9045 |
| G4: G3 training rule + coherent DC30% | 0.9440 | 0.9233 | 0.9185 | 0.9165 |
| G5: G4 + differentiable 17→8→17 raster proxy | 0.9435 | 0.9303 | 0.9195 | 0.9205 |

The pressure columns are the **same fixed original TEST1000**, two new noise
seeds (3041/3042), 1 logical-pixel shift, and a sensitivity noise setting
10× the original mean-scaled CCD proxy (up to 30% frame-mean offset, 10%
frame-mean read noise). Coherent DC30% and the raster proxy are added in the
named cumulative columns. This intentionally strong setting is not calibrated
to CCD electrons or dark frames and is **not physical accuracy**. Normal
evaluation would disable these training-only perturbations, so the evaluator
activates their optical/router flags explicitly while leaving learned modules
in eval mode. All source/checkpoint hashes and splits were checked.

The equal-step control is essential: merely training longer raises clean TEST
from 0.8880 to 0.9435. The final G3/G4/G5 chain uses the **same** 4×40-step
budget, CCD training noise strength 5×, and paired clean/noisy consistency
weight 0.05. Thus the staged differences are CCD, then coherent DC30%, then
the raster proxy. Under their corresponding matched pressure, G3 gains 0.36
percentage points over the equal-step control; G4 gains 1.58 points over G3;
G5 gains 0.40 points over G4. The full-stress G5 score is 2.05 points above
the equal-step control while clean TEST is the same 0.9435. This is directional
evidence for the cumulative **simulation-proxy** measures, strongest for DC.
The CCD-only and raster increments remain small; no per-sample paired
confidence interval or calibrated physical test is yet available. **Do not
describe each trick as proven effective on hardware.** The older unpaired G3
continuation and 3×/5× noise-strength probes did not reliably beat the
equal-step control; their reports are retained for audit. The final paired
chain was designed after seeing earlier TEST engineering feedback, though
every checkpoint epoch was still chosen without TEST, so this is not a blind
holdout claim.

The final G3/G4/G5 selected best weights and SHA-checked reports are in this
folder: `ccd_paired_best.pt`, `paired_r2_ccd_dc30_best.pt`, and
`paired_r3_ccd_dc30_grid_best.pt`. Their server-side last checkpoints remain
in separate run directories. `paired_chain_matched_test.json` is the fixed
TEST pressure report for G1/G2 and G3–G5, plus the diagnostic equal-step
control. `final_matched_test.json`, `ccd_unpaired_matched_test.json`,
and other files preserve earlier candidate/negative-control diagnostics.
No new optical capture was started.

Next, calibrate CCD noise/offset and spatial error from actual dark, repeat,
and signal frames before drawing a physical-robustness conclusion. Any optical
deployment must use whole-dataset, one-layer-at-a-time capture and verify the
simulation/BMP bridge and phase geometry first.
