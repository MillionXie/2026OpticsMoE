# Layered OpenMoji: 30% zero-order and small CCD noise retraining

This is a **new** checkpoint, not a recovered copy of the former 0.8715
checkpoint. The former number was recorded at epoch 30, but that epoch's `.pt`
was overwritten by the old best-checkpoint retention policy. A metric alone
cannot reconstruct its weights. Do not label this release `0.8715.pt`.

## Exact run

- Source training commit: `7121e2e0758ef3aae8c2aa29e8002a6e81e7d204`.
  The subsequent `1ec5c0c0` commit only corrects architecture-report wording.
- Profile: `layered_scene_exp05_dc30_ccdsmall`; seed 73; 100 epochs;
  periodic test at epoch 1, every 5 epochs, and the last epoch.
- Dataset: `openmoji_layered_anchor6_proportional_svg_v3`, 5,000 train /
  1,000 test, no validation split. The test set selects the checkpoint, so
  the selected test result is selection-biased; it is not an untouched holdout.
- GPU: A100-PCIE-40GB, `CUDA_VISIBLE_DEVICES=6`.
- All tested EMA checkpoints are kept in `tested_checkpoints/epoch_XXX.pt`,
  in addition to `best_checkpoint.pt` and `last_checkpoint.pt`.
- Run directory on the laboratory server:
  `/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t04_semantic_interaction/runs/simulation/layered_scene_exp05_dc30_ccdsmall_router_s73_e100_20260924`.

## Optical perturbation contract

During training, router, expert, and global optical calculations use nominal
zero-order **intensity fraction 0.30 per SLM**. This is a simulation parameter,
not a claim that a measured CCD zero-order spot contains exactly 30% intensity;
coherent interference and propagation can change the observed ratio. The CCD
perturbation is additive truncated biased Gaussian noise relative to each
frame's mean intensity: mean 0.01, standard deviation 0.01, clipped to
[-0.01, 0.03]. Existing random gain and phase dropout also remain active.
Random perturbations are disabled in `model.eval()` for deterministic test.

## Selected result and independent same-weight audit

| Item | Value |
| --- | ---: |
| Selected epoch | 70 |
| Changed-cell accuracy | 0.9400 |
| Edit-grid IoU | 0.9459 |
| Scene exact match | 0.8970 |
| Same-weight no-optical changed-cell accuracy | 0.5265 |
| No-optical decrease | 41.35 percentage points |
| Language fusion alpha (layer 1, layer 2) | 0.4852, 0.4842 |
| Vision fusion alpha (layer 1, layer 2) | 0.4706, 0.4769 |

The independent audit loads `tested_checkpoints/epoch_070.pt` and recomputes
the 1,000-case test metric; it matches the training-periodic 0.9400. The
same-weight no-optical result is an ablation, **not** a separately trained
electronic baseline.

Checkpoint SHA-256:
`aea978a6034af68059fe23f49d5fb42a9f71cdb8a72ae5c8d53bf2b44f7ccdb5`.

The language router selected-expert shares are approximately
`[0.2390, 0.2830, 0.2170, 0.2610]`; vision shares are
`[0.2500, 0.2485, 0.2495, 0.2520]`. All four experts are used, and both
router audits passed the configured 0.05–0.45 per-expert share bound.
At epoch 70, every one of the 12 router/expert/global phase arrays changed
from initialization: wrapped RMS change ranges from 0.3471 to 1.0520 rad.

Audit files in the run directory:

- `training_report.json`: selected epoch and checkpoint policy.
- `metrics/training_history.csv`: every epoch and every periodic test.
- `metrics/phase_training_audit.json`: phase changes from initialization.
- `audit_epoch070/audit.json`: fresh normal/no-optical metrics, alpha,
  router shares, checkpoint hash, and perturbation settings.
- `audit_epoch070/normal_predictions.jsonl`: per-case outputs matching the
  selected `.pt`; use these for paper figures, not images from the old run.
- `resolved_config.json`, `run_manifest.json`, `split_contract.json`:
  configuration, source commit, and data split.

For the old 0.8715 paper result, keep the original result table only as a
historical metric with the explicit caveat that its exact checkpoint was not
retained. If a deliverable must contain actual weights, use this documented
0.9400 checkpoint or a separately identified retained epoch checkpoint.
The closest retained score in this new run is epoch 45: 0.8765, independently
recomputed from `tested_checkpoints/epoch_045.pt`; its same-weight no-optical
score is 0.4845. This is **not** 0.8715 and must not be renamed as such.
