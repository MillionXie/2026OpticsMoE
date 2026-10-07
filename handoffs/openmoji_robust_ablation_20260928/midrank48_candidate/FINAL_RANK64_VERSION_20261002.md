# OpenMoji rank-64 final candidate — 2026-10-02

## Verified physical result

Real six-stage TEST CCD replay, serialized checkpoint strict reload, default inference gate .5:

| Version | Changed-cell accuracy | Preserved-cell accuracy | Entire scene exact |
| --- | ---: | ---: | ---: |
| G2 no-trick, original PT | 61.85% | See G2 report | See G2 report |
| G5 robust, original PT | 69.10% | 98.7396% | 47.90% |
| G5 original decoder, TRAIN2000, epoch115 | 91.80% | 98.4511% | 60.50% |
| Same decoder, existing edit bias calibrated, final candidate | **93.05%** | **98.0456%** | **56.90%** |

Original G1/G2 normal ideal simulation on the same bench CPU path is93.90%. Final physical candidate is0.85 percentage points lower, or0.9052% relative: within the requested1% comparison. This compares the original G1 reference to an adapted physical candidate, **not** an assertion that the adapted checkpoint's own normal simulation is93.90%. The completed separate CPU normal-simulation audit of the **same final calibrated PT** is92.95%, with preserved cells98.6488% and scene-exact64.50%; its physical result93.05% is0.10 percentage points above its own ideal simulation. The uncalibrated adapted PT's normal simulation is89.80%, physical91.80%. Original G5 normal simulation is92.70%. These different references and weights are not conflated.

All scores selected using the user-authorized TEST development protocol; they are not independent generalization estimates. TEST1000 is never used for gradients. Do not claim this guarantees performance on another optical bench.

Calibration improves changed cells but lowers scene-exact60.5→56.9% and preserved98.4511→98.0456%. Both checkpoints are retained.

## Checkpoints, local SHA verified

Final candidate: `rank64_train2000_bias_calibrated_best_20261002.pt`

SHA256: `1fa31ec7b30a554280d9115b54f580d40b9c754f805db4ec9714ec28d22af41f`

Uncalibrated scene-exact alternative: `rank64_train2000_decoder_best_20261002.pt`

SHA256: `018470e4e2a21cf572ac8faf5f9884af810ce5e53321aa3fa8c70e6deb26a825`

Bench originals:

- `E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Rank64_SHS_20261002/runs/g5_train2000_bias_calibration/best_full.pt`
- `E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Rank64_SHS_20261002/runs/g5_decoder_train2000_testselected/best.pt`
- Same original-decoder run retains `last.pt`, history, caches and per-sample output.

## Architecture and adaptation

Rank64 was fixed before original simulation training. Shared electronic head271,384 parameters; only its existing final decoder30,162 parameters trained during real-CCD adaptation. No new layer, third branch or extra model. All pre-decoder modules, optical phases, routing and alpha protected by unchanged hash `612c36eadc64be8b43c8e04d150036ffc7096cae3b8eb77b18f27c7817fbfb1b`.

TRAIN2000 source-disjoint real data: initial1000 plus additional1000, each six layers6000CCD. Original TEST1000 six layers6000CCD. TRAIN supplies gradients; TEST every5epoch selects checkpoint (e115); calibration changes only existing `shared_readout.decoder.edit_head.bias`, equivalent to threshold.2, serialized so default.5 inference reproduces93.05%. TRAIN and TEST preservation both constrained>=.98 in calibration selection.

Hardware contract:17µm input,10cm propagation, zero-preserving bounded amplitude `tanh(abs/.5)`, BMP `round(255*a)` without per-batch peak scaling;2000µs/GainX4/wait240, established ROI/directions, CPU hardware model, whole-layer persistent device acquisition.

## Source, data and evidence

Bench isolated project: `E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Rank64_SHS_20261002`.

- Actual source: `source/LightGenV2/tasks/t04_openmoji_robust_ablation`, rank64 modeling support in source semantic/model modules.
- True TEST CCD: `runs/g5_full1000`; true TRAIN CCD: `runs/g5_train1000` and `runs/g5_extra_train1000`.
- Source TRAIN manifests/data: `data_train_adapt1000` and `data_train_extra1000`; original TEST/frontend/assets in sibling `OpenMoji_Lab_SHS_8um`.
- Local sealed reports: `rank64_train2000_decoder_report_20261002.json`, `rank64_train2000_bias_calibration_report_20261002.json`, `rank64_train2000_final_test_samples_20261002.json`, `rank64_train2000_adapted_normal_simulation_report_20261002.json`.

Future inference must strictly load the final full checkpoint into the matching rank64 architecture; a capture entry that still locks original G5 PT will not automatically score with the adapted decoder. Optical masks are unchanged, so existing same-weight upper-boundary CCD can be replayed with the final head, but original G2/G5 ablation entry must not be silently overwritten.

ABO final checkpoint remains untouched. External upload is paused. ThirdTRAIN1000 was only prepared; do not capture it or continue training after this achieved comparison without new evidence/authorization. Normal-simulation audit and checkpoint backups are complete. At06:13 Beijing all relevant tasks wereReady, no Python job remained on the bench, camera/SLM SDKs released, GPU memory324MiB desktop baseline. Any further monitoring is only handoff/file integrity checking, not additional TEST-driven optimization.
