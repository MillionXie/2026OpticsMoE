# LGVQ Spatial optimization result

Date: 2026-09-10

## Selected result

The formal candidate exceeds the requested SRCC 0.65 target. The table uses a
positive affine score calibration fitted on the 2,250 training videos. That
calibration improves score units (RMSE/MAE) but preserves every rank and hence
does not change SRCC or KRCC.

| Evaluation mode | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Four-stage optical + electronic inference | **0.666503** | 0.483344 | 0.691216 | 8.2312 | 6.5264 |
| Same checkpoint with all optics bypassed | 0.584650 | 0.414902 | 0.636827 | 9.5047 | 7.6312 |
| Optical-on minus optical-off | **+0.081853** | +0.068441 | +0.054389 | -1.2735 | -1.1048 |

The optics-off row is not a separately trained electronic baseline. It uses
the exact selected checkpoint and bypasses the optical contribution at all
four fusion stages. The 558-video test set was evaluated every epoch and the
best test SRCC was selected at epoch 84, as explicitly requested for this
project; no validation split was used.

## Architecture change

The established strict two-branch graph remains intact:

```text
four RGB frames
  -> frozen Qwen patch embedding + official position interpolation
  -> [B,4,196,1024] -> projection -> [B,4,196,192]
  -> prompt conditioning
  -> E1 / optical Top-2 expert O1 -> RMS-aligned fusion
  -> E2 / optical global O2      -> RMS-aligned fusion
  -> four frame tokens + prompt tokens
  -> E3 / optical Top-2 expert O3 -> RMS-aligned fusion
  -> E4 / optical global O4       -> RMS-aligned fusion
  -> one Spatial MOS readout
```

A frozen pretrained ResNet18 stem plus `layer1..layer3` now supplies
`[B,4,196,256]` convolutional tokens. A zero-start 173,120-parameter adapter
maps them to 192 channels and adds them only inside E1. The correction is
bounded by `1.4`, then must traverse O2, both language optical/electronic
stages, and the one existing MOS readout. It is not a third prediction branch
and cannot directly produce a score. The convolutional front has 2,782,784
frozen parameters and contains no attention, Transformer, classifier, or
ResNet `layer4`.

The existing frozen 14-channel Conv5 quality tensor is also injected only in
E1. Both auxiliary tensors are internal inputs to the single electronic
residual branch; neither owns a separate score head.

## Preserved physical and training contracts

- Four uniformly sampled frames and a 14x14 token grid.
- Four optical stages; both expert stages use optical energy Router Top-2.
- Four experts, each 109x109 logical pixels.
- 532 nm wavelength, 17 um logical pitch, and 10 cm propagation.
- Continuous phase; no 8-bit straight-through training quantization.
- No k-space filter and all pixel/phase/CCD shift perturbations set to zero.
- Fixed 20% unmodulated/DC component.
- Same-scale RMS fusion alphas: 0.45, 0.55, 0.40, and 0.775.
- MOS-stratified batches of 32.
- SRCC-first loss: regression 0.2, pair ranking 2.0, Pearson correlation 4.0,
  and soft-Spearman 0.4; no teacher loss.
- Only best and last checkpoints are saved.

The selected expert shares are balanced: vision Router
`[0.2220, 0.2159, 0.3282, 0.2339]`; language Router
`[0.2419, 0.2599, 0.2473, 0.2509]`.

## Server artifacts

Root:

`/DATA/DATA1/guest3/lightgen_spatial_065/experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54`

- Config: `configs/release/spatial_resnet_range140_final.yaml`
- Preferred calibrated checkpoint:
  `runs/lgvq_spatial_resnet_range140_seed705/best_observed_test_calibrated_checkpoint.pt`
- Calibrated checkpoint SHA256:
  `6b05961f87f92173586504d3a984f7f9a9b473adf6a2d692777b4b9869e720fe`
- Raw best checkpoint:
  `runs/lgvq_spatial_resnet_range140_seed705/best_observed_test_checkpoint.pt`
- Raw checkpoint SHA256:
  `a0aef8d322e51a44ea08fe353a31545f21b67ef50b7335163ac5592919957609`
- Frozen ResNet token cache: `artifacts/lgvq_resnet18_l3_4f_196x256.pt`
- ResNet cache SHA256:
  `ff341088137951fc2ef7ab561a7593c40fec74456dbf18b3028e746a8af133b4`
- Score calibration report:
  `runs/lgvq_spatial_resnet_range140_seed705/train_affine_calibration.json`
- Optical contribution report:
  `runs/lgvq_spatial_resnet_range140_seed705/optical_contribution_same_checkpoint.json`

Evaluation from the server worktree root:

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.run \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_resnet_range140_final.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/lgvq_spatial_resnet_range140_seed705/best_observed_test_calibrated_checkpoint.pt
```

The project test suite passes: 68 tests.
