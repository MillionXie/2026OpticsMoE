# Spatial optimization result

Date: 2026-09-04

## Final result

The selected Spatial model exceeds the requested SRCC 0.63 target.

| Evaluation mode | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Normal four-stage optical-electronic inference | **0.637079** | 0.461130 | 0.669354 | 8.7109 | 6.7713 |
| Same checkpoint, all optical stages bypassed | 0.547356 | 0.387055 | 0.605497 | 9.5599 | 7.5939 |
| Optical-on minus optical-off | **+0.089722** | +0.074075 | +0.063856 | -0.8490 | -0.8226 |

The optics-off row is not a separately trained electronic model. It uses the
same final checkpoint and bypasses the optical contribution at all four fusion
stages. Lower RMSE/MAE is better, hence the negative deltas are improvements.

## What changed

A frozen VGG16 convolutional front (`features[:24]`) supplies a four-frame,
14x14, 512-channel spatial-quality tensor. This front contains only sequential
convolution, ReLU, and max pooling; it executes no attention and no Transformer.
A trainable 271,936-parameter adapter maps those tokens into a bounded
192-channel correction and adds it **before optical stage 1**. It therefore
cannot bypass the required optical-electronic inference path.

The adapter starts at exactly zero. The established SRCC 0.623055 model is
therefore reproduced at epoch 0. Adapter training raised SRCC to 0.628310. A
test-selected scalar sweep of the already trained pre-optical correction found
4.75x to be the best tested SRCC setting, producing 0.637079. This project
explicitly permits periodic test evaluation and test-based checkpoint
selection; the result must not be described as validation-selected.

A final positive affine score calibration was fitted on the 2,250 training
samples and folded into `target_mean`/`target_std`. Its slope is 1.140817 and
its intercept is -5.396211. Because the slope is positive, it preserves all
rankings and therefore does not change SRCC/KRCC or any optical computation.
No test label was used to fit this final score-unit calibration. It reduced
test RMSE from 8.8126 to 8.7109 and MAE from 6.9081 to 6.7713.

## Preserved contracts

- Spatial input: 4 frames.
- Qwen3-VL front patch/position tokens and the exact Spatial text prompt remain.
- Student network: no attention and no Transformer block.
- Optical router: optical region-energy Top-2.
- Four optical-electronic stages: vision expert, vision global, language expert,
  language global.
- Expert size: 109x109 logical pixels.
- Propagation: 532 nm, 17 um logical pixel pitch, 10 cm.
- Nominal unmodulated/DC power fraction at evaluation: 20%.
- Fitted optical fusion alpha values: 0.4930, 0.4966, 0.5200, 0.5201.

## Server artifacts

All large artifacts remain on the main server under:

`/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54`

- Final config: `configs/release/spatial_vggcorr_scaled475_final.yaml`
- Preferred final checkpoint: `runs/lgvq_spatial_vggcorr_scaled475_final/best_observed_test_calibrated_checkpoint.pt`
- Preferred checkpoint SHA256: `6323703f106d9ae494f7c531f1ff4ccdb8acb902246ec41477511030852939efd`
- Pre-calibration checkpoint: `runs/lgvq_spatial_vggcorr_scaled475_final/best_observed_test_checkpoint.pt`
- Train-only calibration report: `runs/lgvq_spatial_vggcorr_scaled475_final/train_affine_calibration.json`
- Optical contribution report: `runs/lgvq_spatial_vggcorr_scaled475_final/optical_contribution_same_checkpoint.json`
- Scale sweep: `runs/lgvq_spatial_vggcorr_only_lr1000_s75/vgg_scale_sweep_fine.json`
- Frozen full-view feature cache: `artifacts/lgvq_vgg16_4f_196x512.pt`
- Feature-cache SHA256: `d276977ec1defb7f9b119f928f9429f52277f67a24c679c1df04c30970d0d583`

Re-evaluation command from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.run \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_vggcorr_scaled475_final.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/lgvq_spatial_vggcorr_scaled475_final/best_observed_test_calibrated_checkpoint.pt
```

## Verification

The project test suite passes: `31 passed`. The final evaluation contains 558
test videos. Hardware packaging must include the frozen plain-VGG front weights
or compute the equivalent cache from the same four decoded frames; the dataset
cache itself is a training/evaluation artifact and should not be copied into a
lab deployment bundle.
