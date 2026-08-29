# CCD noise robustness commands

Run every command from the repository root.  All four arms strictly resume the
same audited warmstart5 EMA checkpoint (`SHA-256 6a27f54d...e55d`), reset Adam,
run 8 short epochs x 12 optimizer steps, and never inspect the test split while
selecting a checkpoint.

## Parallel training

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1 --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/noise_mild.yaml --phase train

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1 --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/noise_medium.yaml --phase train

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1 --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/noise_strong.yaml --phase train

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1 --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1/configs/release/noise_medium_extreme_phase.yaml --phase train
```

## What differs

| arm | truncated CCD noise, relative to clean frame mean | phase LR | phase-DC weight |
|---|---|---:|---:|
| mild | N(+0.01, 0.01), truncated [-0.01, +0.03] | 0.03 | 0.010 |
| medium | N(+0.03, 0.025), truncated [-0.02, +0.08] | 0.03 | 0.010 |
| strong | N(+0.06, 0.05), truncated [-0.04, +0.16] | 0.03 | 0.015 |
| medium/extreme phase | same as medium | 0.08 | 0.030 |

The perturbation is training-only and is applied independently at each of the
four CCD boundaries, before the same frame-mean/clip/log normalization used by
the hardware path. Existing independent input/mask/CCD shifts, gain variation,
phase dropout, k-space filter, router losses, and CCD operating-point loss stay
enabled. The fusion coefficient is bounded below by 1%; it is a residual
coefficient, not a measured optical-energy percentage.

## Evaluation after training

Do not choose the best arm from repeated test probing. First compare training
loss, phase motion/saturation and numerical stability. Then run one explicit
test evaluation for each predeclared `ema_best_train_loss_checkpoint.pt`.
Replace `<CONFIG>` and `<RUN>` with one row above:

```text
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1 --config <CONFIG> --phase evaluate --checkpoint <RUN>/ema_best_train_loss_checkpoint.pt
```

Key diagnostics are in each `runs/.../train_log.csv`: `phase_delta_run_rms_rad`,
`phase_physical_std_rad`, `phase_sigmoid_saturation_fraction_abs_raw_gt_4`,
`phase_dc_rho_mean`, router coverage and total loss. A mask moving strongly but
saturating toward 0/2pi is not considered a successful result.

