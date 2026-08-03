# Embedding optical phase training audit

This note records the phase-training issue shared by the recent
`qwen3_vl_embedding_2b_*` experiments and the corrective settings that are now
available.  It does not change old checkpoints or reinterpret historical
results.

## Root causes

The nearly flat masks were not caused by `sigmoid(raw_phase) * 2*pi` being
numerically unstable.  That parameterization is bounded and has its largest
derivative at `raw_phase=0`.  The practical causes were a combination of:

1. phase parameters sharing a small learning rate with electronic adapters;
2. electronic heads/adapters and, in some experiments, a residual path being
   able to reduce task loss without requiring strong phase motion;
3. very few optimizer updates per epoch on small datasets;
4. exactly uniform phase initialization;
5. no loss or diagnostic for coherent zero-order transmission;
6. no configured spatial-frequency aperture in the newer embedding runs.

The coherent DC regularizer is evaluated independently on every physical phase
plane:

```text
rho_m = abs(mean(exp(1j * phase_m)))
L_phase_dc = mean_m(rho_m ** 2)
```

It is important that an exactly uniform mask is a stationary maximum of this
loss: its value is one, but its gradient is zero.  Configurations that enable
the DC loss therefore use `phase.init: small_normal` rather than an exact zero
raw phase.  The physical phase remains bounded by the existing sigmoid
parameterization.

The angular-spectrum k-space constraint and the DC regularizer solve different
problems.  The former rejects unsupported high-angle spatial frequencies; it
does not remove the zero-frequency component.  The latter discourages a phase
mask with a strong coherent zero order.  At 532 nm and 8 um sampling,
`theta_max_deg: 2.0` retains about 84% of the sampled frequency grid and is used
as a moderate default.  A one-degree cutoff would retain only about 22% and is
too aggressive as a general default.

## Audited experiments

| Experiment | Finding | Correction |
| --- | --- | --- |
| Grocery10 retrieval | Only about 11 natural PK batches per epoch; weak phase exposure; zero-order not constrained | independent phase LR, phase-only epochs, 100 fresh augmented PK steps/epoch, phase diagnostics, DC loss, k-space aperture, residual disabled in the engaged config |
| ABO retrieval | phase LR was much smaller than the optical task required | split phase optimizer group, DC loss and diagnostics, small-normal phase initialization, k-space aperture |
| FSS-1000 saliency | phase could share the general student LR; no zero-order objective | explicit phase group, DC loss, small-normal initialization, k-space aperture |
| COCO/DUTS optical pretraining | phase and electronic optical parameters shared the same low LR | separate COCO/DUTS phase LRs, DC loss, router regularization during joint DUTS training, k-space aperture |
| FSS COCO/DUTS fine-tuning | pretrained phases used the same fine-tuning LR as other backbone parameters | separate lower fine-tuning phase LR and DC loss |
| ISIC segmentation | scratch/pretrained phases were bundled into the generic optical LR | separate scratch/pretrained phase LR and DC loss |
| LSP pose | phase LR was conservative and zero order unmeasured | stronger phase group, DC loss, small-normal initialization, k-space aperture |
| SALICON saliency | same phase-ownership issue as the segmentation experiments | stronger phase group, DC loss, small-normal initialization, k-space aperture |
| BDD100K/Bench2Drive | optimizer groups existed, but phase started uniformly and zero order was unmeasured | DC loss in pretraining/BC, small-normal initialization, k-space aperture |

## Grocery10 phase-engagement run

Use `configs/grocery10_phase_engaged.yaml` for a fresh ablation.  Its output is
separate from the historical best-reproduction chain.  Training history and
checkpoints record phase gradient/motion statistics and:

```text
phase_dc_loss
phase_dc_weighted_loss
phase_dc_current_loss
phase_dc_rho_mean
phase_dc_rho_max
phase_dc_plane_count
```

The DC term improves physical mask diversity; it is not guaranteed to improve
retrieval accuracy.  For a controlled paper ablation, compare identical fresh
runs with `lambda_phase_dc` equal to `0`, `1`, and `5` while keeping the
k-space setting fixed.
