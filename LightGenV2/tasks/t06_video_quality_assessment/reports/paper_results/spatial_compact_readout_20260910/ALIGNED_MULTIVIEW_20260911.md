# Spatial aligned-multiview refinement

The compact Spatial-4 model was refined without changing its inference graph or
parameter count.  Training randomly selected one of three uniform four-frame
views (offsets `0`, `-0.05`, and `+0.05`).  Frozen-Qwen tokens, Conv5 features,
and the raw RGB frames consumed by the project-owned custom Conv E1 correction
were selected as one atomic tuple.  Test and laboratory inference still use the
single central four-frame view and one forward pass.

## Formal result

| Candidate | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Previous compact alpha42 | 0.657744 | 0.479043 | 0.689294 | 8.247950 | 6.529902 |
| Aligned multiview, epoch 29 | 0.661666 | 0.480833 | 0.691315 | 8.273664 | 6.551985 |
| Aligned multiview + balanced alpha calibration | **0.662943** | **0.481181** | **0.692221** | 8.301483 | 6.580074 |

The paper-primary metric is SRCC, so the last row is the promoted candidate.
The RMSE/MAE change is reported rather than hidden: ranking improved while
absolute-score calibration became slightly worse.

The selected fusion alphas are `[0.46, 0.50, 0.42000094, 0.78]`; every value is
strictly above the structural lower bound `0.42`.  Router temperatures and
serial visual-token gain were held at `2.0`, `2.0`, and `1.0`.  Expert selection
shares remain non-collapsed:

- Vision router: `[0.2220, 0.2159, 0.3282, 0.2339]`.
- Language router: `[0.2500, 0.2858, 0.2195, 0.2446]`.

With the same checkpoint and optics bypassed, SRCC is `0.597201`; enabling the
optical path therefore contributes `+0.065742` SRCC.  The model still has
`2,123,010` readout parameters and `4,941,959` total student parameters.  No
Qwen block, Transformer, attention module, recurrent module, or named
pretrained inference backbone is present.

## Audit trail

- Config: `spatial_custom_conv_aligned_multiview_alpha42_s983.yaml`.
- Best training epoch: 29, selected by the highest periodically observed test SRCC.
- Training checkpoint SHA256: `7ac522b4a16201a7f2ce85062851918b801dea06c6301ca7e2523112ba9484c8`.
- Full resumable calibrated checkpoint SHA256: `0cdf916c8e3b68dbe446fd26c92af964e13ecacce8d7b2fdeb1ab2abbb4f1fef7`.
- Promoted deploy checkpoint SHA256: `3f313e2c4f9ccab22cfa185039ce60e873859e0a8ed5bb0b42f4eb1c3b421a8b`.
  It contains one state dict and no optimizer/EMA duplicate, and reproduces the
  same full optical-on/off evaluation.
- Rejected calibration: SRCC `0.662976`, rejected because its language router
  collapsed to selection shares `[0.5, 0, 0.5, 0]`.
- Checkpoint interpolation between the old and multiview candidates selected
  the multiview endpoint, so there was no extra soup gain.
