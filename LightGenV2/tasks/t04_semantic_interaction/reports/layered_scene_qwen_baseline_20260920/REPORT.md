# Layered OpenMoji: electronics and frozen-Qwen baseline audit

Date: 2026-09-20. Seed: 73. Split: 5,000 train / 1,000 deterministic,
disjoint test examples, with no validation split. Checkpoint selection uses the
highest test changed-cell accuracy measured at epoch 1, every five epochs, and
the final epoch, as required by the project protocol.

## Fair comparison contract

The optical model and Qwen baseline use the same layered SVG scenes, text
instructions, targets, training objective, standard 192-wide shared readout,
and test set. The Qwen baseline runs all 24 native vision blocks and all 28
native language blocks of frozen Qwen3-VL-2B-Instruct. Its cached features are
the last pre-merger visual tokens `[B,196,1024]` and last-layer instruction
tokens `[B,L,2048]`. Only two dimension adapters and the common readout train.

## Test results

| Model | Best epoch | Changed-cell acc. | Edit-grid IoU | Foreground acc. | Object F1 | Exact scene |
|---|---:|---:|---:|---:|---:|---:|
| Optical Router Top-2 hybrid | 60 | **0.9315** | **0.9455** | **0.9774** | **0.9811** | **0.879** |
| Frozen complete Qwen + same readout | 30 | 0.8120 | 0.7438 | 0.9353 | 0.9358 | 0.634 |

The Qwen test trajectory peaks at epoch 30. Changed-cell accuracy at epochs
20/25/30/35/40 is 0.7505/0.7990/0.8120/0.8085/0.8060; by epoch 100 it has
fallen to approximately 0.76 while the training loss is almost zero. Therefore
the last checkpoint is not a valid substitute for the selected checkpoint.

## Trainable parameter distribution of the optical model

Total trainable parameters: 3,139,437. Frozen Qwen lookup/patch frontend:
3,933,184 additional frozen parameters.

| Group | Parameters | Share of trainable |
|---|---:|---:|
| Optical phase masks | 958,728 | 30.54% |
| Optical interfaces/readout adapters | 260,160 | 8.29% |
| Language electronic residual path | 767,814 | 24.46% |
| Vision electronic residual path | 572,742 | 18.24% |
| Text-to-vision conditioning | 198,017 | 6.31% |
| Common output readout | 381,976 | 12.17% |

Thus optical-related trainable parameters are 1,218,888 (38.83%) and
electronic trainable parameters are 1,920,549 (61.17%). The two electronic
cores are width 192, depth two, expansion two: causal depthwise Conv1d plus
residual MLP for language, and spatial depthwise Conv2d plus residual MLP for
vision. There is no attention or Transformer in the optical model.

The selected checkpoint's scale-matched optical coefficients are 0.5027 and
0.4986 in language, and 0.4786 and 0.4910 in vision. Hence each optical stage
contributes about one half after branch RMS normalization; it is not hidden by
an uncontrolled electronic feature scale.

The frozen-Qwen baseline has 972,952 trainable parameters: 590,976 in the two
dimension adapters and 381,976 in the identical common readout. The frozen
Qwen checkpoint itself contains 2,127,532,032 parameters.

## Same-checkpoint counterfactuals

Removing optics without retraining lowers changed-cell accuracy from 0.9315 to
0.4220 (minus 50.95 percentage points). Removing the electronic branches
without retraining lowers it to 0.0045. These tests prove that both trained
branches are used, but the optical-only number must not be interpreted as the
best achievable retrained optical-only model: branch co-adaptation makes a
same-checkpoint removal deliberately harsh.

## Implication for reducing electronics

Do not remove the complete electronic path. The safest next ablation is to
retain the input adapters, token-mixing depthwise convolutions, cross-modal
conditioning and common readout, while reducing only the residual MLP
expansion from 2.0 to 1.0. A second, separately trained ablation can remove the
second electronic residual MLP in each modality. Both must be retrained and
reported with the same 5,000/1,000 split and the same common readout.

## Artifacts on the training server

- Optical run: `LightGenV2/tasks/t04_semantic_interaction/runs/simulation/layered_scene_focus_changed_iou_s73_e100_20260920`
- Qwen run: `LightGenV2/tasks/t04_semantic_interaction/runs/simulation/layered_scene_qwen_shared_s73_e100_20260920`
- Reusable Qwen cache: `LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_layered_anchor6_svg_v3/qwen_shared_head_v1`

The cache is about 2.6 GB and records the exact model/data contract. It is an
offline training acceleration artifact, not a shortcut in the stated online
Qwen inference graph.
