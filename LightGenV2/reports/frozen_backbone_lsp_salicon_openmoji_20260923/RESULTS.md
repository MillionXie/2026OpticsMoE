# Frozen-backbone baselines for LSP, SALICON, and OpenMoji

Date: 2026-09-23

This report records the first complete frozen-backbone comparison for three
downstream tasks. The backbone parameters are frozen and only the task adapter
and readout head are optimized. All dense image features are normalized to a
14 x 14 spatial grid before entering the task-specific Qwen-compatible head.

## Backbones and readouts

| Baseline | Frozen image representation | Frozen text representation for OpenMoji |
| --- | --- | --- |
| DeepSeek-VL2-Tiny | final SigLIP-So400M global-view 27 x 27 tokens, bilinearly resized to 14 x 14 | final DeepSeek decoder instruction-token hidden states |
| OpenAI CLIP ViT-B/32 | final 7 x 7 ViT patch tokens, bilinearly resized to 14 x 14 | native final CLIP text embedding (one token) |
| YOLO11s + CLIP text | YOLO11s layer-10 C2PSA feature map, resized to 14 x 14 | frozen CLIP text tower; this is a composite baseline rather than a pure YOLO multimodal model |

## Primary results

| Task / metric | Qwen reference | DeepSeek-VL2-Tiny | CLIP ViT-B/32 | YOLO11s (+ CLIP text for OpenMoji) |
| --- | ---: | ---: | ---: | ---: |
| LSP PCK@0.2 | 0.7226 | 0.0259 | 0.0213 | 0.0763 |
| SALICON CC | 0.8748 | **0.9093** | 0.8723 | 0.8887 |
| OpenMoji changed-cell accuracy | 0.8120 | 0.7195 | 0.6285 | 0.7225 |

The OpenMoji values above use the current layered-layout dataset, 5,000 train
and 1,000 test samples, seed 73, 100 epochs, and the same shared 192-dimensional
readout used by the Qwen baseline. They are independently reproduced by loading
the selected checkpoint after training.

## Detailed selected checkpoints

| Task | Baseline | Epoch | Primary metric | Trainable head parameters |
| --- | --- | ---: | ---: | ---: |
| LSP | DeepSeek-VL2-Tiny | 19 | PCK@0.2 = 0.025929 | 1,119,630 |
| LSP | CLIP ViT-B/32 | 32 | PCK@0.2 = 0.021286 | 1,069,710 |
| LSP | YOLO11s | 16 | PCK@0.2 = 0.076286 | 1,036,430 |
| SALICON | DeepSeek-VL2-Tiny | 5 | CC = 0.909313 | 307,172 |
| SALICON | CLIP ViT-B/32 | 35 | CC = 0.872336 | 233,444 |
| SALICON | YOLO11s | 20 | CC = 0.888736 | 184,292 |
| OpenMoji | DeepSeek-VL2-Tiny | 45 | changed-cell = 0.7195 | 850,072 |
| OpenMoji | CLIP ViT-B/32 | 45 | changed-cell = 0.6285 | 628,888 |
| OpenMoji | YOLO11s + frozen CLIP text | 40 | changed-cell = 0.7225 | 579,736 |

Additional OpenMoji metrics:

| Baseline | Edit-grid IoU | Object F1 | Scene exact match |
| --- | ---: | ---: | ---: |
| DeepSeek-VL2-Tiny | 0.7313 | 0.9311 | 0.589 |
| CLIP ViT-B/32 | 0.6215 | 0.9032 | 0.499 |
| YOLO11s + frozen CLIP text | 0.6942 | 0.9113 | 0.572 |

## Reproducibility and audit notes

- Backbone trainable-parameter count is asserted to be zero in every feature
  cache. The reported parameter count is the trainable task adapter/readout only.
- An OpenMoji checkpoint bug was found and fixed: evaluation used EMA weights,
  but the original save order restored raw weights before checkpointing. The
  final runs save the exact EMA weights that produced the selected metric.
  Stored and independently reloaded changed-cell accuracy now match exactly for
  all three models (0.7195, 0.6285, and 0.7225).
- Earlier OpenMoji runs made before this fix, and runs accidentally using the
  parser default seed 42, are superseded and must not be quoted.
- LSP uses the Qwen-compatible deconvolution pose head and reports all 1,000 test
  images / 14,000 joints, but all three results are abnormally low. Treat them as
  diagnostic results until the target heatmap scale and head optimization are
  audited; do not place them in the paper's final main table yet.
- SALICON uses 10,000 train and 5,000 official validation images and the same
  aligned adapter/progressive density decoder. The numbers are suitable for the
  next comparison draft, subject to a repeat-seed confidence interval.
- OpenMoji currently follows the Qwen protocol of selecting the best checkpoint
  on test changed-cell accuracy every five epochs. This is selection-biased.
  Before the final Nature submission, introduce a validation split, select on
  validation only, and evaluate the test split once.

## Server artifacts

Code entry point:

`LightGenV2/common/frozen_backbone_dense_tasks.py`

Final OpenMoji result directories:

- `/DATA/DATA1/guest3/LightGenV2_baseline_runs_20260923/openmoji_deepseek_final_seed73`
- `/DATA/DATA1/guest3/LightGenV2_baseline_runs_20260923/openmoji_clip_final_seed73`
- `/DATA/DATA1/guest3/LightGenV2_baseline_runs_20260923/openmoji_yolo11s_final_seed73`

LSP and SALICON results are under the same root as `lsp_<model>` and
`salicon_<model>`. Feature caches and logs are stored in
`/DATA/DATA1/guest3/LightGenV2_baseline_cache_20260923`.
