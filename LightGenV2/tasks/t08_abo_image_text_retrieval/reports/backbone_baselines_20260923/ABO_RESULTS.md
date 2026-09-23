# ABO backbone baseline results (2026-09-23)

All values use the unchanged ABO candidate sets, splits, and ranking metrics.
Every pretrained backbone is frozen. The Qwen and LightGenV2 rows are existing
references and were not rerun.

## Recommended paper table: frozen backbones

This is the primary comparison because Qwen, DeepSeek, and CLIP all have zero
task-trained parameters and never see the 4,800 easy100 training images.

| Method | Image to text R@1 | Image to image R@1 | Text to image Hit@1 | Task-trained parameters |
|---|---:|---:|---:|---:|
| LightGenV2 (Ours) | **0.8058** | 0.8313 | **0.8800** | method-specific |
| Qwen3-VL-Embedding-2B, frozen | 0.7358 | 0.8513 | 0.8200 | 0 |
| DeepSeek-VL2-Tiny, frozen | 0.1379 | **0.8888** | 0.0800 | 0 |
| OpenAI CLIP ViT-B/32, frozen | 0.5700 | 0.8713 | 0.6400 | 0 |
| YOLO11s, frozen | N/A | 0.8713 | N/A | 0 |

YOLO11s is visual-only, so it is excluded from both cross-modal tasks. Its old
YOLO+CLIP composite is not a YOLO-only baseline and must not appear in the main
table.

## Secondary diagnostic: light supervised alignment

The following rows are not zero-shot. For each retrieval direction, all
pretrained weights remain frozen and one image-side rank-64 residual calibration
is trained for 50 fixed epochs at learning rate `1e-4`, residual scale `0.4`,
drift weight `1`, and seed `42`. Image-to-text uses image-to-title CE;
text-to-image uses title-to-image-prototype CE. There is no test-time checkpoint
selection. Parameter counts are per task/direction, not summed across two
separate experiments.

| Frozen backbone + rank-64 calibration | Image to text R@1 | Text to image Hit@1 | Trainable parameters per task |
|---|---:|---:|---:|
| DeepSeek-VL2-Tiny | 0.5017 | 0.0500 | 163,840 |
| OpenAI CLIP ViT-B/32 | 0.8413 | 0.9300 | 65,536 |

These results should be labelled `task-adapted`, not mixed silently with the
frozen rows. DeepSeek's decoder hidden states improve for image-to-title
classification but remain unsuitable for reverse dense retrieval under this
small image-side calibration. CLIP already supplies an aligned contrastive
space and becomes stronger than LightGenV2 after supervised fitting.

## Why score-targeted tuning was rejected

ABO easy100 trains and tests on different views of the same 100 product IDs.
Consequently, an expressive supervised head can encode the closed catalogue
rather than learn unseen-product generalization. A five-epoch dual 64-D readout
already reached DeepSeek 0.9604/1.0000 and CLIP 0.9067/0.9600; a full square
linear map trained for 50 epochs reached DeepSeek 0.9488/0.9100 and CLIP
0.9767/0.9800. Those runs are retained only for audit and are excluded from the
paper table. Continuing to tune until a baseline lands just below LightGenV2
would be test-set score targeting.

## Protocols

- Image to text: 2,400 easy100 test images rank all 100 official titles.
- Image to image: 800 ABO-200 queries rank 1,600 enrolled images; each query has
  eight relevant candidates.
- Text to image: 100 official easy100 titles rank all 2,400 test images; each
  title has 24 relevant images. The paper's `R@1` entry is Hit@1 under the
  original script's terminology.
- DeepSeek uses the final decoder layer at the last valid prompt token. Titles
  are encoded as documents for image-to-text and as queries for text-to-image.
- CLIP uses native image/text embeddings. YOLO11s uses global-average C2PSA
  layer-10 features for image-to-image only.

## Parameter accounting

| Path | Frozen parameters loaded/used | Trainable task parameters |
|---|---:|---:|
| Qwen3-VL-Embedding-2B | 2,127,532,032 | 0 |
| DeepSeek-VL2-Tiny, frozen | 3,370,501,440 total (about 1B active/token) | 0 |
| CLIP ViT-B/32, frozen | 151,277,313 | 0 |
| YOLO11s image-to-image | 9,458,752 | 0 |
| DeepSeek rank-64 calibration | 3,370,501,440 | 163,840 per task |
| CLIP rank-64 calibration | 151,277,313 | 65,536 per task |

These runs measure retrieval performance only. Formal latency and energy require
the separate fixed-hardware timing boundary.

## Reproducibility

- Runner: `abo_backbone_baselines.py`.
- Unit tests: 5 passed.
- Branch: `codex/abo-backbone-baselines`.
- Final runner SHA256:
  `e3b98c42ec9fbe0575c31a12631b7767794041c7637a14578c82e8c92a457961`.
- DeepSeek report SHA256:
  `bffa31c5d551f7bf0eb26e312812adb349b8b1ffb011fb49d93c550897f2ef6d`.
- CLIP report SHA256:
  `dc09bcb0dd733a2e3085265f1b90b43299ec0e3f82bd125ba887e295f58a2ea3`.
- Resumable features, checkpoints, histories, and predictions remain under the
  ignored server directories `runs/backbone_baselines_20260923/` and
  `runs/backbone_baselines_adapted_20260923/`.
