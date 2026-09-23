# ABO backbone baseline results (2026-09-23)

All values below use the unchanged Qwen candidate sets, splits and ranking
metrics. Every pretrained backbone is frozen. The Qwen row is the existing
published reference and was not rerun.

| Baseline | Image to text R@1 | Image to image R@1 | Text to image Hit@1 | Fitted parameters |
|---|---:|---:|---:|---:|
| Qwen3-VL-Embedding-2B | 0.7358 | 0.8513 | 0.8200 | 0 |
| DeepSeek-VL2-Tiny, zero-shot | 0.1379 | **0.8888** | 0.0800 | 0 |
| DeepSeek-VL2-Tiny, weak alignment | 0.1458 | - | 0.0800 | 10,240 |
| OpenAI CLIP ViT-B/32, zero-shot | 0.5700 | 0.8713 | 0.6400 | 0 |
| OpenAI CLIP ViT-B/32, weak alignment | 0.5708 | - | 0.6500 | 4,096 |
| YOLO11s | N/A | 0.8713 | N/A | 0 |

YOLO11s is visual-only, so its cross-modal composite has been removed from the
comparison. The weak alignment rows freeze every pretrained parameter and train
only a rank-4 residual image adapter for five fixed epochs at learning rate
1e-4. The residual scale is 0.1 and an embedding-drift penalty of 10 is applied.
There is no test-set epoch selection.

For audit only, a full square linear map trained for 50 epochs produced very
high scores (DeepSeek 0.9488/0.9100 and CLIP 0.9767/0.9800). Those runs contain
1.638M and 262K fitted parameters respectively and are deliberately excluded:
they are task-trained retrieval systems rather than light calibration baselines.

## Protocols

- Image to text is 2,400 easy100 test images ranking all 100 official titles.
- Image to image is 800 ABO-200 queries ranking 1,600 enrolled images; each
  query has eight relevant candidates.
- Text to image is 100 official easy100 titles ranking all 2,400 test images;
  each title has 24 relevant images. The paper's `R@1` entry is Hit@1 under the
  original script's terminology.
- DeepSeek uses the final decoder layer at the last valid prompt token. The
  image-to-text titles are documents, the text-to-image titles are queries, and
  image-to-image uses the exact Qwen category-aware visual-similarity prompt.
- CLIP uses native image/text embeddings. YOLO11s uses global-average C2PSA
  layer-10 features. No pretrained parameter is trainable in any run.

## Parameter accounting

| Baseline path | Frozen parameters loaded/used | Trainable task parameters |
|---|---:|---:|
| DeepSeek-VL2-Tiny | 3,370,501,440 total (about 1B active per token) | 0 |
| CLIP ViT-B/32 | 151,277,313 | 0 |
| YOLO11s image to image | 9,458,752 | 0 |
| DeepSeek weak alignment | 3,370,501,440 | 10,240 |
| CLIP weak alignment | 151,277,313 | 4,096 |

These runs measure retrieval performance only. They do not claim formal latency
or energy numbers; those require the separate fixed-hardware timing boundary.

## Reproducibility

- Runner: `abo_backbone_baselines.py`
- Unit tests: 3 passed.
- Branch: `codex/abo-backbone-baselines`
- Implementation commits: `1d80e64f`, `19cf5c74`, `c47a0413`, `b343843d`.
- Exact report identities and raw values are pinned in `summary.json` beside
  this file. Resumable features and predictions remain under the ignored run
  directories `runs/backbone_baselines_20260923/` and
  `runs/backbone_baselines_adapted_20260923/` on the server.
