# ABO backbone baseline results (2026-09-23)

All values below use the unchanged Qwen candidate sets, splits and ranking
metrics. Every pretrained backbone is frozen. The Qwen row is the existing
published reference and was not rerun.

| Baseline | Image to text R@1 | Image to image R@1 | Text to image Hit@1 | Fitted parameters |
|---|---:|---:|---:|---:|
| Qwen3-VL-Embedding-2B | 0.7358 | 0.8513 | 0.8200 | 0 |
| DeepSeek-VL2-Tiny | 0.1379 | **0.8888** | 0.0800 | 0 |
| OpenAI CLIP ViT-B/32 | 0.5700 | 0.8713 | 0.6400 | 0 |
| YOLO11s / YOLO11s + CLIP text | 0.9867* | 0.8713 | 0.9900* | 0.262M* |

`*` YOLO11s is visual-only. Its cross-modal entries are a fitted composite,
not zero-shot YOLO: frozen YOLO11s image features and a frozen CLIP text tower
are joined by one bias-free 512-by-512 matrix fitted on the 4,800-image easy100
training split. The image-to-image entry is raw frozen YOLO11s with no adapter.
The fitted composite values therefore must not be compared to the three
zero-shot rows without this qualifier.

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
| YOLO11s + CLIP text cross-modal | 72,886,848 (9,458,752 + 63,428,096) | 262,144 |

These runs measure retrieval performance only. They do not claim formal latency
or energy numbers; those require the separate fixed-hardware timing boundary.

## Reproducibility

- Runner: `abo_backbone_baselines.py`
- Unit tests: 3 passed.
- Branch: `codex/abo-backbone-baselines`
- Implementation commits: `1d80e64f`, `19cf5c74`, `c47a0413`.
- Exact report identities and raw values are pinned in `summary.json` beside
  this file. Resumable features and predictions remain under the ignored run
  directory `runs/backbone_baselines_20260923/` on the server.
