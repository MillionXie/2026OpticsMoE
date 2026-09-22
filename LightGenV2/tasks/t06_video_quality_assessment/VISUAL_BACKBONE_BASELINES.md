# Frozen CLIP and YOLO11 LGVQ baselines

These baselines use the same fixed LGVQ split and four-frame sampling contract as
the Qwen and DeepSeek quality-token baselines.  Each pretrained backbone is fully
frozen.  Four ordered 512-dimensional frame features are concatenated into a
2048-dimensional video feature, and only five bias-free quality rows are trained
(10,240 parameters, matching the Qwen3-VL-2B readout budget).

## Architectures

- `clip_vit_b32_lgvq_4f_r224`: OpenAI CLIP ViT-B/32 final image embeddings.  The
  five rows are initialized from native CLIP text embeddings for `Bad`, `Poor`,
  `Fair`, `Good`, and `Excellent`, repeated across the four frame positions.
- `yolo11s_lgvq_4f_r640`: Ultralytics YOLO11s layer-10 C2PSA feature maps, global
  average pooled per frame.  Since YOLO has no text tower, the five rows use
  deterministic Xavier-uniform initialization with seed 42.

The shared feature cache contains both temporal and spatial MOS targets, so each
backbone is evaluated once and the two independent quality heads are then trained
from the same frozen features.

## Environment

The reproducible server environment uses PyTorch 2.6.0+cu124 and pins
`ultralytics==8.4.159`.  The CLIP implementation is the official `openai/CLIP`
package.  Do not vendor either pretrained checkpoint into Git.  YOLO11 software
and weights are AGPL-3.0; keep the YOLO experiment code and release obligations
explicit when publishing the repository.

## Commands

```bash
PYTHON=/absolute/path/to/lgvq-visual-baselines/bin/python
MANIFEST=/absolute/path/lgvq_train2250_test558.csv
CLIP_MODEL=/absolute/path/ViT-B-32.pt
YOLO_MODEL=/absolute/path/yolo11s.pt

CUDA_VISIBLE_DEVICES=<free-gpu-uuid> "$PYTHON" -m \
  LightGenV2.tasks.t06_video_quality_assessment.visual_backbone_quality \
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/baselines/clip_vit_b32_lgvq_4f_r224.yaml \
  --phase all --model "$CLIP_MODEL" --manifest "$MANIFEST"

CUDA_VISIBLE_DEVICES=<free-gpu-uuid> "$PYTHON" -m \
  LightGenV2.tasks.t06_video_quality_assessment.visual_backbone_quality \
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/baselines/yolo11s_lgvq_4f_r640.yaml \
  --phase all --model "$YOLO_MODEL" --manifest "$MANIFEST"
```

Use physical GPU UUIDs rather than numeric CUDA indices on the shared server,
because CUDA enumeration and `nvidia-smi` indices are not guaranteed to match.
The extractor writes atomic 64-video shards and safely resumes completed shards.
