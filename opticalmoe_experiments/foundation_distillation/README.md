# Foundation Image-Feature Distillation

This experiment trains a 9-expert fair134 AS global-router OpticalMoE against a frozen CLIP image encoder. It is isolated from the legacy `opticalmoe/` code and reuses the validated propagation modules in `opticalmoe_experiments/common/`.

## Scope

- Datasets: CIFAR10 grayscale and Imagenette grayscale.
- Teacher: frozen CLIP image encoder (`ViT-B/32`) only.
- No CLIP text encoder, text features, zero-shot logits, or DINOv2 backend.
- Both teacher and student receive the same grayscale information. The student sees one optical-amplitude channel at 134x134. The teacher sees that exact grayscale tensor resized to 224x224 and replicated to three channels.
- Distillation target: a 256-dimensional feature pooled from detector-plane intensity, never an intermediate complex field.
- The electronic classifier is a small MLP. A separate training-only projector maps the 256-dimensional optical feature to the CLIP image-feature dimension.

## Optical Student

The student uses the existing fair134 path:

```text
input amplitude
-> AS input_to_prompt
-> complex-amplitude global router in the 600x600 prompt aperture
-> AS prompt_to_expert
-> hard 9-expert aperture
-> five shared expert phase layers
-> windowed 600x600 global FC phase
-> AS detector propagation
-> detector intensity
-> 16x16 grid pooling
```

The feature detector optionally normalizes its pooled cells by total detector energy. The resulting 256 values feed both the classifier and the distillation projector. Teacher features are not used at inference time.

## Imagenette Layout

Place Imagenette under:

```text
data/imagenette2-160/
  train/<class_name>/*.JPEG
  val/<class_name>/*.JPEG
```

The default config uses `download: false`. Set it to `true` to download the official `imagenette2-160.tgz`, or download and extract it manually. `train/` is split deterministically into train/validation; Imagenette `val/` is used as the test split.

## Cache Contract

Build the teacher cache before training. Each split file stores normalized features, labels, and deterministic split-local indices. `metadata.json` records dataset split sizes, teacher model, feature dimension, class names, grayscale input mode, and a configuration hash. Training rejects stale or misaligned caches by default.

The CLIP dependency is optional for the rest of the repository. Install `open_clip_torch` to build caches:

```text
pip install open_clip_torch
```

## Outputs

Runs are written to `foundation_distillation/runs/<run_id>/`. Key files include checkpoints, epoch/final metrics, confusion matrix, feature similarity, expert usage, prompt weights, optical-energy diagnostics, light fields, prompt maps, and phase masks. Aggregate CSV files are rebuilt under `foundation_distillation/results/`.

