# Foundation Image-Feature Distillation

This experiment trains a 9-expert `fast120_520` AS global-router OpticalMoE against a frozen image encoder. It supports CLIP and DINOv2 feature teachers and reuses the validated propagation modules in `opticalmoe_experiments/common/`.

## Scope

- Datasets: CIFAR10 grayscale and Imagenette grayscale.
- Teachers: frozen CLIP image encoder (`ViT-B/32`) or frozen DINOv2 image encoder (`facebook/dinov2-small` and `facebook/dinov2-base`).
- No text encoder, text features, zero-shot logits, teacher classification head, or logits distillation.
- DINOv2 defaults to its normalized CLS token; `patch_mean` is also supported.
- Teacher and student receive the same grayscale information. The student sees one optical-amplitude channel at 120x120; teacher input replicates that grayscale image to three channels and applies backend-specific resize/normalization.
- Distillation target: a semantic feature produced from the physical camera crop, never an intermediate complex field.
- The 520x520 array is only the propagation canvas. The physical camera/mask window is the central 450x450 region `[35:485, 35:485]`; propagation padding never enters feature pooling.

## End-to-end MoE baseline

The folder also contains a controlled CE-only baseline for CIFAR10-gray and Imagenette-gray. It uses the same:

- 9-expert `fast120_520` AS global-router optical backbone;
- trainable prompt amplitudes and phase biases;
- five expert phase layers and 450x450 global FC phase;
- detector-plane camera crop followed by a 30x30 grid feature;
- LayerNorm/GELU postprocess and a small MLP classifier;
- dataset split, optimizer, phase dropout, seed, and training schedule.

The baseline removes the teacher cache, semantic projector, and cosine feature loss. Its only objective is label cross-entropy. This makes `end_to_end_optical_moe` the direct control for `feature_distilled_optical_moe`.

In the distilled model the projector is part of inference. It maps the 900-dimensional camera feature into the CLIP/DINOv2 teacher space, and the classifier consumes that same semantic representation. CE and cosine distillation therefore optimize one shared inference path.

Parameter reports define `electronic_parameter_count` as projector plus classifier. The affine LayerNorm parameters are reported separately as `feature_preprocess_parameter_count`; `inference_parameter_count` and `training_parameter_count` include all three components.

## Optical Student

The student uses the `fast120_520` path:

```text
input amplitude
-> AS input_to_prompt
-> complex-amplitude global router in the 450x450 prompt aperture
-> AS prompt_to_expert
-> hard 9-expert aperture
-> five shared expert phase layers
-> windowed 450x450 global FC phase
-> AS detector propagation
-> detector intensity on 520x520 propagation canvas
-> center crop to the physical 450x450 camera region
-> 30x30 sum pooling (900 values, about 15x15 camera pixels per cell)
-> LayerNorm + GELU
-> projector into teacher feature space
-> classifier
```

The default camera feature does not use square-root compression or total-energy normalization. The projector output is L2-normalized by default and is used by both the classifier and cosine feature loss. Teacher features are never needed at inference time. Energy outside the camera crop is recorded as `outside_camera_energy_ratio`; `leak_loss_weight` defaults to `0.0`, so it does not alter training unless explicitly enabled.

The default projector is `900 -> 512 -> teacher_dim`. A single `Linear(900, teacher_dim)` can be selected with:

```yaml
projector:
  type: linear
  input_dim: auto_feature_dim
  output_dim: auto_teacher_dim
  output_l2_normalize: true
```

The student propagation canvas is `520 x 520`, input/expert size is `120`,
pitch is `150`, and prompt/global FC use `[35:485, 35:485]`. Teacher
preprocessing remains independent of student canvas size.
Explicit `fair134_1000` configs remain loadable for legacy reproduction.

## Imagenette Layout

Place Imagenette under:

```text
data/imagenette2-160/
  train/<class_name>/*.JPEG
  val/<class_name>/*.JPEG
```

The default config uses `download: false`. Set it to `true` to download the official `imagenette2-160.tgz`, or download and extract it manually. `train/` is split deterministically into train/validation; Imagenette `val/` is used as the test split.

## Cache Contract

Build the teacher cache before training. Each split file stores L2-normalized features, labels, and deterministic split-local indices. `metadata.json` records dataset split sizes, teacher type/backend/model, feature type and dimension, class names, grayscale input mode, and a configuration hash. CLIP and DINOv2 use the same cache payload format, so the training script does not branch on teacher type. Training rejects stale or misaligned caches by default.

The CLIP dependency is optional for the rest of the repository. Install `open_clip_torch` to build caches:

```text
pip install open_clip_torch
```

DINOv2 uses the optional Hugging Face transformers backend:

```text
pip install transformers
```

The default DINOv2 model is `facebook/dinov2-small` with
`feature_type: cls`. The image encoder remains in `eval()` and all of its
parameters have `requires_grad=False` while cache features are generated.

## Teacher Feature Probe

`train_teacher_feature_probe.py` trains a classifier directly on cached frozen-teacher features. The default `matched_mlp` probe uses the same one-hidden-layer, 128-unit GELU classifier as the optical semantic classifier, providing a controlled estimate of how separable the grayscale teacher features are. A linear probe is available as a secondary reference. Probe runs are stored under `foundation_distillation/teacher_probe_runs/` and aggregated into `master_teacher_probe_*.csv`.

## LeNet Distillation Diagnostic

The `feature_distilled_lenet` baseline checks whether the cached-teacher, projector, classifier, and cosine-loss pipeline works without optical propagation. It uses the same grayscale dataset split and teacher cache as the CIFAR10 CLIP OpticalMoE experiment:

```text
grayscale image
-> three-layer LeNet feature extractor
-> 900-dimensional feature
-> the same LayerNorm + GELU preprocess
-> the same teacher-space projector
-> the same semantic classifier
-> CE + cosine feature distillation
```

The classifier cannot bypass the projector: both classification and distillation use the same semantic feature. If this baseline generalizes well while OpticalMoE does not, the likely bottleneck is optical feature extraction. If both fail, the cache alignment, semantic projector/classifier, data split, or loss weighting should be investigated. LeNet has no camera padding or leak loss; reports use `optical_parameter_count=0` and record its CNN parameters as `lenet_parameter_count`.

The LeNet backbone supports `conv_dropout2d` after its first two pooled convolution blocks and `feature_dropout` after the 900-dimensional projection. Both default to `0.0` for old configurations; the CIFAR10 diagnostic configs use `0.1` and `0.2` respectively. CLIP, DINOv2-small, and DINOv2-base LeNet distillation variants share this architecture.

`SupervisedLeNetClassifier` is the CE-only control. It uses the same grayscale split, LeNet backbone, 900-dimensional feature, preprocess, and classifier, but has no teacher cache, projector, or feature loss. Comparing supervised LeNet against distilled LeNet isolates the benefit or harm of teacher feature alignment.

## Outputs

Runs are written to `foundation_distillation/runs/<run_id>/`. Key files include checkpoints, epoch/final metrics, confusion matrix, feature similarity, expert usage, prompt weights, camera-feature statistics, optical-energy diagnostics, light fields, prompt maps, and phase masks. Light-field sample directories include `input_student_gray.png`, `input_amplitude.png`, `input_teacher_gray_rgb.png`, `label.txt`, and `prediction.txt`; `input_original_rgb.png` is additionally written when the dataset wrapper supplies it. Aggregate CSV files are rebuilt under `foundation_distillation/results/`.

The end-to-end baseline does not require a teacher cache. Its runs are written into the same results tables with `experiment_variant=end_to_end_ce_baseline`, `teacher_type=none`, and `feature_distill_weight=0`. LeNet distillation runs use `experiment_variant=lenet_feature_distillation` and are included in the same master tables.
