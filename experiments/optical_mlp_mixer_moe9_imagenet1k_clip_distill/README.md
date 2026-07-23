# OpticalMixerMoE9: ImageNet-1K CLIP distillation

This is an independent vision pre-training experiment. It contains only the
seven-block `OpticalMixerMoE9`; it does not contain Qwen, a language model,
an electronic MLP-Mixer baseline, fine-tuning, or SAM.

The goal is to train a reusable optical vision representation on the complete
ImageNet-1K training set with a frozen OpenAI CLIP ViT-B/16 teacher.

## Why the patch tensor is `[B,196,224]`

An RGB image is kept at `224x224`. A `16x16`, stride-16 patch embedding produces
a `14x14` grid, therefore exactly 196 real visual patches:

```text
[B,3,224,224]
-> Conv2d(3,224,kernel_size=16,stride=16)
-> [B,224,14,14]
-> [B,196,224]
```

Creating 224 patch tokens directly would require an irregular grid, overlapping
patches, or 28 learned/fabricated tokens. This experiment does none of those.
Instead, the 196 real token rows/columns are strictly zero-padded to the
`224x224` optical field. No bilinear or nearest-neighbor feature interpolation
is used.

## Model

Each of seven folded Mixer blocks has the standard two residual roles:

```text
token-oriented optical mixing -> residual
channel-oriented optical mixing -> residual
```

One block owns five 9-expert phase planes and one shared global phase:

```text
Token half:
  LN([196,224])
  -> transpose [224,196]
  -> zero pad to [224,224]
  -> Softplus
  -> electronic top-3 router (called once)
  -> direct weighted amplitude loading
  -> optical expert stages 1-2 with OEO
  -> shared global phase and central detector
  -> signed non-affine detector LayerNorm
  -> token residual

Channel half:
  LN([196,224])
  -> zero pad rows to [224,224]
  -> Softplus
  -> direct amplitude reload using the SAME top-3 and weights
  -> optical expert stages 3-5 with OEO
  -> the SAME shared global phase and central detector
  -> signed non-affine detector LayerNorm
  -> channel residual
```

Every OEO stage follows the currently validated homogeneous-MoE rule:

```text
phase modulation
-> 10 cm angular-spectrum propagation
-> square-law detection
-> independent non-affine LayerNorm for each selected expert
-> ReLU
-> reapply the same routing weight
-> force unselected expert regions to zero
-> zero-phase amplitude reload
```

The electronic router replaces the old optical prompt/grating router. There is
no prompt phase. It predicts a top-3 selection from the block input and directly
loads weighted feature copies onto the corresponding amplitude-SLM regions.

## Geometry and parameter count

```text
expert_size                 224
expert_gap                   30
expert_pitch                254
active_size                 762
outer_padding_per_side       15
canvas_size                 792
experts                     3x3
```

This preserves the same geometric convention as the earlier
`120/150/450/480` MoE layout.

Phase-only parameter count:

```text
one nine-expert plane          9 * 224^2 =   451,584
five expert planes per block                 2,257,920
one 762x762 global phase                       580,644
one optical block                           2,838,564
seven optical blocks                       19,869,948
```

Electronic patch embedding, routers, normalizations, CLIP projection, and
ImageNet classifier are reported separately in `model.json`.

## CLIP supervision

The frozen teacher is OpenAI CLIP ViT-B/16. The student produces a normalized
512-dimensional embedding. Training combines:

```text
cosine loss against the cached CLIP image embedding
KL distillation against CLIP ImageNet text-prototype logits
supervised ImageNet CrossEntropy
router balance loss
```

The CLIP cache stores the exact deterministic augmented view used by the
student. By default, four views are cached for each training image and the epoch
sampler cycles through them. It is invalid to cache a center-crop teacher
feature and train the student on an unrelated random crop.

CLIP embeddings use a contiguous NumPy memmap cache, approximately 5.2 GB for
four fp16 views of the full ImageNet-1K train split. ImageNet text logits are
computed from the 512-dimensional embeddings and fixed text prototypes, so a
1000-logit cache is unnecessary.

## Training policy

The formal configuration uses:

```text
epochs              90
optimizer           AdamW
weight_decay        0.0
warmup              10 epochs
scheduler           cosine
phase dropout       disabled
full train samples  1,281,167 per epoch
validation samples  50,000
```

Ninety epochs are the selected first full pre-training budget. The best
ImageNet validation top-1 checkpoint and the final checkpoint are both saved.
If the losses and validation metrics are still improving at epoch 90, set
`training.resume_checkpoint` to `checkpoints/last.pt`, increase `epochs`, and
resume; the 90-epoch checkpoint is not an architectural limit.

`weight_decay` is validated to be exactly zero to avoid applying AdamW decay to
raw phase parameters.

## ImageNet authorization and automatic preparation

ImageNet-1K is gated. The formal and smoke configs use the official
`ILSVRC/imagenet-1k` Hugging Face dataset directly; they do not make a second
ImageFolder copy. Complete these one-time authorization steps:

1. Sign in at `https://huggingface.co/datasets/ILSVRC/imagenet-1k`.
2. Review and accept the ImageNet access conditions.
3. Create a read token in the same Hugging Face account.
4. Run `hf auth login` as the server user and enter that token.

After authorization, `--phase prepare_data` downloads/reuses the train and
validation parquet shards automatically under:

```text
data/imagenet1k/huggingface_cache/
```

The token is read from the Hugging Face credential store; it is never written
to `config_resolved.json`, cache metadata, checkpoints, or Git.

The formal config validates 1,281,167 train images, 50,000 validation images,
and 1,000 labels. The complete upstream repository is about 167 GB, and the
cache plus temporary download state requires additional free space. On the
laboratory server this cache belongs under `/DATA`, not the nearly full home
filesystem.

For an already extracted private ImageFolder copy, set
`dataset.source="imagefolder"`, change `validation_split` to `val`, and point
`dataset.root` at:

```text
root/train/<synset>/*.JPEG
root/val/<synset>/*.JPEG
```

The OpenAI CLIP checkpoint may download automatically. Set `clip.cache_dir` to
an existing cache for offline operation.

## Phases

```text
prepare_data  Authorized download/reuse plus ImageNet validation and reports
clip_cache    Cache exact-view frozen CLIP features and text prototypes
train         Train all seven OpticalMixerMoE9 blocks
evaluate      Load best.pt and evaluate ImageNet validation once
all           Run all phases in order
```

## Outputs

```text
config_resolved.json
environment.json
dataset.json
model.json
optical_parameter_formula.json

clip_cache/train_clip_embeddings.npy
clip_cache/validation_clip_embeddings.npy
clip_cache/imagenet_text_prototypes.pt

checkpoints/best.pt
checkpoints/last.pt
checkpoints/epoch_XXXX.pt

metrics/training_history.csv
metrics/training_latest.json
metrics/best_validation.json
metrics/final_validation.json
metrics/validation_per_class.csv

figures/training_curves.png
figures/phase_masks/epoch_XXXX.png
figures/router/
figures/debug_examples/
```

Debug examples distinguish physical nonnegative detector intensity from signed
post-detector LayerNorm values and residual deltas. They also save the direct
amplitude-SLM loads, all five OEO detector intensities, all five zero-phase
reloaded amplitudes, and the sample's selected experts/routing weights.

## Compute warning

A 792x792 propagation is substantially more expensive than the earlier 480x480
canvas. Each folded block performs five expert propagations and two global
readouts; seven blocks perform 49 propagation paths per sample. Top-3 routing
controls optical energy and trainable expert use, but a full-canvas FFT does not
become three times faster merely because six experts are zero.

Multi-GPU H200 training should use `torchrun`. Every original image is covered
once per epoch; when the dataset size is not divisible by the world size, the
sampler repeats at most `world_size - 1` boundary samples so all ranks execute
the same number of optimizer steps.
