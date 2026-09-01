# P11 8-stage large-scale backbone continuation

This is an **independent** continuation recipe.  It does not modify the locked
P11/P13 model or training files and cannot overwrite the original 90-epoch P11
run.  Its source is the SHA-locked, physically copied P11 epoch-88 asset under
`FixedFeedbackSFT/runs/_assets/8stage`; the continuation never follows the
mutable historical run symlink after freezing.

## Implemented recipe

- two full-ImageNet, five-epoch, two-GPU proxy runs at phase peak learning
  rates 0.002 and 0.007 (per-rank batch 96, accumulation 2, global batch 384);
- a five-GPU, 100-epoch formal template (per-rank batch 96, global batch 480)
  that is launched only after a proxy passes the promotion gate;
- online RandomResizedCrop, flip and RandAugment(2,9), inherited from the
  existing ImageNet loader;
- Mixup 0.8, CutMix 1.0 and soft-target BCE with smoothing 0;
- random erasing is initially disabled to avoid over-regularizing this 2.8M
  trainable-parameter model; parameter-free stochastic depth reaches 0.05;
- AdamW with layer-wise LR decay 0.92 and cosine decay; the formal template
  uses five warmup epochs;
- phase LR remains material.  The proxy brackets 0.002 and 0.007 at the last
  stage; layer decay keeps earlier optical layers trainable rather than frozen;
- trainable-parameter EMA, evaluated separately from raw weights each epoch;
- strict source SHA, config digest, dirty-worktree implementation hashes,
  dataset/index identity, interval checkpoints and separate raw/EMA best
  checkpoints;
- resume additionally verifies the frozen stem and original phase-snapshot
  identities before restoring optimizer state.

The experiment changes several recipe components together. It is a
**performance run**, not an attribution ablation.  If it improves P11, follow
with a small three-run attribution set: baseline continuation, +layer decay and
EMA, then +strong regularization. Do not compare its 100-epoch result directly
to a 20-epoch depth-growth result as evidence for depth.

Promotion is deliberately gated.  The epoch-88 locked source is 51.348% Top-1.
A five-epoch proxy must reach at least 51.448% or show a clearly increasing raw
or EMA curve to justify the 100-epoch run.  The formal scientific gain target is
at least +0.30 pp over 51.348%, followed by repeated seeds; a smaller single-run
gain is only a tuning signal.

## Run order

```bash
PHYSICAL_GPU_INDEX=<idle> bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/05_gpu_smoke_large_scale_continue.sh

P11_PROXY_LOW_GPUS=<g0,g1> P11_PROXY_HIGH_GPUS=<g2,g3> \
  P11_PROXY_ACTION=fresh \
  bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/06_launch_large_recipe_proxy_pair.sh

# Only after the promotion gate, and only after confirming five GPUs are idle:
P11_FORMAL_GPUS=<g0,g1,g2,g3,g4> P11_FORMAL_PHASE_LR=<2e3|7e3> \
  P11_FORMAL_ACTION=fresh \
  bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/07_launch_large_recipe_formal_100e_5gpu.sh
```

Resume uses the same launchers with action `resume`. Fresh mode refuses any
existing artifact; it never deletes or replaces a run. Exact DDP resume stores
and restores one RNG state per rank, requires the original world size, and
checks the dataset/index, stem and continuation-start phase identities.

The formal run has **not** been started. `P11_FORMAL_PHASE_LR` must be selected
explicitly from the two proxy results; there are separate immutable configs and
run directories for `2e3` and `7e3`.

## Why supervised first

The loader continues to use the project's existing RandomResizedCrop,
horizontal flip and RandAugment(2,9). This is not a strict reproduction of
DeiT-III ThreeAugment or repeated augmentation, so reports must call it a
DeiT/ConvNeXt-inspired recipe rather than an official recipe reproduction.

The existing frozen Qwen stem already supplies a pretrained tokenization. A
new self-supervised objective would simultaneously change the data, loss and
head, making a plateau diagnosis ambiguous.  First test whether a mature
supervised recipe moves the 8-stage ceiling.  If the gain is below 0.5 pp after
the planned repeated runs, move to a separately named multi-dataset
self-supervised stage (ImageNet-21K or OpenImages images, masked feature
prediction / teacher-student self-distillation), followed by fixed ImageNet-1K
linear probing and fine-tuning.  That stage must not be labeled as implemented
by this entry point.
