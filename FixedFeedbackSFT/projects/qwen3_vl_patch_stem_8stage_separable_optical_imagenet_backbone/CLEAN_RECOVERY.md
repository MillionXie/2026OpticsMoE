# P11 post-strong-pretrain clean recovery

This is a five-epoch **recovery stage**, not a fifth FA method group and not a
new pretraining method. Its only purpose is to test whether the high-phase
strong-augmentation proxy learned useful features while temporarily damaging
the ImageNet linear decision boundary.

## Registered protocol

- source: the completed `phase_lr=7e-3` five-epoch proxy `last.pt`;
- source state: RAW (configurable code path also supports EMA, but the
  registered run is RAW);
- source identity: SHA256, format, role, epoch and config digest are hard
  checked before loading;
- optimizer: newly initialized AdamW; no source optimizer/scheduler/scaler
  state is reused;
- augmentation: RandomResizedCrop + horizontal flip only;
- view schedule: five deterministic RRC/flip views, so no view repeats within
  the registered five epochs;
- disabled: Mixup, CutMix, RandAugment, random erasing, label smoothing and
  stochastic depth;
- retained architecture behavior: the source P11 mixer's internal dropout
  remains `0.10`; "clean" describes the input/loss recovery recipe, not removal
  of every architectural regularizer;
- learning rates: phase `8e-4`, electronic `2.5e-5`, adapter `2e-5`, temporary
  ImageNet head `5e-5`, with one warmup epoch and cosine decay;
- budget: five epochs, one GPU, batch/global batch 96;
- selection: track and save independent RAW and EMA best checkpoints as well as
  exact-resume `last.pt`.

The run records source, dataset, frozen stem, initial phase, config and code
identities. Exact resume rejects a changed source, dataset/index set, stem,
config, implementation, world size or initial phase snapshot.

## Launch

```bash
P11_CLEAN_GPU=5 P11_CLEAN_ACTION=fresh \
  bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/08_launch_clean_recovery_probe_gpu5.sh
```

Use `P11_CLEAN_ACTION=resume` only after an interrupted run has produced
`checkpoints/last.pt`. Fresh mode refuses to overwrite an existing run.

## Interpretation gate

Compare the recovery baseline (the exact source RAW checkpoint), each RAW/EMA
epoch and the original P11 epoch-88 baseline. A recovery result is evidence
about optimization dynamics only; it must not be presented as an independent
FA comparison group.
