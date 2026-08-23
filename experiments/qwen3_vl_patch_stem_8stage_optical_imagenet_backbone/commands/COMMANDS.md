# P08 commands

All commands are run from the repository root on the server.

```bash
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/01_extract_stem.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/02_validate_stem_equivalence.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/02_smoke.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/03_run_100k_screen.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/04_train_imagenet_90e.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/05_watch.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/06_benchmark_batch_size.sh
```

The equivalence command compares the extracted Conv2D/position stem against the
official Qwen image processor and original Conv3D tensors without loading the
full model. Long training should not start unless it passes.

Override GPU selection without editing a script:

```bash
PHYSICAL_GPU_INDEX=3 bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/03_run_100k_screen.sh
PHYSICAL_GPU_INDICES=3,5 bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/04_train_imagenet_90e.sh
PHYSICAL_GPU_INDEX=1 BATCH_SIZES="64 96 128 160 192" bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/06_benchmark_batch_size.sh
```

The batch benchmark performs 300 real training steps per candidate, including
the frozen online stem, eight optical stages, exact backpropagation, AMP,
augmentations and the optimizer update. It writes measured samples/second and
PyTorch peak allocated/reserved memory to a timestamped `summary.tsv`.
