# P08 commands

All commands are run from the repository root on the server.

```bash
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/01_extract_stem.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/02_smoke.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/03_run_100k_screen.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/04_train_imagenet_90e.sh
bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/05_watch.sh
```

Override GPU selection without editing a script:

```bash
PHYSICAL_GPU_INDEX=3 bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/03_run_100k_screen.sh
PHYSICAL_GPU_INDICES=3,5 bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/04_train_imagenet_90e.sh
```
