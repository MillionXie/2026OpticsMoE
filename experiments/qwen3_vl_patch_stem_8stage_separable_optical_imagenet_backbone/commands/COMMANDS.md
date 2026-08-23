# P11 commands

```bash
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/01_test.sh
PHYSICAL_GPU_INDEX=1 bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/02_gpu_smoke.sh
PHYSICAL_GPU_INDICES=3,5 bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/03_launch_imagenet_90e_bs96.sh
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/04_watch.sh
```

The formal launcher intentionally has no default GPU pair, so it cannot start
accidentally. It has been prepared for a future two-free-GPU window but was not
executed during implementation.
