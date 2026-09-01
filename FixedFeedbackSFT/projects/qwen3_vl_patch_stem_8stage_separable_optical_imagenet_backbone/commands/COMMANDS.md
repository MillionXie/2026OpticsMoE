# P11 commands

```bash
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/01_test.sh
PHYSICAL_GPU_INDEX=1 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/02_gpu_smoke.sh
PHYSICAL_GPU_INDICES=3,5 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/03_launch_imagenet_90e_bs96.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/04_watch.sh
```

The formal launcher intentionally has no default GPU pair, so it cannot start
accidentally. The active 2026-08-24 run was launched with
`PHYSICAL_GPU_INDICES=1,2`; always re-check device ownership before any resume
or replacement launch.

The same launcher was used on 2026-08-26 to resume the complete epoch-15
checkpoint on physical GPUs 2 and 3:

```bash
PHYSICAL_GPU_INDICES=2,3 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/03_launch_imagenet_90e_bs96.sh
```

The launcher passes `--resume`, preserves checkpoints/history and restarts at
the next complete epoch. It opens the active log with shell redirection, so an
existing log must be copied to an audit filename before any future resume.
