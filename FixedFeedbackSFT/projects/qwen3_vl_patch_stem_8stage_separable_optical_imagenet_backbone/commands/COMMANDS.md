# P11 commands

```bash
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/01_test.sh
PHYSICAL_GPU_INDEX=1 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/02_gpu_smoke.sh
PHYSICAL_GPU_INDICES=3,5 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/03_launch_imagenet_90e_bs96.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/04_watch.sh
```

The new large-scale continuation is isolated from the locked P11 run. First
freeze the completed 8-stage source and run the one-GPU smoke:

```bash
bash FixedFeedbackSFT/commands/00_freeze_8_16_backbones.sh
PHYSICAL_GPU_INDEX=<idle> bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/05_gpu_smoke_large_scale_continue.sh
```

Then use four non-overlapping GPUs for the two five-epoch learning-rate proxies:

```bash
P11_PROXY_LOW_GPUS=<g0,g1> P11_PROXY_HIGH_GPUS=<g2,g3> \
P11_PROXY_ACTION=fresh \
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/commands/06_launch_large_recipe_proxy_pair.sh
```

The five-GPU/100-epoch launcher is a gated template, not an automatic next
step. It requires `P11_FORMAL_PHASE_LR=2e3` or `7e3` selected from the proxy
results. See `../LARGE_SCALE_PRETRAINING_RECIPE.md` for the numerical gate; no
formal large-recipe run has been started by adding these commands.

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
