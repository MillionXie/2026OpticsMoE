# P10 commands

```bash
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone/commands/01_test.sh
PHYSICAL_GPU_INDEX=1 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone/commands/02_gpu_smoke.sh
PHYSICAL_GPU_INDICES=3,5 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone/commands/03_launch_imagenet_90e_bs96.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone/commands/04_watch.sh
```

The formal launcher intentionally has no default GPU pair, so it cannot start
accidentally. It has been prepared for a future two-free-GPU window but was not
executed during implementation.

The formal run was started on 2026-08-27 on physical GPUs 0 and 3:

```bash
PHYSICAL_GPU_INDICES=0,3 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone/commands/03_launch_imagenet_90e_bs96.sh
```

Always re-check device ownership before a resume. The launcher has no fallback
GPU indices and always passes `--resume` for checkpoint-safe continuation.
