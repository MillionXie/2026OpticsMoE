# P09 commands

Run from the repository root on the server.

```bash
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/commands/01_test.sh
PHYSICAL_GPU_INDEX=5 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/commands/02_gpu_smoke_bs96.sh
PHYSICAL_GPU_INDICES=3,5 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/commands/03_launch_imagenet_90e_bs96.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/commands/04_watch.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/commands/05_compare_with_p08_epoch9.sh
```

The formal launcher uses two ranks, batch 96 per rank and global batch 192. It
therefore has the same 6,672 optimizer steps per ImageNet epoch as P08. The
comparison command only reports epochs completed by both runs; the archived
P08 baseline currently ends after epoch 9.
