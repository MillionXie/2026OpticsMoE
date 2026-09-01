# P08 commands

All commands are run from the repository root on the server.

```bash
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/01_extract_stem.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/02_validate_stem_equivalence.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/02_smoke.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/03_run_100k_screen.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/04_train_imagenet_90e.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/05_watch.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/06_benchmark_batch_size.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/07_benchmark_ddp_batch_size.sh
bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/08_launch_imagenet_90e_bs96.sh
```

The equivalence command compares the extracted Conv2D/position stem against the
official Qwen image processor and original Conv3D tensors without loading the
full model. Long training should not start unless it passes.

Override GPU selection without editing a script:

```bash
PHYSICAL_GPU_INDEX=3 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/03_run_100k_screen.sh
PHYSICAL_GPU_INDICES=3,5 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/04_train_imagenet_90e.sh
PHYSICAL_GPU_INDEX=1 BATCH_SIZES="64 96 128 160 192" bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/06_benchmark_batch_size.sh
PHYSICAL_GPU_INDICES=3,5 BATCH_SIZES="64 96" bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/07_benchmark_ddp_batch_size.sh
PHYSICAL_GPU_INDICES=3,5 bash FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/08_launch_imagenet_90e_bs96.sh
```

The batch benchmark performs 300 real training steps per candidate, including
the frozen online stem, eight optical stages, exact backpropagation, AMP,
augmentations and the optimizer update. It writes measured samples/second and
PyTorch peak allocated/reserved memory to a timestamped `summary.tsv`.
The DDP benchmark repeats the selected candidates on the actual heterogeneous
formal-training pair and reports global throughput, so the slower GPU and NCCL
synchronization are included in the final batch decision.

The optimized launcher starts the selected per-rank batch 96 run under
`nohup`, writes a PID file, refuses to create a duplicate live job, and resumes
the same run after an interruption. `05_watch.sh` now watches this run by
default; pass the old run directory explicitly to inspect the batch-28 archive.
