# Four-rank full-image smoke

Before the formal 20-epoch run, execute one real 224x224 ImageNet DDP smoke with the same batch geometry: batch 24 per rank, four ranks and gradient accumulation 2, for effective global batch 192.

```bash
PHYSICAL_GPU_INDICES=0,1,3,4 P13_ACTION=fresh \
bash experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/commands/05b_gpu_smoke_full_image_4rank_gb192.sh
```

The launcher requires four unique physical GPU indices, converts them to UUIDs, runs `torchrun --standalone --nproc_per_node=4` in the foreground and holds the run-specific `flock` until exit. It runs two training micro-batches and one validation batch, and writes only to `runs/p13_growth16_full_image_4rank_gb192_gpu_smoke`. It is an execution and memory check, not a performance result.
