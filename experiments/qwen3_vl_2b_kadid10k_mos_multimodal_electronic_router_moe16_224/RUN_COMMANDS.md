# Run commands

Run these commands from the repository root. They are intentionally one-line commands without `\`.

## Prepare/download KADID-10k

```bash
python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos.json --phase prepare_data
```

## Smoke run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos_smoke.json --phase all
```

## Full run

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos.json --phase all
```

## Task-driven rerun without SAM or weight decay

This configuration keeps the student head independently initialized. It disables
SAM, weight decay, and phase dropout; uses a non-affine head LayerNorm; reduces
the three teacher-derived losses to weak regularizers; and gives ground-truth
regression/ranking the dominant optimization signal. It writes to a separate
run directory and reuses the shared processor/teacher hidden cache.

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos_task_driven_no_sam_no_wd.json --phase all
```

If the shared caches are already complete, the phases can also be run separately:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos_task_driven_no_sam_no_wd.json --phase teacher_train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos_task_driven_no_sam_no_wd.json --phase teacher_predictions
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos_task_driven_no_sam_no_wd.json --phase student_train
```

## Separate phases

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos.json --phase input_precompute
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos.json --phase teacher_precompute
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos.json --phase teacher_train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos.json --phase teacher_predictions
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos.json --phase student_train
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos.json --phase student_inference
```

## Tests

```bash
pytest experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/tests -q
```
