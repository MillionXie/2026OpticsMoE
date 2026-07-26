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
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_kadid10k_mos_multimodal_electronic_router_moe16_224/configs/kadid10k_mos.json --phase all
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
