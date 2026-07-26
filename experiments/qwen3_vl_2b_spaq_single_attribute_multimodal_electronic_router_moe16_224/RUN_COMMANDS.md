# Run commands

All commands are written as one line for execution from the repository root.

## MOS smoke

```bash
python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_mos_smoke.json --phase all
```

## MOS formal vision + language optical MoE

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_mos.json --phase all
```

## MOS diagnostic: vision optical, frozen electronic language

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_mos_vision_electronic_language.json --phase all
```

## Other attributes

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_brightness.json --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_colorfulness.json --phase all
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_contrast.json --phase all
```

## Reusable precompute phases

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_mos.json --phase teacher_precompute
```

```bash
python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_mos.json --phase input_precompute
```

## Epoch-77 regularized fine-tuning

The source `best` checkpoint must report epoch 77. The optimizer and scheduler
are rebuilt from scratch. Frozen Qwen teacher/processor caches and teacher
predictions are reused.

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_mos_epoch77_regularized_finetune.json --phase student_train
```

## Epoch-77 regularized fine-tuning with SAM

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/configs/spaq_mos_epoch77_regularized_finetune_sam.json --phase student_train
```

After either training run, use the same config with `--phase student_inference`.

## Tests

```bash
pytest experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224/tests -q
```
