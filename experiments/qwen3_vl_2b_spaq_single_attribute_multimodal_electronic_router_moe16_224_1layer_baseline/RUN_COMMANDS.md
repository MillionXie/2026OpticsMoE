# Run commands

Run every command from the repository root. Commands are intentionally written
on one line without continuation backslashes.

## Unit tests

```bash
pytest experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/tests -q
```

## MOS smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos_smoke.json --phase all
```

## Formal MOS

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos.json --phase all
```

## Formal Brightness

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_brightness.json --phase all
```

## Formal Colorfulness

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_colorfulness.json --phase all
```

## Formal Contrast

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_contrast.json --phase all
```

## Vision optical + electronic Language diagnostic

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos_vision_electronic_language.json --phase all
```

## Phases separately

Prepare or download SPAQ:

```bash
python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos.json --phase prepare_data
```

Precompute frozen Qwen teacher and processor caches:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos.json --phase teacher_precompute
```

Train teacher head:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos.json --phase teacher_train
```

Cache teacher predictions:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos.json --phase teacher_predictions
```

Train student:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos.json --phase student_train
```

Final student inference:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe16_224_1layer_baseline/configs/spaq_mos.json --phase student_inference
```

