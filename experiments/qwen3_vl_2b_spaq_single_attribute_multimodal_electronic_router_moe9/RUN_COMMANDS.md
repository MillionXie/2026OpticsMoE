# Commands

All commands are intended to be run from the `2026OpticsMoE` repository root. They are deliberately single-line commands without shell continuation backslashes.

## MOS

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9/configs/spaq_mos.json --phase all
```

## Brightness

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9/configs/spaq_brightness.json --phase all
```

## Colorfulness

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9/configs/spaq_colorfulness.json --phase all
```

## Contrast

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9/configs/spaq_contrast.json --phase all
```

## Smoke

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9/configs/spaq_mos_smoke.json --phase all
```

The other smoke configs are `spaq_brightness_smoke.json`, `spaq_colorfulness_smoke.json`, and `spaq_contrast_smoke.json`.

## Vision optical + frozen electronic language diagnostic

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_electronic_router_moe9/configs/spaq_mos_vision_electronic_language.json --phase all
```
