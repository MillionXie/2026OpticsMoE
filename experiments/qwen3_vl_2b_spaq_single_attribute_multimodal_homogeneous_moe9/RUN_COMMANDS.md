# Commands

Commands are written as single lines for direct use from the repository root.

## Vision optical MoE + frozen electronic language

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9/configs/spaq_mos_vision_electronic_language.json --phase all
```

## Vision optical MoE + language optical MoE

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9/configs/spaq_mos_vision_language_optical.json --phase all
```

## Smoke tests

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9/configs/spaq_mos_vision_electronic_language_smoke.json --phase all
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9 --config experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9/configs/spaq_mos_vision_language_optical_smoke.json --phase all
```

## Tests

```bash
pytest experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9/tests -q
```
