# Qwen3-VL-2B Multimodal Optical Weather-4 Commands

```bash
# Prepare BDD100K Weather-4
python -m experiments.qwen3_vl_2b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --phase prepare_data

# Train electronic teacher MLP
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --device cuda --phase teacher_train

# Cache teacher logits and five groups of input/output hidden states
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --device cuda --phase teacher_cache

# Train student only from cached teacher targets
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --device cuda --phase student_train

# Run every phase
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --device cuda --phase all

# Balanced subset
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/bdd100k_weather4_balanced.json \
  --device cuda --phase all

# Smoke test
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/bdd100k_weather4_smoke.json \
  --device cuda --phase all

# Compare completed teacher/student metrics
python -m experiments.qwen3_vl_2b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --phase compare
```

All model phases support `--cache-dir`, `--model-id`, and `--local-files-only`.
