# Qwen3-VL-8B Multimodal Optical Weather-4 Commands

## Run all phases

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_8b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --device cuda \
  --phase all
```

## Train the electronic teacher MLP

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_8b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --device cuda \
  --phase teacher_train
```

## Evaluate the electronic teacher

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_8b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --device cuda \
  --phase teacher_inference
```

## Run only student training if teacher checkpoint already exists

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_8b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --device cuda \
  --phase student_train
```

## Evaluate the optical student

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_8b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --device cuda \
  --phase student_inference
```

## Compare teacher and student

```bash
python -m experiments.qwen3_vl_8b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_8b_multimodal_optical_weather4/configs/bdd100k_weather4.json \
  --phase compare
```

## Smoke run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_multimodal_optical_weather4 \
  --config experiments/qwen3_vl_8b_multimodal_optical_weather4/configs/bdd100k_weather4_smoke.json \
  --device cuda \
  --phase all
```

All model phases also support `--cache-dir`, `--model-id`, and `--local-files-only`.

