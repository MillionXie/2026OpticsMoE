# Qwen3-VL-8B Vision-only MLP Commands

Set an optional cross-server cache location first, for example:

```bash
export HF_HOME=/path/to/hf_cache
```

## Download checkpoint

```bash
python -m experiments.qwen3_vl_8b_vision_mlp \
  --config experiments/qwen3_vl_8b_vision_mlp/configs/cifar100.json \
  --phase download \
  --cache-dir "$HF_HOME"
```

## Extract frozen visual features

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_vision_mlp \
  --config experiments/qwen3_vl_8b_vision_mlp/configs/cifar100.json \
  --device cuda \
  --cache-dir "$HF_HOME" \
  --local-files-only \
  --phase extract
```

## Train MLP from cached features

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_vision_mlp \
  --config experiments/qwen3_vl_8b_vision_mlp/configs/cifar100.json \
  --device cuda \
  --phase train
```

## Run synchronized end-to-end inference benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_vision_mlp \
  --config experiments/qwen3_vl_8b_vision_mlp/configs/cifar100.json \
  --device cuda \
  --cache-dir "$HF_HOME" \
  --local-files-only \
  --phase inference
```

## Run all phases

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_8b_vision_mlp \
  --config experiments/qwen3_vl_8b_vision_mlp/configs/cifar100.json \
  --device cuda \
  --cache-dir "$HF_HOME" \
  --phase all
```

## Regenerate figures from existing metrics

```bash
python -m experiments.qwen3_vl_8b_vision_mlp \
  --config experiments/qwen3_vl_8b_vision_mlp/configs/cifar100.json \
  --phase visualize
```

