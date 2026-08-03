# Run commands

所有命令均从仓库根目录 `2026OpticsMoE/` 执行。

CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency 
  --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon.yaml 
  --phase all

## 数据准备

```bash
python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency \
  --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon.yaml \
  --phase prepare_data
```

首次运行会下载官方 train/validation 图像和 fixation JSON。下载量和解压后数据均
为 GB 级；中断后保留 `.part`，重新运行即可继续（前提是远端支持 Range）。

## 电子 Teacher

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency \
  --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon.yaml \
  --phase teacher_train

CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency \
  --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon.yaml \
  --phase teacher_evaluate
```

## Optical Student

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency \
  --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon.yaml \
  --phase student_train

CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency \
  --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon.yaml \
  --phase student_evaluate
```

## 可选 final-map KD

先缓存 Teacher map，再用空间增强关闭的 KD 配置：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency \
  --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon_mask_kd.yaml \
  --phase cache_teacher_maps

CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency \
  --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon_mask_kd.yaml \
  --phase student_train
```

## Smoke / tests

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency 
  --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon_smoke.yaml 
  --phase all

pytest experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/tests -q
```
