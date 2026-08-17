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

## Vision2 光电联合训练（正式配置）

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon_vision2_hybrid.yaml --phase student_train
```

本配置不训练 Teacher、不使用 KD。phase/mask 学习率为 `1e-4`。联合训练 checkpoint 为 `experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/runs/salicon_vision2_hybrid/checkpoints/student_best.pt`。

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task salicon --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/runs/salicon_vision2_hybrid/checkpoints/student_best.pt --session-dir experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/hardware_sessions/vision2_run1 --stage vision_expert --phase export

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task salicon --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/runs/salicon_vision2_hybrid/checkpoints/student_best.pt --session-dir experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/hardware_sessions/vision2_run1 --stage vision_expert --phase finetune --epochs 20

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task salicon --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/hardware_sessions/vision2_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/hardware_sessions/vision2_run1 --stage vision_global --phase export

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task salicon --config experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/configs/salicon_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/hardware_sessions/vision2_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/hardware_sessions/vision2_run1 --stage vision_global --phase finetune --epochs 20
```

实验室重建、采集和上传步骤见 `experiments/vision2_hybrid_dense/HARDWARE_PROTOCOL.md`。
