# Run commands

Run all commands from the repository root.

## Download and validate the official dataset

```bash
python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_scratch.yaml \
  --phase prepare_data
```

## Shape smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_scratch_smoke.yaml \
  --phase shape_smoke
```

## End-to-end from scratch

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_scratch.yaml \
  --phase all
```

## COCO -> DUTS pretrained transfer

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_coco_duts_pretrained.yaml \
  --phase all
```

## Fastest recovery from the completed head-only run

This keeps the already ISIC-adapted segmentation head, resets the optimizer,
and genuinely unfreezes the optical core, router and recombiner:

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_resume_previous_pretrained_unfreeze.yaml \
  --phase all
```

## Re-evaluate the selected checkpoint

Replace the config with either formal config:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation \
  --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_scratch.yaml \
  --phase test
```

## Unit tests

```bash
pytest experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/tests -q
```

## Vision2 光电联合训练（正式配置）

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_vision2_hybrid.yaml --phase train
```

本配置从第一步联合训练两级光学、电子 Mixer 和 lesion decoder；不依赖 COCO/DUTS checkpoint。phase/mask 学习率为 `1e-4`。checkpoint 为 `experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/checkpoints/isic_student_best_train_loss.pt`。

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task isic --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/checkpoints/isic_student_best_train_loss.pt --session-dir experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/hardware_sessions/vision2_run1 --stage vision_expert --phase export

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task isic --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/runs/isic2016_vision2_hybrid/checkpoints/isic_student_best_train_loss.pt --session-dir experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/hardware_sessions/vision2_run1 --stage vision_expert --phase finetune --epochs 20

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task isic --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/hardware_sessions/vision2_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/hardware_sessions/vision2_run1 --stage vision_global --phase export

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task isic --config experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/configs/isic2016_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/hardware_sessions/vision2_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/hardware_sessions/vision2_run1 --stage vision_global --phase finetune --epochs 20
```

实验室步骤见 `experiments/vision2_hybrid_dense/HARDWARE_PROTOCOL.md`。
