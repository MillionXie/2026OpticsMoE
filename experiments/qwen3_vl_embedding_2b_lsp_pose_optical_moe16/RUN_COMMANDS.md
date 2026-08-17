# Commands (run from the repository root)

## Data preparation

```bash
python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose.yaml \
  --phase prepare_data
```

首次运行会下载约 2.86 GB 的 HR-LSPET 训练集。下载写入
`data/lsp_pose/lspet_dataset.zip.part` 并支持断点续传；下载或解压中断时，
直接重新执行上述命令。

## Electronic Teacher upper bound

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose.yaml \
  --phase teacher_train

CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose.yaml \
  --phase teacher_inference
```

## Optical Student

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose.yaml \
  --phase student_train

CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose.yaml \
  --phase student_inference
```

## One command and smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose.yaml \
  --phase all

CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_smoke.yaml \
  --phase all
```

## Tests

```bash
python -m pytest experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/tests -q
```

## Vision2 光电联合训练（正式配置）

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_vision2_hybrid.yaml --phase student_train
```

本配置不训练 Teacher、不使用 KD。phase/mask 学习率为 `1e-4`。checkpoint 为 `experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_vision2_hybrid/checkpoints/student_best_train_loss.pt`。

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task lsp --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_vision2_hybrid/checkpoints/student_best_train_loss.pt --session-dir experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/hardware_sessions/vision2_run1 --stage vision_expert --phase export

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task lsp --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_vision2_hybrid/checkpoints/student_best_train_loss.pt --session-dir experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/hardware_sessions/vision2_run1 --stage vision_expert --phase finetune --epochs 20

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task lsp --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/hardware_sessions/vision2_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/hardware_sessions/vision2_run1 --stage vision_global --phase export

CUDA_VISIBLE_DEVICES=4 python -m experiments.vision2_hybrid_dense.hardware_bridge --task lsp --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_vision2_hybrid.yaml --checkpoint experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/hardware_sessions/vision2_run1/checkpoints/after_vision_expert.pt --session-dir experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/hardware_sessions/vision2_run1 --stage vision_global --phase finetune --epochs 20
```

实验室步骤见 `experiments/vision2_hybrid_dense/HARDWARE_PROTOCOL.md`。
