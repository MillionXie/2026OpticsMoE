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
