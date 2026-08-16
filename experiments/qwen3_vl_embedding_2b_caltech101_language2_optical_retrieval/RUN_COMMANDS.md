# 运行命令

以下命令均在仓库根目录执行，不要把本文件当作 shell 脚本运行。

## 1. 仿真训练

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval \
  --config experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/configs/release/caltech101_language2_optical_residual.yaml \
  --phase train
```

## 2. 导出 Language Block 2 光路输入

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval.hardware_bridge \
  --config experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/configs/release/caltech101_language2_optical_residual.yaml \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/runs/caltech101_language2_optical_residual/ema_best_train_loss_checkpoint.pt \
  --session-dir experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/hardware_sessions/language2_run1 \
  --phase export
```

将实测 CCD 文件按 manifest 中相同 basename 放入 `ccd_captured/`，允许
`.pt/.npy/.tif/.tiff/.png`。推荐 CCD ROI 为 `448 x 448`，程序执行严格的
`2 x 2` block mean 得到逻辑 `224 x 224`，不会插值。

## 3. 用实测 CCD 微调下游电子部分

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval.hardware_bridge \
  --config experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/configs/release/caltech101_language2_optical_residual.yaml \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/runs/caltech101_language2_optical_residual/ema_best_train_loss_checkpoint.pt \
  --session-dir experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/hardware_sessions/language2_run1 \
  --phase finetune
```

微调结果保存在 session 目录的 `hardware_finetuned_checkpoint.pt`。正式采集前，
可以追加 `--use-simulation --epochs 1`，先用导出的仿真 CCD 检查完整流程。
