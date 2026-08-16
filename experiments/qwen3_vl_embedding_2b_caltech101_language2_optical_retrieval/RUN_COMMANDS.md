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
`.pt/.npy/.tif/.tiff/.png`。推荐 CCD ROI 为 `448 x 448`，此时程序执行严格的
`2 x 2` block mean 得到逻辑 `224 x 224`。

## 3. 注册并检查实测 CCD

先在配置的 `hardware.ccd` 中设置 ROI、上下/左右翻转和尺寸注册方式，再执行：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval.hardware_bridge \
  --config experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/configs/release/caltech101_language2_optical_residual.yaml \
  --session-dir experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/hardware_sessions/language2_run1 \
  --phase prepare_ccd
```

处理结果写入 `ccd_registered/`，每张图同时生成 JSON 审计记录。若尺寸不是
`224 x 224` 或 `448 x 448`，默认先中心裁成正方形，再按面积缩小/双线性放大到
`224 x 224`。若希望禁止任何近似缩放，将 `registration_mode` 改成 `strict`。

## 4. 用实测 CCD 微调下游电子部分

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval.hardware_bridge \
  --config experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/configs/release/caltech101_language2_optical_residual.yaml \
  --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/runs/caltech101_language2_optical_residual/ema_best_train_loss_checkpoint.pt \
  --session-dir experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/hardware_sessions/language2_run1 \
  --phase finetune
```

微调结果保存在 session 目录的 `hardware_finetuned_checkpoint.pt`。正式采集前，
可以追加 `--use-simulation --epochs 1`，先用导出的仿真 CCD 检查完整流程。
