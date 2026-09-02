# 运行命令

以下命令全部从仓库根目录执行。正式比较使用
`ema_best_observed_test_checkpoint.pt`，因为本实验按要求以周期性 test Top-1 选轮。

## 1. 四组训练

正式训练在 Linux 服务器运行。可分别放在四张空闲 GPU；如果卡被
其他任务占用，请只改 `CUDA_VISIBLE_DEVICES`，不要改配置。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation --config experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/configs/release/alpha_free.yaml --phase train
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation --config experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/configs/release/alpha_low_lt_0p5.yaml --phase train
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation --config experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/configs/release/alpha_high_gt_0p5.yaml --phase train
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation --config experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/configs/release/electronic_only.yaml --phase train
```

Windows PowerShell 本地调试时，先执行
`$env:CUDA_VISIBLE_DEVICES="0"`，再执行对应 `python -m ...` 命令。

每 5 epoch 日志才会出现一次有限的 `test_top1/ema_test_top1`；其他 epoch 为
`nan` 是预期行为。最佳文件是：

```text
runs/<variant>/ema_best_observed_test_checkpoint.pt
runs/<variant>/metrics/ema_best_observed_test.json
```

## 2. 同一 checkpoint 的完整光电结果

以 low 组为例：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation --config experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/configs/release/alpha_low_lt_0p5.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/runs/alpha_low_lt_0p5/ema_best_observed_test_checkpoint.pt --ablation none --evaluation-output-dir experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/runs/eval_low_full
```

## 3. 同一 checkpoint 直接去光（不微调）

`remove_optical` 会跳过四次光学传播，只运行对应电子分支，并把电子输出恢复到其
原 RMS。它不是另训一个模型。

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation --config experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/configs/release/alpha_low_lt_0p5.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/runs/alpha_low_lt_0p5/ema_best_observed_test_checkpoint.pt --ablation remove_optical --evaluation-output-dir experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/runs/eval_low_remove_optical
```

应向老师汇报：

```text
光的因果贡献 = full Top-1 - same-checkpoint remove-optical Top-1
```

也可做辅助的去电子方向对照：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation --config experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/configs/release/alpha_low_lt_0p5.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/runs/alpha_low_lt_0p5/ema_best_observed_test_checkpoint.pt --ablation remove_electronic --evaluation-output-dir experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/runs/eval_low_remove_electronic
```

注意：`remove_electronic` 只移除四个融合点的电子 Mixer 输出；Qwen 的冻结输入
编码、光场编码器、CCD readout 和最终 retrieval head 仍是电子网络，不能称为
“全光学模型”。

## 4. 结果位置

每个评估目录中重点查看：

```text
metrics/evaluation_summary.json
fusion_diagnostics_last_batch.json
confusion_matrix.png
```

诊断中四层的 `fused_to_electronic_rms_ratio` 应接近 1；完整 hybrid 的
`post_optical_to_electronic_rms_ratio` 应为 1。若不是，先停止比较并检查 padding
或 checkpoint/config 是否混用。

导出最佳 checkpoint 中实际学到的四个 alpha（以 low 组为例）：

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.checkpoint_report --config experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/configs/release/alpha_low_lt_0p5.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/runs/alpha_low_lt_0p5/ema_best_observed_test_checkpoint.pt --output experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/runs/alpha_low_lt_0p5/best_alpha_report.json
```
