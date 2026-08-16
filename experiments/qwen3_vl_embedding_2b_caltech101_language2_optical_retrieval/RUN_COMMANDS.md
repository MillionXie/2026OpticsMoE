# 运行命令

以下命令均在仓库根目录执行；本文只是命令记录，不是 shell 脚本。

## 1. 仿真训练（Block 1 专家，Block 2 global）

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/configs/release/caltech101_language2_optical_residual.yaml --phase train
```

## 2. 导出 Language Block 2 全局光路输入与 mask

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/configs/release/caltech101_language2_optical_residual.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/runs/caltech101_language_two_block_moe4_dual_fusion/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/hardware_sessions/language2_run1 --phase export
```

程序先完成 Block 1 的专家 CCD readout、gate1 光电融合，再把融合结果重新编码。
`amplitude_to_play/*.bmp` 是重新编码后的 global 输入；`phase_mask/language_block2_global_phase.bmp`
都包含 `956×956` 的物理有效区。

## 3. 独立处理采集后的 CCD（不翻转）

先在 `experiments/hardware_sdk/configs/process_moe4_ccd_956.yaml` 设置统一
`roi_xywh` 和强度映射，然后执行：

```bash
python -m experiments.hardware_sdk.workflows.process_moe4_ccd --config experiments/hardware_sdk/configs/process_moe4_ccd_956.yaml --input-dir experiments/hardware_sdk/data/ccd_captured_raw --output-dir experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/hardware_sessions/language2_run1/ccd_captured
```

输出必须是与 manifest 同 basename 的 8-bit 灰度 `956×956` PNG。这里缩放的是
完整目标 ROI，不是中心裁剪 `224×224`，也不会做任何翻转。

## 4. 按配置翻转并注册为逻辑 CCD

```bash
python -m experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/configs/release/caltech101_language2_optical_residual.yaml --session-dir experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/hardware_sessions/language2_run1 --phase register_ccd
```

这一步仅执行 `hardware.ccd` 中的上下/左右翻转和严格 `2×2` block mean，输出
`ccd_registered/*.pt`（逻辑 `478×478`）及 JSON 审计记录。

## 5. 用实测 CCD 微调下游电子部分

```bash
CUDA_VISIBLE_DEVICES=6 python -m experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/configs/release/caltech101_language2_optical_residual.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/runs/caltech101_language_two_block_moe4_dual_fusion/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval/hardware_sessions/language2_run1 --phase finetune
```

结果保存为 `hardware_finetuned_checkpoint.pt`。正式采集前可追加
`--use-simulation --epochs 1` 检查下游流程。
