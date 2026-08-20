# Caltech101 四层光电联合训练与新 17 µm 光路命令

所有服务器命令均从仓库根目录执行。新配置不会覆盖旧的 16 µm 实验。

## 1. 新光路正式训练

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um.yaml --phase train
```

该配置从头联合训练电子 Mixer、四组 phase、两个 router、四组 CCD readout、四个融合门和最终 retrieval head。Qwen 主体保持冻结，不读取电子预训练 checkpoint。

关键训练量：

- logical pitch：17 µm；
- phase LR：`5e-4`；router LR：`2e-4`；
- router balance/importance loss：`0.02/0.005`；
- 输入、phase、CCD 的逻辑错位范围均为 `±12 px`；
- 训练输出：`experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um`。

## 2. 导出训练完成后的四组相位 BMP

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.export_phase_bmps --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um/ema_best_train_loss_checkpoint.pt --output-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um/hardware_phase_export
```

输出包括：

```text
hardware_phase_export/
├── compact_phase/     # 4 张 478×478 逻辑相位 PNG
├── phase_bmp/         # 4 张 1920×1200 原生相位 SLM BMP
└── phase_export_report.json
```

逻辑 phase 仍为 `478×478 @ 17 µm`。导出器根据 `17/8=2.125` 映射为约 `1016×1016 @ 8 µm`，沿用原来的纵向翻转，再放到配置中心 `(980,590)`。修改 YAML 的 `hardware.phase_mask.center_x/center_y` 后重新执行即可，不需要重新训练。

## 3. 生成两块 SLM 的对齐图案

```bash
python -m experiments.hardware_sdk.generators.dual_slm_alignment --config experiments/hardware_sdk/generators/slm_patterns/configs/dual_slm_17um_8um.yaml
```

输出位置：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/dual_slm_17um_8um_alignment/
├── amplitude_bmp/     # 1024×1024，17 µm，一像素对一像素
├── phase_bmp/         # 1920×1200，8 µm，保持纵向翻转
└── alignment_manifest.json
```

重点配对：

```text
amplitude checker block 8  ↔ phase checker block 17
amplitude checker block 16 ↔ phase checker block 34
amplitude checker block 32 ↔ phase checker block 68
```

相位中心需要改变时，修改 `dual_slm_17um_8um.yaml` 中的 `phase_slm.center_xy` 后重新生成。

## 4. 逐层实验导出与微调

下文使用：

```text
SESSION=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/four_layer_17um_run1
CKPT=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um/ema_best_train_loss_checkpoint.pt
```

以第一层 Vision expert 为例，服务器生成478×478紧凑 amplitude、紧凑 phase，并直接生成完整 phase BMP：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/four_layer_17um_run1 --stage vision_expert --phase export
```

实验室电脑把 amplitude 一对一放入新 `1024×1024` 输入 SLM：

```bash
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir SESSION/01_vision_expert/compact_amplitude --output-dir SESSION/01_vision_expert/amplitude_to_play --slm-width 1024 --slm-height 1024 --scale-factor 1 --center-x 512 --center-y 512
```

本层完整相位文件已经位于：

```text
SESSION/01_vision_expert/phase_to_play/vision_expert.bmp
```

如果只传输了紧凑 phase，也可以在实验室重建：

```bash
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir SESSION/01_vision_expert/compact_phase --output-dir SESSION/01_vision_expert/phase_to_play --slm-width 1920 --slm-height 1200 --logical-pixel-pitch-um 17 --slm-pixel-pitch-um 8 --center-x 980 --center-y 590
```

采集后的 `478×478 uint8` CCD 使用相同 basename 放入本层 `ccd_captured/`，上传服务器后微调：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint_17um.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint_17um/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/four_layer_17um_run1 --stage vision_expert --phase finetune --epochs 20
```

后续 checkpoint 链：

```text
vision_global   ← checkpoints/after_vision_expert.pt
language_expert ← checkpoints/after_vision_global.pt
language_global ← checkpoints/after_language_expert.pt
```

必须按 `export → 实验采集 → finetune` 顺序逐层推进。后续层 amplitude 依赖前一层实测 CCD，不能在尚未采集前提前导出。

## 5. 旧 16 µm 光路

旧配置和旧整数放大命令仍然保留：

```text
configs/release/caltech101_four_layer_optical_joint.yaml
--scale-factor 2
```
