# 四层联合训练与硬件命令

所有服务器命令从仓库根目录执行；实验室电脑也只需同步 `experiments/hardware_sdk`。

## 1. 四层联合训练

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint.yaml --phase train
```

该命令不读取电子预训练 checkpoint。电子 Mixer、router、phase、CCD readout、四个门控
和最终 retrieval readout 一起训练。

下文令 `SESSION=experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/four_layer_run1`，初始 `CKPT` 为联合训练输出的
`ema_best_train_loss_checkpoint.pt`。

## 2. 每一层的固定流程

以 Vision expert 为例，服务器生成紧凑 payload：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/four_layer_run1 --stage vision_expert --phase export
```

把本层 `compact_amplitude` 和 `compact_phase` 传到实验室，重建完整 SLM BMP：

```bash
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir SESSION/01_vision_expert/compact_amplitude --output-dir SESSION/01_vision_expert/amplitude_to_play --slm-width 1920 --slm-height 1080 --scale-factor 2
python -m experiments.hardware_sdk.workflows.reconstruct_slm --input-dir SESSION/01_vision_expert/compact_phase --output-dir SESSION/01_vision_expert/phase_to_play --slm-width 1920 --slm-height 1200 --scale-factor 2 --center-x 980 --center-y 590
```

实验室的 `tucam_windows.yaml` 已配置为直接保存 `478×478 uint8`。采集文件必须按相同
basename 放入本层 `ccd_captured/`。手动加载本层 `phase_to_play/vision_expert.bmp`
后，可直接指定 session 目录采集，不需要再复制中间文件：

```bash
python -m experiments.hardware_sdk.workflows.acquire_folder --config experiments/hardware_sdk/configs/tucam_windows.yaml --input-dir SESSION/01_vision_expert/amplitude_to_play --output-dir SESSION/01_vision_expert/ccd_captured --clear-output
```

把 `ccd_captured` 上传服务器相同目录后微调：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.hardware_bridge --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/runs/caltech101_four_layer_moe4_joint/ema_best_train_loss_checkpoint.pt --session-dir experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/hardware_sessions/four_layer_run1 --stage vision_expert --phase finetune --epochs 20
```

微调只开放当前实测 CCD 之后的模块，电子、phase、router、最终 readout 分别沿用 YAML
中的学习率；已经实测的 CCD 保持确定性，尚未采集的下游仿真光路继续使用 phase
dropout、错位、增益和噪声扰动。完成后测试性能写入本层 `finetune_metrics.json`，新 checkpoint 写入
`SESSION/checkpoints/after_vision_expert.pt`。

## 3. 后三层 checkpoint 链

依次将 `--stage` 和输入 checkpoint 设置为：

```text
vision_global:
  checkpoints/after_vision_expert.pt

language_expert:
  checkpoints/after_vision_global.pt

language_global:
  checkpoints/after_language_expert.pt
```

每层仍执行一次 `export → 重建/采集/上传 → finetune`。最终 checkpoint 为：

```text
checkpoints/after_language_global.pt
```

禁止提前导出后续层：后续层 amplitude 必须由已采集 CCD 和上一层微调 checkpoint 重新
前向得到。
