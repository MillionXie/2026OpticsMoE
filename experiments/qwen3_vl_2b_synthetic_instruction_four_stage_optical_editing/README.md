# Qwen3-VL-2B 四级光学指令图像编辑

本实验实现一个可控的生成式任务：

```text
简单几何图像 + 英文文本指令 -> 编辑后的 224x224 图像
```

它不是完整 diffusion，也不要求未见组合泛化。训练集与测试集使用相同的任务、模板和组合分布，仅按确定性随机种子划分；没有 validation。为了直接观察学习效果，每个 epoch 结束后都使用当轮 EMA 权重完整测试一次，但测试指标不参与 checkpoint 选择。

## 任务

四类样本均衡生成，文本是模型切换任务和确定操作参数的唯一输入：

1. `attribute`：改变指定物体颜色或形状；
2. `add`：相对指定物体在 left/right/above/below 添加物体；
3. `remove`：删除指定物体；
4. `edge`：输出白底黑线的全场景边缘图。

每个样本都保存 `source.png`、`target.png`、像素类别图、编辑/保留区域掩码和完整 `scene.json`。场景中每个颜色-形状描述唯一，避免语言指代歧义。

## 模型

```text
instruction
  -> frozen full Qwen3-VL-2B language model (offline contextual-token cache)
  -> Language2: MoE4 expert optics -> global optics
  -> mean+max condition, 192-D

source image
  -> frozen Qwen patch/position stem, 196 x 1024
  -> inject the 192-D language condition before the visual router
  -> Vision2: MoE4 expert optics -> global optics
  -> prompt-conditioned spatial editor
  -> electronic structured canvas decoder
  -> 8-class palette logits + edit-mask logits
  -> preserve source outside the predicted edit mask
```

因此语言模型仍然存在，而且使用完整冻结 Qwen language model 的上下文化隐藏状态，不是简单词袋或任务 ID。训练时不重复加载 2B 模型：所有唯一 prompt 只编码一次并缓存。实际紧凑模型依次包含四个物理阶段：`language_expert`、`language_global`、`vision_expert`、`vision_global`。

电子 decoder 是有意保留的：光路负责低维语义/空间变换，decoder 负责把 14x14 表征稳定恢复为 224x224 离散画布。辅助 task head 只提供训练监督，不作为任务输入。

## 配置

- `configs/smoke.yaml`：32 train / 8 test，检查完整 GPU 流程；
- `configs/pilot_gpu.yaml`：512 train / 128 test，快速判断任务是否可学；
- `configs/release.yaml`：20,000 train / 2,000 test，正式训练。

建议先完成 smoke 和 pilot，再根据曲线决定是否直接运行 release。所有配置固定 `224x224` 输入、`7x7` 逻辑网格、8 色离散输出。

## 运行

从仓库根目录执行：

```bash
export CUDA_VISIBLE_DEVICES=3
PYTHON=/home/guest3/miniconda3/envs/xml/bin/python
EXP=experiments/qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing

$PYTHON -m experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing \
  --config $EXP/configs/smoke.yaml --phase all --device cuda
```

也可以依次执行 `prepare_data`、`cache_prompts`、`train`、`test`。命令只使用本地 Hugging Face snapshot，不从网络下载 Qwen。

## 结果与证据

数据保存在配置的 `dataset.data_dir`。所有运行证据都保存在本实验目录自己的 `runs/` 内，例如正式实验为 `experiments/qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing/runs/v1`：

- `dataset_summary.json`、`train.jsonl`、`test.jsonl`、逐样本图像与场景标签；
- `prompt_hidden.pt/json`：冻结 Qwen 的 prompt 隐状态与元数据；
- `resolved_config.json`、`environment.json`、`model.json`；
- `train_log.csv`、`training_summary.json`；
- `test_log.csv`：每个 epoch 的 overall 与四个子任务指标；
- `epoch_tests/epoch_NNN_metrics.json`、`epoch_NNN_predictions.jsonl`、`epoch_NNN_examples.png`；
- `checkpoints/last.pt`、`checkpoints/best_train_loss.pt`、`optical_phases.pt`；
- `test_metrics.json`、`test_predictions.jsonl`、`test_examples.png`。

正式结果固定使用最后一个 epoch 的 EMA 权重，不使用测试集选 checkpoint。`training.resume: true` 时会从 `checkpoints/last.pt` 恢复模型、EMA、optimizer 和 scheduler，并补测尚未记录指标的已有 epoch。
