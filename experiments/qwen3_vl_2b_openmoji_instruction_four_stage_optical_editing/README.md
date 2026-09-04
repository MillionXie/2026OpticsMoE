# OpenMoji 四级光学文本指令图像编辑

这是几何形状实验的语义升级版。输入是由 16 类 OpenMoji 日常对象组成的简单场景，文本指令控制 `add / replace / move / remove`，输出仍是 224×224 目标图像。

## 架构

```text
instruction -> frozen full Qwen3-VL-2B language model (offline cache)
            -> Language2: expert optics -> global optics
            -> 192-D condition

source image -> frozen Qwen patch/position stem
             -> prompt injection
             -> Vision2: expert optics -> global optics
             -> conditioned spatial editor
             -> electronic 6x6 category-grid + edit-grid decoder
             -> fixed OpenMoji alpha compositor -> target image
```

结构化 compositor 有意避免让小网络浪费容量重新学习固定 SVG 笔画；模型必须完成输入场景识别、语言操作解析、对象类别预测和空间关系推理。文本仍是唯一任务切换输入。

## 数据与评价

- 16 类：apple、bicycle、car、bus、dog、cat、bird、tree、flower、cup、book、phone、ball、umbrella、house、light bulb；
- 6×6 网格，每个场景 1–4 个类别唯一的对象；
- train/test 同分布、种子隔离，无 validation、无 OOD；
- 核心指标：changed-cell accuracy、foreground category accuracy、edit-grid IoU、object F1、scene exact match；
- 每个 epoch 完整测试，但不使用测试结果选择 checkpoint。

测试可视化按任务分别保存。每行均显示 sample ID、task、完整 prompt，以及 `INPUT | TARGET | PREDICTION | ERROR CELLS`；同时生成 `index.html`，不再把不同任务无标题混排。

## OpenMoji 许可

本实验固定使用 OpenMoji 17.0.0 官方 72×72 彩色 PNG 的 16 个子集。OpenMoji 图形采用 CC BY-SA 4.0：

> All emojis designed by OpenMoji – the open-source emoji and icon project. License: CC BY-SA 4.0.

来源：https://github.com/hfg-gmuend/openmoji/releases/tag/17.0.0

## 运行

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
PYTHON=/home/guest3/miniconda3/envs/xml/bin/python
EXP=experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing

$PYTHON -m experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing \
  --config $EXP/configs/pilot_gpu.yaml --phase all --device cuda
```

所有数据、runs、逐轮测试、checkpoint 和 gallery 都保存在本实验目录内部。

## Pilot 实跑结果

服务器物理 GPU 3 上的首轮 pilot 使用 5,000 个训练样本和 1,000 个测试样本，训练 20 个 epoch；每轮均对全部测试集评估。训练耗时约 1,004 秒，最终使用第 20 轮 EMA 权重独立测试：

| 范围 | changed-cell accuracy | edit-grid IoU | object F1 | scene exact match |
| --- | ---: | ---: | ---: | ---: |
| overall | 0.8765 | 0.7629 | 0.9016 | 0.664 |
| add | 0.820 | 0.6691 | 0.9229 | 0.528 |
| replace | 0.996 | 0.9740 | 0.9980 | 0.992 |
| move | 0.706 | 0.4505 | 0.6953 | 0.164 |
| remove | 0.984 | 0.9580 | 0.9901 | 0.972 |

最终任务分类准确率为 1.0，说明 prompt 能稳定完成任务切换。主要剩余瓶颈是 `move`：它要求同时清除旧格并在关系目标格恢复正确类别，比单格的 add/replace/remove 更难。

服务器产物位于：

```text
$EXP/data/pilot_gpu/
$EXP/runs/pilot_gpu/
```

其中 `runs/pilot_gpu/epoch_tests/epoch_001` 至 `epoch_020` 保存逐轮完整测试；`runs/pilot_gpu/test_examples/` 保存最终四类独立图册；`runs/pilot_gpu/checkpoints/last.pt`、`optical_phases.pt` 和 `test_metrics.json` 分别保存最终 checkpoint、光学相位和最终指标。
