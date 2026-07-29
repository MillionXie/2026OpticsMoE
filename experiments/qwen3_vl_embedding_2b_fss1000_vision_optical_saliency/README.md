# FSS-1000 Qwen Vision Optical Saliency

该实验把 FSS-1000 的所有类别统一视为前景，完成类别无关二值显著性分割：

```text
RGB image [B,3,224,224] -> binary mask logits [B,1,224,224]
```

## 数据划分

- 使用官方 240 个 test 类别。
- 其余类别合并为训练集。
- train/test 类别严格不相交。
- 不创建 validation。
- checkpoint 仅根据最低训练损失保存；每个 epoch 的 test 指标只是观察值。
- RGB 与 mask 使用相同几何增强，mask 始终使用 nearest-neighbor 插值。
- 源图像与 mask 几何不一致的损坏样本会被隔离，不会强行 resize。

## 电子 Teacher

```text
224x224 RGB
-> frozen Qwen3-VL-Embedding-2B Vision
-> final pre-merger spatial hidden
-> lightweight electronic segmentation head
-> 224x224 mask logits
```

Qwen 全部冻结，只训练分割头。保留的正式 run 中包含 Teacher checkpoint
以及最终 mask-logit cache。

## 最终单层 Optical Student

```text
224x224 RGB
-> frozen Qwen patch/position embedding
-> Linear(Dv,224) + LayerNorm + Softplus
-> electronic Top-4 router
-> routed amplitude copies on 16 expert regions
-> one 224x224 phase-only mask per expert
-> 10 cm propagation
-> square detection + per-expert LayerNorm + ReLU
-> routing-weight reload and unselected-expert zeroing
-> 986x986 global phase
-> 10 cm propagation
-> 986x986 CCD active ROI
-> adaptive pooling / LayerNorm / ReLU to 224x224
-> restore token spatial grid from image_grid_thw
-> lightweight segmentation head
-> 224x224 mask logits
```

不运行 language model，不使用文本 instruction，不使用全局图像池化。
未参与分割路径的 hidden output adapter 保持冻结。

## 最终从零训练配置

`configs/fss1000_saliency_single_layer_from_scratch_100ep.yaml` 是唯一推荐的
正式 Student 配置：

- Student 不加载任何旧 Student checkpoint；
- phase raw parameter 使用零初始化；
- 训练 100 epoch，batch size 16；
- BCE + Dice + soft-IoU + boundary loss；
- 使用对齐 crop/flip 后的 Teacher final-mask KD；
- phase dropout 与 weight decay 关闭；
- cosine learning-rate decay；
- 每 10 epoch 保存 checkpoint；
- 训练完成后自动加载最低训练损失 checkpoint 并保存最终测试结果。

## 可视化输出

最终 run 的 `figures/` 至少包含：

- `student_training_curves.png`
- `student_examples/`：输入、GT、概率图、二值预测和误差图
- `failure_cases/`：最低 IoU 样本
- `optical_parameters/expert_phase_overview.png`
- `optical_parameters/global_phase.png`
- `optical_debug_examples/`：
  - optical input field
  - routed amplitude-SLM canvas
  - expert-stage detector intensity
  - physical CCD intensity
  - 224x224 detector readout
  - Top-4 routing weights and selected experts

详细命令见 `RUN_COMMANDS.md`。
