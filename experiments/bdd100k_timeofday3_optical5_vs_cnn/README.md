# BDD100K TimeOfDay-3: Optical5 Enhanced Readout vs CNN

本实验是 BDD100K TimeOfDay-3 的端到端学习实验，不使用 Qwen、tokenizer、prompt、teacher/student、蒸馏或 MoE。它比较两个使用相同灰度输入和相同 train/validation/test 划分的模型：

1. `Optical5EnhancedReadout`：五次可微光学传播与平方律探测，加两层卷积、普通平均池化和轻量 MLP detector readout。
2. `ElectronicCNNBaseline`：不使用预训练权重的中等强度纯电子 CNN。

目标是判断五层光学传播模型在真实驾驶场景的 daytime/night/dawn_dusk 分类上是否具备可学习性，并与纯电子 CNN 做对照。

## 数据

BDD100K 原始 `train` 用作实验 train，原始 `val` 用作实验 test；train 内部再按类别分层切出 validation。标签 `dawn/dusk` 规范化为 `dawn_dusk`。其他 timeofday 标签写入 manifest 的 `ignored_non_timeofday3`。

数据准备器优先复用仓库其他 BDD100K 实验已有的 `_raw` 图片和标签，并以软链接/硬链接组织：

```text
data/bdd100k_timeofday3/
  train/{daytime,night,dawn_dusk}/
  test/{daytime,night,dawn_dusk}/
  timeofday3_manifest.json
```

只有找不到已有原始数据时才执行下载。

## 光学前向

五层之间传递归一化后的非负 detected intensity。训练前向不会将探测强度执行平方根重新编码。相位和可选幅度 mask 位于传播之前，使相位参数能够影响探测平面并获得梯度。

每个 epoch 都立即保存 history/latest、best/last checkpoint，并根据配置保存 validation predictions。Optical run 额外保存 phase masks、逐层 light fields 和 detector outputs。

光学配置支持训练期 `block_phase_bypass` dropout：默认从第 10 个 epoch 开始，以 8×8 block、`p=0.05` 将部分相位调制临时替换为透明旁路。该正则化只在 `train()` 生效，validation/test 自动关闭，并记录在 resolved config、model report 和 training history 中。
