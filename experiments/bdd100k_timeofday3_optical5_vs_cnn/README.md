# BDD100K TimeOfDay-3: Optical5 O-E-O vs Continuous Optical vs CNN

本实验是 BDD100K TimeOfDay-3 的端到端学习实验，不使用 Qwen、tokenizer、prompt、teacher/student、蒸馏或 MoE。它比较三个使用相同灰度输入和相同 train/validation/test 划分的模型：

1. `Optical5EnhancedReadout`：每层都执行传播、平方律探测、归一化、ReLU 和重新加载，共五次 O-E-O 转换。
2. `Optical5Continuous`：复光场连续经过五组 mask 和角谱传播，层间不探测、不归一化、不加非线性、不重新加载，只在第五次传播后探测一次。
3. `ElectronicCNNBaseline`：不使用预训练权重的中等强度纯电子 CNN。

目标是判断五层光学模型在真实驾驶场景的 daytime/night/dawn_dusk 分类上是否具备可学习性，并分离“连续衍射传播”和“层间光电非线性”的贡献，再与纯电子 CNN 做对照。

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

## 两种光学前向

O-E-O 版本五层之间传递归一化后的非负 detected intensity。每层都有平方律探测和 ReLU，但不会将探测强度执行平方根重新编码。

Continuous 版本只在输入端将灰度编码加载成复光场，随后连续执行五次 `mask -> angular spectrum propagation`。五层之间保持 complex field，仅在最后执行一次 `|E|²`、均值归一化和 detector ReLU。详细对照见 `CONTINUOUS_CONTROL.md`。

两种版本都使用五个独立 phase masks、五个可选 amplitude masks、完全相同的物理参数、phase dropout、三区域 detector 和电子 readout。

## 当前 loss

最终 256×256 探测面中央水平排列三个固定 48×48 方格，顺序为 `daytime`、`night`、`dawn_dusk`：

```text
L_total = L_classification + 1.0 * L_region + 0.1 * L_concentration
```

- `L_classification`：电子 readout 三分类交叉熵。
- `L_region`：三个类别方格的相对能量交叉熵，要求真实类别方格能量最大。
- `L_concentration=-log(E_regions/E_total)`：要求光能进入三个 detector 方格，而不是散落在方格外。它约束的是归一化能量比例，不是无界的绝对光强。

电子 readout 继续处理完整末端强度图，并额外接收三个方格内的相对能量。区域设计见 `DETECTOR_REGIONS.md`。

每个 epoch 都立即保存 history/latest、best/last checkpoint，并根据配置保存 validation predictions。Optical run 额外保存 phase masks、逐层 light fields 和 detector outputs。

光学配置支持训练期 `block_phase_bypass` dropout：默认从第 10 个 epoch 开始，以 8×8 block、`p=0.05` 将部分相位调制临时替换为透明旁路。该正则化只在 `train()` 生效，validation/test 自动关闭，并记录在 resolved config、model report 和 training history 中。

区域 readout 改变了模型结构，旧版没有类别区域的 checkpoint 与当前代码不兼容，需要重新训练。
