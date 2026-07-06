# 五层连续光传播对照组

## 实验问题

该对照用于回答：在输入、五个 masks、传播距离、区域 detector、电子 readout 和训练 loss 都相同时，层间的平方律探测与电子非线性是否有助于 TimeOfDay-3 分类。

## O-E-O 实验组

`model_type=optical5_enhanced`

```text
grayscale encoding
 -> mask 1 -> propagation -> |E|² -> normalize -> bias -> ReLU -> reload
 -> mask 2 -> propagation -> |E|² -> normalize -> bias -> ReLU -> reload
 -> mask 3 -> propagation -> |E|² -> normalize -> bias -> ReLU -> reload
 -> mask 4 -> propagation -> |E|² -> normalize -> bias -> ReLU -> reload
 -> mask 5 -> propagation -> |E|² -> normalize -> bias -> ReLU
 -> three class regions + electronic readout
```

它有五次平方律探测和五次 detector ReLU。

## 连续光传播对照组

`model_type=optical5_continuous`

```text
grayscale encoding -> one complex-field load
 -> mask 1 -> angular spectrum propagation
 -> mask 2 -> angular spectrum propagation
 -> mask 3 -> angular spectrum propagation
 -> mask 4 -> angular spectrum propagation
 -> mask 5 -> angular spectrum propagation
 -> final |E|² -> normalize -> bias -> ReLU
 -> three class regions + electronic readout
```

层间张量始终为 complex64 field `[B,256,256]`。没有以下操作：

- intermediate square-law detection；
- intermediate intensity normalization；
- intermediate ReLU；
- intensity-to-field reload；
- electronic residual bypass。

逐层 light-field 可视化会计算临时的 `|E|²`，但该计算只在 diagnostics 路径中发生，不会反馈给下一层，也不参与训练 forward。

## 保持一致的变量

两种光学模型均使用：

- 灰度 224×224 输入；
- RMS 输入归一化；
- 256×256 optical field；
- 400×400 propagation padding；
- 532 nm 波长；
- 17 µm pixel pitch；
- 每段 5 cm 传播距离；
- 五个独立 256×256 phase masks；
- 五个独立 amplitude masks；
- 相同 phase initialization 和 phase dropout；
- 相同三个 48×48 class regions；
- 相同两层卷积 + average pooling + MLP readout；
- 相同数据划分、优化器、学习率和 loss 权重。

## 不可避免的参数差异

O-E-O 版本每层有一个 trainable scalar detector bias，共五个。Continuous 版本只有最终 detector 的一个 trainable scalar bias。除此之外，phase mask、amplitude mask 和 readout 参数规模相同。

## Loss

两种版本完全相同：

```text
L_total = CE(electronic_logits, label)
        + 1.0 * CE(region_logits, label)
        + 0.1 * [-log(detector_region_energy / total_energy)]
```

第三项是 detector 能量集中比例损失，不是直接最大化绝对光强。使用比例可以避免通过整体放大数值投机降低 loss。

## 公平比较指标

至少比较：

- test top-1；
- macro-F1；
- balanced accuracy；
- detector region 独立准确率；
- mean detector energy fraction；
- mean target-region energy fraction；
- 每类 recall；
- 参数量和训练时间。

两个实验必须使用不同 output directories，不能复用 checkpoint。
