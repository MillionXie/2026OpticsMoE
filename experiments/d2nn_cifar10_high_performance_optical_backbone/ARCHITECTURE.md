# RGB、高光学残差与电子增强架构

输入 CIFAR RGB 强度图先转为振幅并插值到 128×128。三个颜色通道各自使用独立相位面，但共享相同波长和传播几何。每个 OEO stage 执行：

```text
phase-only modulation -> angular-spectrum propagation -> CCD intensity
-> spatial standardization -> ReLU -> RMS balance -> constrained residual reload
```

共使用 8 个 stage。A01--A06 的每层光学分支占比从 0.50 开始学习且不得低于 0.35；
A07--A13 将硬下限提高到 0.50。最后将三路振幅池化到 8×8，再由 LayerNorm、512 维 GELU
隐层和线性分类器输出类别。

该模型仍是混合光电网络：传播与相位调制是光学算子，CCD 后归一化、激活、重载和最终分类头是电子算子。因此“光学处理比例”不只用参数量表述，还必须看关闭光路和破坏已学习相位后的性能下降。

## RGB 三通道究竟如何传播

输入张量为 `[B, 3, 32, 32]`，像素值被解释为强度。双三次插值到 128×128 后执行平方根，得到光场振幅。R/G/B 三个通道在每个 stage 各有一张独立的 128×128 phase-only mask；三个通道共享波长、像素尺寸和传播距离，但传播过程中不互相混合。旧 A01–A07 直到最终 MLP head 才发生 RGB 通道交互。

## 光学分支与旁路的数值平衡

每个样本、每个颜色通道分别在二维空间上处理：

```text
optical = ReLU((intensity - spatial_mean) / sqrt(spatial_variance + eps))
optical = optical / spatial_RMS(optical)
skip    = amplitude / spatial_RMS(amplitude)
output  = alpha * optical + (1-alpha) * skip
```

因此 `alpha` 表示两个已进行 RMS 平衡的分支之间的显式混合权重，而不是被未经控制的数值尺度掩盖的系数。A07 及之后候选使用
`alpha = 0.5 + 0.5*sigmoid(logit)`，逐层保证 `alpha >= 0.5`。

## A08–A10 的受限电子残差

电子增强只发生在权重不超过 0.5 的 bypass 内，之后仍与同层光学输出门控合并：

```text
processed_skip = RMSNorm(ReLU(skip + s * electronic_transform(skip)))
output = alpha * optical + (1-alpha) * processed_skip
alpha >= 0.5, 0 <= s <= 0.25
```

- A08：逐像素 `1×1` bottleneck，只做 RGB 通道混合；
- A09：增加每通道 `3×3` depthwise 空间卷积，再做逐像素通道混合；
- A10：在 A09 上，把早期 stage 0/1/2 的输出分别送到后期 stage 7/6/5 的 bypass，长跳连权重限制在 0.25 以内。

这种 U-Net-like 连接不是完整 U-Net，因为没有编码器下采样和解码器上采样；它只借用了对称跨深度 feature skip。它可能提升优化，也可能形成新的电子旁路，因此必须同时报告 `optical-off`、`phase-random`、`phase-shuffle`、`electronic-skip-off` 和 `long-skip-off`。

## 读出头

当前正式头是：

```text
AdaptiveAvgPool(8×8) -> flatten(192) -> LayerNorm -> Linear(192,512)
-> GELU -> Dropout -> classifier
```

第二轮筛选过的 `conv` 头是：

```text
pool -> 3×3 Conv -> GroupNorm -> GELU
-> stride-2 3×3 Conv -> GroupNorm -> GELU
-> 2×2 pool -> MLP -> classifier
```

它与当前 MLP 头参数量同量级，并保留粗粒度空间布局。另一个 `dual_pool` 候选把 8×8
average-pooled 与 max-pooled 光学图样拼接后送入 MLP，用于同时读取平均能量和局部衍射峰。
A11/A12 在统一协议第 20 轮的 best validation 分别只有 57.88%/58.30%，明显落后原 MLP
候选 A08 的后续曲线，因此均止损。当前保留 104,330 参数的原 MLP 头；这是验证集选择，
不是因为卷积头在一般情况下必然更差。

## 电子预算扩展 A13

A08 的电子残差只有 592 个参数，是极小有效性基线。根据实验室可接受的 1--2M 电子参数
预算，A13 把旁路先平均池化到 32×32，在低分辨率使用 64 通道的两层空间变换，再双线性
上采样回 128×128。电子修正仍以 `s<=0.25` 残差加入旁路，随后通过 `alpha>=0.5` 的门控
与同层光学输出合并。八层电子残差 312,336 参数，加旧 MLP head 后总电子参数 416,666。
这样利用了几十万级合理容量，同时避免直接在 128×128 上运行 64 通道全卷积。

A13 最佳 checkpoint 的 test/optical-off/random-phase/shuffled-phase/electronic-skip-off Top-1
分别为 72.18%/13.39%/12.48%/12.76%/28.06%，normalized optical dependence 为 94.55%。
八层 gate 的最小值/均值为 50.05%/51.18%，因此门控约束和光学因果消融都通过。其估算电子
卷积量仍有 317.82M MAC/sample，所以 `alpha>=0.5` 只表示经过 RMS 平衡后的分支混合权重，
不能称为实际光学计算比例、能耗比例或硬件加速比。
