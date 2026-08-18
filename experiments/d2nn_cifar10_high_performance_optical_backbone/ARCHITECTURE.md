# A01 架构

输入 CIFAR RGB 强度图先转为振幅并插值到 128×128。三个颜色通道各自使用独立相位面，但共享相同波长和传播几何。每个 OEO stage 执行：

```text
phase-only modulation -> angular-spectrum propagation -> CCD intensity
-> spatial standardization -> ReLU -> RMS balance -> constrained residual reload
```

共使用 8 个 stage。每层光学分支占比从 0.50 开始学习且不得低于 0.35。最后将三路振幅池化到 8×8，再由 LayerNorm、512 维 GELU 隐层和线性分类器输出类别。

该模型仍是混合光电网络：传播与相位调制是光学算子，CCD 后归一化、激活、重载和最终分类头是电子算子。因此“光学处理比例”不只用参数量表述，还必须看关闭光路和破坏已学习相位后的性能下降。
