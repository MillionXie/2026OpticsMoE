# MNIST-4 single-layer 10 cm v2

这是一个不覆盖既有 10 cm 正式项目的独立 v2。它保留 notebook 的 `raw_phase=0`
和 `2π·sigmoid(raw_phase)`，但将四个探测区按 `478/400` 映射到真实 478×478
CCD，并加入 k 空间滤波和训练期错位注入。

## 光路与张量

```text
MNIST 28x28
  -> bicubic 336x336, zero pad 32 each side to 400x400
  -> ToTensor [0,1] amplitude (no sqrt or normalization)
  -> zero pad 39 each side: 478x478 input field
  -> cardinal input shift (train only, <=2 px)
  -> multiply shifted 478x478 phase, phase=2π sigmoid(raw), raw init=0
  -> zero pad to physical canvas 518x518
  -> zero pad to numerical grid 1024x1024
  -> circular k-space cutoff, theta_max=0.65 deg
  -> 10 cm angular-spectrum propagation
  -> crop physical 478x478 CCD field
  -> cardinal pre-CCD field shift (train only, <=2 px)
  -> CCD photodetection |E|^2
  -> four raw 59x59 region sums, argmax
```

输入、相位和 CCD 前错位相互独立，只从零位或上下左右四个方向采样，不使用循环
`roll`，移出边界的部分以零填充。正式验证、测试和 BMP 导出自动关闭随机错位。

## Raw CCD loss

对于每张样本，正确 target 区面积为 `59²=3481`，其余背景面积为
`478²-3481=225003`。使用原始强度：

```text
L_target = mean_target((I - 1)^2)
L_background = mean_background(I^2)
L = 1.0 * L_target + 0.5 * L_background
```

这里的 mean 只是固定像素数上的 loss reduction，不依赖单帧曝光或总能量。CCD
输出之后严禁：单帧归一化、LayerNorm、激活、log、动态拉伸、背景扣除和 resize。
硬件分类同样只用四个 raw region sums。

几何和 notebook 原始代码的逐项证据见 `NOTEBOOK_AUDIT.md`。
