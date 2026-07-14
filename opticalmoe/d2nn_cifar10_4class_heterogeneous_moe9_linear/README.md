# CIFAR-10 Heterogeneous Optical MoE9 (Linear)

这是一个独立的 CIFAR-10 四分类异构光学 MoE 实验。它保留参考工程的输入相关 top-k router、480×480 canvas、450×450 有效区域、3×3 专家布局、global FC、探测器区域 loss、日志和可视化，但九个专家不再都是同一种 D2NN。

## 关键限制

专家路径中始终传递 `complex64` 复光场：

```text
[B,120,120] complex field -> expert linear coherent operator -> [B,120,120] complex field
```

专家内部不包含：

- ReLU、GELU 或其他激活；
- 平方探测后重新编码；
- 强度归一化或样本相关功率归一化；
- 电子分类层。

Fiber 的 Gaussian modes 只在模型构造时分别做固定 L2 归一化。这不是对输入样本的功率归一化。相位约束、Fiber 有界模式透射和专家固定标量的参数化不会改变专家对输入复光场的线性关系。

## 数据流

```text
CIFAR-10 RGB
-> grayscale
-> resize 100x100
-> zero pad 120x120
-> centre on 480x480 canvas
-> input-dependent electronic router (top-k=3)
-> prompt amplitude routes coherent copies to 3x3 expert entrances
-> crop nine [B,120,120] complex local fields
-> heterogeneous expert bank
-> reassemble one 480x480 complex canvas
-> 5 cm propagation
-> trainable 450x450 global FC phase plane
-> 10 cm propagation
-> square-law detector
-> four target detector regions / detector-plane MSE
```

只有最终 detector 执行平方探测。专家 bank 到 global FC 之前没有探测和非线性。

## 默认专家布局

YAML 中 `experts.types` 按 row-major 配置：

```text
D2NN      Fourier    Fiber
Fiber     D2NN       Fourier
Fourier   Fiber      D2NN
```

### D2NNExpert

- 五个独立的 120×120 phase-only masks；
- 相邻 phase plane 之间使用角谱传播；
- 每次局部传播前从 120×120 zero-pad 到 180×180，传播后中心裁剪回 120×120；
- padding 只用于传播计算，不是可训练区域。

### FourierExpert

```text
ifftshift -> fft2(norm="ortho") -> fftshift
-> phase-only Fourier mask
-> ifftshift -> ifft2(norm="ortho") -> fftshift
```

不做幅度归一化。

### FiberArrayExpert

- 固定的 10×10 Gaussian single-mode bank；
- 每个 mode 在构造时独立归一化；
- 对复光场执行相干投影；
- 施加逐模式可训练相位和 `[0,1]` 有界幅度透射；
- 相干重构复光场，不经过强度域。

## 参数与 routing 记录

`architecture_report.json` 保存：

- 每个专家的类型与参数量；
- D2NN phase、Fourier phase、Fiber mode phase/amplitude 数量；
- 每种专家类型的合计参数；
- global FC、router 和总可训练参数量。

`metrics/epoch_metrics.csv` 每个 epoch 保存九个专家的：

- selection rate；
- mean routing weight；
- mean input optical power；
- mean output optical power。

同时保存 D2NN/Fourier/Fiber 三种类型聚合后的平均 selection rate、routing
weight、输入光功率和输出光功率。

## 可视化

每个可视化 epoch 保存：

- 输入、prompt amplitude/phase、专家入口；
- 重拼后的 expert-bank output intensity/phase；
- D2NN、Fourier、Fiber 各一份代表输出的 intensity/phase；
- 九个专家各自的输出强度和相位；
- D2NN/Fourier/Fiber 的可训练光学参数；
- global FC 和最终 detector。

## 类型感知的 SLM 导出

`slm_export_best/manifest.json` 明确区分：

- D2NN：按深度导出仅填充 D2NN cells 的 phase mosaic；
- Fourier：导出明确标记为 frequency-domain 的 phase mask；
- Fiber：导出 Gaussian mode centres、sigma、mode phase 和 mode amplitude 的 `.pt` 参数文件。

Fourier/Fiber 不会被伪装成普通空间 D2NN phase mask。
