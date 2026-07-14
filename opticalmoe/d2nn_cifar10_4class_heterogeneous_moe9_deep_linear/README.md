# CIFAR-10 Deep Heterogeneous Optical MoE9 (Linear)

本工程是 `d2nn_cifar10_4class_heterogeneous_moe9_linear` 的独立加深版本。它不修改正在训练的浅层异构实验，也不修改原始同构 MoE 或公共模块。

## 不变部分

- CIFAR-10 前四类、灰度输入、100×100 resize 和 120×120 zero padding；
- 480×480 canvas、450×450 active area 和 3×3 专家布局；
- 输入相关 top-3 router 与 prompt 振幅路由；
- 450×450 global FC phase；
- 四区域 detector-plane MSE；
- 日志、checkpoint、可视化和 SLM 导出流程。

## 线性约束

九个专家均执行：

```text
[B,120,120] complex64 -> coherent linear optical operator -> [B,120,120] complex64
```

专家内部不存在平方探测、强度重编码、ReLU、LayerNorm 或逐样本功率归一化。Fiber mode distribution 的归一化仅用于记录指标，不参与前向光场计算。

## 默认布局

```text
D2NN      Fourier    Fiber
Fiber     D2NN       Fourier
Fourier   Fiber      D2NN
```

由 `expert_bank.assignments` 控制。

## D2NNExpert

```text
SpatialPhase1 -> padded ASM propagation
SpatialPhase2 -> padded ASM propagation
SpatialPhase3 -> padded ASM propagation
SpatialPhase4 -> padded ASM propagation
SpatialPhase5
```

共有五张独立 120×120 phase-only mask。传播时从 120×120 zero-pad 到 180×180，再中心裁剪回 120×120。

## Deep FourierExpert

```text
FourierConv1
-> finite-aperture padded propagation
-> FourierConv2
-> finite-aperture padded propagation
-> FourierConv3
-> SpatialPhase1
-> finite-aperture padded propagation
-> SpatialPhase2
```

每个 FourierConv 包含：

```text
120x120 field
-> zero-pad 180x180
-> centered fft2(norm="ortho")
-> crop a finite 120x120 frequency aperture
-> independent 120x120 phase-only frequency mask
-> embed into a zero 180x180 spectrum
-> centered ifft2(norm="ortho")
-> finite 120x120 spatial center crop
```

三张频域 mask 之间不能简单相乘折叠。原因是每一级之间都有空间中心裁剪和有限孔径传播；空间域截断在 Fourier 域不是对角算子，因此不与频域 phase mask 交换。

最后两张独立的 120×120 spatial phase masks 负责把卷积特征混合后送往共享 global FC。

## Deep FiberArrayExpert

```text
Encoder SpatialPhase1
-> padded ASM propagation
-> Encoder SpatialPhase2
-> coherent Gaussian-mode projection
-> trainable per-mode phase and bounded amplitude
-> coherent mode reconstruction
-> Decoder SpatialPhase1
-> padded ASM propagation
-> Decoder SpatialPhase2
```

- 编码器和解码器共四张独立的 120×120 phase-only masks；
- 默认固定 10×10 Gaussian mode bank，sigma=3 pixels；
- mode bank 初始化时逐模式 L2 归一化，但不对输入样本功率归一化；
- 复振幅投影、复模式调制与相干重构全程保持 complex field；
- 每模式 phase 和 `[0,1]` bounded amplitude 可训练。

记录的 Fiber 指标：

- coupling efficiency；
- 100 维平均 per-mode power distribution；
- effective mode number；
- 每个 Fiber 专家的输入/输出光功率。

## 参数统计

默认启用每专家一个有界标量增益和一个标量相位偏置：

| 类型 | 单专家参数组成 | 单专家参数 |
|---|---|---:|
| D2NN | 5×120×120 phase + 2 scalars | 72,002 |
| Fourier | 3×120×120 frequency phase + 2×120×120 spatial phase + 2 scalars | 72,002 |
| Fiber | 4×120×120 spatial phase + 100 mode phase + 100 mode amplitude + 2 scalars | 57,802 |

具体数值同时写入 `architecture_report.json`，避免把参数规模差异误认为专家类型优势。

## 可视化

除输入、prompt、global FC 和 detector 外，还会保存：

- 三个 Fourier convolution blocks 的逐级复光场强度和相位；
- Fourier 两层 spatial tail 的强度和相位；
- Fiber encoder、mode projection 前光场、coherent reconstruction、decoder 光场；
- Fiber per-mode power distribution 和指标 JSON；
- 所有频域、空间域和模式调制参数。

## 类型感知导出

`slm_export_best/manifest.json` 分别记录：

- D2NN 的五层 phase mosaics；
- Fourier 的三张 frequency-domain masks 和两张 tail spatial masks；
- Fiber 的两张 encoder masks、mode parameters 和两张 decoder masks。

Fourier/Fiber 参数不会被伪装成普通 D2NN mask。

