# 两级 Vision 光电融合骨干

这套共享骨干用于 SALICON 显著性、ISIC 2016 病灶分割和 LSP 姿态估计。三个任务只替换 decoder，骨干和实光路协议完全一致。

## 数据与网络形状

```text
224×224 RGB
  → 冻结 Qwen3-VL patch embedding + position embedding
  → block-major Vision tokens [ΣT, 1024]
  → Linear(1024,192) + LayerNorm
  → Vision Block 1
       电子：2-D depthwise Mixer
       光学：MoE4（2×2 expert，Top-2）→ expert CCD → [224,224] readout → Linear(224,192)
       合并：E1 + sigmoid(g1) × O1
  → Vision Block 2
       电子：2-D depthwise Mixer
       光学：global phase → global CCD → [224,224] readout → Linear(224,192)
       合并：LayerNorm(E2 + sigmoid(g2) × O2)
  → 精确撤销 Qwen 2×2 block-major 排列
  → [B,192,Htoken,Wtoken]
  → task decoder
```

原 Qwen Vision Transformer blocks、主 merger、DeepStack 和 Language model 均不执行。Qwen patch/position stem 冻结；两个电子 Mixer、两级光学模块、CCD readout、融合门和 decoder 从第一步联合训练，不依赖电子预训练 checkpoint。

## 三个 decoder

- SALICON：`192→128` 投影，逐级 bilinear 上采样配合 depthwise/pointwise 卷积，输出一张 density logit；训练继续使用 KLD、CC、SIM、NSS。
- ISIC：相同的渐进上采样骨架，在全分辨率增加两个 depthwise boundary-refinement residual block；输出二值 mask logit；使用 BCE、Dice、Soft-IoU、boundary loss。
- LSP：`192→160` 后上采样至 `56×56`，输出 14 张关节 heatmap；使用 masked heatmap MSE 与 `0.1×` coordinate Smooth-L1。

所有 decoder 都没有注意力，也没有教师蒸馏。它们保留空间结构，同时避免直接把 CCD 全图送入巨型 MLP。

## 优化参数

| 参数组 | 学习率 |
|---|---:|
| 电子 Mixer / adapter | `1e-4` |
| phase/mask | `1e-4` |
| router | `5e-5` |
| CCD readout | `5e-5` |
| task decoder | `3e-4` |

phase/mask 学习率由旧 retrieval 的 `2e-5` 提高到 `1e-4`，即 5 倍；它与电子主干同量级，能实际训练起来，但仍远低于早期 dense optical 配置中过激的 `4e-3`。融合门初值为 `0.05`，模型不会在训练开头被随机光路压垮，随后可学习增加光学比例。

光学仿真仍包括 phase dropout、K-space 限制、输入/phase/CCD 错位、增益、offset 和 read noise。CCD 仅做逐帧强度归一化、截断、log compression 和行 LayerNorm；没有背景图，因此不做背景扣除。

## 对照范围

每个任务只维护两条正式路径：

1. 从任务随机初始化模块开始的正常光电联合训练。
2. 使用联合训练 checkpoint，按 `vision_expert → vision_global` 顺序进行实光路采集与下游微调。

不额外维护纯电子、纯光、教师蒸馏等四组配置，避免当前实验被无关消融分散。
