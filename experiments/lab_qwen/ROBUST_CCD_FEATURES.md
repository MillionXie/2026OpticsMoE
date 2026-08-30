# Robust mask 与 CCD feature 合同

## 1. 当前 Qwen 到底训练了什么

`accuracy_first_full` 不是一张 phase mask，也不是多组噪声消融。模型包含四张彼此
独立的可训练相位 mask：

```text
vision_expert -> vision_global -> language_expert -> language_global
```

它是一组“联合硬件噪声包络”：每个训练 step 重新随机采样 ±16 像素输入/phase/CCD
位移、0.4–2.5 倍强度增益、0–5% 振幅与相位 0 级泄漏、phase dropout、k 空间截止，
以及相对干净单帧均值的截断偏置高斯 CCD 噪声
`N(mean=+2%, std=2%) clipped to [-2%,+6%]`。四个融合门最低为 0.1%，学习结果约
0.626%。固定 200-query 测试 Top-1=85.0%，测试集没有参与 checkpoint 选择。

因此当前实验能回答“联合扰动下训练的 mask 是否可用”，不能单独归因“高斯噪声、
0 级光或位移中哪一项最有效”。如要做论文消融，必须另外训练只改变一个因素的模型，
不能把当前一个 checkpoint 写成多组噪声实验。

## 2. Qwen CCD feature 在哪里

完整会话位于：

```text
experiments/lab_qwen/qwen_theoretical_ccd_accuracy_first_full
```

每层 62 个 capture key：50 个固定 test 输入、8 个确定性人工探针、2 个 key 各重复
3 次（额外 4 帧）。每个 key 同时包含：

- `theoretical_ccd/ideal_model_fp32/*.npz`：连续 FP32 phase/振幅的原始线性强度；
- `theoretical_ccd/transport_quantized/*.npz`：实际 8-bit BMP 量化后重新仿真的原始
  线性强度，是实测 CCD 的主要理论参照；
- `ccd_feature_visualization/*/network_input_224/*.npz`：网络真正读取的 224×224
  feature；
- `CONTACT_SHEET_*.png`：只用于观看。

网络映射固定为：

```text
非负截断 -> 单帧均值归一化 -> 相对强度截断 -> log1p -> AdaptiveAvgPool(224,224)
```

不扣背景，也不做每帧 min-max。实测与仿真必须分别走完全相同的映射后，才能计算
network-input 域指标；linear 域则直接比较非负原始强度。显示 PNG 不能用于正式指标。

BMP 量化本身非常小：四层 transport-vs-ideal 的平均 linear PCC 均约
0.99947–0.99967，network-input PCC 均约 0.99990。实验中若 PCC 明显更低，主要误差
不是 8-bit 文件量化，而是光路、LUT、几何、0 级光、散斑、曝光或 CCD 噪声。

## 3. MNIST-4 为什么也保留两张 mask

MNIST 是无需电子后处理的简单诊断：478×478 原始 CCD 只计算四个 59×59 ROI 的
强度和并 argmax。`post_robust_best` 是原 baseline；`ccd_robust_rv3` 从它热启动，
加入 ±2 像素位移、5% block phase dropout、0–5% 两级 0 级泄漏、0.8–1.2 增益和
`N(+1%,1%) clipped to [-1%,3%]` CCD 噪声。robust candidate 必须从扰动启用后的
epoch 中按固定种子 robust validation 选择，不能把 warmup 初始 mask 改名冒充新 mask。

同一 quick40/formal400 输入分别播放两张 mask 后，才可以公平比较准确率、PCC、SSIM、
shape-NRMSE、余弦、能量比和质心误差。
