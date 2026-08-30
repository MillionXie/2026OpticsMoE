# Robust mask 与 CCD feature 合同

## 1. 当前 Qwen 到底训练了什么

`accuracy_first_full` 不是一张 phase mask，也不是多组噪声消融。实验时依次播放四张
硬件 BMP：

```text
vision_expert -> vision_global -> language_expert -> language_global
```

其中两个 expert BMP 都在同一物理画面内排布 2×2 四个专家相位，两个 global BMP
各含一张全局相位，所以 checkpoint 实际训练的是 `4+1+4+1=10` 个逻辑 phase
plane；日志中的 `phase_dc_plane_count=10` 也与此一致。之所以实验室一次只看到一张
BMP，是因为四层必须逐层播放和采集，而不是模型只有一张 mask。

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

为了避免实验人员在 `.npz` 目录中逐个找图，会话根目录还会生成：

```text
VIEW_THEORETICAL_CCD/
├── OPEN_ME_FIRST_TEST_ONE_PER_CLASS.png
├── OPEN_ME_FIRST_DESIGNED_PROBES.png
├── 01 Vision expert (MoE4)/
├── 02 Vision global/
├── 03 Language expert (MoE4)/
└── 04 Language global/
```

前两张总览图最适合直接查看：每一行是同一个输入，每一列是一个光学层。每层目录还包含
全部样本的灰度图、viridis 伪彩图、网络实际读入的 224×224 伪彩图和联系表。紫色为低强度，
黄色为高强度。PNG 的归一化仅用于显示，正式 PCC/SSIM 仍读取 `.npz`。

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

本次最终结果为：best epoch=6，clean validation=88.08%，三次固定种子 robust
validation=87.49%，完整 4,157 张 sealed test=88.67%，实际 8-bit BMP 重仿真的
固定 formal400=89.0%。quick40 固定诊断子集为 82.5%，不得写成正式准确率。
