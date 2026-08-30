# 10 cm 四层光电网络：早期鲁棒训练与实测流程

本文件是本工程的唯一操作顺序说明。所有命令都在仓库根目录执行。

## 0. 已知基线与目标

| 模型 | 独立仿真测试 Top-1 | 用途 |
|---|---:|---|
| 纯电子、同一 Qwen/2D Mixer 骨干 | 87.0% | 可达到的电子参考上限 |
| 已训练 strong-noise、四门实际约 1.52% | 82.0% | 现有鲁棒优先方案 |
| 本工程 accuracy-first | 训练完成后填写 | 首选实测，目标是提高 78% 达标概率 |
| 本工程 balanced | 训练完成后填写 | 更强光路扰动容忍度的备选 |

82% 是 `noise_strong_mu0p06_sigma0p05_phase0p03` 的 sealed-test
Top-1，不是光路实测值。旧 LUT 四层实测结果 70%→69%→71%→71% 仅作历史
参考；更换 LUT 后必须重新采集，不能混用旧 CCD。

## 1. 两种新方案

### accuracy-first（建议先采）

- 光学融合系数硬下限：0.1%；从 Stage-A checkpoint 载入后实际初始值约
  0.63%，之后可学习。
- 振幅 SLM 与相位 SLM 的未调制强度分量分别在 0–5% 随机采样。
- CCD 截断偏置高斯：均值 2%、标准差 2%、范围 -2% 至 6%。
- 目的：保留更多 87% 电子模型能力，优先满足实际 Top-1 ≥78%。

### balanced

- 光学融合系数硬下限：0.5%；实际初始值约 1.02%，之后可学习。
- 两个 SLM 的未调制强度分量分别在 2–8% 随机采样。
- CCD 截断偏置高斯：均值 4%、标准差 3.5%、范围 -3% 至 12%。
- 目的：在精度和光路鲁棒性之间取折中，若 accuracy-first 对环境变化敏感再采。

这里的 5%/8% 是**强度比例**。仿真先取平方根再做相干光场叠加，并随机化
相对相位；不是把 CCD 图简单加 5%。扰动只在训练时启用，sealed-test 保持
干净。仿真和实测 CCD 使用完全相同的归一化：非负截断 → 单帧均值归一化 →
相对强度截断 → `log1p`。程序不做虚构的背景扣除。

## 2. 服务器训练

可在两张空闲 GPU 上并行运行：

```powershell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/configs/release/accuracy_first_floor0p1.yaml --phase train
```

```powershell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/configs/release/balanced_floor0p5.yaml --phase train
```

两组都从 Stage-A 第 4 epoch 的固定 SHA checkpoint 开始，噪声从第一个续训
epoch 就生效，优化器状态重置。每组额外训练 32 epoch，每 epoch 12 个均衡
batch。训练期间 test 完全封存，checkpoint 只由 train loss 选择。

## 3. 仿真测试（训练完成后各执行一次）

accuracy-first：

```powershell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/configs/release/accuracy_first_floor0p1.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/runs/accuracy_first_floor0p1_leak0to5/ema_best_train_loss_checkpoint.pt
```

balanced：

```powershell
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff --config experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/configs/release/balanced_floor0p5.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff/runs/balanced_floor0p5_leak2to8/ema_best_train_loss_checkpoint.pt
```

## 4. 实验采集前的硬性规则

1. 当前权重/相位 SLM 保持纯黑（灰度 0）；训练和仿真不要求改动硬件。
2. 正式采集前必须确认使用 3500 μs 新 LUT，不能混入旧 LUT 帧。
3. 每个方案有自己的四层 phase mask 和逐层振幅输入，不能跨方案混用。
4. 每层顺序固定为 `vision_expert → vision_global → language_expert → language_global`。
5. 每采完一层，只微调该层之后的电子网络，再从所得 checkpoint 导出下一层。
6. 先完整采 accuracy-first；只有它对扰动过敏或老师需要鲁棒性 trade-off 时，
   再完整采 balanced。现有 strong-noise 作为第三个鲁棒优先对照，不必重训。

## 5. 四层闭环命令模板

下面把 `<CONFIG>`、`<CHECKPOINT>`、`<SESSION_DIR>` 替换为同一个方案的路径。
不要把不同方案拼在一条链中。

导出第一层：

```powershell
python -m experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff.hardware_bridge --config <CONFIG> --checkpoint <CHECKPOINT> --session-dir <SESSION_DIR> --stage vision_expert --phase export
```

然后按实验室总流程重建 `amplitude_to_play`、手动加载该层 phase BMP、采集
CCD。完成后本地执行该层微调并导出下一层；依次对四个 stage 重复。若使用
实验室 ZIP，优先运行其中 `experiments.lab_qwen.local_four_stage` 的短命令，
它会完成“校验输入 → 本地微调 → development 最优选模 → 导出下一层”。

实测 checkpoint 选择使用固定 development 子集，不使用 test。最多 100 epoch，
连续 15 epoch development 无提升可提前停止；最后对 100 张 sealed test 只评一次。

## 6. 判定与 trade-off 记录

每层都记录：当前层后 sealed-test Top-1、PCC、SSIM、gain-aligned NMAE、饱和
比例、实际融合门值、phase 标准差。最终要求是第四层微调后 Top-1 ≥78%。
这不是仿真能够保证的数值；若 accuracy-first 新 LUT 实测仍低于 78%，先检查
PCC/SSIM 与方向、曝光、ROI 合同，再决定是否继续降低门值，不能用 test 反复
挑 epoch。
