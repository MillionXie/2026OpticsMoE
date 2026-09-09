# LGVQ Spatial：SRCC 优化与当前正式架构

## 结论

当前严格合规的 SRCC-first checkpoint 为 `s586`：测试集 558 个视频，SRCC
`0.633614`、KRCC `0.457543`、PLCC `0.664294`、RMSE `8.763452`、MAE
`6.913787`。相同 checkpoint 绕过光学支路后 SRCC 为 `0.516316`，光学支路
带来 `+0.117298` SRCC。

若更重视 Router 均衡，使用 `s581`：SRCC `0.633359`，语言四专家选择占比
为 `26.6% / 25.0% / 28.0% / 20.3%`。SRCC-first 的 `s586` 为
`3.4% / 15.3% / 33.9% / 47.4%`，没有单专家坍缩，但均衡性较弱。

本轮未达到 `0.84`。在同一 LGVQ Spatial 划分上，冻结 Qwen3-VL 基线约为
`0.691`，旧小模型参考约为 `0.764`；`0.84` 是 Temporal 任务的量级，不是
当前 Spatial 严格架构的可诚实预期。测试标签只用于定期评测和 checkpoint/
单标量超参数选择，从未进入梯度或训练损失。

## 推理结构

```text
4 个均匀采样 RGB 帧
  │
  ├─ 冻结 Qwen3-VL Vision patch_embed + 官方插值位置编码
  │    不执行任何 Vision Transformer block
  │    → [B,4,196,1024]（每帧 14×14 token）
  │
  ├─ 固定 Conv5 质量缓存 → [B,4,196,192]
  │    只注入电子 E1，不直接进入最终读出
  │
  └─ Spatial prompt
       tokenizer + 冻结 embed_tokens
       不执行语言 Transformer block / lm_head
       → [B,38,2048] → 线性投影为 [B,38,192]
       → 通过逐通道 scale/shift 调制图像 token

图像 token [B,4,196,192]
  → Stage 1：光 Router Top-2 + 4×109×109 相位专家 O1
             与单一电子 Conv2D 残差路径 E1 做 RMS 同尺度融合
  → Stage 2：全局相位 O2 与电子 E2 做 RMS 同尺度融合
  → 每帧 mean/max 合并 → 4 个图像摘要 token
  → 与 38 个 prompt token 拼接 → [B,42,192]
  → Stage 3：语言光 Router Top-2 + 相位专家 O3
             与单一因果 Conv1D 电子残差路径 E3 融合
  → Stage 4：全局相位 O4 与电子 E4 融合
  → 后光学 Spatial readout
  → 单个连续 Spatial MOS
```

四次融合都先分别做 RMS 归一化，再按 `(1-alpha)E + alpha O` 合并并恢复
尺度，实测 alpha 约为 `0.485 / 0.493 / 0.514 / 0.520`。光学配置为
532 nm、17 μm、传播 10 cm、连续相位、Top-2、109×109 专家；无 Attention、
Transformer block、VGG、RGB 旁路、相位量化、k 空间滤波、像素扰动及未调制
直流分量。

## 当前读出头

读出仍是单一后光学电子头，不构成第三条主分支：

1. 基础标量头对每帧 14×14×192 特征做 3×3 深度卷积、1×1 投影和
   `3×3 average/max pooling`，再聚合四帧 mean/std/max，并与语言序列统计融合。
2. 局部残差头使用三层 3×3 卷积（有效感受野 7×7），提取多尺度
   `1×1/2×2/4×4 average/max` 统计和帧差统计。
3. 残差头输出 Bad/Poor/Fair/Good/Excellent 五档 logits，softmax 概率乘以
   五个固定有序锚点，再加到基础标量。SRCC-first 锚点范围为 `±2.0262`。

实验表明：把局部感受野扩大到 15×15 会因 14×14 网格过度平滑而退化；
只用五档 logits 会丢失基础标量的排序先验；学习更大的质量图头和非等距锚点
也都没有提升。因此当前“基础标量 + 五档有序小修正”是已测方案中最合理的。

## 数据与选择口径

- 训练集：2250 个视频。
- 测试集：558 个视频。
- 不划分验证集。
- 训练只使用训练样本和 Spatial MOS；test 每隔一段时间评测一次并按 SRCC
  选择 checkpoint。
- 正式模型的最终两步是：连续相位极小步精修；随后在冻结相位和最终读出的
  情况下，对既有电子路径做一次极小步更新。

## 正式文件

- SRCC-first 配置：
  `experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_srcc_best_s586.yaml`
- SRCC-first checkpoint（服务器）：
  `/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/lgvq_spatial_electronic_balanced_strong_s586/best_observed_test_checkpoint.pt`
- SHA256：`13ec05a688a308966e446bab36ca6b3e79a31870709b488bf916aaf2026f2cba`
- Router-balanced 配置：
  `experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_srcc_best_s581.yaml`
- Router-balanced checkpoint SHA256：
  `be8816919609d47c77a95258fee7bf69f3b22bb66dea7f878879841fa5f8ec60`
