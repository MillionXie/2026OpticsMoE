# Warmstart5 固定测试结果

## 结论

该独立工程满足预声明目标：四个光电融合门的下限均为 `0.05`，固定测试
Top-1 为 **81.00%**。这里的 5% 是融合系数下限：

```text
alpha = 0.05 + 0.95 * sigmoid(raw_gate)
layer_output = electronic + alpha * optical_delta
```

它不是 CCD 能量占比。固定 checkpoint 的四个实际 alpha 为
`0.055000070–0.055000529`。

## 无测试选轮协议

- Stage A 只训练光支路，电子分支、64 维 head 和 gate 冻结；
- Stage B 低学习率联合训练 12 epoch；
- 两阶段每轮 `test_top1=nan`，`best_observed_test=-inf`；
- checkpoint 只按最小训练总损失选择；
- 预声明的 Stage B epoch 8 EMA checkpoint 固定后，才揭封测试一次；
- 测试结果没有用于换 epoch 或继续调该运行。

固定 checkpoint：

```text
runs/caltech101_warmstart5_stage2_joint_sealed_test/ema_best_train_loss_checkpoint.pt
SHA-256: 6a27f54d8c869cce46150583383a127b0ba47b3d34503f5753aa23974ac1e55d
epoch: 8
train_loss: 2.2300728
train_top1: 0.8583
```

## 固定 retrieval 测试

协议为 10 类、200 query、30 gallery images、mean-prototype gallery：

| 指标 | 结果 |
|---|---:|
| Top-1 | 0.8100 |
| Top-3 | 0.9300 |
| MRR | 0.876345 |
| embedding | 64 维、有限、L2 归一化 |

各类 Top-1：

| 类别 | Top-1 |
|---|---:|
| airplanes | 0.80 |
| Motorbikes | 0.70 |
| Faces | 0.90 |
| Leopards | 0.90 |
| accordion | 0.70 |
| grand_piano | 0.85 |
| scorpion | 0.55 |
| sunflower | 1.00 |
| watch | 0.90 |
| yin_yang | 0.80 |

机器可读证据：

```text
runs/caltech101_warmstart5_stage2_joint_sealed_test/metrics/evaluation_summary.json
```

## 训练完整性

- Stage A：Vision/Language 的 phase 5/5、router 2/2、光后电子 8/8 均更新；
  所有电子主体与最终 head 逐 tensor 保持不变；四个 gate 精确保持 0.055；
- Stage B：Vision 电子 35/35 更新；Language 32/35 更新，按设计冻结的
  `residual_logit` 与未用于 retrieval 的 `output_adapter` 保持不变；
- Stage B 的两个模态 phase/router/光后电子、四 gate 和 readout 4/4 均更新；
- 全程专家覆盖 `active_v/l=4/4`、`unselected_v/l=0/0`；
- 无 NaN、Inf、OOM 或 traceback。

相位确实更新，但 Stage B run 级 `phase_delta≈0.0671 rad` 仍低于配置的
0.08 rad 诊断阈值，因此保留“相位变化幅度未达期望”的 warning；这不是梯度丢失，
每轮 phase gradient 均有限非零。

## 四张正式 phase BMP

全部为 1920×1200、8-bit 灰度，checkpoint SHA 已写入
`hardware_phase_export/phase_export_report.json`：

| stage | SHA-256 |
|---|---|
| vision_expert | `44954a0043eaeaed4535a79c74aba5e0ddc94e4e173a82889cf44dad406f9a118` |
| vision_global | `1fb6e25bdc10206442d15d8bbe942948b25a036026e852443b4ab95e5124114ab` |
| language_expert | `630f66dca8db41cd00f3a771a9ef1d55d3fdc09b644c99c3313bea4fabd12c127` |
| language_global | `7cf04fb852f356ec7e69a78460786991e22b24eed9ebbba97944b9b9ea8f6444e` |

物理合同仍为 532 nm、17 μm 逻辑采样、10 cm 传播、478×478 有效区、
phase SLM 8 μm/1920×1200/中心 `(980,590)`，并保持既定 vertical flip。
