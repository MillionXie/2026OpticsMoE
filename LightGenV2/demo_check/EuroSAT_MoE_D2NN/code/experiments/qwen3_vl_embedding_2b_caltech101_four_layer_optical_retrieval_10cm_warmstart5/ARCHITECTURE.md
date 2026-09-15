# Warm-start5 架构与正式协议

## 目标

本工程保持 10 cm robust 工程的四层光路、硬件尺寸和电子 Mixer 不变，但以已经训练好的 2D/no-DeepStack 电子模型为性能锚点。`Top-1 > 0.80` 是预注册验收线，不是通过查看测试集选择 epoch 的条件。

## 网络

```text
Frozen Qwen3-VL embedding backbone
  Vision 1024 -> 192
    -> 2D electronic Mixer 1 + alpha_v1 * MoE4 expert optical delta
    -> 2D electronic Mixer 2 + alpha_v2 * global optical delta
  Language 2048 -> 192
    -> causal 1D electronic Mixer 1 + alpha_l1 * MoE4 expert optical delta
    -> causal 1D electronic Mixer 2 + alpha_l2 * global optical delta
  mean+max pooling -> 384 -> LN -> Linear -> 64D L2 embedding
```

四个融合系数都使用：

```text
alpha = 0.05 + 0.95 * sigmoid(raw_gate)
```

初始化为 `0.055`。这里的 5% 是光学残差的系数下限，不是 CCD 能量占比或最终特征能量占比。

## 严格双源初始化

Stage A 只接受两个固定 SHA-256 的 EMA/train-best checkpoint：

- 电子源：加载 Vision/Language 的全部共有电子张量和最终 64D readout；
- 10% robust 光源：只加载 `core.optical_branch.*`；
- 不加载任何旧 gate，四个 gate 始终由本工程重新初始化为 0.055；
- 每个 key、shape、checkpoint version、embedding 维度、detector 维度、Qwen model ID、EMA 标记及无 test 选模标记都必须通过检查。

审计结果是每个模态 35 个共有电子 tensor 全部同名同形，robust 模型每模态额外包含 17 个光支路/门控 tensor。loader 不使用宽松的“碰到什么就加载什么”逻辑。

## 两阶段训练

### Stage A：光支路校准

- 4 epochs，每 epoch 固定 12 个 PK batch；
- 冻结两模态电子 Mixer、外层 adapter、最终 readout 和四个 gate；
- 只训练两个 `optical_branch`，包括 phase、router、CCD readout 和光后电子 adapter；
- phase LR `6e-3`，router LR `2e-4`，其余光支路 LR `1e-4`。

### Stage B：低学习率联合微调

- 从 Stage A 的 `ema_best_train_loss_checkpoint.pt` 严格加载，不恢复 Stage A optimizer；
- 12 epochs，每 epoch 固定 12 个 PK batch；
- Mixer/base LR `2e-5`，外层 adapter `5e-6`，64D readout `1e-5`；
- phase LR `6e-3`，router LR `1e-4`，gate 使用 base LR；
- 原 Qwen 始终冻结。

两阶段都使用 `SupCon + episodic prototype CE + CCD operating-point + router balance/importance`，没有教师蒸馏。

## 鲁棒光路

- 532 nm，17 µm，传播 10 cm；
- 518×518 仿真画布，478×478 有效/CCD 区域；
- k-space `theta_max=0.65°`；
- input、phase、CCD 在每个物理 stage 独立采样 `±16` 像素平移；
- gain `0.4–2.5`、offset `0.05`、read noise `0.015`；
- phase dropout `0.08`，block size 8；
- router train-only logit noise `0.10`。

## Sealed test

两个训练配置都强制 `evaluate_test_each_epoch=false`。训练中只能按训练总损失保存 checkpoint。正式结果只允许用 Stage B 预先声明的 `ema_best_train_loss_checkpoint.pt` 执行一次 evaluate。

若该固定 checkpoint 未达到 0.80，应记录为未通过；禁止改用 `best_observed_test`。需要调参时，应从 train 划出开发集并建立新版本，继续封存原 200 张 test。
