# 固定反馈四组实验记录

更新日期：2026-08-18

## 唯一正式方法

主表只允许 NoFT、BP、FA-pretrained、FA-random 四组。head-only、optical-off、phase-random 和 phase-shuffle 只作为内部机制诊断，不是额外方法组。

## 共同 source 与起点

- source optical operator：A03 CIFAR-100 best checkpoint；
- source SHA-256：`f632c57cf851805090686cda81d4b4a0efc07b02c91dc0e0b63c00912247becc`；
- source CIFAR-100 test：32.13%，normalized optical dependence 87.92%；
- 先冻结全部 optical stages，只训练一个共同 CIFAR-10 head；
- head-warmup checkpoint 的模型权重和 SHA-256 必须被四组共同引用；
- 每个 seed 内，BP/FA-pretrained/FA-random 使用相同 batch order、augmentation seed、优化器、LR、epoch 和 validation policy。

NoFT 直接评估共同起点。三个微调组的电子头和 residual 均使用普通 BP；区别只在 optical stage 之间的反向误差连接：

- BP：当前相位算子；
- FA-pretrained：冻结 A03 source phase；
- FA-random：冻结、seed-matched 的随机 phase。

所有方法的前向都使用当前可训练相位。FA 不允许在前向中使用冻结 phase；本层 phase gradient 仍由当前局部算子精确计算，只有传向前一 stage 的误差信号使用固定连接。

## Pilot P01

状态：代码已实现，等待服务器测试与运行。

- 配置：`configs/formal_pilot.yaml`
- head warm-up：10 epoch，seed 4242；
- downstream pilot：seed 2026，20 epoch；
- phase LR：5e-4；electronic/residual LR：3e-4；
- checkpoint：validation best，且 epoch-0 共同起点受到保护；
- 预期用途：验证高性能 RGB 八层骨干上是否仍出现 FA-pretrained 接近 BP、且优于 FA-random 的现象。

必须回填：共同 checkpoint SHA-256、head-warmup best、四组 test accuracy、光学依赖、epoch-0 分层 gradient cosine、endpoint phase drift、BP-FA gap 和是否进入多 seed。

## 工程验证 S01

在 P01 长跑前，用 `configs/formal_smoke.yaml` 对真实 A03 source 执行一次 head warm-up、四种方法各两个训练 batch、四种光学消融和完整性受检的比较汇总。S01 只验证数据、checkpoint、固定反馈 autograd 与文件输出链路，不用于科学结论。

状态：2026-08-18 服务器通过。

- 服务器单元测试：8 passed；
- 真实 A03 checkpoint 的旧格式兼容与 SHA-256 校验通过；
- smoke common checkpoint SHA-256：`a7d5c436ab33ebe15435b27cae1fe16e5d7be86f6083d8ec1970f9f052d0874b`；
- 四组均生成 `best.pt`、`result.json`，比较器生成 `comparison.json` 和 `runs.csv`；
- FA-pretrained 的 epoch-0 八层 gradient cosine 均约为 1，符合“冻结算子等于初始前向算子”预期；
- FA-random 的前七层 gradient cosine 为 `[0.1023, 0.1749, 0.1194, 0.4647, 0.2537, 0.5784, 0.7256]`，末层为 1，符合“只替换跨层 error connector，末层仍由相同 head 误差直接驱动”的预期。

结论：S01 只覆盖极少样本，accuracy 不作解释；工程链路允许进入 P01。
