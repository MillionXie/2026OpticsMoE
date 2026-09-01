# 下一个会话的完整交接说明

## 1. 研究背景

用户调研了光学神经网络原位训练、光电混合训练、PAT 和固定反馈方法。传统
原位 BP 需要在真实光路中产生和测量反向误差光场，并处理前后向光路配准；
PAT 需要依赖当前数字可微模型反复计算梯度。受 fixed-feedback SFT 工作启发，
提出如下光学微调假设：

> 光学网络先离线预训练。部署到真实平台后继续使用真实光路完成当前模型的
> 前向传播，但在电子端复用预训练状态对应的固定反馈算子估计层间误差方向。
> 若微调参数漂移较小，固定反馈可能仍与当前真实 BP 梯度保持较高对齐。

当前仓库只做**理想光学仿真中的初步验证**，暂未引入实际 SLM/CCD 误差，也
没有宣称已经解决真实系统中各层误差观测和局部相位更新的硬件实现问题。

## 2. 必须保持准确的 FA-pretrained 定义

微调第 `t` 步、第 `l` 层：

```text
当前输入 a_l(t)
  -> 当前相位 phi_l(t)
  -> 当前前向光场和 CCD 输出
  -> 当前 loss L(t)
```

反向时：

```text
相位局部梯度 dL/dphi_l：使用当前输入、当前相位和当前 batch error
前层误差 dL/da_l：使用预训练结束时冻结的 phi_l(pre)
```

下面三句话不能被误解：

1. 固定的是 operator，不是一次具体 backward field。
2. 当前 forward 永远不能换成 pretrained phase。
3. 每个 batch 的 loss、output error 和局部相位梯度都必须重新计算。

## 3. 四个主实验组

所有组从同一个 pretrained checkpoint 开始，并匹配随机种子、batch 顺序、
数据增强、optimizer、学习率、epoch 数和数据划分。

### A. BP

- 当前相位前向；
- 当前相位对应的精确反向；
- 用作性能和更新方向参考。

### B. FA-pretrained

- 当前相位前向；
- 预训练相位对应的固定反馈连接器；
- 最关键实验组。

### C. FA-random

- 当前相位前向；
- 每层初始化一次随机、单位模长、维度兼容的固定相位反馈；
- 用于区分“固定本身有效”还是“预训练初始化有效”。

### D. No Fine-Tuning

- 预训练模型直接评估；
- 不更新光学相位或电子参数；
- 用于度量微调的真实增益。

## 4. 共同光学结构

```text
灰度输入
-> bicubic resize 到 400 x 400
-> 20 个独立 OEO stage
-> 电子读出
```

每个 OEO stage：

```text
非负实振幅
-> 400 x 400 phase-only mask
-> 无 padding 角谱传播 5 cm
-> 400 x 400 平方律 CCD
-> 对完整 CCD 平面的无仿射 LayerNorm
-> ReLU
-> 自适应 optical/skip 残差混合
-> 零相位振幅重新加载
```

物理参数：波长 532 nm，像素尺寸 16 µm，相位参数为
`2*pi*sigmoid(raw_phase)`，`raw_phase=0` 对应物理相位 `pi`。没有 MoE、
k-space 约束、DC penalty、phase smoothness、padding 或逐样本功率归一化。

每层都有两项可学习残差 logits：

```text
[w_optical, w_skip] = softmax(residual_logits)
a_next = w_optical * activated_CCD + w_skip * a_previous
```

残差权重始终用普通 BP，不使用固定反馈。

## 5. V1 已完成实验

源码：
`FixedFeedbackSFT/projects/d2nn_cifar100c10_fixed_feedback_20stage400`

协议：

- clean CIFAR-100 预训练，保留 100-way 分类头；
- downstream 是 CIFAR-100-C 中十个原有类别；
- downstream 仍使用原 100-way head 的对应输出坐标；
- 20 stage，残差初始化 optical/skip = 0.10/0.90；
- 80 epoch 预训练，50 epoch 微调，3 个匹配随机种子；
- phase LR 0.01，electronic LR 0.001，AdamW，weight decay 0。

核心结果：

| checkpoint policy | BP | FA-pretrained | FA-random | NoFT |
|---|---:|---:|---:|---:|
| 固定 epoch 50 test | 37.61 ± 0.35% | 37.17 ± 0.50% | 37.00 ± 1.04% | 42.33% |
| validation-selected test | 41.33 ± 1.32% | 40.72 ± 2.11% | 39.11 ± 3.53% | 42.33% |

匹配终点几何：

| epoch | method | relative drift | drift/BP | cosine to BP |
|---:|---|---:|---:|---:|
| 10 | BP | 0.1804 | 1.000 | 1.000 |
| 10 | FA-pretrained | 0.1803 | 0.999 | 0.997 |
| 10 | FA-random | 0.2197 | 1.218 | 0.433 |
| 50 | BP | 0.4033 | 1.000 | 1.000 |
| 50 | FA-pretrained | 0.4019 | 0.997 | 0.972 |
| 50 | FA-random | 0.4984 | 1.236 | 0.364 |

V1 的正确解读：

- FA-pretrained 的更新方向和幅度确实明显比 FA-random 更接近 BP；
- 这是支持核心假设的**几何证据**；
- 但任务性能没有超过 NoFT，不能宣称固定反馈改善了迁移性能；
- downstream 类别和 head 都已在预训练出现，NoFT 本身很强；
- downstream 很小，50 epoch 全参数微调严重过拟合；
- BP 最终 relative drift 约 0.40，远非 fixed-feedback SFT 所依赖的小漂移区间。

## 6. V2 改进实验

源码：
`FixedFeedbackSFT/projects/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400`

目的：去掉预训练固定 100-way 分类头带来的类别坐标绑定，测试真正的跨数据集
迁移。

协议：

- CIFAR-100 使用 supervised contrastive loss 预训练；
- downstream 使用真实 CIFAR-10，而不是 CIFAR-100-C 的旧类别；
- 读出为 128 维有正有负的 L2-normalized embedding；
- CIFAR-10 用固定 support images 构建 prototype；
- 微调损失为 `SupCon + 0.5 * leave-one-out prototype CE`；
- prototype CE 构建本类 prototype 时排除 anchor 自身；
- test 永不用于 checkpoint 选择；
- 残差初始化改为 optical/skip = 0.35/0.65，避免光学路径过弱；
- 电子 embedding readout 使用 `Dropout(0.1)`；phase dropout 关闭；
- 120 epoch 预训练，30 epoch 微调；fine-tune phase LR 0.003；
- 仍使用 BP、FA-pretrained、FA-random、NoFT 和三匹配随机种子。

当前状态：代码、9 个单元测试和正式 CUDA smoke 均已完成；服务器还已完成 120
epoch 共享预训练，以及 NoFT、BP、FA-pretrained、FA-random 三个 matched seeds 的
正式微调和聚合。固定 epoch 30 test accuracy 为 BP 31.00 +/- 0.52%、
FA-pretrained 31.02 +/- 0.52%、FA-random 28.19 +/- 2.24%、NoFT 27.56%。epoch 30
FA-pretrained endpoint cosine to BP 为 0.9975 +/- 0.0001，FA-random 为
0.3865 +/- 0.0202。三个 seed 的逐 epoch batch order 在三方法间完全匹配，test 未
用于 checkpoint 选择。

V2 的关键新问题是 optical bypass：预训练后平均 residual optical weight 只有约
0.070，最小约 0.021。因而结果证明四组现象已经成立，但尚未排除“网络主要走 skip
path，所以 feedback connector 影响很弱”的解释。

## 7. 现有实现最关键的边界

当前 fixed connector 只替换光学 phase+propagation 对前层振幅的 input VJP。
以下部分仍是当前网络的精确 BP：

- 当前层 phase 的局部梯度；
- CCD 平方律；
- LayerNorm；
- ReLU；
- 残差混合；
- 电子读出或 embedding projection。

这与“冻结整个网络完整 Jacobian”不同。后续若改变定义，必须新建消融名称，
不能仍称为当前 `fa_pretrained`。

## 8. 下一会话应优先做什么

1. 不要重跑或覆盖 V1/V2；它们只作为当前 baseline 和反馈实现验证。
2. 暂停新增 FA 训练，新建 CIFAR-10 direct supervised BP backbone 实验。
3. 优先比较 RGB/grayscale、4/8/12/20 stage、residual 和小型 readout，使完整
   CIFAR-10 test accuracy 先达到 60%，再争取 65%-70%。
4. 每个性能运行都同时做 optical-off、phase-random/phase-shuffle，计算 normalized
   optical dependence；不能只用 phase 参数量或 residual weight 代表光学比例。
5. 只有 accuracy >= 60% 且 normalized optical dependence 达到预注册门槛后，才把
   模型固化为 pretrained optical backbone，并恢复 BP/FA-pretrained/FA-random/NoFT。
6. 完整实验架构以 `FixedFeedbackSFT/RESEARCH_PLAN.md` 为准；新模块 CLI 约定见
   `FixedFeedbackSFT/commands/02_performance_first_runbook.md`。

## 9. 可直接贴给下一个 AI 的短提示

```text
请先阅读仓库根目录 FixedFeedbackSFT/ 下的 README.md、HANDOFF.md、METHOD.md、
EXPERIMENTS.md、RESEARCH_PLAN.md 和 CODE_INDEX.md。不要修改或覆盖已经完成的
V1/V2 正式实验。当前优先工作不是继续跑 FA，而是新建性能优先的 CIFAR-10 direct
supervised BP backbone：先达到 full-test accuracy >= 60%，并用 optical-off、
phase-random 和 phase-shuffle 证明 normalized optical dependence。通过两道 gate 后
才固化 pretrained checkpoint 并恢复 fixed-feedback 对比。所有代码/config 修改必须
通过 Git 同步，启动入口放在新实验的 commands/ 中。
```
