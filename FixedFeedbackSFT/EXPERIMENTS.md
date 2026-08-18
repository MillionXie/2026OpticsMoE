# 实验记录与当前结论

## V1：CIFAR-100 → CIFAR-100-C 十类分类

### 实验目的

先用结构最直接的分类任务验证 fixed pretrained optical connector 是否能产生
接近 BP 的累计更新方向。

### 配置摘要

| 项目 | 设置 |
|---|---|
| 输入/CCD | 400 x 400 |
| 光学 stage | 20 |
| 每层传播 | 5 cm |
| 波长/像素 | 532 nm / 16 µm |
| 每层电子操作 | full-plane non-affine LN + ReLU |
| residual init | optical 0.10 / skip 0.90 |
| pretrain | clean CIFAR-100, 80 epochs |
| downstream | CIFAR-100-C，十个旧类别，severity 3 |
| fine-tune | 50 epochs，3 matched seeds |
| optimizer | AdamW, phase LR 0.01, electronic LR 0.001, WD 0 |

### 结果

固定 epoch 50：

- BP：37.61 ± 0.35%
- FA-pretrained：37.17 ± 0.50%
- FA-random：37.00 ± 1.04%
- NoFT：42.33%

validation-selected：

- BP：41.33 ± 1.32%，平均 epoch 7.3
- FA-pretrained：40.72 ± 2.11%，平均 epoch 10.7
- FA-random：39.11 ± 3.53%，平均 epoch 6.7
- NoFT：42.33%

epoch 50 累计更新：

- FA-pretrained drift/BP = 0.997，cosine to BP = 0.972；
- FA-random drift/BP = 1.236，cosine to BP = 0.364。

### 可以支持的结论

预训练反馈显著保留了 BP 累计更新方向；随机固定反馈偏离更大。这支持
“pretrained feedback initialization 包含有用结构”，而不是“任意固定反馈都
等价”。

### 不能支持的结论

不能声称 FA-pretrained 改善了 downstream accuracy，也不能声称已经实现真实
光路无反传训练。NoFT 最好，且所有微调组都明显过拟合。

### V1 的设计缺陷

1. downstream 十类已在 CIFAR-100 pretraining 中出现；
2. 原 100-way 分类头直接保留；
3. NoFT 已经知道这些类别的输出坐标；
4. 1,800 张微调图面对约 320 万相位参数，容易过拟合；
5. phase LR 和 50 epoch 使参数漂移达到约 0.40，不符合小漂移假设。

## V2：CIFAR-100 → 真实 CIFAR-10 对比迁移

### 改进目的

将表示学习和分类坐标解耦，让下游任务真正使用新的数据集和类别定义。

### 配置摘要

| 项目 | 设置 |
|---|---|
| 光学结构 | 与 V1 相同的 20 stage/400 x 400/5 cm |
| residual init | optical 0.35 / skip 0.65，可学习 |
| readout | LN(400) → Dropout(0.1) → Linear(400,128) → L2 norm |
| pretrain | CIFAR-100 SupCon，120 epochs |
| downstream | actual CIFAR-10 |
| support | 100 images/class，仅构建 prototype，不反传 |
| validation | 200 images/class |
| fine-tune | SupCon + 0.5 prototype CE，30 epochs |
| phase LR | pretrain 0.01；fine-tune 0.003 |
| methods | BP / FA-pretrained / FA-random / NoFT，3 seeds |

### 正式结果

服务器已完成 120 epoch 共享预训练、NoFT、BP、FA-pretrained 和 FA-random 三个
matched seeds 的 30 epoch 微调。三个 seed 的逐 epoch batch-order hash 在三种训练
方法间完全一致，test 未用于 checkpoint 或超参数选择。

| policy | BP | FA-pretrained | FA-random | NoFT |
|---|---:|---:|---:|---:|
| 固定 epoch 30 test | 31.00 +/- 0.52% | 31.02 +/- 0.52% | 28.19 +/- 2.24% | 27.56% |
| validation-selected test | 30.70 +/- 0.28% | 30.77 +/- 0.14% | 29.91 +/- 0.49% | 27.56% |

epoch 30 的 matched endpoint：

| method | relative drift | drift/BP | cosine to BP |
|---|---:|---:|---:|
| BP | 0.1744 +/- 0.0015 | 1.000 | 1.000 |
| FA-pretrained | 0.1743 +/- 0.0015 | 1.000 | 0.9975 +/- 0.0001 |
| FA-random | 0.2385 +/- 0.0090 | 1.368 | 0.3865 +/- 0.0202 |

### 当前可以支持的结论

- 在真实跨数据集迁移中，FA-pretrained 同时匹配 BP 的固定终点性能、累计更新幅度
  和方向；
- FA-random 的性能更低、方差更大、漂移更远且更新方向明显偏离；
- BP 和 FA-pretrained 都超过 NoFT，V1 的“微调没有收益”问题已解决；
- V2 的评估规则比参考论文的混合 test/validation checkpoint selection 更严格。

### 当前最重要的限制

预训练结束时，20 层平均 optical residual weight 已从初始化的 0.35 降至约 0.070，
最小值约 0.021，最大值约 0.346；微调后平均值仍只有约 0.072。因此网络大部分走
skip path。BP 与 FA-pretrained 极度接近既可能说明固定反馈准确，也可能说明反馈
连接器的影响被旁路衰减。必须通过 optical occlusion、相位扰动和残差约束实验排除
这一混杂因素后，才能把 V2 作为强机制证据。

V2 约 31% 的绝对准确率说明当前 pretrained embedding 仍弱。它不妨碍四组因果
比较，但限制了结果的展示力度。后续应在保证光学路径真正被使用的前提下构造更强
source pretraining，而不是只延长现有训练。

## 结果报告规则

应同时报告：

1. **固定预算终点**：所有训练方法使用相同最后 epoch；
2. **validation-selected**：各方法独立按 validation 选择，随后只在 test 评估；
3. **匹配 epoch 几何**：只比较同 seed、同 epoch 的更新向量；
4. **NoFT**：固定模型，单独列出，不伪造更新 cosine。

V1 结果原始表位于：

- `experiments/d2nn_cifar100c10_fixed_feedback_20stage400/results/main/source_data/`
- `aggregate_performance.csv`
- `aggregate_geometry.csv`
- `checkpoint_performance.csv`
- `endpoint_geometry.csv`
- `training_trajectories.csv`

V2 服务器结果位于：

- `experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/runs/main/comparison/`
- `aggregate.csv`
- `task_metrics.csv`
- `endpoint_geometry.csv`
- `comparison.json`

## 当前最有价值的科学问题

1. 能否在完整 CIFAR-10 test split 上先把 BP optical/hybrid backbone 做到 60%-65%？
2. RGB 编码、stage 深度、传播设置、residual 和 readout 中，什么限制了当前性能？
3. 在高性能模型中，optical-off/phase-random 会损失多少性能，因果光学依赖度多高？
4. 在 accuracy-optical-dependence Pareto 前沿选出的 backbone 上，FA-pretrained 是否
   仍能匹配 BP 并优于 FA-random？
5. gradient cosine 和任务差距如何随 operator phasor distance 增大而变化？
6. 在 SLM/CCD 噪声与校准误差下，固定反馈和 periodic refresh 的有效边界是什么？
