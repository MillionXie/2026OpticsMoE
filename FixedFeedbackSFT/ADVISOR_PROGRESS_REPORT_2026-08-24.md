# 预训练固定反馈算子在光学神经网络微调中的阶段汇报

更新时间：2026-08-24
用途：课题组阶段汇报；本文只把已经有实验记录支持的内容写成“结果”，将规划、pilot 和正式
多随机种子实验明确区分。

## 0. 一页结论

本课题研究的问题是：光学神经网络完成预训练并部署后，微调时能否保持**当前真实光学前向**，
但复用预训练终点保存的固定光学反馈算子传递跨层误差信号，从而避免每一步都重新构造完整的
当前反向光学连接。

目前最有力的结果有三层：

1. **理想数字光学环境，三配对随机种子正式实验（P02）**：BP、FA-pretrained、
   FA-random、NoFT 的 CIFAR-10 test Top-1 分别为 `72.30±0.54%`、`71.44±0.42%`、
   `63.29±0.91%`、`51.18%`。FA-pretrained 距 BP 仅 `0.86±0.26 pp`，并比随机固定反馈高
   `8.15±1.32 pp`。
2. **冻结推理部署扰动，3 个训练种子 × 3 个部署种子正式实验（P03）**：在相位误差、
   探测器噪声、0.0625/0.125 pixel 横向偏移及联合工作点中，FA-pretrained 仍接近 BP 且
   优于 FA-random；0.25 pixel 时高性能表征相对理想值严重退化并出现排序反转，构成明确
   失效边界。
3. **部署后适配与预防性鲁棒训练 pilot（P04/P05）**：当前 BP 确实能够获得错位后的最新
   Jacobian 并恢复性能；部署前最后一次训练保存的反馈算子在 0.125--2 pixel 固定偏移适配中
   始终优于随机反馈。连续错位“疫苗化”进一步把七环境平均准确率从 `33.06%` 提高到
   `60.94%`。

因此，当前证据支持的准确表述是：

> 在光学表征确实被使用、微调漂移受控的条件下，预训练终点的固定光学反馈算子能够在理想
> 环境和一定部署失配工作区内近似当前 BP，并显著优于同形状随机反馈；其有效性随算子失配
> 增大而退化。这提示应继续正式验证鲁棒预训练，并研究是否需要周期性刷新；刷新策略的收益
> 尚未得到实验验证，不能把固定反馈宣称为任意条件下 BP 的普遍替代。

## 1. 背景、动机与研究问题

多层光学神经网络的原位 BP 通常要求生成、测量或数值重建与当前前向光路严格匹配的反向误差
场。真实系统中还会遇到前后向配准、相位标定、器件漂移、CCD 噪声和横向错位等问题。PAT
等方案可以依赖当前数字可微模型计算梯度，但仍要求数字模型持续跟踪真实光路。

受 fixed-feedback 微调工作的启发，本项目提出以下光学假设：如果预训练后微调造成的光学算子
漂移不大，预训练终点的反向连接可能仍与当前精确 BP 保持较高对齐，从而在下游微调中复用。

需要回答的核心问题依次是：

1. 预训练固定反馈是否只是“任意固定随机反馈也能工作”，还是包含预训练结构信息？
2. 在性能较强且真正依赖光学路径的模型中，它是否仍接近 BP？
3. 反馈算子与当前部署光路失配后，适用工作区和失效边界在哪里？
4. 能否通过鲁棒预训练扩大工作区，并通过少量校准恢复剩余性能？
5. 该结论能否从 CIFAR-10 扩展到 ImageNet 预训练的通用视觉 backbone 和多个下游任务？

## 2. 方法定义：固定的是什么，仍然更新什么

### 2.1 当前前向始终参与训练

FA 描述的是**反向误差传播规则**，不是一种相似性匹配任务。P01--P05 主线均为带标签的监督
分类；只有早期 V2 为了验证真实跨数据集迁移，使用过 SupCon、prototype 和相似度分类。

微调第 `t` 步、第 `l` 个光学层始终使用当前相位 `phi_l(t)`：

```text
当前输入 a_l(t)
  -> 当前相位 phi_l(t)
  -> 当前光学传播和 CCD 读出
  -> 当前 loss L(t)
```

每个 batch 都重新计算当前输出、loss 和输出误差。FA-pretrained 不会把前向相位替换成预训练
相位，也不会复用旧 batch 的误差信号。

### 2.2 只替换跨光学层的误差连接器

```text
BP-current:     使用当前光学算子对应的 input VJP/Jacobian 传递层间误差
FA-pretrained:  使用预训练终点保存的光学算子传递层间误差
FA-random:      使用一次初始化后固定的随机 phase-only masks 与相同传播算子传递层间误差
```

FA-random 不是显式稠密随机矩阵；它保留与真实光学层相同的 phase-only mask 形状和传播模型，
只将反馈相位随机初始化后固定。本层相位参数的局部梯度仍由当前输入、当前相位和当前误差信号
精确计算。CCD 平方律、
LayerNorm、ReLU、光电门控、轻量电子残差和读出头继续使用普通 BP。因此本方法不能表述成
“完全不需要 BP”，更准确的名称是“复用预训练光学跨层反馈连接器”。

### 2.3 唯一四个正式方法组

| 方法 | 当前前向 | 跨光学层反馈 | 参数是否更新 | 作用 |
|---|---|---|---|---|
| NoFT | 当前部署/预训练终点 | 不反传 | 否 | 微调增益和部署即时下限 |
| BP / BP-current | 当前相位和当前部署算子 | 当前精确算子 | 是 | 性能与适配上限 |
| FA-pretrained | 同 BP | 预训练或部署前最后算子 | 是 | 核心方法 |
| FA-random | 同 BP | 同形状固定随机算子 | 是 | 排除“固定本身就有效” |

架构搜索编号 A07/A13、部署阶段 P03/P04/P05、head-only、optical-off、phase-random 和
phase-shuffle 都不是额外反馈方法组。

## 3. 公平比较和审计规则

四组实验遵守以下共同控制：

- 从完全相同、SHA-256 可核验的 source/common checkpoint 开始；
- 每个 seed 内使用相同数据划分、batch 顺序、增强随机数、优化器、学习率和 epoch；
- P02/P03 正式四组协议锁定后，checkpoint 只由 validation 选择，test 不参与 method-specific
  调参或 epoch 选择；A01--A13 曾在 CIFAR-10 上反复开发结构，因此仍需如实报告同数据集选择
  偏差；
- 电子读出和电子残差在所有可训练组中都使用普通 BP；
- 同时报告任务性能、逐层梯度 cosine、相位漂移和光学破坏消融；
- `normalized optical dependence` 定义为
  `(normal_accuracy - optical_off_accuracy) / (normal_accuracy - chance_accuracy)`；
- optical gate 只表示经过分支 RMS 归一化后的数值混合下限，不等同于真实能耗、延迟或硬件
  计算占比。

证据等级：

| 等级 | 实验 | 可用于什么结论 |
|---|---|---|
| 正式多 seed test | P02、P03 | 当前主结论 |
| 已完成多 seed test、但有早期设计限制 | V1、V2、P01 | 方法演进和机制佐证 |
| 单 source/deployment seed validation pilot | P04、P05 | 可行性、边界与下一步依据，不能冒充正式泛化结论 |
| 进行中的 backbone 训练 | P09、P11 | 下一阶段基础设施，目前还不是 FA 结果 |

## 4. 实验路线总览

| 阶段 | 任务 | 主要问题 | 状态 |
|---|---|---|---|
| V1 | CIFAR-100 -> CIFAR-100-C 十类 | 固定反馈更新几何是否接近 BP | 完成；性能负结果 |
| V2 | CIFAR-100 SupCon -> CIFAR-10 | 真实跨数据集迁移 | 完成；存在 optical bypass |
| P01 | 高光学依赖骨干上的小漂移微调 | 排除弱光学路径混杂 | 3 seeds 完成 |
| A13 | RGB 八层强光学架构 | 提高性能并限制电子预算 | 4 seeds 架构复验完成 |
| P02 | A03 source operator + A13 架构上的唯一四组 | 理想环境核心对比 | 3 paired seeds 正式 test 完成 |
| P03 | 冻结模型 + 部署扰动 | 零微调工作区和失效边界 | 3×3 seeds 正式 test 完成 |
| P04 | 固定错位后继续训练 | 当前 BP 是否正确、旧算子能否校准 | validation pilot 完成 |
| P05 | 连续错位疫苗化 + held-out 校准 | 预防脆弱性并保留 FA 优势 | validation pilot 完成 |
| P09/P11 | ImageNet-1K 视觉预训练 | 构建可迁移通用光学 backbone | 正在训练，尚未进入四组 FA |

## 5. 早期实验：从几何现象到强骨干

### 5.1 V1：更新方向成立，但任务性能失败

V1 使用 20 个 400×400 OEO stages，将 clean CIFAR-100 预训练模型微调到 CIFAR-100-C 中
十个已有类别。

| 方法 | 固定 epoch-50 test | validation-selected test |
|---|---:|---:|
| NoFT | 42.33% | 42.33% |
| BP | 37.61±0.35% | 41.33±1.32% |
| FA-pretrained | 37.17±0.50% | 40.72±2.11% |
| FA-random | 37.00±1.04% | 39.11±3.53% |

epoch 50 时 FA-pretrained 的 `drift/BP=0.997`、更新 cosine 为 `0.972`；FA-random 为
`1.236/0.364`。这说明预训练反馈的更新几何更接近 BP，但所有微调方法都没有超过 NoFT，不能
作为任务有效性证据。

### 5.2 V2：真实跨数据集迁移，但光学路径被旁路

V2 使用 CIFAR-100 supervised contrastive 预训练和真实 CIFAR-10 下游任务。

| 方法 | 固定 epoch-30 test | validation-selected test |
|---|---:|---:|
| NoFT | 27.56% | 27.56% |
| BP | 31.00±0.52% | 30.70±0.28% |
| FA-pretrained | 31.02±0.52% | 30.77±0.14% |
| FA-random | 28.19±2.24% | 29.91±0.49% |

FA-pretrained endpoint cosine 为 `0.9975±0.0001`，FA-random 为 `0.3865±0.0202`。但预训练
后 20 层平均 optical residual weight 只有约 `0.070`，所以该结果仍可能受到 skip path 过强的
混杂，绝对性能也只有约 31%。

### 5.3 P01：高光学依赖、小漂移验证

| 方法 | test Top-1 | optical dependence | phase RMS drift |
|---|---:|---:|---:|
| NoFT | 55.53% | 86.65% | 0 |
| BP | 58.36±0.09% | 89.27% | 0.0676 rad |
| FA-pretrained | 58.39±0.09% | 89.28% | 0.0680 rad |
| FA-random | 57.48±0.15% | 88.08% | 0.0987 rad |

三个 seed 中 FA-pretrained 相对 FA-random 均为正，平均高 `0.90 pp`；相对 BP 的平均差值为
`FA-pretrained - BP = +0.02 pp`。但 BP/FA-pretrained 漂移只有约 0.068 rad，operator
coherence 约 0.9977，属于较容易的小漂移区间。

### 5.4 A13：强性能、强光学依赖且电子预算受控的架构

A13 不是一种反馈算法，而是 P02 锁定使用的 backbone 架构。P02 的预训练光学 source operator
仍来自 A03 CIFAR-100 checkpoint；A13 复验的作用是证明这套架构和电子预算能够承载强性能：

- RGB 输入，8 个 OEO stages；每个 stage 有 3 个 RGB bank 的独立 128×128 mask，共 24 张
  可学习相位 mask，532 nm、16 μm、5 cm；
- 光学相位参数 `393,216`；
- 每层光学主分支 gate 被约束为不低于 0.5；
- 低分辨电子残差：下采样 4 倍、hidden width 64，共 `312,336` 参数；
- MLP readout `104,330` 参数，总电子参数 `416,666`；
- 电子卷积估算约 `317.82M MAC/sample`，因此“参数量合规”不等于已经证明硬件能耗合规。

四 seed backbone 复验（前三 seed 来自原 aggregate，确认性 seed 2028 的 evaluation 加入后
重算四 seed 统计）：

| 指标 | 结果 |
|---|---:|
| CIFAR-10 test Top-1 | 72.34±0.14% |
| optical-off | 13.70% |
| random-phase | 11.28% |
| shuffled-phase | 12.81% |
| electronic-skip-off | 26.29% |
| normalized optical dependence | 94.07±1.19% |

这一步显著缓解了早期实验“性能弱、主要走电子旁路”的问题，并通过 optical-off/phase
破坏排除了纯电子旁路可以单独维持 72% 性能的解释；模型本身仍是光电协同系统。

## 6. P02：理想环境核心四组正式结果

P02 使用同一个 A03 CIFAR-100 source optical operator，在锁定的 A13 架构内先建立共同
CIFAR-10 head-warmup checkpoint，再运行 3 个严格配对 seeds，每组 50 epochs。这里**没有
加载 A13 的 CIFAR-10 best 权重**：只复用 A13 的结构和超参数；phase 来自 A03，新加入的
low-resolution residual 为零初始化，head warm-up 时全部 optical stages 冻结。随后四组共享
完全相同的 common checkpoint。

| 方法 | test Top-1 | optical-off | electronic-skip-off | optical dependence | phase RMS drift |
|---|---:|---:|---:|---:|---:|
| NoFT | 51.18±0.00% | 11.59% | 51.18% | 96.14% | 0 |
| BP | **72.30±0.54%** | 13.70% | 32.06% | 94.06% | 0.304 rad |
| FA-pretrained | **71.44±0.42%** | 12.76% | 32.87% | 95.51% | 0.295 rad |
| FA-random | 63.29±0.91% | 15.01% | 16.37% | 90.59% | 0.480 rad |

配对差值：

- `BP - FA-pretrained = 0.86±0.26 pp`；
- `FA-pretrained - FA-random = 8.15±1.32 pp`；
- FA-pretrained 起点逐层 gradient cosine 约为 `1.000`；
- FA-random 的起点 gradient cosine 均值为 `0.224`，逐运行最小层均值为 `-0.203`；
- 所有方法最小 optical gate 均不低于 `0.50008`。

P02 是当前理想环境下最重要的主结果：性能较强、光学依赖高、相位漂移明显大于 P01，并且
预训练反馈与随机反馈之间形成稳定的大差距。

## 7. P03：冻结推理部署非理想的正式结果

P03 不再训练模型，只对 P02 best checkpoints 注入固定部署误差。正式实验包含 3 个 training
seeds × 3 个 deployment seeds；先对同一训练模型的三个 deployment seeds 求均值，再在三个
独立训练模型间报告 mean±sample std。

表中 `±` 是跨 3 个独立训练模型的 sample SD；NoFT 在三个 training-seed 槽位复用同一 common
checkpoint，所以该层级的 SD 为 0。这不代表不同 deployment perturbation seeds 之间没有波动，
内部波动需查看原始 `by_training_seed`/run rows。

正式配置中未额外注明 geometry 的 shift 均为各层独立的 `layerwise` 位移。`combined operating`
为：每个 deployment seed 固定采样的零均值逐像素 Gaussian phase error（目标标准差 0.15 rad）、
layerwise 0.0625 pixel shift，以及按 spatial-intensity RMS 标度的 5% Gaussian detector noise；
不是早期计划草案中的 2 pixel 联合偏移。

| test 条件 | NoFT | BP | FA-pretrained | FA-random |
|---|---:|---:|---:|---:|
| ideal | 51.18±0.00 | 72.30±0.54 | 71.44±0.42 | 63.29±0.91 |
| phase 0.05 rad | 50.76±0.00 | 72.05±0.79 | 70.98±0.58 | 63.28±0.93 |
| phase 0.15 rad | 41.66±0.00 | 68.55±0.98 | 66.52±1.23 | 61.83±0.98 |
| shift 0.0625 pixel | 49.04±0.00 | 69.99±0.59 | 68.67±0.40 | 61.61±1.08 |
| shift 0.125 pixel | 41.34±0.00 | 62.00±0.77 | 59.44±0.58 | 56.09±0.71 |
| shift 0.25 pixel（失效边界） | 22.90±0.00 | 30.91±1.80 | 28.45±1.92 | 35.81±0.48 |
| detector 1% RMS | 51.08±0.00 | 72.33±0.57 | 71.37±0.45 | 63.32±0.95 |
| detector 5% RMS | 50.96±0.00 | 72.25±0.76 | 71.16±0.55 | 63.27±0.89 |
| combined operating | 41.12±0.00 | 65.92±0.58 | 63.92±0.40 | 59.72±1.16 |

关键配对差：

| 条件 | BP - FA-pretrained | FA-pretrained - FA-random |
|---|---:|---:|
| ideal | 0.86±0.26 pp | 8.15±1.32 pp |
| phase 0.15 rad | 2.03±0.31 pp | 4.69±2.20 pp |
| shift 0.125 pixel | 2.56±0.19 pp | 3.35±1.17 pp |
| detector 5% RMS | 1.08±0.29 pp | 7.89±1.43 pp |
| combined operating | 2.00±0.23 pp | 4.20±1.56 pp |
| shift 0.25 pixel | 2.46±0.20 pp | **-7.36±2.38 pp** |

0.25 pixel 下 random 反超不是“正常 BP 获取不到最新关系”：P03 根本没有部署后反传，且三种
高性能模型都已经从约 63%--72% 严重退化到约 28%--36%。该点在看 test 前已定义为装调失效
边界，而不是有效工作区。

## 8. P04：固定错位后的部署适配 pilot

P04 修复了早期不可微部署路径。其前向始终使用发生固定横向偏移后的**当前相位**；BP-current
使用部署后当前算子的精确 Jacobian；FA-pretrained 仅将跨层 error connector 固定为部署前最后
算子。以下结果为 validation-only、单 training seed 2026、单 deployment seed 9101，应视为
机制和可行性 pilot。

| 固定偏移 | NoFT | BP-current | FA-pretrained | FA-random |
|---|---:|---:|---:|---:|
| global 0.125 px | 57.94% | 73.54% | 73.36% | 69.24% |
| layerwise 0.125 px | 62.20% | 73.48% | 73.42% | 70.54% |
| global 0.25 px | 24.70% | 72.20% | 71.96% | 65.86% |
| layerwise 0.25 px | 29.66% | 72.56% | 72.50% | 67.40% |
| global 0.5 px | 12.06% | 66.04% | 65.44% | 61.96% |
| global 1 px | 49.68% | 73.28% | 71.42% | 69.20% |
| global 2 px | 28.76% | 72.74% | 70.04% | 68.04% |
| layerwise 0.5 px | 13.26% | 67.74% | 67.96% | 62.44% |
| layerwise 1 px | 14.84% | 71.08% | 67.68% | 65.22% |
| layerwise 2 px | 16.86% | 71.16% | 67.52% | 63.64% |

主要观察：

- 0.25 pixel 时 BP-current 从 `24.70/29.66%` 恢复至 `72.20/72.56%`，证明实现中的正常 BP
  能得到部署后的最新关系；
- FA-pretrained 在 0.125/0.25 pixel 与 BP 只差 `0.06--0.24 pp`；
- 在 0.5--2 pixel 的六个条件中，FA-pretrained 全部优于 FA-random，但与 BP 的差距随错位
  增大而总体扩大；
- 0.25 pixel 起点 FA-pretrained 对 BP 的分层 gradient cosine，global mean/min 为
  `0.9546/0.8879`，layerwise 为 `0.9675/0.9257`；random 均值只有 `0.3243/0.3659`。

更新归因进一步表明：

| 0.25 px | 方法 | 完整适配 | 仅相位更新 | 仅电子/门控更新 |
|---|---|---:|---:|---:|
| global | BP | 72.20% | 51.20% | 45.14% |
| global | FA-pretrained | 71.96% | 51.64% | 47.02% |
| global | FA-random | 65.86% | 26.86% | 54.34% |
| layerwise | BP | 72.56% | 61.28% | 44.76% |
| layerwise | FA-pretrained | 72.50% | 61.40% | 46.40% |
| layerwise | FA-random | 67.40% | 37.94% | 52.28% |

BP/FA-pretrained 的仅相位更新明显强于随机反馈，完整模型又优于任一单侧更新，说明恢复来自
光学与电子协同，而不是纯电子分支保底。

## 9. P05：连续错位疫苗化与 held-out 校准 pilot

P05 将问题从“错位后再恢复”推进到“部署前扩大容忍工作区 + 部署后校准”。训练每个 batch
连续采样 global/layerwise 位移，最大偏移在前 8 epochs 从 0.25 增加到 2 pixel：

```text
L = 0.35 * CE(ideal)
  + 0.65 * CE(shifted)
  + 0.10 * KL(stopgrad(p_ideal) || p_shifted)
```

这里 ideal 分布是停止梯度的 teacher、shifted 分布是 student；实现还按蒸馏惯例乘以
temperature squared。旧 P05 计划曾把 KL 记号方向写反，2026-08-24 已按实际训练代码更正，
不改变任何已完成运行或结果。

validation split 上的 deployment perturbation seed 9201 七环境模型选择结果：

| checkpoint | ideal | 七环境平均 | 最差环境 |
|---|---:|---:|---:|
| source / epoch 0 | 73.40% | 33.06% | 13.60% |
| vaccinated epoch 18 | 72.84% | **60.94%** | **50.36%** |

七环境平均提高 `27.89 pp`，最差环境提高 `36.76 pp`，理想环境只下降 `0.56 pp`。

训练未见过的 held-out deployment perturbation seed 9301 上，四组 10-epoch validation 校准
结果（它仍不是 test split）：

| held-out 固定偏移 | NoFT | BP-current | FA-pretrained | FA-random |
|---|---:|---:|---:|---:|
| global 1 px | 71.76% | 74.48% | 73.94% | 73.18% |
| global 2 px | 66.86% | 74.14% | 73.84% | 72.50% |
| layerwise 1 px | 55.50% | 72.80% | 71.08% | 70.18% |
| layerwise 2 px | 48.78% | 72.10% | 70.54% | 69.94% |

FA-pretrained 四个条件都优于 FA-random，距离 BP 为 `0.30--1.72 pp`。但 P05 目前仍是单
训练 seed 的 validation pilot，后续必须补多 seed/test 才能升级为主结论。

## 10. ImageNet 通用 backbone：当前扩展状态

这一阶段的目标是先训练一个性能足够强、光学参数达到百万量级且光学占比不低于 50% 的通用
视觉 source，再在下游任务中恢复唯一四组。当前 ImageNet 训练全部使用 exact BP；尚未得到
ImageNet 规模的 FA 四组结果。

### 10.1 P09：Qwen 静态 Stem + 八层光学 + width-96 电子 mixer

- 冻结 Qwen Patch/Position Stem，不加载 Transformer 或语言模型；
- 1024 -> 224 adapter，3 个 optical banks、8 个 stages；每个 stage 有 3 个独立 224×224
  phase masks，共 24 张 mask（代码中为 8 个 `[3,224,224]` 相位张量）；
- 光学相位参数 `1,204,224`；电子残差 `733,472`；
- backbone（不含临时 ImageNet head）光学参数占比 `55.51%`；
- 90 epochs、每卡 batch 96、双卡全局 batch 192。

截至本报告快照，P09 已完成 epoch 32 并进入 epoch 33：

| epoch | ImageNet-1K validation Top-1 | Top-5 |
|---:|---:|---:|
| 1 | 6.932% | 19.066% |
| 5 | 24.712% | 47.384% |
| 10 | 32.198% | 56.932% |
| 20 | 38.978% | 63.928% |
| 25 | 40.854% | 65.842% |
| 30 | 42.580% | 67.534% |
| 32 | **43.112%** | **68.118%** |

epoch 32 八层 phase gradients 全部 finite/nonzero，平均绝对相位移动约 `1.919 rad`；当前结果
仍在上升，不能用 epoch 32 代替最终 90-epoch 结果。

### 10.2 P10/P11 光学混合方式

| 版本 | 光学处理 | 训练状态 |
|---|---|---|
| P10 | `[5 mm local 2-D -> 50 mm global 2-D] × 4` | 单元测试和 GPU smoke 通过，未正式训练 |
| P11 | `[token-axis 1-D -> channel-axis 1-D] × 4` | 已在 GPU 1/2 正式启动 90 epochs |

P11 与 P09 的数据、batch、学习率、电子结构、参数量和训练周期保持一致，只改变光学 mixing
operator，因而可以做受控架构比较。启动检查已确认两个 DDP ranks 实际占用 GPU 1/2，日志已
出现 baseline 和 `epoch 1 batch 300/6672`；这只证明训练链路正常，不是性能结果。

## 11. 当前可以和不可以支持的结论

### 可以支持

1. 在 A13 强骨干和理想数字光学环境下，FA-pretrained 多 seed 性能接近 BP，并稳定优于
   FA-random。
2. 优势与逐层梯度对齐一致：预训练反馈 cosine 高，随机反馈低且部分层为负。
3. 模型性能依赖学习到的光学相位；关闭光路或随机/打乱相位会大幅退化。
4. 在冻结推理的中等相位、读出噪声和亚像素偏移工作区，P02 的方法排序仍基本保持。
5. 部署错位后，BP-current 能获得最新关系；旧预训练算子能够承担有效校准，但失配越大越不
   接近 BP。
6. 连续错位鲁棒训练能够显著扩大零微调工作区，并保留后续 FA 校准优势的初步证据。

### 目前不能支持

1. 不能声称固定反馈在任意任务、任意漂移或任意光学系统中都能替代 BP。
2. 不能声称已经实现真实硬件上的“无反向传播训练”；本层相位局部梯度和电子模块仍用 BP。
3. 不能将 P04/P05 的单 seed validation pilot 写成多 seed test 泛化结论。
4. 不能将 gate≥0.5、相位参数占比或 optical dependence 直接解释成真实能耗/延迟中的光学比例。
5. 当前探测器实验只是经过归一化的相对高斯噪声，尚未覆盖 shot noise、ADC 量化、饱和、
   动态范围、坏点和绝对光功率。
6. 尚无真实 SLM/CCD 多次独立部署 session，也没有实测校准次数、反向场测量次数、通信量、
   wall time 或能耗收益。
7. P09/P11 仍是 exact-BP backbone 预训练，不能提前作为 ImageNet FA 结论。
8. A13 在 CIFAR-10 上完成架构筛选，P02 的强下游任务仍是 CIFAR-10，存在同数据集选择偏差；
   当前正式证据还不能代替第二、第三个独立下游数据集。

## 12. 下一阶段建议的正式实验

### 12.1 完成并冻结通用 source

先完成 P09/P11 的相同 90-epoch预算，根据预注册的 validation 指标选择一个 source：

- ImageNet Top-1/Top-5；
- optical-off、random-phase 和 phase-shuffle；
- 八层梯度和相位移动；
- optical/electronic 参数、电子 MAC、显存和吞吐；
- 不使用 test 为两个光学 operator 单独调参。

### 12.2 在通用 backbone 上恢复唯一四组

对同一个冻结 ImageNet source，统一建立 downstream common head，再比较 NoFT、BP、
FA-pretrained、FA-random。建议至少覆盖：

1. 一个标准分类任务，用于与现有 CIFAR 结果衔接；
2. 一个检索或度量学习任务，验证反馈结论不依赖单一 softmax 分类头；
3. 一个难度适中的密集/感知判别任务，验证空间表征迁移。

每个任务至少 3 个严格配对 seeds；预先冻结数据划分、训练预算、validation 选模规则和
BP--FA non-inferiority margin。主表仍只出现四组，任务专用 head 均使用普通 BP。

### 12.3 将部署误差从 pilot 升级为正式结论

- 对 P04/P05 补 3 个 source seeds × 3 个 held-out deployment seeds 和最终 test；
- 报告错位幅度--零样本准确率--校准后准确率连续曲线；
- 加入反馈刷新周期：never、每 N epochs、基于 operator/gradient alignment 阈值刷新；
- 扩展到相位偏置、相位时间噪声、shot/read noise、量化、饱和和坏点的联合分布；
- 在真实硬件上报告校准/刷新次数、反向测量次数、传输状态量、wall time 和能耗代理指标。

## 13. 建议的论文/汇报图表

1. **方法图**：当前可训练 forward、固定 pretrained feedback path、四组定义。
2. **理想环境主表**：P02 四组 test、光学破坏消融和配对差。
3. **机制图**：operator drift -> layerwise gradient cosine -> BP--FA task gap。
4. **部署相图**：扰动强度 × 零样本/校准后性能，标出工作区和失效边界。
5. **预防 + 校准图**：P05 vaccination 前后连续错位曲线及四组校准效率。
6. **规模扩展图**：P09/P11 ImageNet source 及多个下游任务的唯一四组结果。
7. **系统表**：光学相位、电子参数/MAC、显存、吞吐、校准与反馈刷新预算。

## 14. 复现和证据入口

- 准确方法定义：[METHOD.md](METHOD.md)
- 早期 V1/V2：[EXPERIMENTS.md](EXPERIMENTS.md)
- A13、P01、P02 正式记录：
  [FORMAL_EXPERIMENT_LOG.md](../experiments/d2nn_cifar10_high_performance_optical_backbone/FORMAL_EXPERIMENT_LOG.md)
- 骨干优化与运行记录：
  [OPTIMIZATION_LOG.md](../experiments/d2nn_cifar10_high_performance_optical_backbone/OPTIMIZATION_LOG.md)
- P03 正式部署鲁棒性：
  [DEPLOYMENT_ROBUSTNESS_PLAN.md](../experiments/d2nn_cifar10_high_performance_optical_backbone/DEPLOYMENT_ROBUSTNESS_PLAN.md)
- P04 部署适配：
  [P04_DEPLOYMENT_ADAPTATION_PLAN.md](../experiments/d2nn_cifar10_high_performance_optical_backbone/P04_DEPLOYMENT_ADAPTATION_PLAN.md)
- P05 错位疫苗化：
  [P05_MISALIGNMENT_VACCINATION_PLAN.md](../experiments/d2nn_cifar10_high_performance_optical_backbone/P05_MISALIGNMENT_VACCINATION_PLAN.md)
- 全部历史启动命令：
  [commands/COMMANDS.md](../experiments/d2nn_cifar10_high_performance_optical_backbone/commands/COMMANDS.md)

本文中的 P02/P03 数值来自对应正式汇总；P04/P05 数值来自 validation pilot 汇总；P09/P11
状态来自 2026-08-24 服务器运行快照。后续若正式运行继续推进，应更新本文件的日期和状态，
而不是覆盖旧实验的原始结果。

P03 的标准差必须读取正式 `comparison.json` 中的 `hierarchical_summary` 和
`hierarchical_paired_accuracy_deltas`：先在每个训练模型内平均 deployment seeds，再在三个
独立训练模型间统计；不能把 9 次相关部署运行当作 9 个独立模型读取普通 `summary`。
