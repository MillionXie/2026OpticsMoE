# 光学固定反馈微调：研究架构与实验路线

更新日期：2026-08-18

## 1. 建议聚焦的核心命题

本项目不应只复现“FA-pretrained 和 BP 精度接近”这一现象，而应回答一个更适合
光学系统、也更容易形成完整论文的问题：

> 对已经预训练的光学网络，在受控的小算子漂移微调中，冻结于预训练状态的
> 光学反馈连接器能否在不逐步同步当前反馈算子的情况下，保持接近 BP 的更新
> 方向和任务性能？它的有效边界由什么决定，在光学非理想条件下是否仍成立？

这里的“预训练骨干”不要求存在类似 ImageNet ResNet 的现成公共模型。源任务上
训练完成并保存相位掩模的光学网络，本身就是后续微调所需的 pretrained optical
backbone。当前 CIFAR-100 SupCon checkpoint 已经满足这个形式定义；真正的问题是
它的表示能力和光学路径使用率都偏低，需要进一步加强。

## 2. 论文原文实际支持什么

参考论文在 Qwen2.5-3B/7B 的短程全参数 SFT 中观察到：

- FA-pretrained 以预训练权重转置作为固定反馈连接器；
- 固定的只是选定线性模块向前一级发送的 error connector；
- 当前 forward、当前 loss、当前 batch error 和当前局部参数更新仍然重算；
- LLM SFT 的相对参数漂移仅约 0.004-0.005；
- 在这一小漂移区间，FA-pretrained 的任务性能和 BP 接近，终点更新 cosine 较高；
- 论文只称结果与 small-drift explanation 一致，没有证明因果，也没有证明相对 BP
  的计算优势；
- 论文的 checkpoint 选择规则并不统一：GSM8K 以及 BP 的 SAMSum 报告中使用了
  test-selected scheduled checkpoint。当前项目坚持 validation selection，统计上更干净。

因此，不能把论文结论扩张为“微调时普遍可以使用预训练算子”。更准确的外推是：
当反馈算子相对预训练状态变化有限时，预训练连接器可能仍是当前 BP 连接器的好近似。

## 3. 当前两代实验已经得到的证据

### 3.1 V1：几何现象成立，但任务设计失败

V1 的 FA-pretrained 终点 cosine 明显高于 FA-random，但 NoFT 准确率最高，且 BP
relative drift 达到约 0.40。它只能作为早期几何验证，不能作为主性能结果。

### 3.2 V2：四组性能关系已经成立

服务器上的 V2 正式实验已经完成，所有方法使用同一 pretrained checkpoint，三个
matched seeds 的 batch order 在 30 个 epoch 中逐 epoch 一致，test 未用于选择。

| policy | BP | FA-pretrained | FA-random | NoFT |
|---|---:|---:|---:|---:|
| 固定 epoch 30 test | 31.00 +/- 0.52% | 31.02 +/- 0.52% | 28.19 +/- 2.24% | 27.56% |
| validation-selected test | 30.70 +/- 0.28% | 30.77 +/- 0.14% | 29.91 +/- 0.49% | 27.56% |

epoch 30 的匹配终点几何为：

| method | relative drift | drift/BP | cosine to BP |
|---|---:|---:|---:|
| BP | 0.1744 +/- 0.0015 | 1.000 | 1.000 |
| FA-pretrained | 0.1743 +/- 0.0015 | 1.000 | 0.9975 +/- 0.0001 |
| FA-random | 0.2385 +/- 0.0090 | 1.368 | 0.3865 +/- 0.0202 |

这比 V1 更接近一个可展示的主结果：FA-pretrained 同时匹配 BP 的性能、更新幅度和
更新方向，而且明显优于随机反馈及 NoFT。

### 3.3 当前最大混杂因素：网络绕开了光学路径

V2 把 residual optical/skip 初始化从 0.10/0.90 提高到 0.35/0.65，但在 120 epoch
预训练后：

- 平均 optical weight 约为 0.070；
- 最小 optical weight 约为 0.021；
- 最大 optical weight 约为 0.346。

微调到 epoch 30 后平均 optical weight 仍只有约 0.072。于是 BP 与 FA-pretrained
极度接近存在两种解释：

1. pretrained feedback connector 的确准确；
2. 光学路径被 skip path 强烈衰减，connector 的选择对输出本来就不敏感。

在排除第二种解释前，V2 还不能作为强机制证据。绝对准确率约 31% 也主要反映了
当前 pretrained representation 较弱，而不是固定反馈方法本身的上限。

## 4. 应该回答的五个研究问题

### RQ1：现象是否真实

在公平控制下，FA-pretrained 是否持续比 FA-random 更接近 BP，并且能保留 BP 的
下游收益？V2 已经给出肯定的初步答案。

### RQ2：现象是否由光学反馈路径造成

当网络不能绕开光学路径时，FA-pretrained 是否仍然匹配 BP？这是下一步最优先的
因果问题。

### RQ3：固定反馈的有效范围是什么

随着当前相位算子远离预训练算子，instantaneous gradient cosine、endpoint cosine
和任务性能差距如何变化？是否存在可复现的漂移阈值？这是最有论文价值的主机制。

### RQ4：预训练结构为何有用

收益来自“任意物理可实现的固定传播”、预训练相位的层特异性，还是与当前前向
算子的初始精确对齐？需要 identity、layer-shuffled、noisy-pretrained 等对照。

### RQ5：是否产生光学系统收益

在相位量化、SLM/CCD 噪声、配准误差和模型失配下，固定反馈能否减少逐步同步、
反向校准或当前数字模型更新的需求？当前仿真实现尚未证明速度或能耗优势。

## 5. 分阶段实验矩阵

### E0：锁定并展示 V2 已有结果（不重跑）

目的：形成当前 baseline，避免丢失已完成实验。

- 固化 resolved config、checkpoint digest、batch-order hash 和结果 CSV；
- 画四组 accuracy trajectory、固定终点和 validation-selected 并列表；
- 画 drift/cosine trajectory；
- 单独画 20 层 optical residual weight 热力图；
- 报告当前局限：低绝对精度、optical bypass、理想仿真。

### E1：光学路径使用率与参数更新归因（最高优先级）

先做低成本诊断，再决定是否重训架构。

1. **post-hoc optical occlusion**：评估时把各层 optical contribution 置零，观察
   accuracy/embedding 的下降；若几乎不下降，当前 backbone 不是有效光学骨干。
2. **phase permutation/noise at evaluation**：打乱或扰动相位后若性能不变，也说明
   phase 未承载主要表示。
3. **更新范围对照**：
   - readout-only；
   - phase-only BP / FA-pretrained / FA-random；
   - phase + readout；
   - full parameter（现有设置）。
4. **残差消融**：先用单 seed 筛选以下设置，再用三个 seeds 复验：
   - learned residual（现有）；
   - 固定 optical/skip = 0.35/0.65；
   - 对 optical weight 加软下限或路径使用正则；
   - 无 skip 或更少 stage 的诊断网络。

关键成功条件不是强行让 optical weight 越大越好，而是证明移除/扰动光学路径会
造成显著性能下降，同时网络仍可稳定训练。

### E2：算子漂移-反馈失配响应曲线（核心机制实验）

这部分应成为论文主图，而不是只报告一个 epoch 终点。

1. 从同一 pretrained checkpoint 出发，改变单一因素控制漂移：
   - phase LR：0.0003、0.001、0.003；
   - horizon：5、10、20、30 epoch；
   - 独立 trust-region 配置：
     `lambda * mean(|exp(i*phi)-exp(i*phi_pre)|^2)`。
2. 对每个 checkpoint 同时记录：
   - phase circular RMS；
   - operator phasor distance/coherence；
   - per-layer instantaneous gradient cosine 和 norm ratio；
   - endpoint cosine；
   - FA-pretrained 与 BP 的 validation/test gap。
3. 主横轴使用光学算子距离，而不是 raw phase 参数的相对 L2 漂移。对 phase-only
   传播，phasor distance 直接对应当前算子与预训练算子的相对失配，更有物理意义。
4. 画 `operator distance -> gradient cosine -> task gap` 的链式图，拟合或分箱展示
   fixed feedback 的有效区间。

为减少算力，先对 seed 1234 做筛选；确定 3-4 个代表性漂移点后再跑三个 matched
seeds。最终主结论配置建议增加到五个 seeds。

### E3：反馈连接器结构消融

保留现有四组作为主比较，增加以下机制对照：

- `fa_identity_phase`：无 phase screen、只保留传播 adjoint；
- `fa_shuffled_pretrained`：预训练 phase 在层间打乱；
- `fa_noisy_pretrained_sigma_*`：在预训练 phase 上加入受控角度噪声；
- `fa_random_phase`：现有物理形状兼容随机 baseline；
- `fa_periodic_refresh_K`：每 K 个 epoch 或达到漂移阈值后刷新一次 connector。

其中 noisy-pretrained sweep 最重要，因为它能主动控制 connector 失配并验证因果；
shuffled-pretrained 用于判断“预训练统计分布”与“正确层配对”哪一个重要。

### E4：构造更强的 pretrained optical backbone

没有公共光学骨干不是障碍，可以在本项目中明确构造并发布自己的 source checkpoint：

1. 先解决 optical bypass，再比较 5/10/20 stage，选择能稳定训练且确实使用相位的
   最小架构；
2. 使用更强的源任务预训练：supervised CE、SupCon 或 self-supervised objective；
3. 可使用电子 teacher 对 optical embedding 做特征蒸馏。teacher 只负责构造更强的
   光学预训练状态，不参与 BP/FA 微调比较；
4. 新任务先冻结 optical backbone，仅训练新 readout，得到所有方法共享的 downstream
   起点；再执行 phase/full fine-tuning，避免随机新 head 把反馈比较淹没；
5. 至少覆盖两类迁移：
   - 同标签域迁移：clean -> corruption/domain shift；
   - 新标签迁移：CIFAR-100 -> CIFAR-10 或更强 source -> disjoint target。

绝对精度的目标不是追逐纯电子 SOTA，而是让 NoFT、BP 和 FA 之间有足够动态范围，
同时证明任务性能确实依赖光学相位。

### E5：光学非理想与系统级实验

在理想仿真结论稳定后依次加入，避免一开始混入过多因素：

- SLM phase quantization、phase response bias 和 temporal noise；
- CCD shot/read noise、饱和与有限位深；
- 横向位移、旋转、传播距离和波长误差；
- forward model 与 feedback model 的校准失配；
- fixed feedback、periodic refresh 与 current-BP/PAT 的比较。

系统指标至少包括：每次微调需要的当前相位同步次数、反馈校准次数、FFT/数字传播
次数、显存、wall-clock 和硬件测量次数。若固定连接器仍通过相同 FFT 实现，就不能
宣称计算量低于 BP；更合理的主张是减少 weight transport、逐步校准和当前反馈路径
同步。

## 6. 主实验与消融的推荐优先级

| 优先级 | 实验 | 回答的问题 | 运行策略 |
|---|---|---|---|
| P0 | V2 结果固化和作图 | 当前现象是否完整 | 不重跑 |
| P0 | optical occlusion / phase permutation | 网络是否真的使用光学路径 | 现有 checkpoint 直接评估 |
| P1 | 固定/受约束 residual，小规模架构筛选 | 排除 optical bypass | 1 seed 筛选，3 seeds 复验 |
| P1 | drift sweep + gradient diagnostics | 固定反馈有效边界 | 1 seed 扫描，代表点多 seed |
| P2 | noisy/shuffled/identity connector | 预训练结构为何有用 | 复用同一 checkpoint |
| P2 | 更强 source pretraining / distillation | 提升可展示性能 | 独立新实验，不覆盖 V2 |
| P3 | 光学误差与 periodic refresh | 是否有系统实用性 | 仿真稳定后再做 |
| P4 | 实物平台 | 硬件证据 | 明确可观测量后再做 |

## 7. 预先注册的评价指标与成功标准

### 任务指标

- primary：固定训练预算的 test metric；
- secondary：validation-selected checkpoint 对应的 test metric；
- paired seed difference：FA-pretrained - BP、FA-pretrained - FA-random、方法 - NoFT；
- 不使用 test 选择 checkpoint 或超参数。

### 几何与机制指标

- operator phasor distance/coherence；
- phase circular RMS；
- per-layer instantaneous gradient cosine、norm ratio、relative error；
- matched-epoch endpoint cosine 和 drift ratio；
- residual optical weight 的 layer-epoch 分布；
- optical occlusion/permutation 导致的性能下降。

### 建议的成功判据

1. 在多个 matched seeds 上，FA-pretrained 的任务性能接近 BP 且优于 FA-random；
2. FA-pretrained 的 gradient/endpoint cosine 显著高于随机与错配 connector；
3. 任务性能对相位扰动和 optical occlusion 明显敏感，排除 skip-only 解释；
4. 随 operator distance 增大，gradient cosine 和任务差距呈稳定、可解释的响应；
5. 在预先定义的漂移区间内，FA-pretrained 保留 BP 的主要下游增益；
6. 最终关键配置至少五个 seeds；探索阶段三个 seeds，不把 n=3 的小差异写成显著性结论。

## 8. 建议的论文/汇报图表

1. **方法图**：current forward、current local phase update、frozen pretrained
   feedback connector 三条信息流；
2. **V2 主结果图**：四组任务性能与 paired seed 连线；
3. **光学使用率图**：20 层 residual heatmap，加 optical occlusion 前后性能；
4. **核心机制图**：operator distance vs instantaneous gradient cosine；
5. **因果链图**：operator distance vs endpoint cosine vs BP-FA task gap；
6. **连接器消融图**：pretrained、noisy、shuffled、identity、random；
7. **非理想鲁棒性图**：噪声/失配强度下 BP、FA-pretrained、periodic refresh；
8. **系统代价表**：同步、校准、数字传播、显存和 wall-clock。

## 9. 给老师汇报时的建议表述

当前可以明确说：

> 我们已经在理想光学仿真中完成两代实验。第二代真实跨数据集迁移中，固定预训练
> 反馈在三个匹配随机种子上同时复现了 BP 的任务性能与更新几何，并优于随机固定
> 反馈和不微调。但我们审计发现网络大部分走 skip path，因此下一步不会只堆更多
> accuracy，而是先证明任务确实依赖光学路径，然后系统测量固定反馈随光学算子漂移
> 的失效边界，最后再加入物理误差和周期刷新策略。

这套叙事比“把 LLM 论文搬到光学分类”更完整：它把论文中的 small-drift hypothesis
转化成可控、可测、具有光学系统意义的实验问题。
