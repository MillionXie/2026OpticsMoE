# 给 GPT-5.6 Pro 的研究规划简报：预训练固定反馈在光学神经网络微调中的迁移

更新日期：2026-08-19
项目状态基线：Git commit `f7aee642`（本简报写入前）

## 0. 希望 GPT-5.6 Pro 帮助解决什么

请把本文当作一份**事实核对后的项目输入**，在此基础上规划：

1. 一条足以支撑高水平论文的中心思想，而不是若干松散实验的拼接；
2. 主张、创新点和适用边界，尤其要判断哪些内容真正具有论文新意；
3. 从当前单一下游数据集扩展到多任务、多数据集、物理非理想和大模型的实验架构；
4. 最小可发表版本、理想顶刊版本以及二者之间的优先级和算力预算；
5. 主文图表、机制分析、关键对照、统计方案和可能被审稿人攻击的混杂因素；
6. 一条现实可执行的大模型路线：光学模块应接在大模型的哪里，预训练算子从何而来，
   固定反馈究竟替代哪一段 backward connector。

请不要默认本文后半部分列出的候选路线都正确或都需要执行；希望你进行取舍、反驳和重构。

---

## 1. 一句话项目概括

本项目研究：**在预训练光学神经网络的下游微调中，能否保持前向光学算子可训练，
但把跨光学层传播误差信号所用的 backward connector 固定为预训练时的物理算子，
从而在不持续同步反向物理模型的条件下，获得接近标准反向传播的性能。**

目前已有结果说明，这一现象在一个 CIFAR-100 预训练、CIFAR-10 微调的仿真光学网络上
可以出现；但证据仍集中于一个下游数据集、一个架构和小算子漂移区域，距离顶刊级结论
还有明显差距。

---

## 2. 原始论文背景：它实际证明了什么

参考论文是：

> *On the Surprising Effectiveness of Fixed Feedback Weights for Language Model Supervised Fine-Tuning*

原论文讨论的是大语言模型全参数监督微调中的 weight transport / backward connector 问题。
对于被替换的线性层，三种反向连接为：

```text
BP:             当前前向权重的转置 W_s^T
FA-pretrained:  预训练终点权重的固定转置 W_0^T
FA-random:      形状匹配的固定随机矩阵 B
```

必须强调：FA-pretrained **没有冻结前向权重，也没有复用旧 batch 的误差**。当前 batch
仍由当前模型计算 loss 和 output error；当前权重仍在更新；只是在把误差信号继续传给更早
的计算图时，使用固定的预训练连接器。

论文在 Qwen2.5-3B/7B 上使用 GSM8K 和 SAMSum，比较 BP、FA-pretrained、FA-random。
主要现象是：

- FA-pretrained 的任务性能通常接近 BP；
- FA-random 通常更差，但 7B GSM8K 存在随机反馈不差于 BP 的例外；
- FA-pretrained 的参数更新终点与 BP 高度相似；
- FA-random 漂移更大，更新方向与 BP 明显不同。

作者把结果解释为与“小漂移假设”一致：监督微调时模型仍靠近预训练点，因此
`W_0^T` 仍是 `W_s^T` 的良好近似。

原论文自己的局限也很重要：只有两种模型规模、两个任务、三个随机种子和短程微调；
论文**没有证明固定反馈具备普遍性，也没有证明相对 BP 的计算、能耗或硬件优势**。
它更没有直接讨论光学神经网络。

---

## 3. 为什么把这个问题迁移到光学神经网络

### 3.1 光学场景中的潜在价值

真实光学网络的前向算子由相位调制、自由空间传播、探测和光电转换共同决定。标准 BP
在训练时要求获得与当前前向光路相匹配的伴随/转置算子。如果硬件状态持续变化，反向过程
可能涉及反复校准、数字孪生同步、误差场生成或额外测量。

因此，一个有物理意义的问题是：

> 预训练结束后，能否把当时校准得到的光学 backward operator 固定下来，在下游微调中
> 继续更新前向相位，却不在每一步重新更新该反向连接器？

若答案在明确边界内为“可以”，潜在意义不是简单地说“FA 也能在光学网络运行”，而是：
**预训练物理算子可能同时充当前向初始化和可复用的反向通道。** 这可能降低在线微调时
反向模型持续同步或校准的要求。

### 3.2 这不是对 LLM 论文的机械复现

原论文替换的是 Transformer 中实值线性投影的转置。本项目的连接器来自物理传播：

```text
A_l(phi) = P_z diag(exp(i phi))
```

它包含复数场、phase-only 调制、衍射传播、平方律探测、非线性和 OEO 重加载。
本项目只在跨光学 stage 的误差传播中替换当前/冻结的物理伴随算子；CCD、LayerNorm、
ReLU、残差混合和电子读出仍用普通 BP。

光学迁移还多出几个原论文没有的核心问题：

- 没有社区现成的“光学基础模型”，预训练 backbone 必须自行构造；
- 相位参数的欧氏距离不等于实际光学算子距离，需要使用 phasor/operator 指标；
- 混合光电网络可能通过电子分支或 skip path 绕开光学计算；
- 仿真中的固定反馈是否能转化为校准次数、通信量、存储或能耗收益，需要系统测量，
  不能从准确率结果直接推导；
- 相位量化、噪声、错位、波长漂移和模型—硬件失配可能决定结论是否有实际价值。

---

## 4. 当前方法的准确数学和实现边界

每个光学 OEO stage 的简化前向过程为：

```text
phi_l = 2*pi*sigmoid(raw_phi_l)
E_l = a_l * exp(i phi_l)
U_l = P_z(E_l)
I_l = |U_l|^2
a_(l+1) = optical/skip residual mixing(LayerNorm + ReLU(I_l), a_l)
```

所有正式微调方法的前向完全一致，始终使用当前可训练相位 `phi_current`。区别只在
返回前一光学 stage 的误差信号：

```text
BP:             A_l(phi_current)^H delta_l
FA-pretrained:  A_l(phi_pretrained)^H delta_l
FA-random:      A_l(phi_random)^H delta_l
```

其中 `delta_l` 永远来自当前 batch、当前模型和当前 loss。当前层相位参数的局部梯度
仍由当前输入、当前相位和当前误差精确计算；固定连接器只改变传给更早 stage 的
`dL/da_l`。这不是冻结完整 backward graph，也不是直接用随机投影更新当前层相位。

由于光场维度很高，FA-random 没有显式构造巨大稠密随机矩阵，而是采用与真实反馈
相同形状和传播尺度的“固定随机 phase-only screen + 相同自由空间传播算子”。

### 正式主表永远只有四组

1. **NoFT**：共同起点，不做下游微调；
2. **BP**：当前物理算子作为 backward connector；
3. **FA-pretrained**：source 预训练终点的固定物理算子作为 connector；
4. **FA-random**：固定随机物理形状算子作为 connector。

`head-only`、`optical-off`、`phase-random`、`phase-shuffle`、噪声扫描等是机制消融，
不是第五、第六种正式方法。

---

## 5. 当前网络、数据与研究流程

当前高性能版本是 RGB、8 个 OEO stage 的混合光电分类网络。现阶段流程是：

```text
CIFAR-100 监督预训练
        |
        v
冻结 source optical checkpoint / 保存 pretrained feedback phases
        |
        v
冻结 optical stages，训练一个共同 CIFAR-10 head warm-up checkpoint
        |
        +----------------+------------------+------------------+
        |                |                  |                  |
      NoFT              BP          FA-pretrained         FA-random
        |                |                  |                  |
        +----------------+------------------+------------------+
                         |
                         v
       test accuracy + 光学因果消融 + 相位/算子漂移 + 梯度几何
```

四组共享同一 source、同一 head-warmup 起点、数据划分和 checkpoint policy；在每个
训练 seed 内，三种微调方法共享 batch order、augmentation seed、优化器、学习率和预算。

当前“光学处理比例”没有被粗暴等同于相位参数占比，而是分成：

- 结构指标：光学 stage、传播/探测次数、相位参数和电子参数/MAC；
- 路径指标：各层 optical/skip mixing weight、分支激活能量；
- 因果指标：关闭光学路径、随机化/打乱相位后性能下降。

主要因果指标为：

```text
normalized_optical_dependence
  = (Acc_full - Acc_optical_off) / (Acc_full - Acc_chance)
```

注意：当前的 residual optical weight 和 normalized optical dependence 都**不是**真实硬件
能耗、时延或 FLOPs 中的“光学百分比”。如果论文要声称系统优势，必须另行定义并测量。

---

## 6. 已完成工作的真实含义

### 6.1 旧版 V1/V2：建立现象，但不适合当主结果

早期 V1 使用较弱的任务设置。V2 实现了 CIFAR-100 到真实 CIFAR-10 的迁移，三 seed
固定终点结果为：

| 方法 | CIFAR-10 test Top-1 |
|---|---:|
| NoFT | 27.56% |
| BP | 31.00% ± 0.52 pp |
| FA-pretrained | 31.02% ± 0.52 pp |
| FA-random | 28.19% ± 2.24 pp |

虽然关系符合预期，但平均 optical residual weight 只有约 0.07，网络大部分走 skip path；
绝对性能也弱。因此 V2 只能作为开发历史和动机，不能成为强主结果。

### 6.2 A01–A07 是工程尝试编号，不是论文方法

`A01`、`A03`、`A07` 等只是内部性能优化和 checkpoint 追踪编号，类似实验日志的 run ID。
它们不是七个需要放入主表的算法，也不应拿来和四种正式反馈方法并列。

关键节点如下：

| 编号 | 作用 | 关键结果 | 应如何使用 |
|---|---|---|---|
| A01 | RGB 8-stage 直接 CIFAR-10 基线 | test 59.93%，光学依赖 90.15% | 证明新骨干设计有效 |
| A03 | CIFAR-100 source 预训练 | test 32.13%，光学依赖 87.92% | 正式 pretrained optical source |
| A04 | A03 → CIFAR-10 的 BP 迁移 | test 60.71%，光学依赖 94.77% | 普通光学权重下限参考 |
| A05 | 从 A01 低学习率精修 | test 61.02% | 当前最高准确率参考，但不是真正跨数据集 source |
| A07 | A03 → CIFAR-10 高光学权重 BP 可行性 | test 60.25%，光学依赖 97.71% | 下一轮四组的共同高光学设置 |

A07 相对 A04 唯一结构性变化是把每层 residual optical weight 的硬下限从 0.35 提到
0.50，训练 50 epochs。A07 最佳 checkpoint 的八层光学权重为
`[0.6245, 0.5059, 0.5031, 0.5018, 0.5006, 0.5005, 0.5007, 0.5013]`，均值 51.73%。
关闭光学、随机相位和打乱相位后的准确率分别只有 11.15%、10.95%、9.73%，说明模型
不能靠电子旁路维持 60.25% 的完整性能。

但是 A07 目前只运行了 BP，它只是“高光学设置可行性”结果，**不是第五个正式方法，
也尚未证明 FA-pretrained 在该设置下仍然有效。**

### 6.3 P01：当前最严格的四组固定反馈结果

P01 从同一个 A03 source 出发，先得到一个共同 CIFAR-10 head-warmup checkpoint，再运行
三种微调方式。三 seed 正式结果为：

| 方法 | test Top-1 mean ± std | normalized optical dependence | phase RMS drift |
|---|---:|---:|---:|
| NoFT | 55.53% ± 0.00 pp | 86.65% | 0.0000 rad |
| BP | 58.36% ± 0.09 pp | 89.27% | 0.0676 rad |
| FA-pretrained | 58.39% ± 0.09 pp | 89.28% | 0.0680 rad |
| FA-random | 57.48% ± 0.15 pp | 88.08% | 0.0987 rad |

配对关系：

- FA-pretrained − BP：`[0.00, +0.01, +0.06]` pp，均值 `+0.02 pp`；
- FA-pretrained − FA-random：`[+1.05, +0.62, +1.04]` pp，均值 `+0.90 pp`；
- BP − NoFT 均值 `+2.83 pp`；FA-pretrained − NoFT 均值 `+2.86 pp`。

机制检查也符合实现预期：起点处 FA-pretrained 的八层梯度 cosine 均约为 1；随机反馈
的浅层 cosine 较低，最后一层为 1（最后一层不经过更后的固定 connector）。

P01 可以支持：在当前高光学依赖、**小相位更新**的 regime 内，预训练固定反馈几乎复现
BP，而随机固定反馈稳定较差。

P01 不能支持：固定反馈在大算子漂移、高光学权重下限、其他数据集、其他任务、其他网络
规模、硬件非理想或真实光学平台上仍然有效。BP/FA-pretrained 的 operator coherence 约
0.9977，说明目前仍处于非常接近 source operator 的区域。

### 6.4 当前两个结果还没有合并

这是理解项目现状最重要的一句话：

```text
P01 = 四组关系已经成立，但只训练 20 epoch，test 约 58.4%，相位漂移小。
A07 = BP 达到 60.25% 且光学依赖 97.71%，但其他三组尚未运行。
```

因此下一项已经锁定的实验应是：在不再改结构、不按方法单独调参的前提下，使用 A07 的
`main_min=0.50 + 50 epoch` 共同协议，完整运行 NoFT、BP、FA-pretrained、FA-random
三个 seeds。它会回答 P01 结论能否延伸到更高光学占比和更强更新，而不是继续增加方法组。

---

## 7. 当前证据与尚不能声称的内容

### 已有直接证据

- 自行训练的光学 source backbone 可以提供 frozen pretrained feedback operator；
- 在 CIFAR-100 → CIFAR-10 迁移中，FA-pretrained 可接近 BP，并稳定优于物理形状匹配的
  FA-random；
- 该现象不只存在于早期 31% 弱模型，也存在于 58% 左右且具有强光学因果依赖的骨干；
- A07 说明 60% 准确率、每层至少 50% optical mixing 和接近随机水平的光学破坏结果可以
  同时实现；
- 固定反馈效果与小算子漂移、高梯度对齐相伴出现。

### 当前不能声称

- 不能声称固定预训练反馈普遍等价于 BP；
- 不能声称已经在多数据集、多任务或大模型上成立；
- 不能声称 A07 的高光学设置已验证四组结论；
- 不能把 residual mixing weight 称为实际“光学计算占比”或“能耗占比”；
- 不能声称省显存、加速或节能，当前实现的自定义 autograd 仍是数字仿真，尚未做系统收益测量；
- 不能声称对真实光学硬件的噪声和校准误差鲁棒；
- 不能以 60% CIFAR-10 准确率和单一架构宣称视觉性能竞争力；
- 不能把原 LLM 论文的结论直接当作本项目创新或本项目证据。

---

## 8. 目前最合理但尚待论证的中心思想

下面是给 GPT-5.6 Pro 评估和重构的**候选中心思想**，不是已经成立的论文结论：

> **Pretrained physical operators can be reused as fixed backward pathways for downstream
> adaptation, provided that operator drift is controlled. This decouples a trainable forward
> physical system from continual backward-model synchronization.**

中文可表述为：

> 在物理神经网络的下游适配中，预训练物理算子不仅是前向初始化，也可以在受控算子漂移
> 区间内充当可复用的固定反向通道，从而把“可训练的当前前向光路”与“需要持续同步的反向
> 物理模型”解耦。

这条叙事若要成立，至少要形成以下因果链，而不只是报四个准确率：

```text
source operator quality / source-target relatedness
        -> downstream operator drift
        -> frozen-vs-current feedback alignment
        -> layerwise gradient alignment and update trajectory
        -> downstream performance gap
        -> backward calibration/synchronization burden
```

可能的创新点候选如下，请 GPT-5.6 Pro 判断强弱并删减：

1. 把 pretrained fixed feedback 从实值 Transformer 线性层推广到由衍射传播产生的复值、
   phase-only 物理算子；
2. 给出“算子漂移—反馈对齐—任务性能”的可预测边界，而不是只复现一个正结果；
3. 同时约束任务性能与光学因果依赖，排除电子旁路造成的虚假成功；
4. 研究固定反馈对 forward/backward 光学失配、校准误差和物理噪声的容忍范围；
5. 把这一机制扩展到带预训练视觉/语言骨干的混合大模型，形成 optical module 的低同步成本
   适配方式。

这些创新是否足以达到目标期刊，仍需系统检索相邻工作后判断，尤其是 photonic in-situ
training、adjoint training、hardware-aware training、feedback alignment、physical neural
network transfer learning 和 optical accelerator calibration 等方向。

---

## 9. 当前实验为什么仍显单调

尽管已有多轮 run，强证据本质上仍是一个 source-target 组合：CIFAR-100 → CIFAR-10。
CIFAR-100 在当前设计中主要承担 source pretraining，并不等于已经有两个独立下游任务。

主要缺口包括：

1. **任务外部有效性不足**：只有 CIFAR-10 下游分类，没有细粒度、跨域、场景、纹理、
   corruption 或 dense prediction/generation；
2. **模型外部有效性不足**：只有一个 RGB 8-stage 模拟架构，没有深度、分辨率和容量缩放；
3. **机制范围不足**：正式四组主要落在小漂移区，没有系统找出失效拐点；
4. **物理真实性不足**：没有 phase quantization、shot/read noise、错位、波长漂移、器件漂移、
   measured transfer matrix 或 hardware-in-the-loop；
5. **系统收益不足**：没有测量反向校准次数、状态传输、显存、时延和能耗；
6. **统计力度不足**：目前三 seeds 适合 pilot，但若主张差异小且要做等效性结论，需要置信区间、
   配对设计和预先定义的 equivalence/non-inferiority margin；
7. **潜在选择偏差**：A01–A07 是在 CIFAR-10 上逐步开发出来的，后续需要锁定协议并在未参与
   调参的新数据集上验证；
8. **没有大模型桥梁**：尚未说明光学算子在 ViT/VLM/LLM 中替代哪一部分，也没有大模型结果。

所谓“fancy”不应来自堆积大量方法名，而应来自多个正交证据共同支撑同一中心思想：
跨任务复现、失效边界、物理鲁棒性、系统收益和规模扩展。

---

## 10. 请优先规划的实验工作包

以下是候选工作包，而非要求全部执行。主方法仍只比较四组；其他变化是数据、架构、漂移和
物理条件的实验轴。

### WP0：先完成已经欠缺的闭环

- 固定 A07 架构、`main_min=0.50`、50-epoch schedule 和 source/common start；
- 运行唯一四组、三个或五个 matched seeds；
- 同时报 accuracy、optical-off/random/shuffle、operator drift/coherence、逐层 gradient cosine；
- 预注册 validation、checkpoint 和等效性判断规则；
- 若 FA-pretrained 在更大漂移下开始偏离 BP，不应隐藏，这正好成为边界研究的起点。

### WP1：多数据集和 source-target 距离

目标不是把十个数据集各跑一次，而是构造有解释力的 source-target 梯度，例如：

- 近域迁移：自然图像 source → 自然图像分类；
- 中等域差：通用自然图像 source → 细粒度物体、场景或纹理；
- 强域差：自然图像 source → 遥感、医学或合成图形；
- 同类 clean → corruption/domain shift，用于测试 source operator 在分布漂移中的复用。

候选数据集可从 Tiny ImageNet/ImageNet 子集、STL-10、Caltech-101、Flowers-102、Food-101、
Oxford-IIIT Pets、DTD、EuroSAT、CIFAR-10-C 中选择，但应由 GPT-5.6 Pro 根据科学问题、
许可、预处理兼容性和算力选出最小充分集合。

理想设计应让“source-target 表征距离/初始迁移性能/算子漂移”成为连续解释变量，而不是只做
互不相干的数据集榜单。至少保留一个完全未参与当前开发的新数据集作为 confirmatory test。

### WP2：小漂移假设和失效相图

在锁定架构上系统改变：

- phase learning rate、训练时长和 trust-region 强度；
- source-target 距离、下游样本量和标签重映射程度；
- 允许更新的 optical layers 数量和位置；
- 固定 connector 的层数、周期性 refresh 间隔和校准误差；
- pretrained feedback 的质量：完整 source、早期 source、受噪 source、随机物理形状算子。

希望得到的不是“某超参数最好”，而是一张失效相图：

```text
operator distance / feedback mismatch
              x
task distance or update budget
              -> BP - FA-pretrained gap
```

同时检验 layerwise feedback alignment、instantaneous gradient cosine、endpoint update
cosine 是否能够预测任务差距，并找出 fixed feedback 需要 refresh 的阈值。

### WP3：物理非理想与真实系统意义

最低限度的仿真压力测试可包括：

- SLM 相位位深、静态 bias、时间噪声和坏点；
- CCD shot noise/read noise、位深、饱和和动态范围；
- 横向错位、传播距离误差、波长漂移；
- forward operator 与保存的 feedback operator 之间的系统失配；
- 校准频率扫描：每步更新、周期 refresh、从不 refresh。

需要区分两类问题：固定反馈是否保持任务性能，以及它是否真的减少校准/同步负担。后者应报告
明确的系统量，例如保存/传输的 operator 状态、refresh 次数、模拟 wall time、峰值显存，
以及在给定硬件模型下的能耗估算。若无法可靠估计，就只表述为潜在硬件动机。

### WP4：架构和尺度扩展

可考虑但不应无目的穷举：

- 光学 stage 深度、空间分辨率、通道/波长数；
- 纯 optical、受控 skip 和 hybrid optical-electronic 的 Pareto 曲线；
- optical stem / patch embedding / token mixer 接入预训练 ViT；
- 多光学分支或多探测面，仅在单分支确实成为性能瓶颈时加入。

主图可以是 accuracy、normalized optical dependence、BP-FA gap 与电子 MAC/校准预算之间的
Pareto front，而不是只追求 CIFAR-10 单点最高准确率。

### WP5：大模型路线

大模型不应只是换成更大的分类器，而要明确“光学物理算子”和“固定 feedback”在计算图中的
位置。可比较以下三条路线：

#### 路线 A：先复现 LLM 原论文，作为方法校准

在可承受的 Qwen 模型上复现 BP/FA-pretrained/FA-random，用于确认代码语义和大模型训练
经验。它本身接近原论文，不足以成为本项目的核心创新，但可作为基准或 appendix。

#### 路线 B：光学视觉前端 + 预训练 ViT/VLM

让光学网络承担 image formation、optical stem、patch embedding 或早期 token mixing，后接
预训练视觉/多模态大模型。先预训练/适配光学模块，再在分类、检索、VQA 或 image-to-text
任务上比较四组。固定反馈仅替换 optical module 内或跨 optical stages 的 connector，数字大
模型仍用普通 BP/PEFT。

优点是物理意义清晰、与现有光学代码距离较近；难点是建立公平的大模型 baseline，并证明
性能来自光学表征而非强数字模型绕过。

#### 路线 C：把大模型中的线性算子映射到光学硬件

选择 ViT/LLM 的 Q/K/V/O、MLP projection 或低秩 adapter，使用可由光学系统表示的复数/结构化
线性算子；预训练终点的物理映射充当 frozen feedback connector。可以研究全参数、LoRA/PEFT
和只更新光学 adapter 三种预算。

这条路线与原论文联系最直接，也可能最有规模意义，但工程风险最高：需要解决符号/复数编码、
矩阵分块、非负探测、校准误差、attention 非线性以及真实硬件映射，不能只在代码里把一个
`Linear` 改名为 optical layer。

请 GPT-5.6 Pro 判断：哪条路线最能形成一篇统一论文，哪条应作为后续独立工作。建议优先选择
一个“近程可完成”的 large-model bridge，再保留一个高风险方向，而不是三条同时铺开。

---

## 11. 可能的主文图表结构

这只是候选展示逻辑，请 GPT-5.6 Pro 重排：

1. **Fig. 1 方法图**：当前可训练 forward optical path、冻结 pretrained feedback path、四组定义；
2. **Fig. 2 主结果**：多数据集四组性能，同时标出 optical dependence；
3. **Fig. 3 机制图**：operator drift → gradient alignment → BP-FA task gap；
4. **Fig. 4 失效相图**：任务距离/更新预算 × feedback mismatch；
5. **Fig. 5 物理鲁棒性**：量化、噪声、错位、波长漂移和 refresh frequency；
6. **Fig. 6 规模或大模型图**：不同网络规模/光学插入位置下的性能—光学依赖—系统代价；
7. **Table 1**：所有数据集唯一四组主结果和配对置信区间；
8. **Table 2**：计算、显存、校准/同步次数和硬件假设；
9. **Supplement**：完整种子、layerwise curves、旧 V1/V2、实现正确性测试和 checkpoint digest。

如果篇幅有限，最重要的不是每张图都存在，而是每张图回答不同层次的问题：是否有效、为何有效、
何时失效、物理上是否有意义、是否可扩展。

---

## 12. 希望 GPT-5.6 Pro 明确回答的问题

请在规划中给出明确选择，而不只继续罗列可能性：

1. 最强的一句话 thesis 和 2–3 条可检验 claim 是什么？
2. 与原 LLM 论文相比，本项目真正的新意在哪里；哪些表述只是应用迁移？
3. 要支撑每条 claim，最少必须有哪些对照和数据集？
4. 从 WP0–WP5 中，哪些属于主线、哪些属于补充、哪些应删除？
5. 第二、第三个下游数据集应选什么，理由是什么？source pretraining 应统一还是分别建立？
6. 应如何预注册 non-inferiority/equivalence margin，而不是用“数值看起来接近”声称匹配 BP？
7. 如何定义比 residual weight 更严谨的 optical processing share 和系统收益？
8. 漂移应使用 phase RMS、phasor distance、operator norm、输出相关还是其他指标？
9. 哪种大模型路线最现实，光学层的输入输出、预训练来源和 backward connector 应如何定义？
10. 最可能导致拒稿的三个问题是什么，分别需要哪个实验解除？
11. 请给出按依赖关系排序的实验表：配置数 × 方法数 × seeds × 预计训练预算，并区分
    minimum publishable package 与 top-journal package；
12. 请给出最终论文题目候选、摘要逻辑、主文 figure/table storyboard 和阶段性止损标准。

---

## 13. 代码、结果和复现入口

专题入口：`FixedFeedbackSFT/`

- 原论文：`FixedFeedbackSFT/literature/paper.pdf`
- 准确方法定义：`FixedFeedbackSFT/METHOD.md`
- 旧实验与边界：`FixedFeedbackSFT/EXPERIMENTS.md`
- 既有研究计划：`FixedFeedbackSFT/RESEARCH_PLAN.md`
- 高性能优化复盘：
  `experiments/d2nn_cifar10_high_performance_optical_backbone/OPTIMIZATION_LOG.md`
- P01 四组正式日志：
  `experiments/d2nn_cifar10_high_performance_optical_backbone/FORMAL_EXPERIMENT_LOG.md`
- 高性能实验命令：
  `experiments/d2nn_cifar10_high_performance_optical_backbone/commands/COMMANDS.md`

关键 checkpoint：

| 用途 | SHA-256 |
|---|---|
| A03 CIFAR-100 source optical operator | `f632c57cf8518050904...247becc` |
| P01 shared CIFAR-10 head-warmup start | `deceeec8dfad0026904...0c31a` |
| A07 high-optical BP endpoint | `a9b3784ad392dc19546...487084` |

完整 digest 以实验日志为准。服务器项目路径为 `/DATA/DATA1/guest3/2026OpticsMoE`，正式训练
应在服务器执行。所有新增启动指令应进入对应实验的 `commands/` 目录；每次代码或文档修改后
通过 Git 保持本地、远端仓库和服务器一致。本文不记录服务器密码或其他凭证。

---

## 14. 阅读本简报时最容易误解的四点

1. **A07 不是算法名。** 它只是一次高光学 BP 可行性 run；正式算法始终只有四组。
2. **FA-pretrained 不冻结前向。** 前向相位继续训练，只固定跨层误差传播所用的 source operator。
3. **现在不是已经有两个强下游数据集。** CIFAR-100 是 source，强结果的下游仍只有 CIFAR-10。
4. **当前结论是“小漂移下有效”，不是“固定反馈普遍替代 BP”。** 把边界找清楚可能比继续追求
   一个更高的 CIFAR-10 数字更有研究价值。
