# 性能优先的光学固定反馈微调研究架构

版本：V2，更新日期：2026-08-18

## 1. 总体顺序

本项目的实验顺序调整为：

```text
强 BP 光学模型
-> 证明模型真正依赖光学路径
-> 固化可迁移的 pretrained optical backbone
-> 比较 BP / FA-pretrained / FA-random / NoFT
-> 漂移边界、非理想条件和系统收益
```

固定反馈不能再作为第一阶段的优化对象。只有 BP 下的模型已经具备不错性能，且
关闭或扰动光学路径会明显破坏性能，才能有说服力地讨论固定反馈是否适用于光学
神经网络。

## 2. 建议形成的论文叙事

完整工作可分为四部分：

1. **高性能光学骨干**：设计一个在 CIFAR-10 等非平凡数据集上达到可展示性能的
   OEO/光电混合网络；
2. **高光学依赖度**：证明主要特征变换确实由相位调制、传播和探测完成，而不是由
   skip path 或大电子后端完成；
3. **固定反馈微调**：从同一 pretrained optical checkpoint 出发，验证
   FA-pretrained 在任务性能和更新几何上接近 BP；
4. **光学系统意义**：研究算子漂移、SLM/CCD 非理想、校准失配和周期刷新策略。

对应的核心命题是：

> 对一个性能足够强且任务决策显著依赖光学路径的预训练光学网络，在受控小算子
> 漂移的微调区间内，固定预训练反馈能否保留 BP 的主要适应收益，并减少当前反馈
> 算子的逐步同步或重新校准需求？

## 3. 数据集与任务阶梯

### 3.1 调试数据集：不作为主结果

- MNIST：检查传播、训练、读出和硬件闭环；
- Fashion-MNIST：检查比 MNIST 更复杂的灰度分类；
- 小规模类别子集：只用于单 batch、过拟合和梯度正确性验证。

这些任务可以快速暴露实现问题，但不能作为论文的主要性能证据。

### 3.2 第一主数据集：CIFAR-10 全测试集

CIFAR-10 应作为第一性能目标：类别数适中、训练成本可控，也是光学分类常用基准。
必须使用完整标准 test split，不使用几百张子集作为主数字。

建议按以下层级定义阶段门槛：

| 层级 | CIFAR-10 full-test accuracy | 用途 |
|---|---:|---|
| 工程最低线 | >= 50% | 说明模型不只是勉强高于 chance |
| 可展示线 | >= 60% | 可以进入光学比例与微调实验 |
| 较好目标 | >= 65% | 适合作为主要 backbone |
| 强目标 | >= 70% | 可与较强自由空间/混合光学工作对话 |

这些数字是项目 gate，不是跨硬件的统一 SOTA 判断。已有原始工作中，CIFAR-10
结果随架构和光电划分差异很大：三层 loose-neuron 实验约 45.6%，衍射网络集成约
61%-62%，自由空间光学编码器加数字后端约 72%-73%，某些带 CNN/光学非线性的
仿真可以超过 80%。因此本项目第一阶段以完整 test set 上稳定达到 60%-65% 为合理
目标，再尝试 70%。

### 3.3 第二主数据集：CIFAR-100

CIFAR-10 达标后再转向 CIFAR-100，用于：

- 检查 backbone 是否只适合十分类；
- 生成类别更丰富的 source pretrained checkpoint；
- 为 CIFAR-100 -> CIFAR-10 label-transfer 提供预训练状态。

CIFAR-100 初期不强行设置绝对阈值，应同时报告：

- full-model accuracy；
- 相同电子后端的 phase-random/optical-off baseline；
- 参数规模匹配的小型数字模型；
- 相对 current V2 source representation 的提升。

### 3.4 固定反馈的两个下游任务

性能骨干确定后，至少覆盖两个不同迁移类型：

1. **同标签域迁移**：CIFAR-10 clean -> CIFAR-10-C；
   - 保留同一分类头；
   - 适合展示强 pretrained model 在硬件/环境漂移下的小幅适应；
   - 最符合 small-drift fine-tuning 假设。
2. **新标签迁移**：CIFAR-100 -> CIFAR-10；
   - 先冻结 optical backbone 训练新 head；
   - 把该 head-warmup checkpoint 固定为所有反馈方法的共同起点；
   - 再比较 BP、FA-pretrained、FA-random 和 NoFT。

第一篇完整结果可以以 CIFAR-10/CIFAR-10-C 为主线，CIFAR-100 -> CIFAR-10 作为
跨标签补充。不要一开始同时扩展到 ImageNet。

## 4. 第一阶段：只优化 BP 性能

### 4.1 先把任务改回直接监督分类

当前 V2 的 SupCon + prototype 流程适合研究迁移，但不适合快速定位 backbone 的
性能上限。架构筛选阶段建议使用：

```text
optical backbone
-> compact electronic readout
-> 10 logits
-> cross-entropy
```

先在 CIFAR-10 direct supervised classification 下找出强架构；通过 gate 后再给该
backbone 增加 embedding/SupCon 目标。这样可以区分“架构性能不足”和“对比学习协议
不足”。

### 4.2 颜色信息优先

当前灰度输入丢失了 CIFAR-10 的重要信息。性能优先阶段至少比较：

- grayscale 单路；
- RGB 三个独立光学通道；
- 若硬件允许，三波长/三次曝光共享或独立 phase；
- RGB 经固定线性变换后的有限通道编码。

主结果应明确颜色编码需要的光学通道数、曝光数和电子融合操作。不能为了精度默默
引入一个大型电子 RGB encoder。

### 4.3 重新筛选深度，而不是默认 20 stage

20 个 OEO stage 并不自动意味着更强。反复 CCD 探测、归一化、ReLU 和振幅重载
可能损失信息，也容易迫使 residual 走 skip path。建议首轮比较：

- 4 stage；
- 8 stage；
- 12 stage；
- 20 stage 作为现有对照。

筛选时同时看 accuracy、训练稳定性、显存、单 epoch 时间和 optical dependence。
如果 8 stage 明显优于 20 stage，应优先使用较浅模型；固定反馈研究并不要求网络
必须很深。

### 4.4 优化传播和相位参数化

按单因素顺序验证：

1. 传播 padding/band-limit，排除无 padding 带来的循环边界伪影；
2. zero/pi phase init 与小随机相位 init；
3. phase LR、warmup 和 cosine decay；
4. 相位平滑、频谱带宽或硬件可实现性正则；
5. 逐样本/逐通道功率归一化与现有 full-plane LayerNorm 的比较。

每次只改变一个机制。架构筛选阶段可以使用单 seed；进入正式结果后必须 matched
multi-seed。

### 4.5 限制电子读出的容量

建议从小到大比较：

1. detector regions / global pooling + linear classifier；
2. 当前 20 x 20 pooling + 小型线性或两层 MLP；
3. 中等电子后端，仅作为 performance upper bound。

最终主模型不一定要电子参数最少，但必须同时给出电子 MACs、参数数和 head-only
baseline。若大部分性能来自电子后端，就不能把结果称为高光学处理比例。

### 4.6 训练协议

在结构稳定后，再逐项加入标准性能策略：

- random crop + horizontal flip；
- cross-entropy label smoothing；
- AdamW 或 SGD+momentum 对照；
- warmup + cosine decay；
- 合理 weight decay；
- Cutout/Mixup 只作为后期消融，不在首轮同时开启；
- validation 选择，test 只做最终评估。

若仍无法达到 60%，可使用电子 teacher 做 feature/logit distillation。teacher 只帮助
构造强 optical checkpoint，不能参与之后 BP/FA 方法的差异比较。

## 5. 第二阶段：提高并量化光学处理比例

“光学比例”不能只用 residual optical weight 表示。建议分成三个维度。

### 5.1 结构比例

- optical phase parameter count / total trainable parameter count；
- optical stage 数、光学传播次数和探测次数；
- electronic trainable parameters 和 MACs；
- 输入编码和读出的电子操作列表。

光学 phase 参数很多不代表它们真的被使用，所以结构比例只能作为辅助指标。

### 5.2 路径权重

- 每层 residual optical/skip weight；
- min/mean/max 以及 layer-epoch heatmap；
- optical 与 skip 分支激活 RMS/能量比；
- 两分支输出相关性。

只提高 softmax weight 也不够；如果 optical 输出与 skip 输出相同，仍不能证明光学
变换承担了关键计算。

### 5.3 因果光学依赖度：主指标

定义：

```text
optical_occlusion_drop = Acc_full - Acc_optical_off

normalized_optical_dependence
  = (Acc_full - Acc_optical_off) / (Acc_full - Acc_chance)
```

同时报告：

- optical-off；
- pretrained phase -> random phase；
- layer-shuffled phase；
- phase noise sweep；
- 只保留 electronic head 的结果。

建议的进入固定反馈阶段 gate：

- CIFAR-10 full accuracy >= 60%；
- normalized optical dependence >= 0.5；
- phase shuffle/randomization 至少造成清晰、跨 seed 稳定的性能下降；
- 平均 optical weight 不再接近当前的 0.07；
- 电子 head 单独不能复现 full-model 性能。

`0.5` 是项目预注册门槛，可以在第一次正式运行前根据 chance 和模型结构锁定，但
不能看完 test 后再调整。

### 5.4 提高光学依赖度的手段

按推荐顺序：

1. 固定 optical/skip mixing，先比较 0.35/0.65、0.5/0.5 和无 skip；
2. learned residual 加 optical-weight lower-bound/soft constraint；
3. 减少或移除跨越多个 optical stage 的长旁路；
4. 限制 electronic head 容量；
5. 用 RGB/多通道光学变换增加有效光学特征；
6. 通过 distillation 让 optical feature 对齐强 teacher feature；
7. 多探测平面或多光学分支，只在单分支性能饱和后考虑。

优化目标应是 accuracy 与 optical dependence 的 Pareto frontier，而不是单独最大化
某个 residual weight。

## 6. 推荐的性能-光学比例筛选矩阵

第一轮不要做全因子组合。建议采用逐级筛选。

### Round A：结构筛选，单 seed

固定 direct CIFAR-10 CE 任务：

| 轴 | 候选 |
|---|---|
| 输入 | grayscale / RGB 3-channel optical |
| stage | 4 / 8 / 12 / 20 |
| residual | learned / fixed 0.5 / no skip |
| readout | GAP+linear / small MLP |

先筛输入和 stage，再筛 residual，最后筛 readout；不要一次跑完所有组合。每个候选
必须同时输出 accuracy 和 optical occlusion 指标。

### Round B：训练与传播优化，三个 seeds

对 Round A 的前 2-3 个 Pareto 候选比较：

- padding/band-limit；
- phase initialization；
- optimizer/scheduler；
- optical constraint；
- 轻量数据增强。

### Round C：正式 backbone，五个 seeds

选一个主配置和一个计算量较低的备选配置：

- 完整训练；
- validation-selected checkpoint；
- full test evaluation；
- optical-off/random/shuffle/noise；
- 参数、MACs、运行时间和显存；
- 保存可公开复用的 pretrained checkpoint 和 digest。

## 7. 第三阶段：构造 pretrained optical backbone

性能和光学依赖度达标后，固定模型结构，不再为 FA 单独调 backbone。

建议保存两个 checkpoint：

1. **CIFAR-10 supervised checkpoint**：用于 clean -> CIFAR-10-C；
2. **CIFAR-100 supervised/CE+SupCon checkpoint**：用于 CIFAR-100 -> CIFAR-10。

每个 checkpoint 必须记录：

- model/config digest；
- phase/operator snapshot；
- source validation/test performance；
- optical dependence 指标；
- residual weights；
- input encoding、readout 和电子 MACs；
- 是否使用 distillation。

这两个自行训练并冻结的 checkpoint 就是项目的 pretrained optical backbone，不需要
等待社区出现现成光学基础模型。

## 8. 第四阶段：固定反馈微调

### 8.1 主比较组

- NoFT；
- BP；
- FA-pretrained；
- FA-random。

建议额外加入 head-only，区分下游收益来自电子读出还是光学相位更新。

### 8.2 公平控制

- 相同 pretrained/head-warmup checkpoint；
- 相同 seed、batch order 和 sample augmentation；
- 相同 optimizer、LR、epoch 和 checkpoint policy；
- test 不参与选择；
- geometry 只比较 matched seed、matched epoch。

### 8.3 主结果

固定反馈阶段应同时满足：

1. BP 明显优于 NoFT/head-only，证明 optical fine-tuning 有任务收益；
2. FA-pretrained 接近 BP；
3. FA-pretrained 优于 FA-random；
4. optical-off 和 phase shuffle 仍会破坏微调后性能；
5. 算子距离、gradient cosine 和 BP-FA task gap 呈可解释关系。

### 8.4 漂移边界

在主配置上改变 phase LR、horizon 和 trust-region，画：

```text
operator phasor distance
-> instantaneous gradient cosine
-> endpoint cosine
-> FA-pretrained - BP task gap
```

光学相位模型应以 phasor/operator distance 为主漂移指标，raw parameter relative
drift 只作为辅助量。

## 9. 第五阶段：物理非理想与系统指标

在性能、光学依赖和固定反馈三道 gate 全部通过后再加入：

- SLM phase quantization、bias 和 temporal noise；
- CCD shot/read noise、位深和饱和；
- 横向位移、旋转、距离和波长误差；
- forward/feedback calibration mismatch；
- fixed feedback 与 periodic refresh；
- current BP/PAT 对照。

系统报告必须包括：

- 当前相位/反馈算子同步次数；
- 反馈校准和硬件测量次数；
- 数字 FFT/传播次数；
- electronic MACs、显存和 wall-clock；
- 达到相同性能所需的 refresh 频率。

如果固定反馈仍使用与 BP 相同数量的数字 FFT，就不能声称计算复杂度更低；更可靠的
主张是降低 weight transport、当前反馈路径同步或校准频率。

## 10. 阶段 gate 总表

| Gate | 必须达到的结果 | 未通过时做什么 |
|---|---|---|
| G1 性能 | CIFAR-10 full test >= 60% | 继续优化 BP backbone，不运行 FA |
| G2 光学依赖 | normalized dependence >= 0.5，phase 扰动显著降性能 | 约束 residual、缩小 head、调整架构 |
| G3 预训练 | source checkpoint 稳定、可迁移、digest 固化 | 改进 source objective 或 distillation |
| G4 微调收益 | BP > NoFT/head-only | 修正 downstream task/LR/horizon |
| G5 固定反馈 | FA-pretrained 约等于 BP 且优于 random | 分析 operator drift 和连接器失配 |
| G6 系统意义 | 非理想下仍有效，或 periodic refresh 有明确收益 | 限定结论到理想仿真 |

## 11. 建议的最终图表

1. CIFAR-10 backbone accuracy 与 optical dependence Pareto 图；
2. 不同 stage/RGB/residual/readout 的消融表；
3. optical-off、phase-random、phase-shuffle 对照；
4. strong source checkpoint 的训练曲线和光学路径热力图；
5. 下游 NoFT/BP/FA-pretrained/FA-random/head-only 主结果；
6. operator distance -> gradient cosine -> task gap；
7. 非理想强度与 periodic refresh；
8. optical/electronic 结构比例、MACs、同步和校准代价表。

## 12. 给老师汇报的简洁版本

> 我们会先暂时放下固定反馈比较，优先在完整 CIFAR-10 上把 BP 光学模型做到至少
> 60%-65%。随后不只看相位参数数量或 residual weight，而是通过 optical-off、相位
> 随机化和层打乱定义因果光学依赖度，在准确率与光学依赖度的 Pareto 前沿上选择
> backbone。只有 backbone 同时通过性能和光学依赖两道 gate，才固化成预训练光学
> checkpoint，并进一步比较 BP、固定预训练反馈、随机反馈和不微调。最后研究算子
> 漂移以及 SLM/CCD 非理想条件下的有效边界。

## 13. CIFAR-10 性能目标的参考范围

不同论文的输入编码、光学非线性、电子后端、仿真/实测和 test 规模不同，以下数字
只用于设定项目 gate，不能直接横向排名：

- [Loose neuron array and functional learning](https://www.nature.com/articles/s41467-023-37390-3)：
  三层 LFNN 的 CIFAR-10 实测约 45.62%；
- [Ensemble learning of diffractive optical networks](https://www.nature.com/articles/s41377-020-00446-w)：
  多个衍射网络集成的完整 CIFAR-10 blind test 约 61%-62%；
- [Transferable polychromatic optical encoder](https://www.nature.com/articles/s41467-025-61338-4)：
  自由空间光学编码器加数字后端约 72.1%，重训后端约 73.2%；
- [Integrated reconfigurable photonic tensor processor](https://www.nature.com/articles/s41467-026-71599-2)：
  报告 99% optical workload，但 CIFAR-10 光子推理 72% 是在 400 张子集上；
- [Picosecond pulsed optical neural network](https://www.nature.com/articles/s41377-025-02175-4)：
  使用 CNN/ResNet 型结构和光学非线性仿真得到 82.96%。

因此当前自由空间 OEO 项目以完整 test set 的 60% 为进入下一阶段的 gate、65% 为
较好目标、70% 为强目标；达到后再比较光学依赖和固定反馈，比直接追逐跨架构最高
数字更可信。
