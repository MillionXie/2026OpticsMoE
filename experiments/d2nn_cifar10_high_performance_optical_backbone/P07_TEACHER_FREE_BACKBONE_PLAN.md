# P07 无教师通用光学 Backbone 训练与验证计划

## 1. 本阶段的判断

P06 的最佳 8x224 主干包含 1,204,224 个光学相位参数，完整 ImageNet-1K 验证
Top-1 为 9.748%。它证明了百万量级光学主干可以稳定执行全主干 BP，并且关闭光学或随机化
相位后性能接近随机；但它仍同时优化 ImageNet 标签和 CLIP 教师目标，不能回答“教师是否阻碍了类别
判别”，也还没有完成跨任务迁移验证，因此暂时不能把“通用 backbone”作为已经成立的结论。

P07 去掉 CLIP 教师。纯监督配置必须同时满足：

- 不创建 CLIP projection head；
- 不实例化 `ClipFeatureStore`，不读取 teacher embedding memmap；
- 不加载 CLIP 文本原型；
- 损失仅为 ImageNet 监督交叉熵及标签正则；
- 结果只报告 CE、Top-1/Top-5、梯度、光学门控和光学破坏消融。

当前 P06 最佳权重曾受 CLIP 训练，因此从它继续的 P07-S 只能用于快速选读出头、判断“从现在开始
移除教师”是否有收益，不能称为全程无教师证据。正式 P07-F 将从未接触 CLIP 的 P05 算子扩展
初始化开始，重新完成长周期 ImageNet 监督预训练。

## 2. 什么才算一个可复用 backbone

训练数据集不是定义 backbone 的充分条件。P07 需要同时交付以下四项：

1. **可丢弃任务头**：ImageNet 读出头不属于部署主干；预训练结束额外导出只含 `encoder` 的
   `backbone.pt`。
2. **稳定特征接口**：`forward_backbone()` 返回最后一层以及八层 O/OEO stage maps。分类任务可
   池化第 2/4/6/8 层，分割等密集任务可直接使用空间特征。
3. **规范预训练记录**：完整 ImageNet-1K train/validation、明确 epoch、有效 batch、增强、优化器、
   warm-up、cosine schedule、checkpoint SHA-256 和全层梯度审计。
4. **跨任务可迁移**：至少通过冻结线性探测、全量微调和一个密集/回归任务，证明提升来自可迁移
   表征而非 ImageNet 分类头记忆。

这与经典 ResNet 将 ImageNet 表征迁移到检测/分割、ConvNeXt 同时在 ImageNet、COCO、ADE20K
验证通用性的做法一致。自监督工作也通常使用冻结 linear probe 与完整 fine-tuning，而不是只看
预训练损失：

- ResNet: https://arxiv.org/abs/1512.03385
- SimCLR: https://arxiv.org/abs/2002.05709
- ConvNeXt: https://arxiv.org/abs/2201.03545
- DINOv2: https://arxiv.org/abs/2304.07193
- FPN（多层空间特征用于密集任务）: https://arxiv.org/abs/1612.03144

## 3. 有限的读出头筛选

只筛三个单头候选，不增加正式反馈方法组数。三者从同一个 P06 encoder checkpoint 仅加载主干，
所有读出参数重新初始化；训练样本、增强、学习率和 epoch 完全一致。

| 内部候选 | 结构 | 临时分类头约束 | 要回答的问题 |
| --- | --- | --- | --- |
| S1 Linear | 2/4/6/8 层各自 8x8 avg/max，拼接后 Linear | residual+head <2M | 当前表征是否已线性可分 |
| S2 MLP | 同一 1536 维描述符，1536→256→1000 | residual+head <2M | 少量非线性是否足够 |
| S3 Conv+MLP | 四层各自池化到 8x8，轻量 3x3+depthwise/1x1 融合，双池化后 MLP | residual+head 约0.65M | 保留局部邻域后是否更易读出 |

三者都没有第二个 CLIP 头。筛选使用每类 100 张训练图、完整 50k validation，先 1 epoch 冻结主干
校准新头，再 2 epoch 全主干正常 BP；采用相同 RandAugment、label smoothing=0.1、50% 概率的
Mixup/CutMix。选择规则首先看 joint-BP 后的完整验证 Top-1，其次看 Top-5、电子参数量与稳定性。

## 4. 正式 ImageNet 监督预训练

选定读出后启动 P07-F，不能把三轮筛选当成正式 backbone 训练。锁定建议配方如下：

- 初始化：P05 的 8 层、128x128、无 CLIP 算子；相位 bicubic 扩到 224x224，其余可兼容 OEO
  参数加载。该线从头到尾不接触教师特征。
- 数据：ImageNet-1K 全部 1,281,167 train 和 50,000 validation；每张图每 epoch 一种确定性增强
  view，四个 epoch 完成一次 view cycle。
- 时长：先跑 30 epoch 主周期；若 epoch 20--30 仍持续改善，延长至 50 epoch，而不是预先用
  10 epoch 宣称训练完成。
- 优化：AdamW，phase/residual/head 分组学习率；5 epoch 等价 warm-up，cosine decay；
  label smoothing、RandAugment、Mixup/CutMix；AMP 溢出跳步必须计数。
- 模型选择：只按完整 50k validation Top-1 选 joint-BP checkpoint；head-only 权重不能成为
  backbone best。
- 硬约束：八层 gate 均不低于0.5；光学相位1.204M；残差电子+临时预训练头低于2M；八层相位
  梯度 finite/nonzero。
- 产物：`best.pt`（可恢复全部训练）、`backbone.pt`（只含主干）、配置 digest、SHA-256、逐 epoch
  metrics 和 optical-off/phase-random 消融。

为了避免混淆，P07-S 的“现有最佳主干继续训练”和 P07-F 的“全程无 CLIP 正式训练”必须分表
报告；前者用于工程选择，后者才是消除教师变量后的论文证据。

## 5. Backbone 验证协议

ImageNet 性能是准入门槛，不是终点。读出锁定后按三类任务验证：

1. **ImageNet-1K**：Top-1/Top-5；同时报告参数、耗时、光学破坏依赖。
2. **Caltech101 分类**：冻结 backbone 只训练 linear probe；再做全量 fine-tune。与同结构随机初始化
   比较，证明迁移增益。
3. **KADID-10k 质量回归**：使用冻结特征和轻量回归头，报告 SRCC/PLCC，验证表征不只记类别。
4. **ISIC2016 分割**：读取多层 stage maps，接小型 U-Net/FPN 类解码器，报告 Dice/IoU，验证
   空间特征可用于密集预测。

进入固定反馈论文主实验后，NoFT、BP、FA-pretrained、FA-random 仍然只有这四组；读出头筛选不是
第五/第六组。四组在锁定 backbone 和下游任务后共享数据划分、初始化和电子头，比较的唯一核心
变量仍是反向算子。

## 6. 判定标准与停止规则

- **分类器达标**：P07-F 明显超过当前 9.748%，且不是仅由预训练头增大换来；首先以完整
  ImageNet Top-1 15% 作为阶段门槛，再根据曲线决定是否扩容或延长。
- **backbone 达标**：冻结 linear probe 在至少两个下游数据集稳定优于随机初始化，并且完整微调
  与非分类任务均有正迁移。
- **光学成立**：optical-off/phase-random 显著破坏性能，所有 gate>=0.5，主干电子残差受预算约束。
- 若 10 个正式 epoch 连续无新 best，先检查训练/验证差距、增强强度和各层梯度，再决定调整 LR
  或读出；不靠盲目增加 epoch 掩盖平台期。
- 若 ImageNet 提高但 frozen transfer 不提高，不能称为更好的 backbone，应回到表征接口或预训练
  目标，而不是继续堆分类头。
