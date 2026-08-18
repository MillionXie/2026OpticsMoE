# FA-pretrained 光学神经网络微调专题

本目录是 `2026OpticsMoE` 中 fixed pretrained feedback（简称
FA-pretrained）研究的统一入口，用于把研究动机、数学定义、现有实现、
已完成结果和下一步计划交接给后续会话或其他合作者。

## 最重要的结论

FA-pretrained 固定的是**预训练结束时各光学层对应的反馈算子**，不是某个
样本或某个 batch 的误差信号。微调时：

- 前向传播始终使用正在更新的当前相位；
- 当前模型每一步都重新计算 loss 和 output error；
- 返回前一层的光学误差连接器使用预训练时保存的相位；
- 当前层相位参数的局部梯度仍使用当前输入、当前相位和当前误差；
- CCD、LayerNorm、ReLU、残差权重和电子读出仍使用普通 BP。

因此，它研究的是：在小漂移微调中，能否复用预训练光学网络的固定反馈
连接器，减少真实光路中反向误差场生成和测量的需求。

## 阅读顺序

1. [HANDOFF.md](HANDOFF.md)：给下一个会话的完整背景和当前状态。
2. [METHOD.md](METHOD.md)：数学定义、光学数据流和代码中的准确实现。
3. [EXPERIMENTS.md](EXPERIMENTS.md)：两代实验、已有结果和已知问题。
4. [RESEARCH_PLAN.md](RESEARCH_PLAN.md)：面向汇报和后续论文的完整实验架构。
5. [NEXT_STEPS.md](NEXT_STEPS.md)：最近一轮的执行优先级。
6. [CODE_INDEX.md](CODE_INDEX.md)：源码、配置、结果和测试入口。
7. [COMMANDS.md](COMMANDS.md)：可直接执行的命令。

## 为什么没有把源码物理移动到本目录

两套源码是可导入的 Python 实验包，已有服务器命令、checkpoint 和结果文件
均使用 `experiments.<module>` 路径。直接移动会破坏：

- `python -m experiments...` 启动方式；
- checkpoint 中记录的配置与输出路径；
- 结果生成脚本使用的相对目录；
- 现有测试和服务器复现实验。

所以本目录作为稳定的专题入口，源码继续保留在 `experiments/`。后续若确实
要重构，应新建第三代实验，而不是移动或覆盖已经完成的实验。

## 当前两套实现

| 代次 | 实验 | 状态 | 用途 |
|---|---|---|---|
| V1 | `d2nn_cifar100c10_fixed_feedback_20stage400` | 已完成三随机种子正式实验 | 验证反馈方向与参数漂移 |
| V2 | `d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400` | 预训练、四组正式实验和三随机种子聚合均已完成 | 验证真实跨数据集迁移，并暴露 optical bypass 问题 |

V2 固定 epoch 30 的 test accuracy 为 BP 31.00 +/- 0.52%、FA-pretrained
31.02 +/- 0.52%、FA-random 28.19 +/- 2.24%、NoFT 27.56%。这说明主性能关系
已经成立，但绝对性能和光学依赖均不足：预训练后平均 optical residual weight 只有
约 0.07。当前路线已调整为先在 BP 下建立 full-test accuracy >= 60% 的 CIFAR-10
backbone，再通过 optical-off/phase-random 等因果消融把光学依赖度提高到预注册门槛，
最后才恢复 fixed-feedback 主实验。

本专题最后整理日期：2026-08-18。
