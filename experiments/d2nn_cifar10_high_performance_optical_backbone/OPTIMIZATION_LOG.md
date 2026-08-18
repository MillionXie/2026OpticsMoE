# 光学骨干性能优化复盘

更新日期：2026-08-18

## 目标与边界

当前阶段只优化标准反向传播（BP）下的光学神经网络骨干。正式论文比较最终只保留四组：NoFT、BP、FA-pretrained、FA-random。本文档中的消融与候选结构只是内部优化诊断，不增加正式方法组。

第一性能门槛是在 CIFAR-10 完整 10,000 张测试集上达到至少 60% Top-1；65% 视为良好，70% 以上视为强结果。模型选择只使用固定的 5,000 张验证集，测试集不用于调参。性能过门后再以关闭光路、随机相位和层间相位置换检查光学依赖，并将最优骨干冻结后用于四组微调比较。

## 固定记录规范

每次尝试必须记录：提交号、配置文件、改动变量、假设、启动命令、验证集最优轮次、完整测试集准确率、光路关闭准确率、相位破坏准确率、耗时、结论和下一步。不得只记录成功尝试。

## 已完成诊断：A00

- 来源：`d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400` 已完成结果。
- 现象：BP 测试准确率约 31%，与 FA-pretrained 基本相同；FA-random 和 NoFT 更低。
- 关键问题：预训练表征本身只有约 31% 的原型分类能力；训练后平均光学残差权重从 0.35 附近塌缩到约 0.07，电子旁路主导。
- 决策：暂停 FA 横向扩展，先构造性能足够的 BP 光学骨干；残差光学权重设置硬下限，并添加光学依赖诊断。

## 当前尝试：A01（RGB-8stage constrained residual）

状态：服务器单元测试首次运行 4 passed / 1 failed。失败来自 float32 中 0.35 表示为 0.349999994，而测试使用了无容差比较；模型行为、相位梯度与消融前向无异常。改为 1e-6 容差后复测为 5 passed。两批次端到端 smoke test 已完成，训练、best checkpoint、四种消融评估和聚合均成功，无 NaN/崩溃；两条支路 RMS 均约为 1，光学权重约为 0.50。smoke 只评估 8 个样本，其准确率不作为性能结论。

2026-08-18 阶段检查：正式训练在 GPU 1 启动，PID 3323520。epoch 1/5/10 的 train Top-1 分别为 21.73%/35.12%/42.70%，validation Top-1 分别为 28.18%/37.96%/45.22%；曲线仍在上升，满足“epoch 10 训练准确率高于 35%”的继续条件。平均光学权重为 0.4800/0.3890/0.3732，正在接近但尚未触及 0.35 下限。结论：A01 不提前停止，继续观察中后期平台。

2026-08-18 光学依赖中期检查：epoch 28 best validation 为 52.72%，在完整 10,000 张测试集上 normal/optical-off/random-phase/shuffled-phase Top-1 分别为 52.87%/15.83%/13.31%/15.95%。normal 相比 optical-off 高 37.04 个百分点，归一化光学依赖度为 86.4%。phase/electronic 参数量分别为 393,216/104,330。结论：当前性能主要依赖学习到的光学相位，而不是电子旁路；继续优化的核心缺口是将正常准确率从约 53%提高到 60%门槛。

2026-08-18 epoch-50 检查：A01 best validation 已提高到 57.26%，相对 epoch 10 提升 12.04 个百分点，继续跑完余下 cosine schedule。A03 在 epoch 23 达到 26.74% best validation（CIFAR-100），训练正常；等待其最终 best backbone 后才启动 A04，避免迁移一个尚未收敛的中间算子。

假设：旧结构丢弃颜色、电子读出过弱、残差可自由塌缩，是性能上限偏低的主要原因。先在不引入额外预训练变量的条件下，同时保留 RGB 信息、提升非线性深度、平衡各分支数值尺度并增强读出，可确定从零训练的真实上限。

相对 A00 的动作：

- 输入由灰度单路改成 RGB 三路独立相位调制；输入像素按强度解释并转为平方根振幅。
- 使用 8 个 OEO stage；每层为相位调制、角谱传播、CCD 平方律探测、全平面标准化、ReLU、重新加载。
- 光学支路和旁路逐通道 RMS 归一化，避免仅因尺度不同导致混合权重偏置。
- 残差光学权重初始为 0.50、硬下限为 0.35，禁止再次塌缩至接近零。
- 画布为 128×128，兼顾空间自由度和训练吞吐；电子头只接收 8×8 池化读出，使用 512 维单隐层分类器。
- 直接优化 CIFAR-10 交叉熵；AdamW，phase LR=3e-3，electronic/residual LR=1e-3，5 epoch warmup + cosine，80 epoch。
- 固定 45,000/5,000 的分层训练/验证划分，最终测试使用官方 10,000 张测试集。
- 增加三个内部诊断：optical-off、deterministic random phase、layer-wise phase shuffle。

配置：`configs/main.yaml`

正式启动：

```bash
PHYSICAL_GPU_INDEX=<空闲物理卡号> bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/03_train_a01.sh
```

结果待回填：

| 项目 | 结果 |
|---|---:|
| commit | 实现 b3ab2ba8；测试修正 56549567 |
| 最优验证轮次 | 76 |
| 最优验证 Top-1 | 59.86% |
| 测试 Top-1 | 59.93% |
| optical-off Top-1 | 14.92% |
| random-phase Top-1 | 13.07% |
| shuffled-phase Top-1 | 10.97% |
| full - optical-off | 45.01 个百分点；归一化依赖 90.15% |
| 单轮/总耗时 | 约 45 秒/约 60 分钟 |

继续/停止规则：

- 若前 10 epoch 训练准确率仍低于 35%，先查梯度、传播尺度和优化器，不盲目延长训练。
- 若训练准确率明显上升但验证准确率低，优先正则化/增强；若二者都低，优先结构容量、学习率和初始化。
- 若最终测试低于 60%，A02 只做同数据集优化器与残差/读出小范围筛选。
- 若从零训练接近或超过 60%，进入 A03 的 CIFAR-100 光学骨干预训练，再切回 CIFAR-10 微调。
- 达到性能门槛后才比较光学依赖；若 full 与 optical-off 几乎相同，则降低旁路或改为无旁路的后段 stage，而不是用高总准确率掩盖电子主导。

最终结论：A01 相比旧项目约 31% 的结果实现了大幅提升，并证明性能高度依赖学习到的光学相位；但测试 59.93% 比预设 60% 门槛低 0.07 个百分点，严格记为“接近但未通过”。因此不修改门槛，进入 A05 低学习率精修。

## 后续候选（尚未执行）

### A02：低成本定向调参

状态：已在 epoch 10 停止，判定无收益。

A01 在 epoch 10 的 train/validation Top-1 为 42.70%/45.22%，曲线仍呈欠拟合。A02 只将最终光学场池化从 8×8 改为 16×16，hidden dim 仍为 512，其他变量全部保持 A01 不变。假设是 8×8 读出过早丢失空间信息。A02 的 phase 参数约 39.3 万，电子头参数也约 40 万，参数量不会变成压倒性的电子主导；主要计算仍是 8 层三通道角谱传播。

配置：`configs/a02_pool16.yaml`

启动：

```bash
PHYSICAL_GPU_INDEX=<空闲物理卡号> bash experiments/d2nn_cifar10_high_performance_optical_backbone/commands/06_train_a02_pool16.sh
```

判定：比较同 epoch 验证曲线和各自 best validation；只有最终选中的候选才进入完整测试结论。若 A02 没有稳定提升，不继续扩大电子头。

实测：A02 epoch 10 的 train/validation Top-1 为 40.47%/42.64%，A01 同期为 42.70%/45.22%，A02 均低 2 个百分点以上。保留 epoch-10 checkpoint 和训练日志后终止进程。结论：更大池化读出没有改善优化或泛化，不采用，也不继续扩大电子头。

### A03/A04：CIFAR-100 监督预训练与 CIFAR-10 迁移

状态：A03 已完成；A04 正在服务器 GPU 2 训练。

A03 将 A01 的同一光学骨干分类头改为 100 类，在 CIFAR-100 的固定 45,000/5,000 分层划分上训练。A04 载入 A03 best checkpoint 中的全部 stage（相位、传播 buffer 与残差），丢弃 100 类电子头并新建 10 类头，再以较小 phase LR 做 CIFAR-10 全骨干微调。该方案直接产生“现成的预训练光学算子”，解决当前没有光学预训练骨干的问题。

- A03 配置：`configs/a03_cifar100_pretrain.yaml`
- A04 配置：`configs/a04_cifar100_to_cifar10.yaml`
- A03 启动：`commands/07_pretrain_a03_cifar100.sh`
- A04 启动：`commands/08_finetune_a04_cifar10.sh`（必须等待 A03 完成）

A03 实测：best validation 为 30.80%（epoch 78），完整 CIFAR-100 测试 normal/optical-off/random-phase/shuffled-phase Top-1 为 32.13%/4.76%/3.26%/1.22%，归一化光学依赖为 87.92%。结论：得到一个具有实质 100 类辨别能力、且强依赖学习相位的光学预训练骨干。A04 已在 GPU 2 启动，PID 3864506；随机初始化的新 10 类头使 epoch-0 validation 为 10.38%，该基线已按修正后的协议登记。

A04 最终实测：best validation 为 60.48%（epoch 43），完整 CIFAR-10 测试 normal/optical-off/random-phase/shuffled-phase Top-1 为 60.71%/12.65%/11.99%/13.64%。normal 比 optical-off 高 48.06 个百分点，归一化光学依赖为 94.77%。结论：A04 测试准确率比 A05 低 0.31 个百分点，但光学依赖高 3.67 个百分点，并且直接建立了 CIFAR-100 预训练光学算子到 CIFAR-10 微调的链条。建议 A05 保留为最高准确率参考，A03→A04 作为正式四组固定反馈实验的主预训练骨干。

### A06：教师蒸馏（仅在预训练迁移不足时）

使用固定的高性能电子教师提供软标签或中间表征，但学生推理仍只能使用已定义的光学骨干和紧凑读出。必须同时报告无蒸馏学生和光路破坏消融，避免把教师性能误记为光学能力。

### A05：A01 低学习率连续精修

状态：已完成并通过性能门槛。

A01 已接近收敛且仅差 0.07 个百分点达到测试门槛。A05 从 A01 epoch-76 best checkpoint 完整加载光学 stage 和电子头，不改变任何架构；phase/electronic LR 均降至 5e-4，residual LR 为 2e-4，继续 30 epoch cosine 精修。若仍不能稳定越过 60%，则不继续堆叠相同训练，而等待 A03/A04 的预训练迁移结果。

首次启动的 epoch-1 validation 为 57.92%，低于初始化 A01 的 59.86%。检查发现训练器原本只在新 run 内选 best，会忘记 epoch-0 初始化性能。已停止进程并修改协议：所有带初始化 checkpoint 的 run 先评估并保存 epoch-0 best，后续只有严格超过它才覆盖。这一修正同时适用于 A04。A05 将用 `FORCE_RESTART=1` 从 A01 checkpoint 干净重跑；首次两轮结果不纳入结论。

最终实测：best validation 为 60.54%（A05 epoch 28），完整 CIFAR-10 测试 normal/optical-off/random-phase/shuffled-phase Top-1 为 61.02%/14.54%/13.44%/11.29%。normal 比 optical-off 高 46.48 个百分点，归一化光学依赖为 91.10%。phase/electronic 参数分别为 393,216/104,330。结论：严格通过 60% 性能门槛，且相位破坏后的性能接近随机；A05 可作为当前最优 BP 光学骨干候选。继续 A04 的目的不是补救 A05，而是验证光学预训练能否给出更高性能或更强论文叙事。

- 配置：`configs/a05_refine_a01.yaml`
- 启动：`commands/09_refine_a05_from_a01.sh`

## 冻结 checkpoint 清单

以下 SHA-256 在服务器训练全部结束后计算。正式四组实验应引用 A03 source checkpoint 的路径与 digest；若使用 head warm-up 后的公共起点，应另行生成并记录新 digest，不能直接复用 A04 的 BP endpoint 作为其他方法起点。

| 用途 | checkpoint | SHA-256 | 大小（byte） |
|---|---|---|---:|
| 正式 source optical operator | `runs/a03_cifar100_pretrain/seed_1234/best.pt` | `f632c57cf851805090686cda81d4b4a0efc07b02c91dc0e0b63c00912247becc` | 9,188,450 |
| 预训练迁移 BP endpoint | `runs/a04_cifar100_to_cifar10/seed_1234/best.pt` | `679f9552cd402a71b0a37734640547edaf7f2d6c136a68a063fca4b0024d4486` | 8,629,858 |
| 最高准确率参考 | `runs/a05_refine_a01/seed_1234/best.pt` | `b549bd322e39aa847b20458aa09fd19373e0f49a9e14c64ce8559a933bcf5938` | 8,627,938 |
