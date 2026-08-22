# 光学骨干性能优化复盘

更新日期：2026-08-19

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

## 已完成尝试：A01（RGB-8stage constrained residual）

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

最终结果：

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

## 后续尝试

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

状态：A03/A04 均已完成。

A03 将 A01 的同一光学骨干分类头改为 100 类，在 CIFAR-100 的固定 45,000/5,000 分层划分上训练。A04 载入 A03 best checkpoint 中的全部 stage（相位、传播 buffer 与残差），丢弃 100 类电子头并新建 10 类头，再以较小 phase LR 做 CIFAR-10 全骨干微调。该方案直接产生“现成的预训练光学算子”，解决当前没有光学预训练骨干的问题。

- A03 配置：`configs/a03_cifar100_pretrain.yaml`
- A04 配置：`configs/a04_cifar100_to_cifar10.yaml`
- A03 启动：`commands/07_pretrain_a03_cifar100.sh`
- A04 启动：`commands/08_finetune_a04_cifar10.sh`（必须等待 A03 完成）

A03 实测：best validation 为 30.80%（epoch 78），完整 CIFAR-100 测试 normal/optical-off/random-phase/shuffled-phase Top-1 为 32.13%/4.76%/3.26%/1.22%，归一化光学依赖为 87.92%。结论：得到一个具有实质 100 类辨别能力、且强依赖学习相位的光学预训练骨干。A04 训练启动时，随机初始化的新 10 类头使 epoch-0 validation 为 10.38%，该基线已按修正后的协议登记。

A04 最终实测：best validation 为 60.48%（epoch 43），完整 CIFAR-10 测试 normal/optical-off/random-phase/shuffled-phase Top-1 为 60.71%/12.65%/11.99%/13.64%。normal 比 optical-off 高 48.06 个百分点，归一化光学依赖为 94.77%。结论：A04 测试准确率比 A05 低 0.31 个百分点，但光学依赖高 3.67 个百分点，并且直接建立了 CIFAR-100 预训练光学算子到 CIFAR-10 微调的链条。建议 A05 保留为最高准确率参考，A03→A04 作为正式四组固定反馈实验的主预训练骨干。

### A06：教师蒸馏（仅在预训练迁移不足时）

使用固定的高性能电子教师提供软标签或中间表征，但学生推理仍只能使用已定义的光学骨干和紧凑读出。必须同时报告无蒸馏学生和光路破坏消融，避免把教师性能误记为光学能力。

### A05：A01 低学习率连续精修

状态：已完成并通过性能门槛。

A01 已接近收敛且仅差 0.07 个百分点达到测试门槛。A05 从 A01 epoch-76 best checkpoint 完整加载光学 stage 和电子头，不改变任何架构；phase/electronic LR 均降至 5e-4，residual LR 为 2e-4，继续 30 epoch cosine 精修。若仍不能稳定越过 60%，则不继续堆叠相同训练，而等待 A03/A04 的预训练迁移结果。

首次启动的 epoch-1 validation 为 57.92%，低于初始化 A01 的 59.86%。检查发现训练器原本只在新 run 内选 best，会忘记 epoch-0 初始化性能。已停止进程并修改协议：所有带初始化 checkpoint 的 run 先评估并保存 epoch-0 best，后续只有严格超过它才覆盖。这一修正同时适用于 A04。A05 将用 `FORCE_RESTART=1` 从 A01 checkpoint 干净重跑；首次两轮结果不纳入结论。

最终实测：best validation 为 60.54%（A05 epoch 28），完整 CIFAR-10 测试 normal/optical-off/random-phase/shuffled-phase Top-1 为 61.02%/14.54%/13.44%/11.29%。normal 比 optical-off 高 46.48 个百分点，归一化光学依赖为 91.10%。phase/electronic 参数分别为 393,216/104,330。结论：严格通过 60% 性能门槛，且相位破坏后的性能接近随机；A05 是该阶段的最优 BP 光学骨干候选。继续 A04 的目的不是补救 A05，而是验证光学预训练能否给出更高性能或更强论文叙事。

- 配置：`configs/a05_refine_a01.yaml`
- 启动：`commands/09_refine_a05_from_a01.sh`

## 冻结 checkpoint 清单

以下 SHA-256 在服务器训练全部结束后计算。正式四组实验应引用 A03 source checkpoint 的路径与 digest；若使用 head warm-up 后的公共起点，应另行生成并记录新 digest，不能直接复用 A04 的 BP endpoint 作为其他方法起点。

| 用途 | checkpoint | SHA-256 | 大小（byte） |
|---|---|---|---:|
| 正式 source optical operator | `runs/a03_cifar100_pretrain/seed_1234/best.pt` | `f632c57cf851805090686cda81d4b4a0efc07b02c91dc0e0b63c00912247becc` | 9,188,450 |
| 预训练迁移 BP endpoint | `runs/a04_cifar100_to_cifar10/seed_1234/best.pt` | `679f9552cd402a71b0a37734640547edaf7f2d6c136a68a063fca4b0024d4486` | 8,629,858 |
| 最高准确率参考 | `runs/a05_refine_a01/seed_1234/best.pt` | `b549bd322e39aa847b20458aa09fd19373e0f49a9e14c64ce8559a933bcf5938` | 8,627,938 |
| 高光学占比 BP endpoint | `runs/a07_high_optical_cifar100_to_cifar10/seed_1234/best.pt` | `a9b3784ad392dc19546266c0c804dc8f77a2c90d92812ae425b2d3c52a487084` | 8,630,498 |

## P01 正式反馈对性能优化的结论（2026-08-18）

基于 A03 预训练光学算子的唯一四组实验已完成三个 seed。NoFT、BP、FA-pretrained、FA-random 的 CIFAR-10 test Top-1 分别为 55.53%、58.36%±0.09 pp、58.39%±0.09 pp、57.48%±0.15 pp；后三组平均归一化光学依赖分别为 89.27%、89.28%、88.08%。FA-pretrained 在三个 seed 中相对 FA-random 分别提高 1.05、0.62、1.04 pp，同时与 BP 的差值只有 0.00、0.01、0.06 pp。

这证明当前骨干已经能在不是电子旁路主导的条件下展示预训练固定反馈的价值，但也暴露出两个下一阶段性能缺口：正式协议只训练 20 epoch，低于 A04 的 50 epoch，因此 58.39% 仍低于 A04 的 60.71%；BP/FA-pretrained 的平均相位 RMS 漂移约 0.068 rad，属于小更新范围。下一优化动作是保持正式方法仍为四组，只构造一个更强的共同设置：延长 schedule、适度提高 phase LR，并提高 residual optical weight 的硬下限。先用 BP 验证该设置能兼顾接近/超过 60% 的性能和更高光学占比，再冻结设置运行其余三组，避免按方法单独调参。

## A07：预训练迁移下的高光学占比骨干

状态：已完成并通过 60% 性能门槛；不计为第五个正式方法组。

- 配置：`configs/a07_high_optical_cifar100_to_cifar10.yaml`；
- 启动：`commands/17_train_a07_high_optical.sh`；
- source：A03 checkpoint，SHA-256 `f632c57cf851805090686cda81d4b4a0efc07b02c91dc0e0b63c00912247becc`；
- 相对 A04 的唯一实验变量：`residual.main_min` 从 0.35 提高到 0.50；
- 保持不变：RGB 八层结构、CIFAR-100 source、随机种子、50 epoch、phase LR=1e-3、electronic LR=1e-3、residual LR=5e-4、数据划分和 validation-best 规则；
- 目标：test Top-1 尽量达到 60%，同时所有 stage 的光学混合权重不低于 0.50，并通过 optical-off/random/shuffle 诊断；
- 判定：若性能明显低于 A04，A07 仍作为“提高物理处理比例的代价”保留记录，不通过增加电子头容量掩盖问题。

最终结果：

| 项目 | A07 | A04 参考 |
|---|---:|---:|
| selected epoch | 48 | 43 |
| best validation Top-1 | 60.40% | 60.48% |
| test Top-1 | 60.25% | 60.71% |
| optical-off Top-1 | 11.15% | 12.65% |
| random-phase Top-1 | 10.95% | 11.99% |
| shuffled-phase Top-1 | 9.73% | 13.64% |
| normalized optical dependence | 97.71% | 94.77% |
| selected-checkpoint mean optical weight | 51.73% | 约 37.58% |

A07 最佳 checkpoint 的八层光学权重为 `[0.6245, 0.5059, 0.5031, 0.5018, 0.5006, 0.5005, 0.5007, 0.5013]`。相对 A04，测试性能只下降 0.46 pp，平均光学权重提高约 14.15 pp，归一化光学依赖提高 2.94 pp；三种相位/光路破坏后的准确率均降到接近随机水平。结论：A07 在该阶段是更适合作为“高光学占比”主设置的骨干，严格通过 60% 门槛。

冻结 checkpoint：`runs/a07_high_optical_cifar100_to_cifar10/seed_1234/best.pt`，SHA-256 `a9b3784ad392dc19546266c0c804dc8f77a2c90d92812ae425b2d3c52a487084`，大小 8,630,498 byte。当时计划把 A07 的 `main_min=0.50` 与 50 epoch schedule 固定为四组共同协议；随后按用户要求先继续优化骨干性能，形成 A08--A13。

## A08–A10：高光学约束下的轻量电子残差筛选

状态：2026-08-19 已完成训练与统一消融。它们是 BP 骨干优化 run，不是新增正式方法。

动机：A01–A07 的三个颜色通道在八层光学传播中完全独立，只有最终 MLP 读出能够混合 RGB；每层 bypass 也只是原振幅的 RMS-normalized identity。用户允许在保证每层 optical gate 不低于 0.5 的前提下，对 bypass 做少量电子处理，并建议尝试 U-Net 式跳连。

共同控制：

- A03 CIFAR-100 source checkpoint 和 `load_backbone_only=true`；
- CIFAR-10 split、seed 1234、数据增强、50 epochs 和 A07 的全部 optimizer 设置；
- 每层 `residual.main_min=0.50`，电子分支永远只位于剩余不超过 0.50 的 bypass 内；
- 电子残差修正系数限制为 `<=0.25`，U-Net-like 长跳连权重限制为 `<=0.25`；
- MLP readout 在第一轮保持不变，避免把 branch 与 head 的效果混在一起；
- validation 选择 checkpoint，test 不参与选择。

候选：

| run | 唯一新增结构 | 目的 |
|---|---|---|
| A08 | 每层 pointwise bottleneck | 用极少电子参数实现 RGB 交互 |
| A09 | depthwise 3×3 + pointwise bottleneck | 在 bypass 中补充局部空间归纳偏置 |
| A10 | A09 + stage 0→7、1→6、2→5 的有界长跳连 | 检验跨深度特征复用是否改善优化 |

完整模型之外必须报告：

- optical-off：电子路径单独能达到什么性能；
- phase-random / phase-shuffle：学习到的物理相位是否必要；
- electronic-skip-off：新增电子局部变换的直接消融；
- long-skip-off：长跳连的直接消融；
- 每层 optical gate、电子修正系数、长跳连权重与参数量。

保留标准：相对 A07 的 60.25%，优先选择 validation/test 提升且 normalized optical dependence 仍高的 Pareto 候选。若完整性能提高但 optical-off 同幅提高、相位破坏不再接近随机水平，则判定为电子旁路扩张，不进入下一阶段。第一轮胜者再与小型卷积 readout 做单变量第二轮比较。

中期预注册补充：A08 在 epoch 34 达到 63.34% validation，A09/A10 到 epoch 40 时若
validation-best 仍落后 A08 超过 0.5 pp，则判定为被支配候选，停止剩余 10 epochs，保留 best
并补齐统一消融。A08 仍跑满 50 epochs。该止损规则在读取 A09/A10 epoch-40 结果前写入。

第二轮固定 A08 pointwise bypass，只改变读出头：A11 使用参数量与旧 MLP 同量级、保留 2×2
空间布局的小卷积头；A12 拼接 8×8 average/max pool 后使用 MLP。两组仍使用 A03 source、
`main_min=0.50` 和相同 50-epoch 优化协议，并与 A08 当前 MLP 头直接比较。

电子预算补充：实验室允许 residual electronic processing 总参数在 1–2M 以内，经验上几十万
参数合理。A08 的 592 参数虽然高效，但不能代表允许预算下的性能上限。因此新增 A13：每层
旁路降采样到 32×32，以 64 通道进行低分辨率空间处理，再上采样并以 `scale<=0.25` 加回旁路；
八层约 0.31M residual 参数，连同约 0.10M MLP head 总电子参数约 0.42M。它仍严格位于
`1-alpha<=0.5` 的 bypass 中，并使用与 A08/A07 相同 source 和训练协议。

第一轮完整结果（单 seed，仅用于架构筛选）：

| 项目 | A07 参考 | A08 pointwise | A09 depthwise | A10 depthwise + long skip |
|---|---:|---:|---:|---:|
| selected epoch | 48 | 48 | 41 | 48 |
| best validation Top-1 | 60.40% | **63.64%** | 62.72% | 63.22% |
| test Top-1 | 60.25% | 62.05% | **63.05%** | 62.79% |
| optical-off Top-1 | 11.15% | 15.47% | 16.26% | 14.95% |
| random-phase Top-1 | 10.95% | 13.66% | 15.24% | 12.65% |
| shuffled-phase Top-1 | 9.73% | 11.18% | 10.93% | 11.99% |
| electronic-skip-off Top-1 | 不适用 | 38.52% | 32.87% | 29.64% |
| long-skip-off Top-1 | 不适用 | 62.05% | 63.05% | 60.60% |
| normalized optical dependence | 97.71% | 89.49% | 88.20% | 90.62% |
| residual electronic parameters | 0 | 592 | 856 | 859 |

A08/A09/A10 都比 A07 提高约 1.8--2.8 pp，同时 optical-off 和两种相位破坏仍接近随机，
因此提升不能用电子旁路单独分类解释。A09 的单次 test 最高，但 A08 的 validation 最高；在只做
单 seed 架构筛选时不能依据 test 反选 A09。A10 的 `long-skip-off` 比完整模型低 2.19 pp，说明
长跳连确实被使用，但完整性能没有形成新的 Pareto 优势，因此不保留 U-Net-like 长跳连。

读出头筛选的 A11/A12 都在第 20 轮按验证曲线止损：

| 项目 | A08 原 MLP，跑满 50 轮 | A11 conv head，止于 20 轮 | A12 avg/max MLP，止于 20 轮 |
|---|---:|---:|---:|
| best validation Top-1 | 63.64% | 57.88% | 58.30% |
| test Top-1 | 62.05% | 58.46% | 58.74% |
| optical-off Top-1 | 15.47% | 12.68% | 15.55% |
| random-phase Top-1 | 13.66% | 14.08% | 9.98% |
| shuffled-phase Top-1 | 11.18% | 14.76% | 13.31% |
| electronic-skip-off Top-1 | 38.52% | 41.02% | 42.42% |
| normalized optical dependence | 89.49% | 94.47% | 88.61% |
| head electronic parameters | 104,330 | 88,362 | 203,018 |

A11/A12 的验证曲线都没有显示出追平 A08 的趋势；继续训练会消耗大量共享 GPU 时间，且不能
改变由验证集作出的架构选择，因此保留原 `8x8 pool -> 192x512 MLP` 读出。A13 的最终结果
见下一节。

## A13：预算内低分辨率电子残差

状态：2026-08-19 跑满 50 epochs 并完成统一消融；当前单 seed 性能骨干候选。

| 项目 | A07 高光学参考 | A08 极小 pointwise | A13 低分辨率电子残差 |
|---|---:|---:|---:|
| selected epoch | 48 | 48 | 49 |
| best validation Top-1 | 60.40% | 63.64% | **73.08%** |
| test Top-1 | 60.25% | 62.05% | **72.18%** |
| optical-off Top-1 | 11.15% | 15.47% | 13.39% |
| random-phase Top-1 | 10.95% | 13.66% | 12.48% |
| shuffled-phase Top-1 | 9.73% | 11.18% | 12.76% |
| electronic-skip-off Top-1 | 不适用 | 38.52% | 28.06% |
| normalized optical dependence | 97.71% | 89.49% | **94.55%** |
| mean optical gate | 51.73% | 51.72% | 51.18% |
| residual electronic parameters | 0 | 592 | 312,336 |
| head electronic parameters | 104,330 | 104,330 | 104,330 |
| total electronic parameters | 104,330 | 104,922 | **416,666** |
| estimated electronic MAC/sample | 约 0.10M | 9.54M | **317.82M** |

A13 相对 A07/A08 的 test Top-1 分别提高 11.93/10.13 pp，而 optical-off 只有 13.39%，
random/shuffle phase 也只有约 12%--13%。因此 72.18% 不能由电子路径单独分类解释，学习到的
光学相位仍是必要条件。关闭新增电子变换后下降到 28.06%，说明几十万参数的低分辨率电子
残差对光电协同同样关键；A13 应表述为“强光学因果依赖的混合光电网络”，不能表述为纯光学
网络。

最佳 checkpoint 的八层 optical gate 为
`[0.5827, 0.5039, 0.5023, 0.5016, 0.5005, 0.5007, 0.5010, 0.5020]`，最小值 50.05%，
全部满足 `alpha>=0.5`。residual electronic processing 共 312,336 参数，低于实验室 1--2M
上限且处于“几十万”经验范围；连同读出头总电子参数 416,666。与此同时，估算电子卷积为
317.82M MAC/sample，参数合规不等于时延或能耗已经合规，后续系统实验必须单独报告。

冻结 checkpoint：`runs/a13_lowres_electronic_residual/seed_1234/best.pt`，SHA-256
`69c3a680e5f53f7c49b7d657daafa10174192000611b30a1c570c5604ae97cf6`，大小 12,458,658 byte。
A13 目前只是在 CIFAR-10 上、seed 1234 的架构筛选胜者；下一步应先做独立 seeds 复验，再
冻结它并回到唯一四组 NoFT/BP/FA-pretrained/FA-random，不能把 A13 当作第五种反馈方法。

### A13 多 seed 复验协议

2026-08-19 在读取额外 seed 结果前锁定协议：增加 seed 2026/2027，与已有 seed 1234 组成
三 seed 复验；复用项目正式实验的随机种子体系。除 `training.seeds` 外，复制配置经单元测试
保证与 A13 原配置的 output、optical、data、optimizer 和全部 training 字段相同。两张空闲卡
并行训练，仍按各自 validation-best 选择 checkpoint，并自动执行六项统一消融。启动和汇总
只能使用 `commands/25_train_a13_replica.sh` 与 `commands/26_aggregate_a13_replicas.sh`。

预先判定：若三 seed test Top-1 均不低于 70%，均值不低于 71%，且每个 seed 的 normalized
optical dependence 不低于 90%、所有 stage optical gate 不低于 0.5，则 A13 通过复验并冻结；
否则保留完整离散结果，分析方差或失败原因，不再依据 test 结果修改同一数据集上的架构。

补充确认性运行：用户在 seed 2026 完成后提供另一张低占用卡，因此增加 seed 2028 作为第四个
确认性 seed。它使用完全相同的 replication config，结果无论好坏都报告，但不改变上面的三
seed 通过门槛。唯一启动入口为 `commands/27_train_a13_confirmatory_seed2028.sh`。

复验最终结果：

| seed | selected epoch | validation | test | optical-off | random phase | phase shuffle | electronic-skip-off | optical dependence | min/mean gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 49 | 73.08% | 72.18% | 13.39% | 12.48% | 12.76% | 28.06% | 94.55% | 50.05% / 51.18% |
| 2026 | 48 | 73.10% | 72.52% | 14.74% | 10.17% | 11.47% | 24.76% | 92.42% | 50.05% / 51.09% |
| 2027 | 47 | 73.72% | 72.33% | 13.67% | 11.34% | 13.49% | 25.67% | 94.11% | 50.05% / 51.11% |
| 2028（确认） | 49 | 72.84% | 72.31% | 12.98% | 11.13% | 13.50% | 26.68% | 95.22% | 50.05% / 51.11% |

预注册的前三 seed test 为 **72.34% ± 0.17 pp**；加入确认性 seed 2028 后为
**72.34% ± 0.14 pp**。四 seed optical-off/random/shuffle/electronic-skip-off 均值分别为
13.70%/11.28%/12.81%/26.29%，normalized optical dependence 为 94.07% ± 1.19 pp。
前三 seed 全部满足 test 不低于 70%、均值不低于 71%、光学依赖不低于 90% 和逐层 gate
不低于 0.5 的预注册门槛；确认性第四 seed 同样满足。

冻结决定：A13 通过复验，不再在 CIFAR-10 上修改架构或超参数。原筛选配置 SHA-256 为
`beafc5521e81cf05be84a6c142ad2935d04f0d49541f20c09418dced97dc7239`，三 seed replication
配置 SHA-256 为 `eb4f3d3659b48b5f4920c7710d5b4a45c5ec512e2bb96b5d96bbad6d1db47376`。
下一阶段只运行 NoFT/BP/FA-pretrained/FA-random 四组 P02，不再增加正式方法。

## P05：连续错位疫苗化，而不是继续堆固定点消融

状态：2026-08-20 完成协议和代码，等待服务器 smoke 后启动。

P04-S2 显示原 P02 BP source 在 0.5--2 pixel 固定错位下零样本只有 12.06%--49.68%，但 BP-current 适配后可以恢复到 66.04%--73.28%。因此本轮不再改变 A13 结构、电子预算或增加反馈组，而是建立“部署前预防 + 部署后校准”的方法链。

具体动作：新增连续 batch-wise global/layerwise 位移采样；最大位移在 8 epoch 内从 0.25 增到 2 pixel；使用理想 CE、错位 CE 和 teacher-consistency KL 联合训练；用训练未见的 seed 9201 上七环境平均 validation accuracy 选择 checkpoint；保留 epoch 0 source 作为失败兜底。疫苗化结束后，再用 seed 9301 的 1/2 pixel 固定偏移运行唯一四组。光学 gate 下限和 A13 的 416,666 个电子参数完全不变。

所有配置、指标、门槛和命令见 `P05_MISALIGNMENT_VACCINATION_PLAN.md`。这一轮的首要结果不是某个网络变体胜出，而是零样本错位工作区能否扩大、以及预训练反馈算子能否在残余校准中保持接近 BP 且优于随机反馈。

服务器动作记录：提交 `43fa9b61` 同步后 22 个测试通过，真实 checkpoint 一 batch smoke 通过；物理 GPU 4 已运行 command 45，CPU watcher 将在结束后自动使用 GPU 3/4 运行唯一四组 held-out 校准。epoch 3 的七环境平均由 source 33.06% 提高到 45.64%，最差环境由 13.60% 提高到 21.34%，理想 validation 从 73.40% 降到 70.70%；这是运行中检查点，完整逐条件值和限制见 P05 计划第 8 节。

P05 最终 validation-best 为 epoch 18：七环境平均 60.94%，相对 source 提高 27.89 pp；最差环境 50.36%，提高 36.76 pp；理想 validation 72.84%，只下降 0.56 pp。held-out seed 9301 的 1/2 pixel global/layerwise 四组适配也全部完成：FA-pretrained 距离 BP 为 0.30--1.72 pp，并在四条件全部优于 FA-random。完整表见 P05 计划第 9 节。

## P06/P07：ImageNet 通用 backbone 与跨任务迁移

2026-08-21 完成服务器资产和已有路线审计。ImageNet-1K/CLIP 4-view cache 完整；Caltech-101、KADID-10k、ISIC2016 均可直接使用。已有 792×792 OpticalMixerMoE9 在 epoch 8 的 ImageNet validation Top-1 为 6.59%，单 epoch 约 3--5.4 小时，因此不继续把重型 MoE 当主线。

新主线固定从 P05 epoch-18 checkpoint 初始化紧凑 128×128 八层骨干，增加无参数 stage 2/4/6/8 特征导出，用 ImageNet CE + CLIP cosine/KL 训练通用表示，再在分类/检索、IQA 回归和医学分割三种输出形态上运行唯一四组。新增电子 residual 仍为 0；预训练 projector/classifier 约 0.71M 且下游时丢弃。完整阶段、门槛、数据防泄漏规则和计算预算见 `P06_GENERAL_OPTICAL_BACKBONE_PLAN.md`。

### P06-E0/S 实现与启动记录（2026-08-21）

本轮没有改动已冻结的 A13/P05 光学—电子主体，而是实现了通用预训练入口
`general_backbone_pretraining.py`。具体动作如下：

- 严格校验 P05 epoch-18 checkpoint SHA-256
  `5bd7889700f16eae112776f622b10cd52e59c064384558d85ae7be999b1aa8aa`，加载全部八层相位、
  逐层门控和低分辨率电子残差；原 CIFAR-10 head 随后丢弃。
- CLIP cache 对应的输入先用 CLIP mean/std 反标准化并截断到 `[0,1]` RGB intensity，之后才由
  光学模型缩放到 128×128 并开平方编码为 amplitude。训练首 batch 会把标准化/反标准化数值范围
  写入 `metrics/latest.json`，防止静默重复标准化。
- 从物理 stage 2/4/6/8 分别做 4×4 average/max pooling，得到 384 维描述，再通过
  `LayerNorm(384) -> Linear(384,512)` 和 ImageNet 线性分类器。预训练头在下游丢弃；原有
  residual electronic processing 参数量不增加。
- 损失固定为 `0.5 CE + 1.0 cosine + 0.5 CLIP-text KL`。第一轮只训练读出头，后五轮使用
  当前精确 BP 联合更新八层相位、电子残差和读出头；只有联合 BP epoch 可以成为 best checkpoint。
- 100k screen 固定每类 100 张 train、每类 10 张 validation，并保留原始 128 万 cache index；
  DDP sampler 每个原图每 epoch 只取四个缓存 view 中一个，避免复制 ImageNet 或错配 teacher。
- 每个联合 BP 运行记录八层 raw-phase gradient norm/finite/nonzero、逐层 optical gate、Top-1/Top-5、
  CLIP cosine/zero-shot、两项光学破坏消融和完整配置摘要。`last.pt` 可恢复，`best.pt` 仅按联合 BP
  validation Top-1 选择。

对应配置为 `p06_imagenet_smoke.yaml` 和 `p06_imagenet_100k_screen.yaml`；command 51 负责真实
单卡 smoke，command 52/53 负责 GPU 3、5 双卡 DDP screen 与后台启动。服务器实测结果和首轮
速度将在运行后追加到本节，不能用本地 Windows 的损坏 PyTorch DLL 状态代替服务器验证。

首次真实 smoke 暴露并修复了两项只会在服务器数据链上出现的问题。第一，服务器已缓存全部
ImageNet Arrow 文件但 Hugging Face token 已过期；共享 ImageNet loader 现在会在无 token 时
强制 `local_files_only` 复用已审计缓存，缓存缺失仍明确报错。第二，AMP 初始 loss scaling 可能
跳过一次非有限更新；batch scheduler 现在只在 optimizer 确实更新后前进，避免悄然缩短 warm-up。

修复前的真实数据 smoke 已完成两次 head-only 和两次 joint-BP batch：八层 phase gradient norm 为
`[0.1895, 0.0437, 0.0486, 0.1492, 0.1073, 0.1863, 0.0924, 0.0861]`，全部 finite/nonzero；
最小 optical gate 为 `0.50021`。审计到 phase/residual/pretraining-head 参数分别为
393,216 / 312,336 / 710,888，384 维 stage descriptor 符合设计。该 smoke 只评估工程正确性；
validation 仅运行 8 张图，0% Top-1 不具有统计意义。修正 AMP scheduler 后将重新执行 smoke，
再启动 100k screen。

AMP 修正后的 smoke 再次通过：检测到的两次 overflow 分别使 scale 从 65536 降到 32768、
再降到 16384，scheduler 均未前进；之后 optimizer 正常更新且不再出现 PyTorch 调度顺序警告。
输入诊断为 CLIP-normalized `[-1.7923, 2.1459]`、反标准化 intensity `[0,1]`；八层 phase
gradient 再次全部 finite/nonzero，norm 为
`[0.1942, 0.0456, 0.0489, 0.1563, 0.1085, 0.1837, 0.0950, 0.0872]`。

2026-08-21 11:25（Asia/Shanghai）使用 command 53 在物理 GPU 3（RTX 4090）和 GPU 5
（RTX 3090）启动 P06-S 双卡 DDP；launcher PID 792349、torchrun PID 792561，完整日志为
`runs/p06_imagenet_100k_screen/train.log`。manifest 已确认 world size 2、train/validation
分别 100,000/10,000、train cache 每图 4 views、P05 source selected epoch 18。每卡 batch 32，
有效 batch 64，共 1,563 optimizer batches/epoch。head warm-up batch 50 时 running Top-1 0.06%、
CLIP cosine -0.0003、loss 9.2143，耗时 50.4 秒；这些是随机新 head 的起始在线指标，不能当作
epoch 结果。按首 50 batch 粗估 warm-up epoch 约 26 分钟，联合 BP 速度需以第二 epoch 实测修正。

### P06-S 完成审计与防塌缩精炼（2026-08-21）

P06-S 并非进程丢失，而是在 11:57 正常完成全部 6 epochs 后释放 GPU。head warm-up 首次读取
Arrow 数据耗时 1109.7 秒；数据进入系统 cache 后，五个 joint-BP epoch 各约 127--133 秒。
validation Top-1 从 joint epoch 2 的 1.70% 单调上升到 epoch 6 的 3.84%，Top-5 11.78%，
CLIP cosine 0.7014；八层 phase gradient 全部 finite/nonzero，最小 optical gate 0.50022。
optical-off / phase-random Top-1 分别只有 0.22% / 0.14%，相对破坏下降 94.27%，所以模型确实
更新并依赖光学路径，但未通过预注册的 10% Top-1 门槛，不能进入 full ImageNet。

进一步直接读取相同 10k validation 的 teacher cache：CLIP teacher zero-shot Top-1/Top-5 为
64.61%/87.38%，但 teacher embedding 与其全局平均方向的 cosine 已高达 0.7210±0.0648。
P06-S student cosine 0.7014 而 student zero-shot 仅 1.27%，说明单样本 cosine 很大程度可由靠近
公共均值方向获得，类别/实例结构没有同步学到。因而不原样延长 S1，也不把 cosine 过线误写成
通用 backbone 成功。

P06-S2 从 S1 epoch-6 best（SHA-256
`8212bb07c453b9d8723ce1e52e0c75c57240095651110b3a6930d8704d3730b5`）严格继续，保留同一
100k/10k split 和 0.5 optical gate floor。损失改为
`1.0 CE + 0.25 cosine + 0.5 text-KL + 1.0 paired contrastive`；batch 内双向
student↔teacher InfoNCE 以 0.07 temperature 强制每个 student 对应自己的 teacher，而不是所有
样本靠向相同均值。训练器同时新增 teacher zero-shot 指标，使 teacher 上限、student zero-shot 和
监督 classifier Top-1 可逐 epoch 对照。精炼运行 24 个 joint-BP epochs，重新 warm up 500 steps
并使用新的 cosine 周期；command 54/55 默认在物理 GPU 2、4 双卡运行。

S2 在 GPU 2、4 上实跑两轮后按在线门槛止损。epoch 1/2 的 validation classifier Top-1 为
3.41%/3.10%，均低于 S1 的 3.84%；student zero-shot 由 1.27% 小幅升至 1.55%/1.60%，但
cosine 同时降到 0.620/0.550。对比损失从 3.535 降到 3.398，证明一一对应目标被优化，然而
`contrastive_weight=1.0` 在当前低维 backbone 上过强，破坏原表征快于恢复类别结构。torchrun
PID 1226147 经精确命令行核验后收到 TERM，两个 rank 正常退出；两轮日志/checkpoint 保留。

由此新增 S3 保守精炼：始终从未被 S2 覆盖的 S1 best 开始，将 contrastive weight 降到 0.1，
保留 cosine weight 1.0，CE 提高到 1.0；phase/residual/head LR 降为
`3e-5/3e-5/1.5e-4`，运行 24 joint epochs。训练器现在会先在同一 10k validation 上评估初始
checkpoint，并将它保存为 epoch-0 best；只有后续 Top-1 严格超过 3.84% 才能替换，因而精炼
失败时最终模型也不会性能倒退。无 head warm-up 的 DDP 同时关闭 unused-parameter 图遍历。

S3 经服务器 26 项测试通过后，在物理 GPU 2、4 启动；launcher/torchrun PID 为
1289248/1289267，两个 guest3 rank 为 1289577/1289578。epoch-0 完整 10k validation 复测
Top-1/Top-5 为 3.84%/11.80%，student/teacher zero-shot 为 1.27%/64.52%，并成功写为保护性
best。epoch 1/2 Top-1 为 3.74%/3.64%，未替换 best；epoch 3 上升到 **4.03%**，Top-5
11.67%、student zero-shot 1.40%、cosine 0.7035，首次严格超过 S1 并替换 best。epoch 4 正在
继续运行。该在线提升只说明 S3 方向优于 S2，仍未通过 10% Top-1 的 full-ImageNet 准入门槛。

S3 最终完成 24 epochs，validation-best 为 epoch 19：Top-1 **5.01%**、Top-5 14.62%、
student zero-shot 1.70%、cosine 0.7075；best SHA-256 为
`590eb01c55498596f4784750084cef7fd37762d67d14cd6581ffb2e65e94f8b1`。optical-off 和
phase-random Top-1 仅 0.12%/0.11%，相对光学破坏下降 97.60%；八层 phase gradient 全部
finite/nonzero，最小 gate 0.50023。Top-1 从 3.84% 提升 1.17 pp，说明保守精炼有效，但
epoch 19--24 已在 4.86%--5.01% 平台且仍低于 10% 门槛，不能继续重复相同 100k subset。

P06-F1 因此不是放宽门槛，而是扩大数据支持集：严格从 S3 epoch-19 best 接续，训练使用全部
1,281,167 张 ImageNet train 图，每图每 epoch 轮换四个 cache view 中的一个；验证改为完整
50,000 张。保留 S3 损失与较低 LR，重启 1,000-step warm-up 后运行 10 joint-BP epochs；同样先
建立 full-validation epoch-0 best，只有更高 Top-1 才替换。command 58/59 默认使用物理 GPU
2/5 双卡、有效 batch 64。该阶段的停止标准仍为 Top-1 10%、cosine 0.65、gate 0.5 和光学
破坏下降 30%，没有因 100k 失败而降低标准。

P06-F1 首次于 2026-08-21 13:50 在 GPU 2/4/5 启动。完整 50k validation 的保护性起点为
Top-1 5.05%、Top-5 14.18%、student/teacher zero-shot 1.78%/64.62%、cosine 0.7068；但三卡
在首个反向批次之后不再前进：GPU 2/4 持续执行而 rank 2 在 GPU 5 等待，且 GPU 4 同时存在
其他用户训练和本项目硬件导出进程。该次没有产生 optimizer checkpoint，torchrun PID 1766442
经命令行核验后于 14:00 收到 TERM，全部子进程退出，未影响 S3 immutable best。

随后在真正空闲且同为 RTX 3090 的 GPU 2/5 上，以 `NCCL_P2P_DISABLE=1`、
`NCCL_IB_DISABLE=1` 完成双 rank 通信 smoke：连续 20 次 64 MiB all-reduce 数值一致，各 rank
耗时 0.67--0.68 秒。14:00 使用 command 59 重新启动，launcher/torchrun PID 为
1833065/1833108，rank PID 为 1833388/1833389。双卡 full-validation 起点复现为 Top-1 5.05%、
Top-5 14.18%；batch 500/20019 在 44.9 秒到达，running loss 9.6729、Top-1 3.02%、cosine
0.7562、phase LR 1.5e-5。仅 batch 1 的 AMP scale 从 65536 自动降至 32768；但 500 batch
之后不能按 44.9 秒线性外推：全局逐样本随机顺序令 Hugging Face Arrow 压缩图像发生大范围
随机读，DataLoader worker 明确进入 `wait_on_page_bit_common`，两个 DDP rank 轮流等待且超过
5 分钟仍未到 batch 1000。该进程没有 epoch checkpoint，遂停止以免继续浪费 GPU 时。

数据集标签顺序审计显示原始 ImageNet train 本身已经充分打散：首个 64/256/1024/4096 连续
窗口平均分别覆盖 62.2/226.1/643.0/983.0 个类别，连续同标签 run 平均 1.001、最大 3。因此新增
可配置 locality-aware sampler：每 epoch 随机排列 4096-image blocks，block 内保持正向连续，
同时继续保证全数据覆盖、DDP 等长 padding 和每图四 view 逐 epoch 轮换。这将每图一次随机 seek
改为每 4096 图一次，且不会形成类别排序批次。对应覆盖性、确定性、局部连续性和 view-cycle
单元测试已加入；command 58/59 的默认设备和 NCCL 环境继续锁定为验证过的 GPU 2/5 路径。

locality sampler 提交 `dfdd8752` 在服务器通过 28 项测试后，于 14:13 再次由 command 59
正式启动；launcher/torchrun PID 为 1912262/1912343，rank PID 为 1912617/1912618。manifest
确认 world size 2、train/validation 1,281,167/50,000、`train_shuffle_block_size=4096`，配置
digest 为 `be3bc639c22fbc2d5afc8641cc24e001cb2dacba45eaeb216d853588bbec2ce4`。完整验证起点第三次
复现 Top-1/Top-5 5.05%/14.18%，说明采样器改动没有改变初始化权重或验证语义。

连续吞吐验证已越过旧运行的失速点：batch 500/1000/1500 累计耗时分别为
89.4/173.2/260.5 秒，两个后续 500-batch 区间为 83.8/87.3 秒，未再出现分钟级随机 I/O
停顿。batch 1500 的即时 loss 为 9.3515，累计训练 Top-1 3.12%、cosine 0.7553；warm-up 在
batch 1000 完成，phase LR 达到 3e-5。AMP 在 batch 1/220 将 scale 从 65536 依次降至
32768/16384，此后至 batch 1500 没有再次跳过更新。按稳定实测速度，单个 20,019-batch epoch
约 58 分钟，10 epochs 约 9.7 小时再加逐轮验证和最终消融；最终耗时和性能只按落盘 epoch
metrics 报告，不再用首 500 batch 估计。

为避免约十小时的无人值守运行静默退出，新增 command 60/61 监督层：每 5 分钟记录最新训练
行和日志年龄，进程丢失时通过 duplicate-safe command 59 从 `last.pt` 恢复；训练日志连续
20 分钟无更新才判为失速并精确终止 P06-F1 进程，最多允许 3 次重启；`result.json` 生成后
自动退出。20 分钟阈值显著长于当前 500-batch 的约 1.5 分钟和完整 50k validation 的实测时间，
不会把正常验证误判成训练挂起。

command 61 已在服务器实际启动 watcher PID 1971968；首条监督记录正确读到 batch 2500、
日志年龄 68 秒，没有误启动重复训练。训练至 batch 2500 的累计耗时为 458.8 秒，GPU 2/5
进程与显存均正常。数据相关的非有限梯度又令 AMP 在 batch 2130/2245 将 scale 从 16384 降至
8192/4096；这两批同样只跳过 optimizer/scheduler update，累计 4/2500 次（0.16%），当前继续
训练并观察 scale 4096 是否稳定，不为消除少量自动保护事件而重启已正常运行的长任务。

### P06-F1 最终结果与 F2 容量扩展决定（2026-08-22）

P06-F1 于 2026-08-21 21:53 正常完成，watcher 未发生重启并在 `result.json` 出现后退出。完整
ImageNet validation 从 epoch-0 Top-1/Top-5 5.05%/14.184% 提高到 epoch-10 的
**8.468%/21.16%**；student zero-shot 从 1.784% 提高到 3.394%，CLIP cosine 为 0.7133。
optical-off/phase-random Top-1 仅 0.142%/0.108%，相对光学破坏下降 98.32%；八层 phase
gradient 全部 finite/nonzero，最小 optical gate 0.500248。best SHA-256 为
`08366dd0010fc74168e870f8750cffcb9e8ee037174a026e8bddb11c8f6dea5d`。总墙钟时间约 7 小时
40 分钟，其中前两轮 Arrow/OS cache 冷读分别约 2 小时 15 分和 1 小时 57 分，后八轮每轮约
22--27 分钟。epoch 9/10 Top-1 为 8.450%/8.468%，原配方已经平台且未过 10% 门槛。

原 F1 光学参数仅为 `8*3*128*128=393,216`，同时把 224 输入降到 128；这对于 ImageNet
通用 backbone 偏小。因此 F2 不只调整电子头：新增 12-stage、192x192 RGB 光学主干，光学
相位参数为 `12*3*192*192=1,327,104`。已有 8x128 raw phase 先做 bicubic 空间插值，再按
相对深度线性映射到 12 层；旧 stage 2/4/6/8 对应新 stage 3/6/9/12，读出语义得以保留。
每层电子残差同样按深度插值，但 downsample factor 从 4 改为 8，使 192 平面的电子支路只在
24x24 上处理，避免电子 MAC 随分辨率暴涨。预计残差电子参数 468,504，连同 710,888 预训练
读出头共约 1.18M，仍在约束内；所有 12 个 optical gate 继续硬约束 `>=0.5`。

F2A/F2B 是瓶颈诊断而不是正式第五/第六反馈组：F2A 保持 F1 结构，以更高 CE 权重和重启的
低 LR 检查训练配方；F2B 把监督分类从 CLIP projector 解耦为 `384->512->1000` MLP，检查
线性读出瓶颈。F2C 才是面向 backbone 容量的主试验：从 F1 best 扩展初始化 12x192，先一轮
冻结 trunk 校准读出，再四轮 exact joint-BP。所有配置、smoke 和长跑入口固定在 command
62--70；结果仍以完整 50k validation、光学破坏消融和 10% Top-1 门槛判断。

### P06-F2C 生产批量与设备复核（2026-08-22）

小批量真实数据 smoke 在 `amp_initial_scale=256`、禁止自动增长的设置下完成两次有效更新，
12 层 raw-phase 梯度全部 finite/nonzero；因此没有缩回 10x192。为避免只验证数值而遗漏显存
问题，command 66 的 smoke 改为与正式训练一致的每卡 batch 32，并各执行一个 train/validation
batch。长跑默认设备从当时有低占用进程的 GPU 2/5 改为真正空闲的 GPU 4/5；command 67--70
及命令文档同步更新。只有生产批量 smoke 和双卡通信都通过，才允许启动 F2C 长跑。

生产批量 smoke 随后在物理 GPU 5（RTX 3090）通过：一批 32 图的 joint-BP 约 1.1 秒，未
OOM、未发生 AMP skipped update；12 层梯度范数为
`[4.5350, 2.5005, 1.8999, 1.3810, 0.9024, 0.8300, 0.5389, 0.4688, 0.4924, 0.2612, 0.2492, 0.2985]`。
F2C 于本地/GitHub/服务器提交 `2388c84d` 后在物理 GPU 4/5 启动，launcher/torchrun PID 为
674154/674173。双卡完整 50k baseline 已正常完成并进入训练；扩展后未经校准的 Top-1 为
0.14%，说明改变传播分辨率和插入四个传播层并非函数保持变换，因此预先设计的一轮 head-only
校准是必要步骤，而不能把插值权重直接当成已有 8.468% 模型。batch 250/500 分别在 19.0/36.2
秒到达，loss 从 37.2854 降到 26.0750，证明双卡通信、数据和优化器均实际前进。新增 command
71/72 每五分钟监督长跑，30 分钟无日志才判定失速，并可从 `last.pt` 安全恢复。

### P06-F2C 结果审计与 P06-F3 八层 224 扩容（2026-08-22）

F2C 于 03:30 正常结束，best 为 epoch 5：Top-1/Top-5 5.212%/14.584%，CLIP cosine
0.6939；optical-off/phase-random 为 0.746%/0.186%，12 层梯度全部 finite/nonzero，门控下限
0.500225。它证明百万相位模型可以稳定 exact-BP 且依赖光学，但性能低于 8x128 F1 的
8.468%。主要诊断是扩容初始 Top-1 从 8.468% 破坏到 0.14%，且首/末层梯度范数比从 F1
的约 1.4 增大到 22.2；直接同时增加分辨率与传播深度不是有效的函数保持扩容。

按用户决定，F3 不改变 16 um 像素尺寸，保持八个传播 stage，只把 RGB phase plane 扩为
224x224，总光学相位 `8*3*224*224=1,204,224`。电子残差下采样从 4 改为 7，使其内部平面
继续为 32x32，电子计算和参数不随光学画布膨胀。旧 stage 2/4/6/8 仍一一对应新 stage
2/4/6/8，相位只做空间 bicubic 初始化，不再做深度插值。所有 F3 训练均设置
`head_warmup_epochs=0`，即从第一个 batch 开始以最新前向关系对相位、残差和读出头执行正常
BP。三个 100k 配方 screen 分别检查原 projected-linear 读出、解耦的 descriptor MLP 读出、
以及更高监督 CE 权重；它们只用于选择一个 full-ImageNet 配方，不增加正式反馈方法组数。

服务器10项配置/加载测试和生产 batch 32 smoke 均通过。实测相位/残差/临时 projected head
参数为 1,204,224/312,336/710,888；一批 joint-BP 约0.7秒，无OOM或AMP跳步，八层梯度
`[3.4554, 2.1489, 1.3791, 1.3361, 0.9095, 1.0041, 0.5217, 0.5757]` 全部有限非零，
首末比约6.0，明显小于12层F2C的22.2。启动前GPU 2已被其他用户新任务占用，因此并行命令
增加可审计的设备覆盖参数；本轮实际使用低占用GPU 1和空闲GPU 4/5，不抢占GPU 2。

另锁定第四个、也是最后一个配方 screen：保持相同8x224光学主干和监督目标，只把 stage
2/4/6/8 的 parameter-free average/max pooling 从4x4提高到8x8，descriptor 从384增至1536，
检查原读出是否过早丢弃224平面的空间信息。该 projected readout 连同残差电子仍低于2M预算；
它在前三条中的一张卡释放后运行。至此内部配方筛选封顶为四条，不再继续枚举读出变体。

四条 screen 的 epoch-3 统一10k validation Top-1/Top-5 为：projected-4x4
2.41%/7.82%，MLP-4x4 1.56%/6.15%，supervised-MLP-4x4 1.49%/5.88%，以及
projected-8x8 **3.66%/11.18%**。MLP 虽有更高 cosine，但没有转化为分类性能；提高 CE 权重
也没有弥补。因此锁定 projected-8x8，不再保留MLP。胜出模型八层梯度全部 finite/nonzero，
best SHA-256 为 `35e1503cf41ad442cd161d90ba4a26871846a3784a71b1e59ea1b6e1378b050e`；相位、
残差电子、临时读出分别为1,204,224/312,336/1,303,016，后两者合计1,615,352，低于2M。

全量阶段从该100k best 严格加载，不重新扩容；取消所有 head-only epoch，从首 batch 对8层
224相位、残差和读出执行 exact BP。使用完整1,281,167 train/50,000 validation，12个joint
epochs，有效batch 64，phase/residual/head LR 为 `2e-5/1e-5/1.5e-4`，重启1000-step warm-up。
command 77--80 固定双卡启动、恢复和监督流程；正式判定仍要求Top-1至少10%、cosine至少0.65、
逐层gate至少0.5及光学破坏相对下降至少30%。

启动时GPU 4被其他用户在数分钟内新增两个任务，占用升至6.8GB并高负载；首次4/5 DDP尚未
产生baseline或训练checkpoint即被精确终止。随后迁移到低占用GPU 1和空闲GPU 5，唯一DDP
实例 launcher/torchrun PID 为1323910/1324007，watcher PID 1324898并继承1/5设备映射。
完整50k起始验证为Top-1/Top-5 **3.90%/11.51%**，高于10k screen的3.66%；首个250-batch
检查点在52.5秒到达，running Top-1 2.89%、cosine 0.7271，证明双rank、数据和正常联合BP
均已实际前进。command 77--79默认设备同步改为当前验证过的1/5。

### P06-F3 并行错位压力测试（2026-08-22）

全量8x224训练至epoch 4时，完整ImageNet validation Top-1/Top-5 已达7.734%/19.714%，明显
优于12x192在四轮joint-BP后的5.212%，且正在继续第5/12轮。为利用空闲GPU 4而不重复训练
配方，新增通用backbone部署评估入口：Compact student前向现在可接收冻结的deployment state，
在不改变训练时BP语义的前提下评估相位横向偏移。command 81/82固定使用epoch-4 checkpoint
及其SHA-256，在完整50k validation上运行ideal和global/layerwise 0.5/1/2 pixel七种条件。
该轮只建立百万相位backbone的错位基线；最终best产生后复用同一入口复测，不增加反馈方法组。

用户随即明确当前阶段只优化预训练性能，不提前进入错位实验。刚启动的部署评估进程已精确
终止，未生成result，GPU 4恢复空闲；8x224主训练进程不受影响。空闲卡改跑唯一一个全量性能
分支F3B：从epoch-4不可变checkpoint兼容加载完全相同的1.204M相位主干和1536维空间描述符，
把监督分类与CLIP projector解耦为独立的`1536->256->1000` GELU MLP。该结构的残差电子、
CLIP projector和分类读出合计约1.75M，仍低于2M；先一轮冻结主干校准新分类器，再四轮正常
联合BP。它检验的是多目标共享线性瓶颈，而不是增加新的光学架构或反馈方法组。command 83--86
固定GPU 4的启动、恢复和监督入口。
