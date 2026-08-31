# P12 多随机种子、无预训练对照与深度扩展阶段报告

更新日期：2026-09-01

## 1. 本轮要回答的问题

本轮不再增加主表算法数量，仍保留四组：

1. `Frozen-backbone / head-only`（代码键 `noft`）；
2. `Exact BP`；
3. `FA-source`（代码键 `fa_pretrained`）；
4. `FA-random`。

新增工作只回答四个问题：结果是否跨随机种子稳定、ImageNet 预训练是否真的形成了可迁移 backbone、电子残差是否掩盖了光学反馈差异、现有 8 层模型能否渐进扩展到百层和千万级相位参数。

## 2. 正式 3-seed 下游结果

正式目录：

`experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/runs/p12_downstream_fa_50e`

训练协议为每个任务、每个方法、每个种子统一 50 epoch；种子为 2026、2027、2028。36/36 个任务完整结束，失败数为 0。下表为测试集主指标的均值 ± 样本标准差。

| 任务 | Frozen/head-only | Exact BP | FA-source | FA-random |
|---|---:|---:|---:|---:|
| Caltech-101 Top-1 | 76.96 ± 1.93% | 79.13 ± 0.85% | **79.46 ± 1.50%** | 78.83 ± 0.68% |
| ISIC2016 mIoU | 81.35 ± 0.38% | 84.02 ± 0.15% | 83.99 ± 0.15% | **84.28 ± 0.12%** |
| LSP PCK@0.2 | 50.33 ± 0.41% | 71.17 ± 0.34% | **71.20 ± 0.08%** | 70.75 ± 0.21% |

配对结果：

- Caltech：FA-source 相对 BP 为 +0.34 个百分点；FA-random 相对 BP 为 -0.30 个百分点。FA-source 与 BP 的三种子配对 bootstrap 区间跨 0，不能宣称稳定优于 BP。
- ISIC：FA-source 相对 BP 为 -0.03 个百分点；FA-random 相对 BP 为 +0.26 个百分点。FA-random 三个种子均略高于 BP，但尚不能据此归因于随机反馈本身。
- LSP：FA-source 相对 BP 为 +0.03 个百分点；FA-random 相对 BP 为 -0.42 个百分点。FA-source 与 BP 基本重合，FA-random 三个种子均较低。

当前最稳妥的主结论是：**在三个不同下游任务上，固定的源算子反馈可以恢复到 exact BP 的性能水平；FA-random 偶尔胜出，但并不是跨任务、跨种子的稳定规律。**

## 3. 梯度与分支依赖

在选中 checkpoint 上，第 1--7 层相位的 FA 梯度相对当前 exact-BP 梯度平均余弦为：

| 任务 | FA-source | FA-random |
|---|---:|---:|
| Caltech-101 | 0.9985 | 0.6198 |
| ISIC2016 | 0.9999 | 0.7527 |
| LSP | 0.9913 | 0.6241 |

这说明 FA-source 的高性能与高度对齐的光学更新方向一致；FA-random 虽能产生可用方向，但与真实梯度的偏差明显更大。

现有 joint-adaptation 中，电子 mixer、融合门、归一化和任务头仍使用 exact BP。这是合理的硬件问题定义：难点是跨光学传播阶段的伴随关系，而普通数字电子模块可以直接反传。因此论文口径应为：

> 固定层间光学反馈，配合精确电子优化框架。

不能写成“全网络不需要 BP”。现有关闭分支测试也说明两条路径均重要，但 ISIC 的电子依赖更强：ISIC 上关闭光学分支平均下降约 7.5 个 mIoU 点，而关闭电子 skip 平均下降约 42.6 个点。因此 ISIC 中 FA-random 略高于 BP 可能包含电子补偿或正则化效应，不能单凭最终分数解释为随机光学反馈更优。

## 4. No-ImageNet body initialization 对照

该对照保留同一个冻结 Qwen Patch/Position stem，但把 8 层相位、adapter、mixer 和门控随机初始化。准确名称是 `No-ImageNet body initialization`，不是“全模型从零”，因为 Qwen stem 仍然来自预训练。

单源初始化、seed 2026 的 12/12 个 50-epoch 任务已经全部结束，失败数为 0：

| 任务 | Frozen/head-only | Exact BP | FA-source-init | FA-random |
|---|---:|---:|---:|---:|
| Caltech-101 Top-1 | 22.38% | **34.78%** | 33.58% | 26.46% |
| ISIC2016 mIoU | 68.84% | 81.69% | **81.83%** | 78.59% |
| LSP PCK@0.2 | 31.36% | 47.70% | **48.06%** | 40.33% |

与 ImageNet 预训练 body 的 78.64% head-only 相比，随机 body 的 22.38% 明显更低，证明 P11 ImageNet 训练确实形成了可迁移视觉表征。与此同时，随机 body 上 FA-source-init 比 FA-random 高 7.12 个百分点并接近 BP，说明“使用与起始前向算子匹配的固定反馈”在缺少成熟表征时更重要。

这一排序在三个任务上完全一致：FA-source-init 相对 FA-random 分别高 7.12 Top-1、3.24 mIoU 和 7.74 PCK 个百分点，并且相对 BP 仅为 -1.20、+0.14、+0.36 个百分点。与正式 ImageNet body 相比，随机 body 在 Caltech、ISIC、LSP 的 BP 下分别低约 44.35、2.33 和 23.47 个百分点。由此可以把两个结论分开：ImageNet 预训练负责形成可迁移语义；source-matched fixed feedback 负责在适配时逼近 BP。该面板目前只有一个随机初始化，后续 run 用于估计初始化方差，不改变四组方法定义。

## 5. 电子掩盖假设的最小验证

不把电子模块也改成 FA。那会回答“整个混合网络是否都能不用 BP”这一不同问题，并削弱当前针对光学伴随传播痛点的聚焦。

采用同样四组、增加一个训练范围面板：

- `joint adaptation`：现有最高性能设置，训练相位、轻量电子 backbone 与任务头；
- `phase-only adaptation`：冻结 adapter、mixer、norm、fusion gate 和全部其他电子 backbone，仅训练 8 层 phase 与任务头；任务头仍为 exact BP。

phase-only 实现已经完成，并在锁定的 e305 P12 基底通过 69 项测试。真实模型参数审计为 1,204,224 个可训练 phase 参数、965,120 个冻结电子 backbone 参数；适配组 optimizer 中只有 `phase` 和 `head`。基础实现 SHA 仍为 `c61ee3bbbabe6937f574987bb48452c0bb7d74502ef839628e55c698910d6fbd`。

如果冻结电子后 FA-random 与 BP 的差距显著扩大，则支持“电子补偿”解释；如果 FA-source 仍接近 BP 且明显优于 FA-random，则更直接支持源算子反馈的中心结论。

seed 2026 的 Caltech 四组已经完成测试：head-only 78.64%、BP 78.59%、FA-source 78.52%、FA-random 78.31%。FA-source 仍与 BP 基本重合，但该 seed 的 phase 更新没有超过 head-only，因而不能用这一个任务夸大方法差异。ISIC/LSP 正在继续；更强的电子补偿证据来自下一节已经完成的 P/E/H 析因审计。

## 6. P/E/H 机制审计

对现有 checkpoint 做零训练反事实审计，每个任务/种子/endpoint 共 40 个唯一状态：

- Phase / Electronics / Head 的完整 `2^3` 析因与 Shapley；
- 6 个定向 phase 交换；
- 6 个定向 electronics 交换；
- 前 7 层与第 8 层 phase reset。

Caltech 与 ISIC、seed 2026 的一批 smoke 已完整通过，每个任务均得到 40 个状态、6 行 Shapley、48 行定向交换和 6 行深度重置结果。完整测试集 pilot 随后也已结束，得到以下关键结果：

- Caltech 的 FA-random 总增益为约 +0.94 个百分点，其中 phase Shapley 约 +0.26，electronics 约 +0.27，head 约 +0.41；该种子的异常高分并非主要来自随机反馈训练出的 phase。
- ISIC 的 BP / FA-source / FA-random phase Shapley 分别只有约 +0.19 / +0.19 / +0.12 个 mIoU 点，而 electronics Shapley 约为 +3.33 / +3.30 / +3.47 个点，确认该任务当前主要由电子适配驱动。
- 在固定 BP electronics/head 的情况下，移植 FA-source phase 可恢复 BP 自身 phase 收益的 95.0%（Caltech）和 98.5%（ISIC）；FA-random phase 只恢复 45.0% 和 29.6%。这是比最终总分更直接的 source-feedback 有效证据。
- phase-depth reset 表明前 7 层通常恢复接近全部 phase 收益，而第 8 层只占很小部分。因此结果不是只依赖“最后一层相位天然仍使用 exact local gradient”。

三任务、三种子、best endpoint 的完整 360-state 审计现已全部完成：9/9 个 run、每个 40 个唯一状态，失败数为 0。聚合后结论更清楚：

- Caltech 中 BP / FA-source / FA-random 的 phase Shapley 分别为 +0.39 ± 0.08、+0.33 ± 0.14、+0.20 ± 0.09 个 Top-1 点，而 electronics 分别为 +1.51 ± 1.37、+1.75 ± 1.89、+1.47 ± 1.05 个点；
- ISIC 中三者的 phase Shapley 仅为 +0.15 ± 0.03、+0.12 ± 0.01、+0.08 ± 0.04 个 mIoU 点，而 electronics 均约为 +3.31--3.39 个点；
- LSP 中 BP 与 FA-source 的 phase Shapley 都为约 +1.96 个 PCK 点；FA-random 的 phase Shapley 为 **-5.12 ± 6.23** 个点，但其 electronics 与 head 分别贡献 +19.45 ± 2.84 和 +6.10 ± 3.71 个点。这直接解释了为什么 FA-random 最终分数看起来不差：随机反馈训练出的 phase 可以很差，而电子路径和头把总分补回来；
- 把 donor phase 放进同一个 BP electronics/head scaffold 后，FA-source 在 Caltech 三个 seed 平均恢复 BP phase 收益的 97.7%，在 LSP 为 100.0%；ISIC 因 BP phase 分母很小而比例不稳定，但三个 seed 的 source phase 收益均为正。FA-random 在 ISIC 平均只恢复 21.0%，在 LSP 三个 seed 均产生大幅负收益；
- 对 BP 与 FA-source，保留第 1--7 层、reset 第 8 层通常仍恢复约 99%--112% 的 phase 收益。结论不是由最后一层的精确局部梯度单独造成。

因此 FA-random 的偶然高总分不再是中心矛盾：它是“exact-electronic scaffold 可以补偿较差光学更新”的证据，而 FA-source 的关键证据是跨 scaffold 可运输的 phase 收益和接近 BP 的梯度方向。

## 7. 百层 / 千万相位参数工程验证

采用函数保持扩深：

`y = x + alpha * (OpticalStage(x) - x)`

`alpha=0` 只用于验证迁移前后输出严格一致；工程梯度测试使用 `alpha=0.01`，正式继续预训练必须将新增层 alpha 强制渐进升到 1。8 个 P11 anchor 保留宽度 96 的电子 mixer；新增层只保留 identity electronic skip 和标量深度门，因此电子参数几乎不随深度增长。

RTX 4090、batch=1、post-adapter 合成光场、activation checkpoint、一次 warmup + 一次独立梯度审计 + 两个计时 SGD step 的结果如下。该表只证明工程可训练性，不是 ImageNet 性能结果。

| 深度 | 相位参数 | 电子 backbone | 光学参数占比 | 峰值 allocated | step 时间 | 吞吐 | 新增 phase 非零梯度 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2,408,448 | 965,128 | 71.39% | 0.086 GiB | 0.115 s | 8.67 sample/s | 8/8 |
| 32 | 4,816,896 | 965,144 | 83.31% | 0.128 GiB | 0.213 s | 4.69 sample/s | 24/24 |
| 64 | 9,633,792 | 965,176 | 90.89% | 0.212 GiB | 0.344 s | 2.91 sample/s | 56/56 |
| 100 | 15,052,800 | 965,212 | 93.97% | 0.306 GiB | 0.548 s | 1.83 sample/s | 92/92 |

因此 64 层可以作为“近千万相位参数”主实验，100 层是“百层、1505 万相位参数”规模点。当前尚不能声称百层模型获得更强语义，因为 alpha 还没有在正式 ImageNet 继续预训练中升到 1，也没有验证移除新增层会导致性能下降。

全深度 feedback 实现已经补齐：16/32/64/100 层的每一个 stage 都有独立 connector，不循环复用 P11 的 8 个算子；支持 `bp_current`、`fa_source`、`fa_random`，source phase 持久化而 runtime feedback 不进入 `state_dict`。`fa_random` 使用逐层独立 SHA-256 派生 seed 和单位模随机相位，严格匹配对应复线性光场算子的谱与 Frobenius 范数；不把这一点过度表述为任意下游实 Jacobian 的全谱相同。服务器相关测试 15/15 通过。正式 resume 必须显式恢复并断言 feedback mode/seed，防止静默回到 BP。

## 8. 百层正式实验的下一步

不需要从头重训，但必须继续预训练新增相位；直接插入随机相位后推理只是在堆参数。建议采用 `8 -> 16 -> 32 -> 64 -> 100` 渐进增长：

1. 每个 `8 -> 16 -> 32 -> 64 -> 100` growth segment 训练 20 epoch；前 10 epoch 把本轮新增层从 `alpha=0.01` 线性升到 1，后 10 epoch 才是真正的 full-depth 训练；
2. 本轮新增 phase 保留 P11 成功使用的 `7e-3` 学习率，carried phase 使用 `1.75e-3`；新增/旧电子模块分别建议 `3.5e-4` / `1e-4`，迁移的 ImageNet head 为 `3e-4`；
3. 同一个 P11 epoch-88 起点同时做等数据预算的 8 层 continuation，并在累计 20/40/60/80 epoch 边界与深模型同步重置 optimizer/scheduler，排除“深模型只是多看了数据”的解释；
4. 全局有效 batch 固定为 192；根据 full-image CE smoke 选择 microbatch，并用 gradient accumulation 保持相同 optimizer update 与样本预算；
5. 每一段使用 epoch-20 `last.pt` 继续扩深，只有 `alpha=1` 的 checkpoint 有资格进入 `best_full_depth.pt` 或 backbone export；
6. 64 层完成四种方法和 3 seeds；100 层先完成 1 seed 的正式训练与下游迁移；深层 FA source 必须从已经训练好的完整深层 checkpoint 捕获，不能把迁移起点的随机新增 phase 冒充 pretrained source；
7. 报告所有新增 phase 的梯度覆盖、相位圆周位移、分层激活变化、reset/drop-stage 精度下降、ImageNet 曲线和下游迁移。

规模主表仍限制为四组：等训练预算的 8-layer BP、deep BP、deep FA-source、deep FA-random。若 100 层只完成单批反向传播，只能称为百层梯度可行性；只有 alpha=1、完整训练并完成迁移后，才能称为百层有效光学 backbone。

## 9. 复现入口与 Git 身份

- 多种子主队列与汇总：`commands/p12_downstream_fa_50e.sh`
- 无 ImageNet body 对照：`commands/p12_scratch_downstream_50e.sh`
- phase-only 四组：`commands/p12_phase_only_fa_50e.sh`
- P/E/H 机制审计：`commands/p12_mechanism_audit.sh`
- 16/32/64/100 工程 sweep：P13 的 `commands/03*_gpu_engineering*.sh`

相关主线提交：

- `638d96e0`：P13 CUDA engineering sweep；
- `d5487ac3`：P13 16/32/64/100 层全深度 fixed-feedback 与逐连接审计；
- `35233da6`：锁定正式 P12 worktree 的机制审计路径支持；
- `e7ae69e7`：phase-only 四组面板。

所有新启动命令均位于各自实验的 `commands/` 目录；正式训练、控制实验与机制审计使用互不覆盖的输出目录。

## 10. 2026-09-01 追加执行与百层论证边界

### 多 run 与 No-ImageNet body

No-ImageNet body 对照已经从下游 seed 2026 扩展到 2027、2028。三次下游
run 共用同一个 `init_seed=2026` 随机 body，因此这一批估计的是“给定同一
随机 backbone 起点时，数据划分、任务头初始化与训练随机性”的方差，不把
不同随机 backbone 初始化混进同一个误差条。服务器队列为 36 个状态：原
seed 2026 的 12 个结果保持只读复用，新增 seed 2027/2028 的 24 个结果使用
GPU 1/2 继续训练。若正文需要把“随机 backbone 初始化方差”也作为独立结论，
再另建 `init_seed=2027/2028` 两个 source panel；不要把两类随机性混称为普通
三 seed。

### phase-only 数值审计与续跑

phase-only seed 2026 中 Caltech、LSP 以及三个任务的 NoFT 共 9/12 项已经
完成。最初 ISIC 的三个适配任务在 epoch 1 前被重复 CUDA reduction 的
bitwise-equality 检查拒绝；该失败不是训练发散。首次修复后的 FA-source 在
epoch 9 又出现最大绝对差 `1.715e-4`，但相对 L2 仅 `2.164e-4`、余弦为
`0.99999999`，因此把单元素绝对门限校准为 `<=5e-4`；相对 L2 仍要求
`<=1e-3`、余弦仍要求 `>=0.99999`，并逐 tensor 记录。旧
NoFT/common-start 只通过一组明确 allow-list 的基础实现 SHA、面板 SHA、配置
摘要、task/seed、数据 manifest、source checkpoint 和完成 epoch 联合校验，
没有一般性的跳过身份检查。ISIC 的 BP、FA-source、FA-random 已分别在 GPU
0/3/5 重启并真实进入 epoch 1；专用入口为
`commands/p12_phase_only_isic_numeric_retry.sh`，拒绝覆盖任何已有
`result.json`。

### 为什么电子仍然使用 exact BP

当前论文问题不是证明“所有电子与光学模块都不需要反向传播”，而是避免为
深光路逐次获取、标定并实现当前物理伴随算子。因此正式定义保持：

- 光学 stage 之间的反馈连接比较 current BP、source-fixed 与 random-fixed；
- adapter、轻量 mixer、归一化、门控和读出头仍使用普通电子 BP；
- phase-only 面板冻结电子 backbone，专门判断高总分是否来自电子补偿；
- P/E/H 交换审计判断真正可运输的是 phase 还是 electronics/head。

把电子模块也强制改用随机反馈会引入另一个研究问题，而且不能更直接回答
“光学伴随传播难以部署”的痛点。更合适的强化方式是提高光学占比，而不是
故意削弱容易获得精确梯度的电子部分。

### 百层验证必须满足的四个证据层级

百层/1505 万相位参数可以成为抓眼的规模验证，但不能只靠构造模型或单批
反传。按以下层级报告，避免把工程可运行性写成语义有效性：

1. **函数保持迁移**：8→16→32→64→100 每次在新增 `alpha=0` 时，旧模型
   feature/logit 位级一致；这保证“继续加参数”不会先毁掉已有语义。
2. **alpha=1 全深度梯度**：在 BP、FA-source、FA-random 下，所有 carried
   与 new phase 都必须有 finite、non-zero 梯度，同时输入 amplitude 梯度
   存在；`alpha=0.01` 结果只算显存/吞吐工程点。
3. **ImageNet 继续预训练**：每次 growth 前 10 epoch 把新增层升到 1，后
   10 epoch 训练真正全深模型；同时运行相同样本/更新预算的 8 层 continuation。
4. **语义与迁移贡献**：报告 ImageNet Top-1、分层 phase motion、drop/reset
   新增层造成的精度下降，并在 Caltech/ISIC/LSP 四组下游协议中验证。只有
   这一层成立，才能称为“百层有效通用光学 backbone”。

还要严格区分两种 `FA-source`：growth 时新插入层尚未预训练，其固定反馈只能
称为 `source-init feedback`；只有先把完整 64/100 层 backbone 训练好，再冻结
其全深度算子用于下游适配，才是 `pretrained deep feedback`。正文关于“大模型
预训练算子复用”的主证据应来自后者。前者可以作为“扩深期间也不需要追踪最新
光学伴随算子”的附加实验，不能混为同一结论。

P13 的首次代码审查还要求正式 checkpoint 锁定 train/model/migration/optics
实现哈希、独立 DataLoader RNG、防止任一 stage 静默退回 BP，并为每一段
growth 提供只接受上一深度 `best_full_depth.pt` 的启动门。上述保护全部通过
CPU测试和真实 CUDA smoke 之前，不启动长时间 ImageNet 训练。

## 11. 2026-09-01 最新执行状态

### phase-only 四组已经全部完成

这里冻结 965,120 个电子 backbone 参数，只训练 1,204,224 个光学 phase 与任务头；任务头继续使用 exact BP。seed 2026 的 12/12 个结果如下：

| 任务 | Head-only / NoFT | BP | FA-source | FA-random |
|---|---:|---:|---:|---:|
| Caltech-101 Top-1 | **78.64%** | 78.59% | 78.52% | 78.31% |
| ISIC2016 mIoU | 81.71% | **82.45%** | **82.45%** | 82.33% |
| LSP PCK@0.1 | 50.29% | 61.71% | **61.71%** | 59.96% |

这个面板比 joint adaptation 更直接：电子 backbone 冻结后，FA-source 在 ISIC 与 LSP 仍分别只落后 BP 0.0007 和领先 0.007 个百分点；FA-random 在 LSP 低 1.74 个百分点。Caltech 的 phase 更新没有优于 head-only，说明该任务的当前协议对相位适配不敏感，不应拿它单独夸大算法收益。

### No-ImageNet 多 run 队列

固定同一个 `init_seed=2026` 随机 body、改变下游 seed 的 36 项队列当前完成 21 项、运行 2 项、等待 13 项、失败 0 项，GPU 1/2 继续执行。它估计的是给定同一个随机 body 时的数据划分、任务头与优化方差，不是独立 backbone 初始化方差。正文若要报告“随机初始化 backbone 的均值方差”，还必须另建 `init_seed=2027/2028` 两个独立 body panel。

### P13 百层路径的真实 CUDA 验证

P13 训练器在服务器通过 52 项测试。真实 ImageNet smoke 已完成官方 P11 epoch-88 双 checkpoint SHA 校验、在线冻结 Qwen stem、8→16 严格迁移、224×224 ImageNet、FA-source、两次 micro-batch、一次 optimizer update、验证及 alpha=1 backbone export。该 smoke 的 Top-1 因只看一个验证 batch 而没有性能意义。

在物理 GPU 4（RTX 4090，按 UUID 锁定）上，64/100 层、三种反馈方式均完成精确 `alpha=1` 全深度审计：

| 深度 | 方法 | 非零 phase 梯度 | 输入 amplitude 梯度 | 峰值 allocated | 平均 step |
|---:|---|---:|---|---:|---:|
| 64 | BP | 64/64 | finite、non-zero | 0.247 GiB | 0.387 s |
| 64 | FA-source | 64/64 | finite、non-zero | 0.250 GiB | 0.367 s |
| 64 | FA-random | 64/64 | finite、non-zero | 0.250 GiB | 0.374 s |
| 100 | BP | 100/100 | finite、non-zero | 0.362 GiB | 0.528 s |
| 100 | FA-source | 100/100 | finite、non-zero | 0.365 GiB | 0.536 s |
| 100 | FA-random | 100/100 | finite、non-zero | 0.365 GiB | 0.510 s |

这已经把结论从“小 alpha 探针”推进到“完整百层计算图可反传”，但仍只是合成 post-adapter 光场的工程证据。它不能替代 alpha=1 ImageNet 继续预训练、层贡献 reset/drop 和下游迁移。旧工程脚本还曾暴露 CUDA 逻辑序号与 `nvidia-smi` 物理序号不一致的问题；现已改为 GPU UUID 锁卡，上表只采用修复后在真实 RTX 4090-4 上重跑的结果。

### 当前正式推进顺序

1. 先用四张空卡完成 world-size=4、每 rank batch=24、累积 2 次的真实 DDP smoke，验证 global batch 192 的显存、NCCL、sampler 与在线 stem；
2. smoke 通过后启动 8→16 的 20-epoch FA-source growth，并同步运行等数据预算的 8-layer BP continuation；资源不足时优先 growth，随后补 control；
3. 16 层只能由 `best_full_depth.pt` 继续到 32，再依次到 64/100；
4. 100 层先完成一条完整 chain 作为 headline-scale demonstration；64 层再补三条独立 growth/优化 seeds。若三条都从同一个 P11 epoch-88 起点出发，只能估计 growth 随机性，不能称为三个独立预训练 backbone 初始化。

电子模块暂时保持 exact BP。研究痛点是深光路的当前物理伴随算子难以逐层获得，而轻量电子 mixer、gate、norm 与任务头的梯度在数字域可直接得到。若将电子模块也改成随机反馈，回答的会变成“全混合网络是否完全不需要 BP”这一不同问题。当前更干净的归因证据是 phase-only 四格与 P/E/H transplant/Shapley 审计；论文口径应写成 `fixed optical feedback with exact optimization of lightweight electronic modules`，不能写“整个网络不需要 BP”。

### P13 正式训练已启动

四卡真实 DDP smoke 随后已在物理 GPU 0/3/4/5 通过：world size 4、每 rank batch 24、gradient accumulation 2、有效 global batch 192，四个 rank 完成同一个 optimizer update，状态为 `complete`，无 OOM、NCCL 或 sampler 错误。

在此基础上已经启动正式 `8→16` FA-source growth：

- 服务器 PID：`3527459`；
- GPU：物理 `0,3,4,5`，均通过 UUID 映射，未使用 GPU 6 的 A100；
- 训练：20 epoch，前 10 epoch 将新增 8 层从 `alpha=0.02` 升到 1，后 10 epoch 全深度训练；
- batch：`24 × 4 ranks × accumulation 2 = 192`；
- 学习率：新增 phase `7e-3`，carried phase `3.5e-3`，新增/已有电子分别 `3.5e-4` / `2e-4`，临时 ImageNet head `5e-4`，均经过 warmup/cosine schedule；
- 首次监督时已经到 epoch 1、micro-batch 100/13345、optimizer step 50，四卡均有真实计算负载；
- 日志：`experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/logs/p13_growth16_fa_source_20e_gb192.latest.log`。

这条 run 使用增长起点捕获的新层反馈，因此当前严格称为 `source-init feedback`。只有完成 16→32→64/100 的 ImageNet 训练，再将完整深层算子冻结用于下游，才能称为 `pretrained deep feedback`。
