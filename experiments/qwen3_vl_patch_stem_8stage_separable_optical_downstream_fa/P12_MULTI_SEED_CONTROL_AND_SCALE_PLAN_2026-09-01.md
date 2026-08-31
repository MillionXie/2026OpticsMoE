# P12 多随机种子、无预训练对照与深层光学扩展计划

日期：2026-09-01

本文冻结 2026-09-01 开始的三项增量工作：补齐 P12 下游迁移的多随机种子、增加无 ImageNet 光学骨干预训练的严格对照，以及把 P11 渐进扩展到约千万光学参数。本文是实验与口径记录，不替代各实验目录中的可执行配置、checkpoint SHA 和最终结果。

## 1. 当前 P12 主实验

主表仍只保留四组，不因后续诊断而增加方法列：

1. `NoFT`：冻结 P11 backbone，只训练临时任务头 50 epoch；
2. `BP-current`：从同一 common start 出发，使用当前前向算子的精确 BP 适配 50 epoch；
3. `FA-pretrained`：从同一 common start 出发，层间反馈固定为 ImageNet 预训练相位；
4. `FA-random`：从同一 common start 出发，层间反馈固定为可复现随机相位。

三个任务分别覆盖分类、分割和关键点定位：Caltech-101、ISIC-2016、LSP。每个任务的 common checkpoint、split、数据增强、batch、学习率、epoch 数和 checkpoint 选择规则在三种更新方法间完全相同。

Seed 2026 的首轮结果已经证明 P11 表征可迁移，并显示 `FA-pretrained` 在 ISIC 和 LSP 上几乎完全恢复 BP 增益；但单 seed 不能用于统计结论。首轮中 `FA-random` 也接近 BP，因此需要多 seed 和机制审计，不能只凭绝对性能猜测原因。

## 2. 2027/2028 多 seed 补齐

2026-09-01 00:24 CST 在服务器提交了缺失的 18 个适配 run：

```bash
P12_GPU_LIST=0,1,3,4,5 \
P12_ADAPTATION_SEEDS=2026,2027,2028 \
P12_REPO_ROOT=/DATA/DATA1/guest3/2026OpticsMoE_p12_e305e0b \
P12_POLL_SECONDS=20 \
P12_MAX_RETRIES=2 \
bash experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.sh launch
```

固定运行 worktree 为 commit `e305e0b`。队列先验证已有 18 个完整 artifact，再只补 seed 2027/2028 的三任务乘三种适配方法；相应 NoFT/common 已经存在。启动时五个物理 GPU 均由独立 worker 占用，并以 GPU UUID、PID、显存和 queue state 交叉确认。不要在该 worktree 内 pull 或修改科学代码，直至 36/36 完整结束。

正式汇总报告均值、样本标准差和逐 seed 配对差值。主要量为原始 test metric、`method - NoFT` 增益以及 FA 对 BP 增益的恢复比例；当 BP 增益接近零时不报告不稳定的恢复百分比。

## 3. 无 ImageNet 预训练的严格对照

实验名应写成 **Scratch-P11-body + frozen Qwen stem**，不能写成 fully scratch。冻结 Qwen Patch/Position Stem 与预训练组保持 byte-identical，因为本课题采用它作为固定的图像 token 化前端；随机初始化且不加载 P11 checkpoint 的部分包括：

- 1024 到 224 的 TokenAdapter；
- 8 个三 bank 光学相位面；
- 8 个 width-96 Slim Spatial Token Mixer；
- 各层光电融合门和 mixer 门；
- 下游临时读出头。

实现上一次性导出 seed 2026 的未训练 P11 body source artifact，登记文件 SHA、stem SHA、模型 signature、参数报告和初始化 seed。随后复用 P12 的 strict source loader、common-start、训练循环和队列，不在核心训练代码中增加特殊逻辑。scratch 与 pretrained 使用不同 output root，禁止共用 common checkpoint。

首个工程 gate 为三个任务、seed 2026、同样四组共 12 个 run。scratch 语境中的 `FA-pretrained` 在论文中改称 `FA-init`，即固定使用该随机初始化 source 的相位作为反馈。若工程 gate 正常，再扩成三任务乘三 seed 乘四组，与预训练 P12 逐 seed 配对。

这个对照回答“P11 的 ImageNet 光学 backbone 预训练是否改善下游迁移”。它不回答“完全没有任何预训练视觉前端能否训练”，也不是等训练算力比较：预训练组额外看过 90 epoch ImageNet，因此结果必须按预训练收益解释。

## 4. FA-random 的光学/电子贡献审计

电子模块继续采用精确 BP。其理由是实际部署中电子算子的导数可直接由软件获得，难点是光学传播链的反向或物理伴随；强制电子模块也采用 FA 会额外降低性能，却不能更直接回答光学训练问题。

为了排除“FA-random 只靠电子参数”的混淆，每个已训练 endpoint 做一个无需重训的 2×2 checkpoint 反事实：

| 光学相位 | 非光学可训练状态 | 解释 |
| --- | --- | --- |
| common | common | 共同起点 |
| adapted | common | 只移植训练后相位 |
| common | adapted | 只移植训练后 adapter/mixer/gate/head |
| adapted | adapted | 完整 endpoint |

四种组合在同一数据 split 上推理，报告相对 common 的性能、两类主效应和交互项，并同时记录 feature drift。它是机制诊断，不扩展四组主表。必要时再增加逐 stage phase reset / electronic reset，定位哪些层真正贡献性能。

## 5. 深层、千万级光学参数扩展

现有几何中每层独立光学参数为：

```text
3 banks x 224 x 224 = 150,528 phase parameters
```

因此准确规模为：

| 深度 | 独立光学相位参数 |
| ---: | ---: |
| 8 | 1,204,224 |
| 16 | 2,408,448 |
| 32 | 4,816,896 |
| 64 | 9,633,792 |
| 100 | 15,052,800 |

约千万参数的主规模应选择 64 层；100 层可以作为 15M 参数的进一步扩展，不能把它误写成 10M。当前实现每层包含相位调制、轴向传播、CCD 强度探测、归一化/非线性、光电残差融合和重新加载，因此准确物理口径是多级或时分复用的 OEO token/channel mixer，而不是 100 个纯被动 D2NN 相位面。

### 5.1 不能直接随机插层

单个 OEO stage 包含 square-law 探测，不是 identity，也不可通过简单复制得到 Net2Deeper 的严格等价层。直接插入 56 或 92 个 active stage 会破坏 P11 已学表示。新增 stage 外使用生长门：

```text
y = x + alpha * (Stage(x) - x)
```

`alpha=0` 时插层严格为 identity，原 8 层 P11 函数保持；`alpha>0` 后新相位才逐步参与。训练需要把 alpha 从 0/很小值逐步推到 1，并在最终 `alpha=1` 下复测。如果 gate 长期接近 0，就不能宣称深层光路有效。

### 5.2 电子预算

不能给 64/100 层各复制一个独立 width-96 mixer，否则电子参数约为 6.1M/9.4M，违反残差电子总量 1--2M 的预算。主设计只保留从 P11 迁移的 8 个 anchor mixer；新增层使用无参数 identity electronic skip，只增加光学相位和极少量标量 gate。预计唯一电子 backbone 参数仍约 0.965M：

- 64 层光学占可训练 backbone 参数约 90.9%；
- 100 层光学占可训练 backbone 参数约 94.0%。

同时必须单独披露 OEO 的 CCD、归一化和 reload 次数，因为“唯一电子参数少”不等同于“电子操作次数少”。

### 5.3 验证顺序

1. 工程等价：64/100 层在新增 alpha=0 时与 P11 final feature/logit 一致；
2. 可训练性：alpha>0 时所有新增相位梯度有限且非零；
3. 资源 sweep：8/16/32/64/100 的峰值显存、吞吐和单 step 时间；
4. 5--10 epoch ImageNet continued-pretrain screen，优先比较 8/16/32/64；
5. 证明新增层被使用：逐层 phase drift/gradient norm、强制 alpha=1、截断或随机化新增层；
6. 只有深层 BP ceiling 正常后，才把最优深层 source 带入四组 FA 下游实验。

100 层 activation 远大于 8 层。正式训练前必须加入分段 activation checkpoint，并验证 checkpoint on/off 的输出和梯度一致；还应把当前形式上使用二维 FFT 的轴向传播改成真正单轴 FFT，并同时覆盖 BP 与固定反馈 connector。未经这些工程验证，不直接排 90 epoch 百层训练。

## 6. 结论边界

- 多 seed P12 回答固定反馈相对当前 BP 的下游适配能力；
- scratch-P11 对照回答 ImageNet optical-backbone pretraining 的迁移收益；
- 2×2 swap 回答 FA-random 的收益主要来自光学、电子还是交互；
- 64/100 层扩展回答方法是否在更深、更高光学参数量下仍可训练。

四者互补，但不能互相替代。特别是“能够实例化并反传 100 层”只属于工程可训练性证据；只有在 alpha 最终激活、性能不退化且新增层干预会造成可重复性能下降时，才可以把它写成有效的深层光学 backbone 结果。
