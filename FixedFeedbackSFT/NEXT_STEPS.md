# 最近一轮执行优先级：先性能，再光学比例

完整方案见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。当前 V1/V2 固定反馈结果保留，
但暂停新增 FA 训练，直到新 backbone 通过性能和光学依赖两道 gate。

## P0：锁定已有结果

已有 V2 不重跑，只作为当前 31% baseline 和反馈实现正确性证据：

```bash
bash FixedFeedbackSFT/commands/01_verify_v2_results.sh
```

## P1：新建性能 backbone 实验

新建独立实验目录，不能覆盖 V1/V2。第一任务固定为 CIFAR-10 direct supervised
classification：

```text
RGB/grayscale input
-> optical OEO backbone
-> compact electronic readout
-> 10 logits
-> cross-entropy
```

第一轮先完成以下最小筛选：

1. grayscale vs RGB optical encoding；
2. 4/8/12/20 stage；
3. learned residual vs fixed 0.5/0.5 vs no skip；
4. GAP+linear vs small MLP readout。

每个运行都必须同时保存 normal accuracy、optical-off accuracy 和 phase-random
accuracy。首轮单 seed，达到候选线后再扩展 seeds。

## P2：达到性能 gate

阶段目标使用完整 CIFAR-10 test split：

- 最低工程线：50%；
- 可展示线：60%；
- 较好目标：65%；
- 强目标：70%。

若最佳结构低于 60%，按顺序检查：

1. RGB 信息是否丢失；
2. 20-stage 是否因重复探测/重载导致退化；
3. propagation padding/band-limit；
4. phase initialization 和 LR/scheduler；
5. normalization 与 readout；
6. 最后才加入 distillation。

未通过 60% gate 前不做新 fixed-feedback 主实验。

## P3：达到光学依赖 gate

对达到 60% 的模型计算：

```text
normalized_optical_dependence
  = (Acc_full - Acc_optical_off) / (Acc_full - Acc_chance)
```

并报告 phase-random、phase-shuffle、phase-noise、head-only、每层分支能量和 residual
weight。建议进入下一阶段的条件：

- normalized optical dependence >= 0.5；
- phase 随机/打乱造成稳定性能下降；
- 平均 optical weight 不再接近 0.07；
- 小电子 head 不能单独复现 full-model 性能。

如果性能高但光学依赖低，优先固定/约束 residual、缩小电子 head 或减少旁路，不把
这个模型直接用于 FA 论文主结果。

## P4：固化 pretrained optical backbone

在 accuracy-optical-dependence Pareto 前沿选择：

- 一个主配置；
- 一个低计算量配置。

正式运行五个 seeds，保存 config/model/operator digest。随后构造：

1. CIFAR-10 source checkpoint，用于 CIFAR-10-C；
2. CIFAR-100 source checkpoint，用于 CIFAR-100 -> CIFAR-10。

## P5：恢复固定反馈实验

只有 P2-P4 完成后再比较：

- NoFT；
- head-only；
- BP；
- FA-pretrained；
- FA-random。

此时再做 phase LR/horizon/trust-region 的 operator drift sweep，以及 noisy/shuffled/
identity/periodic-refresh connector 消融。

## 命令维护约定

新性能实验必须在其 `commands/` 中提供：

```text
01_prepare_data.sh
02_smoke.sh
03_train_bp_backbone.sh
04_evaluate_optical_dependence.sh
05_aggregate.sh
COMMANDS.md
```

每次源代码或 config 修改都提交并推送 Git，再由服务器 `git pull --ff-only` 同步。
