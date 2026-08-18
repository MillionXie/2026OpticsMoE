# 最近一轮执行优先级

完整研究架构见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。本文件只记录下一轮应按什么
顺序执行，避免同时改变过多因素。

## P0：固化现有 V2，不重跑

服务器 V2 的四组正式实验已经完成。先运行只读验证入口：

```bash
bash FixedFeedbackSFT/commands/01_verify_v2_results.sh
```

保存并报告：

- shared pretrained checkpoint digest；
- 三方法逐 epoch matched batch-order hash；
- fixed endpoint 和 validation-selected 两套任务指标；
- matched epoch 10/20/30 的 parameter delta；
- phase phasor distance/coherence；
- per-layer instantaneous gradient cosine；
- 20 层 residual optical weight trajectory。

现有主数字不需要通过重新训练“优化”。它们应作为冻结的 V2 baseline。

## P1：用现有 checkpoint 检查 optical bypass

这是下一次代码修改和服务器运行的首要任务。

1. 增加 evaluation-only optical occlusion：把 optical contribution 置零但不改 checkpoint；
2. 增加 evaluation-only phase permutation/noise；
3. 报告原始、occlusion、phase-shuffled 三种状态的 validation/test accuracy 和 embedding
   变化；
4. 输出每层 residual optical weight 和 occlusion sensitivity；
5. 先只分析现有 BP/pretrained checkpoint，不启动新训练。

判断规则：如果关闭或打乱光学路径几乎不影响结果，当前 V2 只能保留为算法 smoke，
不能作为光学机制主结果。

## P2：独立新建 residual/架构筛选实验

不要覆盖 V2。先用 seed 1234 比较：

- learned residual（现有 baseline）；
- fixed optical/skip = 0.35/0.65；
- optical-weight soft constraint；
- 更少 stage 或无 skip 的诊断配置。

筛选标准同时包括：训练稳定、源任务表示、光学 occlusion drop 和平均/最小 optical
weight。选出 1-2 个设置后再跑三个 matched seeds。

## P3：算子漂移 sweep

在选定的、确实使用光学路径的 backbone 上，从同一 checkpoint 出发逐项改变：

1. phase LR：0.0003、0.001、0.003；
2. horizon：5、10、20、30 epoch；
3. 独立 trust-region 配置。

主横轴使用 phase operator phasor distance，不把 raw-parameter relative drift 作为唯一
漂移指标。画出 operator distance、instantaneous gradient cosine、endpoint cosine 和
BP-FA task gap 的响应关系。

## P4：连接器结构消融

在同一 backbone 和同一微调协议下增加清晰的新方法名：

- `fa_identity_phase`；
- `fa_shuffled_pretrained`；
- `fa_noisy_pretrained_sigma_*`；
- `fa_periodic_refresh_K`；
- 保留现有 `fa_random`。

不能静默改变现有 `fa_pretrained` 定义。noisy-pretrained sweep 优先于构造完整
full-stage frozen Jacobian，因为它更直接检验 connector mismatch 的因果作用。

## P5：更强 backbone 与物理误差

完成 P1-P4 后再投入大规模预训练或硬件仿真：

- source-task CE/SupCon/self-supervised pretraining；
- teacher feature distillation 到 optical embedding；
- 新 readout 先冻结 backbone warm-up，再从共享 checkpoint 联合微调；
- SLM phase quantization/noise、CCD noise、配准和传播参数失配；
- current BP/PAT、fixed feedback、periodic refresh 的系统代价对比。

硬件版本必须明确 current forward、local phase update 和 error signal 分别如何观测或
计算。固定 feedback 不等于不需要层间状态，也不自动等于计算量低于 BP。
