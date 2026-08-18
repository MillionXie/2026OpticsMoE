# CIFAR 高性能光学骨干

这个独立实验先用标准 BP 优化可复用的光学骨干。它不增加最终论文的方法数量；骨干性能和光学依赖达标后，才固定架构并回到 NoFT、BP、FA-pretrained、FA-random 四组比较。

核心入口：

- `configs/main.yaml`：A01 正式配置。
- `configs/smoke.yaml`：两批次端到端检查。
- `OPTIMIZATION_LOG.md`：每次优化动作、实测结果与去留结论。
- `FORMAL_EXPERIMENT_LOG.md`：唯一四组 fixed-feedback 公平协议与结果。
- `commands/`：服务器唯一推荐启动入口。

产物写入 `runs/<attempt>/seed_<seed>/`，包括断点、逐轮历史、验证集选出的 best checkpoint 和完整测试/光学消融结果。`runs/` 不应提交 Git。

截至 2026-08-19，预算内低分辨率电子残差候选 A13 的单 seed CIFAR-10 完整测试 Top-1 为
72.18%，关闭光路后 13.39%，随机/置换相位后为 12.48%/12.76%，归一化光学依赖 94.55%。
八层 optical gate 均不低于 0.50；residual electronic processing 为 312,336 参数，连同原
MLP readout 总电子参数 416,666。A13 是当前待多 seed 复验的性能骨干，不是新的反馈方法；
正式方法仍只有 NoFT、BP、FA-pretrained、FA-random 四组。详细动作、早停候选和消融见
`OPTIMIZATION_LOG.md`，结构与 RGB/归一化/读出解释见 `ARCHITECTURE.md`。
