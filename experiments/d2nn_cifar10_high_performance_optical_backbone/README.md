# CIFAR 高性能光学骨干

这个独立实验先用标准 BP 优化可复用的光学骨干。它不增加最终论文的方法数量；骨干性能和光学依赖达标后，才固定架构并回到 NoFT、BP、FA-pretrained、FA-random 四组比较。

核心入口：

- `configs/main.yaml`：A01 正式配置。
- `configs/smoke.yaml`：两批次端到端检查。
- `OPTIMIZATION_LOG.md`：每次优化动作、实测结果与去留结论。
- `FORMAL_EXPERIMENT_LOG.md`：唯一四组 fixed-feedback 公平协议与结果。
- `commands/`：服务器唯一推荐启动入口。

产物写入 `runs/<attempt>/seed_<seed>/`，包括断点、逐轮历史、验证集选出的 best checkpoint 和完整测试/光学消融结果。`runs/` 不应提交 Git。

截至 2026-08-18，最高准确率候选 A05 的 CIFAR-10 完整测试 Top-1 为 61.02%，关闭光路后 14.54%，归一化光学依赖 91.10%。CIFAR-100 预训练迁移 A04 的测试 Top-1 为 60.71%，关闭光路后 12.65%，归一化光学依赖 94.77%。建议保留 A05 作为性能参考，采用 A03→A04 作为正式固定反馈研究的主预训练骨干。
