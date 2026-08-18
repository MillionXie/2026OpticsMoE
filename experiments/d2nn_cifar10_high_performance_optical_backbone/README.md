# CIFAR 高性能光学骨干

这个独立实验先用标准 BP 优化可复用的光学骨干。它不增加最终论文的方法数量；骨干性能和光学依赖达标后，才固定架构并回到 NoFT、BP、FA-pretrained、FA-random 四组比较。

核心入口：

- `configs/main.yaml`：A01 正式配置。
- `configs/smoke.yaml`：两批次端到端检查。
- `OPTIMIZATION_LOG.md`：每次优化动作、实测结果与去留结论。
- `commands/`：服务器唯一推荐启动入口。

产物写入 `runs/<attempt>/seed_<seed>/`，包括断点、逐轮历史、验证集选出的 best checkpoint 和完整测试/光学消融结果。`runs/` 不应提交 Git。
