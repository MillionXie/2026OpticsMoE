# T11 光学终身学习

本任务把“连续到达的新任务 + 保留旧知识 + 光学路由均衡 + 无任务 ID 推理”作为一个可验证合同。当前提供依赖轻量、确定性的参考后端，用于先验证实验架构；后续可替换 `continual.py` 中的线性学习器为真实光学前端，而不改变数据流、回放和审计接口。

每个任务按顺序到达。模型只在当前任务和回放样本上更新，回放容量固定；每次更新记录物理槽位选择。正式候选必须检查每个任务的槽位覆盖和跨样本计数，报告平均准确率、当前任务准确率和遗忘量。推理接口不接收 task id，padding 样本必须由有效位掩码排除。

## 冒烟运行

```powershell
python -m LightGenV2.tasks.t11_lifelong_optics --config LightGenV2/tasks/t11_lifelong_optics/configs/smoke.json --out LightGenV2/tasks/t11_lifelong_optics/runs/smoke/t11_smoke
```

输出 `metrics.json` 和实际 `config.json`，后续正式 run 还应补齐命令、环境、Git commit 和数据 manifest。
